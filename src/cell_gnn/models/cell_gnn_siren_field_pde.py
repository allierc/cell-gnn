import numpy as np
import torch
import torch.nn as nn
from cell_gnn.models.MLP import MLP
from cell_gnn.models.Siren_Network import Siren
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_model


@register_model("particle_spring_force_diffusion_field_siren_pde")
class CellGNNSirenFieldPDE(nn.Module):
    """GNN for pair forces + vector SIREN field + PDE residual loss.

    d_pos = GNN_pair(edges) + SIREN(x, y, z, t)[:, 1:]

    The SIREN outputs dim+1 channels:
      - channel 0: scalar concentration c(x,t) — used for PDE loss only
      - channels 1..dim: chemotactic velocity vector

    PDE residual (computed via autograd on channel 0):
      R = dc/dt - D * laplacian(c) + lambda * c [- S(x,t)]

    This combines the direct vector-output simplicity of CellGNNSirenField
    with the PDE physics constraint of CellGNNSirenGradFieldPDE.
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super().__init__()

        self.aggr_type = aggr_type

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update
        self.model = model_config.cell_model_name
        self.n_dataset = train_config.n_runs
        self.dimension = dimension
        self.delta_t = simulation_config.delta_t
        self.n_cells = simulation_config.n_cells
        self.embedding_dim = model_config.embedding_dim
        self.n_frames = simulation_config.n_frames
        self.prediction = model_config.prediction
        self.bc_dpos = bc_dpos
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.reflection_augmentation = train_config.reflection_augmentation
        self.recursive_loop = train_config.recursive_loop
        self.state = simulation_config.state_type
        self.remove_self = train_config.remove_self
        self.sigma = simulation_config.sigma
        self.n_ghosts = int(train_config.n_ghosts)
        self.cell_of_interest = 0

        # --- PDE coefficients ---
        fp = getattr(simulation_config, 'field_params', {}) or {}
        def _fp_get(key, default):
            if isinstance(fp, dict):
                return fp.get(key, default)
            return getattr(fp, key, default)
        self.pde_D = float(_fp_get('diffusion_coeff', 0.01))
        self.pde_lambda = float(_fp_get('lambda_decay', 0.1))

        # --- Source-aware PDE residual (optional) ---
        self.pde_source_aware = bool(getattr(train_config, 'pde_source_aware', False))
        self.pde_alpha = float(_fp_get('source_strength', 0.0))
        self.pde_source_fraction = float(_fp_get('source_fraction', 1.0))
        self.pde_pulse_period = int(_fp_get('pulse_period', 0))
        self.pde_pulse_duty = float(_fp_get('pulse_duty', 1.0))
        self.pde_seed = int(getattr(simulation_config, 'seed', 0))
        self.register_buffer('_src_mask', torch.zeros(0, dtype=torch.bool), persistent=False)
        self.register_buffer('_src_phase', torch.zeros(0, dtype=torch.float32), persistent=False)

        # --- GNN branch ---
        self.input_size = self.dimension + 1 + self.embedding_dim
        self.embedding_trial = config.training.embedding_trial

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size,
                            nlayers=self.n_layers, hidden_size=self.hidden_dim, device=self.device)

        if self.update_type == 'mlp':
            self.input_size_update = self.dimension + self.embedding_dim + self.output_size
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update,
                               nlayers=self.n_layers_update, hidden_size=self.hidden_dim_update,
                               device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_cells) + self.n_ghosts, self.embedding_dim)),
                         device=self.device, requires_grad=True, dtype=torch.float32))

        # --- SIREN: (x, y, z, t) -> (c, vx, vy, vz) ---
        n_layers_field = getattr(model_config, 'n_layers_field', 3)
        hidden_dim_field = getattr(model_config, 'hidden_dim_field', 128)
        omega_field = getattr(model_config, 'omega_field', 30.0)

        self.siren_field = Siren(
            in_features=self.dimension + 1,
            out_features=self.dimension + 1,  # c + velocity vector
            hidden_features=hidden_dim_field,
            hidden_layers=max(n_layers_field - 2, 1),
            outermost_linear=True,
            first_omega_0=omega_field,
            hidden_omega_0=omega_field,
        ).to(self.device)

        # Filled by forward(); trainer reads these.
        self.last_pde_residual = None
        self.last_pde_reg = None

    def forward(self, state: CellState, edge_index: torch.Tensor, data_id=[], training=[], has_field=False, k=[]):

        self.data_id = data_id
        self.training = training
        self.has_field = has_field

        if self.remove_self:
            edge_index = remove_self_loops(edge_index)

        pos = state.pos
        d_pos = state.vel / self.vnorm

        if self.rotation_augmentation & self.training:
            if self.dimension == 2:
                self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
                self.rotation_matrix = torch.stack([
                    torch.stack([torch.cos(self.phi), torch.sin(self.phi)]),
                    torch.stack([-torch.sin(self.phi), torch.cos(self.phi)])
                ])
                self.rotation_matrix = self.rotation_matrix.permute(
                    *torch.arange(self.rotation_matrix.ndim - 1, -1, -1)).squeeze()
            else:
                rand_matrix = torch.randn(3, 3, dtype=torch.float32, device=self.device)
                q, r = torch.linalg.qr(rand_matrix)
                q = q * torch.sign(torch.diag(r))
                if torch.det(q) < 0:
                    q[:, 0] = -q[:, 0]
                self.rotation_matrix = q

            d = self.dimension
            d_pos[:, :d] = d_pos[:, :d] @ self.rotation_matrix

        cell_id = state.index.unsqueeze(-1)
        embedding = self.a[self.data_id.clone().detach(), cell_id, :].squeeze()

        # --- GNN pair force branch ---
        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = pos[dst], pos[src]
        d_pos_i, d_pos_j = d_pos[dst], d_pos[src]
        embedding_i, embedding_j = embedding[dst], embedding[src]

        messages = self.message(dst, src, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j)

        n_nodes = pos.shape[0]
        out_pair = scatter_aggregate(messages, dst, n_nodes, self.aggr_type)

        if self.update_type == 'mlp':
            out_pair = self.lin_phi(torch.cat((out_pair, embedding, d_pos), dim=-1))

        # --- SIREN field branch ---
        if not torch.is_tensor(k):
            k = torch.full((pos.shape[0], 1), float(k), dtype=pos.dtype, device=pos.device)
        else:
            k = k.to(pos.dtype)

        with torch.enable_grad():
            pos_field = pos.detach().requires_grad_(True)
            t_field = (k * self.delta_t).detach().requires_grad_(True)
            siren_input = torch.cat([pos_field, t_field], dim=1)

            siren_out = self.siren_field(siren_input)  # (N, dim+1)
            c = siren_out[:, 0:1]          # scalar concentration for PDE
            out_field = siren_out[:, 1:]    # velocity vector

            # --- PDE residual on scalar c ---
            dc_dt = torch.autograd.grad(
                outputs=c, inputs=t_field,
                grad_outputs=torch.ones_like(c),
                create_graph=True, retain_graph=True,
            )[0]

            grad_c = torch.autograd.grad(
                outputs=c, inputs=pos_field,
                grad_outputs=torch.ones_like(c),
                create_graph=True, retain_graph=True,
            )[0]

            laplacian_c = torch.zeros_like(c)
            for i in range(self.dimension):
                d2c_dxi2 = torch.autograd.grad(
                    outputs=grad_c[:, i].sum(),
                    inputs=pos_field,
                    create_graph=True, retain_graph=True,
                )[0][:, i:i+1]
                laplacian_c = laplacian_c + d2c_dxi2

            pde_residual = dc_dt - self.pde_D * laplacian_c + self.pde_lambda * c

            # Optional source term
            if self.pde_source_aware and self.pde_alpha > 0:
                n_cells = pos.shape[0]
                if self._src_mask.numel() != n_cells:
                    gen = torch.Generator(device=pos.device)
                    gen.manual_seed(self.pde_seed)
                    n_src = max(1, int(round(self.pde_source_fraction * n_cells)))
                    perm = torch.randperm(n_cells, generator=gen, device=pos.device)
                    mask = torch.zeros(n_cells, dtype=torch.bool, device=pos.device)
                    mask[perm[:n_src]] = True
                    phase = torch.rand(n_cells, generator=gen, device=pos.device)
                    self._src_mask = mask
                    self._src_phase = phase

                if self.pde_pulse_period and self.pde_pulse_period > 0:
                    k_col = k if k.dim() == 2 else k.unsqueeze(-1)
                    step = k_col.squeeze(-1)
                    phase_t = (step / float(self.pde_pulse_period)
                               + self._src_phase) % 1.0
                    d_ = torch.minimum(phase_t, 1.0 - phase_t)
                    sigma_p = max(self.pde_pulse_duty, 1e-3) / 2.0
                    pulse = torch.exp(-0.5 * (d_ / sigma_p) ** 2)
                    pulse = pulse * self._src_mask.float()
                    gain = 1.0 / max(self.pde_pulse_duty, 1e-3)
                    source_rate = (self.pde_alpha * gain * pulse).unsqueeze(-1)
                else:
                    source_rate = (self.pde_alpha * self._src_mask.float()).unsqueeze(-1)

                pde_residual = pde_residual - source_rate

        # Stash for trainer
        self.last_pde_residual = pde_residual
        self.last_pde_reg = c

        # Un-rotate GNN branch only
        if self.rotation_augmentation & self.training:
            d = self.dimension
            out_pair[:, :d] = out_pair[:, :d] @ self.rotation_matrix.T

        out = out_pair + out_field
        return out

    def message(self, dst, src, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j):
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        if self.rotation_augmentation & self.training:
            d = self.dimension
            delta_pos[:, :d] = delta_pos[:, :d] @ self.rotation_matrix

        in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
        out = self.lin_edge(in_features)

        if self.training == False:
            if out.shape[0] == 0:
                self.msg = torch.zeros(1, out.shape[1], device=out.device) if out.dim() > 1 else torch.zeros(1, device=out.device)
            else:
                pos = torch.argwhere(dst == self.cell_of_interest)
                if pos.numel() > 0:
                    self.msg = out[pos[:, 0]]
                else:
                    self.msg = out[0]
        return out

    def psi(self, r, p1, p2=None):
        k_rep, r0, kadh, r_on, delta, mu_f = p1[0], p1[1], p1[2], p1[3], p1[4], p1[5]
        delta_safe = max(delta, 1e-8)
        F_rep = k_rep * torch.relu(r0 - r)
        g_on = torch.sigmoid((r - r0) / delta_safe)
        g_off = torch.sigmoid(-(r - r_on) / delta_safe)
        F_adh = -kadh * g_on * g_off * (r - r0)
        return mu_f * (F_rep + F_adh)
