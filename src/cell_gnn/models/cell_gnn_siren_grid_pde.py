import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_gnn.models.MLP import MLP
from cell_gnn.models.Siren_Network import Siren
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_model


@register_model("particle_spring_force_diffusion_field_siren_grid_pde")
class CellGNNSirenGridPDE(nn.Module):
    """GNN pair forces + SIREN scalar field on a FIXED MESH + learned PDE dynamics.

    Structural differences from ``CellGNNSirenGradFieldPDE``:

      1. SIREN is evaluated on a fixed spatial grid each forward pass
         (instead of at particle positions).  Particles receive c and
         grad_x c via trilinear interpolation.

      2. Source is spread from emitting particles onto the grid via a
         normalised Gaussian kernel.

      3. A small MLP ``lin_pde`` learns the local PDE law:

             dc/dt  ≈  lin_pde([ c,  Laplacian(c),  source ])

         evaluated per grid point (shared weights).  The PDE residual

             R = dc/dt_siren - lin_pde([ c, Laplacian(c), source ])

         is stored in ``self.last_pde_residual`` for the trainer to add
         to the loss.

    Particle velocity:
        v_i = GNN_pair_i + mu_chem * grad_c_interp_i
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

        fp = getattr(simulation_config, 'field_params', {}) or {}
        def _fp_get(key, default):
            if isinstance(fp, dict):
                return fp.get(key, default)
            return getattr(fp, key, default)

        # --- Fixed spatial grid ---
        # Domain assumed to be [0, 1]^D (periodic boundary in the config).
        # grid_size: scalar or list of ints, per-axis count of grid points.
        gs = _fp_get('model_grid_size', None)
        if gs is None:
            # Fall back to generator's grid_resolution but much coarser for training cost.
            gs = _fp_get('grid_resolution', 32)
            gs = min(int(gs), 48)
        if isinstance(gs, (list, tuple)):
            self.grid_shape = tuple(int(x) for x in gs)
        else:
            self.grid_shape = tuple([int(gs)] * self.dimension)
        assert len(self.grid_shape) == self.dimension, "grid_shape must match dimension"

        box_lo = float(_fp_get('box_lo', 0.0))
        box_hi = float(_fp_get('box_hi', 1.0))
        self.box_lo = box_lo
        self.box_hi = box_hi

        # Grid coords: (G, D) flat list of grid-point positions in physical space.
        axes = [torch.linspace(box_lo, box_hi, n, device=device) for n in self.grid_shape]
        mesh = torch.meshgrid(*axes, indexing='ij')
        grid_coords = torch.stack([m.reshape(-1) for m in mesh], dim=-1)  # (G, D)
        self.register_buffer('grid_coords', grid_coords, persistent=False)

        # Grid spacing (assumes uniform per axis; >1 point per axis).
        self.dx = torch.tensor(
            [(box_hi - box_lo) / max(n - 1, 1) for n in self.grid_shape],
            device=device, dtype=torch.float32,
        )

        # --- Source-schedule parameters (match generator's pulsed-source logic) ---
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

        # --- SIREN scalar field: (x, y, z, t) -> c ---
        n_layers_field = getattr(model_config, 'n_layers_field', 3)
        hidden_dim_field = getattr(model_config, 'hidden_dim_field', 128)
        omega_field = getattr(model_config, 'omega_field', 30.0)

        self.siren_field = Siren(
            in_features=self.dimension + 1,
            out_features=1,
            hidden_features=hidden_dim_field,
            hidden_layers=max(n_layers_field - 2, 1),
            outermost_linear=True,
            first_omega_0=omega_field,
            hidden_omega_0=omega_field,
        ).to(self.device)

        # --- Learned PDE MLP: [c, laplacian_c, source] -> dc/dt ---
        pde_hidden = int(getattr(model_config, 'pde_hidden_dim', 32))
        pde_layers = int(getattr(model_config, 'pde_n_layers', 3))
        self.lin_pde = MLP(
            input_size=3,
            output_size=1,
            nlayers=pde_layers,
            hidden_size=pde_hidden,
            device=self.device,
        )

        # --- Learnable chemotaxis parameters (linear OR saturating) ---
        # Match the generator: if chem_saturation_scale is null/<=0 the true
        # dynamics is linear (F = mu * grad c), so the model uses the linear
        # form too — no log1p squashing that would stall early training.
        init_mu = float(_fp_get('mu_chem', 0.1))
        init_g0 = _fp_get('chem_saturation_scale', None)
        self.linear_chemotaxis = (init_g0 is None) or (float(init_g0) <= 0.0)
        self.log_mu_chem = nn.Parameter(torch.tensor(np.log(init_mu), dtype=torch.float32, device=self.device))
        if not self.linear_chemotaxis:
            self.log_chem_g0 = nn.Parameter(torch.tensor(np.log(float(init_g0)), dtype=torch.float32, device=self.device))

        # Filled by forward().
        self.last_pde_residual = None
        self.last_pde_reg = None
        self.last_scalar_field = None
        self.last_source_field = None

    # -------------------- grid helpers --------------------

    def _finite_diff_laplacian(self, c_grid):
        """Compute Laplacian via central differences on a D-D grid.

        c_grid: (*grid_shape,) tensor.  Returns same shape.
        Uses replicate padding (Neumann-like) so the interior stays clean;
        fine for periodic runs because sources rarely sit at the boundary.
        """
        d = self.dimension
        # Add singleton batch + channel dims for F.pad convenience.
        x = c_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, *grid)
        lap = torch.zeros_like(c_grid)
        for axis in range(d):
            # Build central-difference stencil along axis only.
            # Shift forward and backward along the axis.
            dx_axis = self.dx[axis]
            # Use torch.roll with replicate padding manually: take slices.
            size_a = c_grid.shape[axis]
            # Forward-shifted (pad end with replicate)
            fwd = c_grid.roll(shifts=-1, dims=axis)
            bwd = c_grid.roll(shifts=+1, dims=axis)
            # Replicate BCs: overwrite the wrapped edges with their inside neighbour.
            idx_first = [slice(None)] * d
            idx_last = [slice(None)] * d
            idx_first[axis] = 0
            idx_last[axis] = -1
            # At axis==0 after roll(-1), the *last* slice of fwd came from index 0 (wrap).
            fwd_fix = fwd.clone()
            bwd_fix = bwd.clone()
            # Replace wrapped boundary with the edge value (replicate padding).
            idx_edge_fwd = [slice(None)] * d
            idx_edge_bwd = [slice(None)] * d
            idx_edge_fwd[axis] = -1
            idx_edge_bwd[axis] = 0
            fwd_fix[tuple(idx_edge_fwd)] = c_grid[tuple(idx_edge_fwd)]
            bwd_fix[tuple(idx_edge_bwd)] = c_grid[tuple(idx_edge_bwd)]
            lap = lap + (fwd_fix - 2.0 * c_grid + bwd_fix) / (dx_axis * dx_axis)
        return lap

    def _finite_diff_gradient(self, c_grid):
        """Central-difference gradient on the grid -> (D, *grid_shape)."""
        d = self.dimension
        grads = []
        for axis in range(d):
            dx_axis = self.dx[axis]
            fwd = c_grid.roll(shifts=-1, dims=axis)
            bwd = c_grid.roll(shifts=+1, dims=axis)
            # Replicate boundary.
            idx_edge_fwd = [slice(None)] * d
            idx_edge_bwd = [slice(None)] * d
            idx_edge_fwd[axis] = -1
            idx_edge_bwd[axis] = 0
            fwd = fwd.clone()
            bwd = bwd.clone()
            fwd[tuple(idx_edge_fwd)] = c_grid[tuple(idx_edge_fwd)]
            bwd[tuple(idx_edge_bwd)] = c_grid[tuple(idx_edge_bwd)]
            grads.append((fwd - bwd) / (2.0 * dx_axis))
        return torch.stack(grads, dim=0)  # (D, *grid_shape)

    def _trilinear_interp(self, field_chan, pos):
        """Interpolate a multi-channel field at particle positions.

        field_chan: (C, *grid_shape) tensor, values at grid points.
        pos:       (N, D) particle positions in physical coords.
        returns:   (N, C)

        Uses F.grid_sample with align_corners=True so grid corners map
        exactly to the physical box corners.
        """
        N = pos.shape[0]
        d = self.dimension
        C = field_chan.shape[0]

        # Normalise positions to [-1, 1].
        # F.grid_sample expects coords as (x, y, z) for last dim in 3D.
        # Our grid_shape is in (axis_0, axis_1, ..., axis_{D-1}) order which
        # we defined along the same axes as pos components. So normalise
        # each component and reverse the order to match grid_sample (which
        # reads last-axis-of-input as first-coord-of-grid).
        u = 2.0 * (pos - self.box_lo) / max(self.box_hi - self.box_lo, 1e-12) - 1.0  # (N, D)

        if d == 2:
            # field_chan: (C, H, W) with axis0 = y (H), axis1 = x (W).
            # We stored grid with axes[0]=x, axes[1]=y, so need to transpose:
            inp = field_chan.permute(0, 2, 1).unsqueeze(0)  # (1, C, axis1, axis0) -> (1, C, H, W)
            # grid_sample wants grid shape (1, H_out, W_out, 2) with (x, y).
            grid = u.view(1, 1, N, 2)  # (1, H_out=1, W_out=N, 2)
            out = F.grid_sample(inp, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
            # out: (1, C, 1, N) -> (N, C)
            return out.squeeze(2).squeeze(0).transpose(0, 1)
        elif d == 3:
            # field_chan: (C, Nx, Ny, Nz).  grid_sample 3D wants (1, C, D, H, W)
            # where D, H, W correspond to z, y, x respectively.
            # Permute axes (x, y, z) -> (z, y, x).
            inp = field_chan.permute(0, 3, 2, 1).unsqueeze(0)  # (1, C, Nz, Ny, Nx)
            grid = u.view(1, 1, 1, N, 3)  # (1, D_out=1, H_out=1, W_out=N, 3); last dim = (x, y, z)
            out = F.grid_sample(inp, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
            # out: (1, C, 1, 1, N) -> (N, C)
            return out.squeeze(3).squeeze(2).squeeze(0).transpose(0, 1)
        else:
            raise NotImplementedError(f"Trilinear interp only supports D=2,3; got {d}")

    def _particles_to_grid(self, pos, emission_rate):
        """Cloud-in-cell (trilinear) deposit of per-particle emission onto the grid.

        Each particle sits inside one voxel bounded by 2^D grid corners;
        its rate is divided by the cell volume (giving a volumetric source
        density) and split across those corners with trilinear weights —
        the transpose of the trilinear read-back used for grad c.

        pos:             (N, D) particle positions in [box_lo, box_hi]
        emission_rate:   (N,) or (N, 1) per-particle rate at this instant.
        returns:         (*grid_shape,) source density on the grid.

        Matches the generator's ``deposit_to_grid_weighted`` (same stencil
        as the read-back, so total deposited mass ≈ sum of emission rates /
        cell volume exactly when particles are interior).
        """
        if emission_rate is None:
            return torch.zeros(self.grid_shape, device=pos.device, dtype=pos.dtype)

        d = self.dimension
        e = emission_rate.reshape(-1)  # (N,)

        # Normalise positions to [0, N-1] index space (align_corners grid).
        # Grid has N nodes spanning [box_lo, box_hi] with spacing dx=1/(N-1).
        L = max(self.box_hi - self.box_lo, 1e-12)
        sizes = torch.tensor([s for s in self.grid_shape], device=pos.device, dtype=pos.dtype)
        u = (pos - self.box_lo) / L                     # [0, 1]
        scaled = u * (sizes - 1.0)                      # [0, N-1]
        idx0 = scaled.floor().long().clamp(min=0)       # (N, D)
        # Make sure idx0+1 is a valid index (clamp just below top node).
        idx0 = torch.minimum(idx0, (sizes.long() - 2).unsqueeze(0))
        idx1 = idx0 + 1
        frac = scaled - idx0.to(scaled.dtype)           # (N, D) in [0, 1]

        # Volumetric density = per-particle rate / cell volume.
        cell_volume = torch.prod(self.dx).item()
        density = e / max(cell_volume, 1e-24)           # (N,)

        S = torch.zeros(self.grid_shape, device=pos.device, dtype=pos.dtype)

        if d == 3:
            # Grid stored as field[ix, iy, iz] (axes from meshgrid in __init__).
            ix0, iy0, iz0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
            ix1, iy1, iz1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]
            fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
            corners = [
                (ix0, iy0, iz0, (1 - fx) * (1 - fy) * (1 - fz)),
                (ix1, iy0, iz0, fx       * (1 - fy) * (1 - fz)),
                (ix0, iy1, iz0, (1 - fx) * fy       * (1 - fz)),
                (ix1, iy1, iz0, fx       * fy       * (1 - fz)),
                (ix0, iy0, iz1, (1 - fx) * (1 - fy) * fz),
                (ix1, iy0, iz1, fx       * (1 - fy) * fz),
                (ix0, iy1, iz1, (1 - fx) * fy       * fz),
                (ix1, iy1, iz1, fx       * fy       * fz),
            ]
            for ix, iy, iz, w in corners:
                S.index_put_((ix, iy, iz), density * w, accumulate=True)
        elif d == 2:
            ix0, iy0 = idx0[:, 0], idx0[:, 1]
            ix1, iy1 = idx1[:, 0], idx1[:, 1]
            fx, fy = frac[:, 0], frac[:, 1]
            corners = [
                (ix0, iy0, (1 - fx) * (1 - fy)),
                (ix1, iy0, fx       * (1 - fy)),
                (ix0, iy1, (1 - fx) * fy),
                (ix1, iy1, fx       * fy),
            ]
            for ix, iy, w in corners:
                S.index_put_((ix, iy), density * w, accumulate=True)
        else:
            raise NotImplementedError(f"CIC deposit only supports D=2,3; got {d}")

        return S

    def _compute_source(self, pos, k_time):
        """Compute per-particle emission rate at frame k, using the same
        pulsed-source schedule as the generator, then spread to the grid.
        Returns source density on the grid.  Returns zeros if alpha=0.
        """
        if self.pde_alpha <= 0.0:
            return torch.zeros(self.grid_shape, device=pos.device, dtype=pos.dtype)

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

        # Scalar / tensor k -> per-particle step index.
        if torch.is_tensor(k_time):
            step = k_time.reshape(-1).to(pos.dtype)
            if step.shape[0] == 1:
                step = step.expand(n_cells)
        else:
            step = torch.full((n_cells,), float(k_time), dtype=pos.dtype, device=pos.device)

        if self.pde_pulse_period and self.pde_pulse_period > 0:
            phase_t = (step / float(self.pde_pulse_period) + self._src_phase) % 1.0
            d_ = torch.minimum(phase_t, 1.0 - phase_t)
            sigma_p = max(self.pde_pulse_duty, 1e-3) / 2.0
            pulse = torch.exp(-0.5 * (d_ / sigma_p) ** 2)
            pulse = pulse * self._src_mask.float()
            gain = 1.0 / max(self.pde_pulse_duty, 1e-3)
            e = self.pde_alpha * gain * pulse
        else:
            e = self.pde_alpha * self._src_mask.float()

        return self._particles_to_grid(pos, e)

    # -------------------- forward --------------------

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

        # --- GNN pair-force branch ---
        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = pos[dst], pos[src]
        d_pos_i, d_pos_j = d_pos[dst], d_pos[src]
        embedding_i, embedding_j = embedding[dst], embedding[src]

        messages = self.message(dst, src, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j)
        n_nodes = pos.shape[0]
        out_pair = scatter_aggregate(messages, dst, n_nodes, self.aggr_type)

        if self.update_type == 'mlp':
            out_pair = self.lin_phi(torch.cat((out_pair, embedding, d_pos), dim=-1))

        # --- SIREN on fixed grid (autograd through t for dc/dt) ---
        if not torch.is_tensor(k):
            k_scalar = float(k)
            t_scalar = torch.tensor(k_scalar * self.delta_t,
                                    dtype=pos.dtype, device=pos.device)
        else:
            # Use the first entry (trainer batches a single frame per call in
            # the graph_trainer path for this residual).  Broadcasting one t
            # across the grid is what the PDE residual requires.
            t_scalar = k.reshape(-1)[0].to(pos.dtype) * self.delta_t

        # Use a length-G t vector so autograd gives per-grid-point dc/dt
        # (a scalar t input would only yield the summed gradient).
        with torch.enable_grad():
            t_vec = t_scalar.detach().clone().expand(self.grid_coords.shape[0]).clone()
            t_vec = t_vec.requires_grad_(True)
            siren_in_grid = torch.cat([self.grid_coords, t_vec.unsqueeze(-1)], dim=1)
            c_flat = self.siren_field(siren_in_grid)  # (G, 1)
            dc_dt_per = torch.autograd.grad(
                outputs=c_flat,
                inputs=t_vec,
                grad_outputs=torch.ones_like(c_flat),
                create_graph=True, retain_graph=True,
            )[0]  # (G,)
        c_grid = c_flat.view(*self.grid_shape)
        dc_dt_grid = dc_dt_per.view(*self.grid_shape)

        # Laplacian on grid (finite differences).
        laplacian_grid = self._finite_diff_laplacian(c_grid)

        # Source on grid (from particles).
        source_grid = self._compute_source(pos, k)

        # PDE MLP prediction: (G, 3) -> (G, 1)
        pde_inputs = torch.stack([
            c_grid.reshape(-1),
            laplacian_grid.reshape(-1),
            source_grid.reshape(-1),
        ], dim=-1)
        dc_dt_pred = self.lin_pde(pde_inputs).reshape(*self.grid_shape)

        # PDE residual for the trainer to add: (dc/dt from SIREN) vs MLP prediction.
        pde_residual = dc_dt_grid - dc_dt_pred
        self.last_pde_residual = pde_residual.reshape(-1, 1)
        self.last_pde_reg = c_grid.reshape(-1, 1)
        self.last_scalar_field = c_grid.detach()
        self.last_source_field = source_grid.detach()

        # --- Interpolate field gradient to particle positions (no rotation) ---
        # Compute grad_c on the grid via finite differences, then trilinear-interp.
        grad_c_grid = self._finite_diff_gradient(c_grid)  # (D, *grid_shape)
        grad_c_at_p = self._trilinear_interp(grad_c_grid, pos)  # (N, D)

        mu_chem = torch.exp(self.log_mu_chem)
        if self.linear_chemotaxis:
            v_chem = mu_chem * grad_c_at_p
        else:
            g0 = torch.exp(self.log_chem_g0)
            g_norm = grad_c_at_p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g_hat = grad_c_at_p / g_norm
            v_chem = mu_chem * torch.log1p(g_norm / g0) * g_hat
        out_field = v_chem / getattr(self, 'ynorm', 1.0)

        # Un-rotate ONLY the GNN branch output.
        if self.rotation_augmentation & self.training:
            d = self.dimension
            out_pair[:, :d] = out_pair[:, :d] @ self.rotation_matrix.T

        out = out_pair + out_field
        return out

    # -------------------- message --------------------

    def message(self, dst, src, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j):
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        if self.rotation_augmentation & self.training:
            d = self.dimension
            delta_pos[:, :d] = delta_pos[:, :d] @ self.rotation_matrix

        in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
        out = self.lin_edge(in_features)

        if self.training is False:
            if out.shape[0] == 0:
                self.msg = torch.zeros(1, out.shape[1], device=out.device)
            else:
                pos = torch.argwhere(dst == self.cell_of_interest)
                if pos.numel() > 0:
                    self.msg = out[pos[:, 0]]
                else:
                    self.msg = out[0]
        return out

    def psi(self, r, p1, p2=None):
        # Same spring-force closed form as CellGNN for plot diagnostics.
        k_rep, r0, kadh, r_on, delta, mu_f = p1[0], p1[1], p1[2], p1[3], p1[4], p1[5]
        delta_safe = max(float(delta), 1e-8)
        F_rep = k_rep * torch.relu(r0 - r)
        g_on = torch.sigmoid((r - r0) / delta_safe)
        g_off = torch.sigmoid(-(r - r_on) / delta_safe)
        F_adh = -kadh * g_on * g_off * (r - r0)
        return mu_f * (F_rep + F_adh)
