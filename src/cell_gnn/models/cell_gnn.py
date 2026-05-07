import numpy as np
import torch
import torch.nn as nn
from cell_gnn.models.MLP import MLP
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_model


@register_model("arbitrary_ode", "boids_ode", "gravity_ode", "particle_spring_force_ode", "particle_spring_force_prescribed_field", "particle_spring_force_diffusion_field")
class CellGNN(nn.Module):
    """Interaction Network for cell dynamics — learns pairwise interaction from relative positions/velocities.

    acceleration_i = aggr_j(lin_edge(delta_pos, delta_vel, a_i, a_j)) * ynorm
    """

    PARAMS_DOC = {
        "model_name": "CellGNN",
        "description": "GNN for cell dynamics — learns pairwise interaction from relative positions/velocities. "
                       "acceleration_i = aggr_j(lin_edge(delta_pos, delta_vel, a_i, a_j)) * ynorm",
        "equations": {
            "message_arbitrary": "msg_j = lin_edge(delta_pos_ij / max_r, r / max_r, a_i)",
            "message_boids": "msg_j = lin_edge(delta_pos_ij / max_r, r / max_r, dpos_i / vnorm, dpos_j / vnorm, a_i)",
            "message_gravity": "msg_j = lin_edge(delta_pos_ij / max_r, r / max_r, dpos_i / vnorm, dpos_j / vnorm, a_j)",
            "update_none": "acceleration = aggr(messages) * ynorm",
            "update_mlp": "acceleration = lin_phi(aggr(messages), a_i, dpos / vnorm) * ynorm",
        },
        "graph_model_config": {
            "lin_edge (MLP0)": {
                "description": "Pairwise interaction function — force from relative state",
                "input_size": {
                    "arbitrary_ode": "dimension + 1 + embedding_dim  (delta_pos, r, a_i)",
                    "boids_ode": "dimension + 1 + 2*dimension + embedding_dim  (delta_pos, r, dpos_i, dpos_j, a_i)",
                    "gravity_ode": "dimension + 1 + 2*dimension + embedding_dim  (delta_pos, r, dpos_i, dpos_j, a_j)",
                },
                "output_size": "dimension (force vector)",
                "hidden_dim": {"typical_range": [64, 256], "default": 128},
                "n_layers": {"typical_range": [3, 7], "default": 5},
            },
            "lin_phi (MLP1, update_type='mlp')": {
                "description": "Node update — acceleration from aggregated messages + embedding + velocity",
                "input_size_update": "output_size + embedding_dim + dimension",
                "hidden_dim_update": {"typical_range": [32, 128], "default": 64},
                "n_layers_update": {"typical_range": [2, 5], "default": 3},
            },
            "embedding a": {
                "shape": "(n_datasets, n_cells + n_ghosts, embedding_dim)",
                "description": "Learned per-cell embedding encoding cell type",
            },
        },
        "training_config": {
            "learning_rate_start": {"description": "LR for MLP params", "typical_range": [1e-5, 1e-3]},
            "learning_rate_embedding_start": {"description": "LR for embedding", "typical_range": [1e-6, 1e-4]},
            "batch_size": {"description": "Frames per gradient step", "typical_range": [1, 16]},
            "data_augmentation_loop": {"description": "Iterations = n_frames * aug_loop / batch_size"},
            "coeff_edge_diff": {"description": "Same-type edge similarity penalty", "typical_range": [0, 100]},
            "coeff_edge_norm": {"description": "Monotonicity penalty on lin_edge", "typical_range": [0, 10]},
            "recursive_training": {"description": "Enable multi-step unrolling during training"},
            "recursive_loop": {"description": "Number of recurrent unroll steps", "typical_range": [0, 8]},
        },
        "simulation_config": {
            "n_cells": "Number of cells",
            "n_cell_types": "Number of distinct cell types",
            "max_radius": "Interaction radius for edge construction",
            "delta_t": "Integration time step",
            "boundary": "Boundary condition: periodic, no, wall",
        },
    }

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

        # Auto-compute input_size from model type to stay consistent with embedding_dim
        match self.model:
            case 'arbitrary_ode' | 'particle_spring_force_ode' | 'particle_spring_force_prescribed_field':
                self.input_size = self.dimension + 1 + self.embedding_dim
            case 'boids_ode' | 'gravity_ode':
                self.input_size = 3 * self.dimension + 1 + self.embedding_dim
            case _:
                self.input_size = model_config.input_size

        # Auto-compute input_size_update for MLP1
        if self.update_type != 'none':
            self.input_size_update = self.dimension + self.embedding_dim + self.output_size
        else:
            self.input_size_update = model_config.input_size_update
        self.embedding_trial = config.training.embedding_trial

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

        # self.lin_edge = FusedMLP(in_dim=self.input_size, hidden_dim=self.hidden_dim, out_dim=self.output_size, n_hidden=self.n_layers, activation='ReLU', output_activation=None, device=self.device)

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers, hidden_size=self.hidden_dim, device=self.device)

        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update,
                               nlayers=self.n_layers_update,
                               hidden_size=self.hidden_dim_update, device=self.device)

        if self.state == 'sequence':
            self.a = nn.Parameter(torch.ones((self.n_dataset, int(self.n_cells*100 + 100 ), self.embedding_dim), device=self.device, requires_grad=True,dtype=torch.float32))
            self.embedding_step = self.n_frames // 100
        else:
            self.a = nn.Parameter(
                    torch.tensor(np.ones((self.n_dataset, int(self.n_cells) + self.n_ghosts, self.embedding_dim)), device=self.device,
                                 requires_grad=True, dtype=torch.float32))

    def get_interp_a(self, k, cell_id, data_id):
        id = cell_id * 100 + k // self.embedding_step
        alpha = (k % self.embedding_step) / self.embedding_step
        return alpha * self.a[data_id.clone().detach(), id+1, :].squeeze() + (1 - alpha) * self.a[data_id.clone().detach(), id, :].squeeze()


    def forward(self, state: CellState, edge_index: torch.Tensor, data_id=[], training=[], has_field=False, k=[]):

        self.data_id = data_id
        self.training = training
        self.has_field = has_field

        if self.remove_self:
            edge_index = remove_self_loops(edge_index)

        if has_field:
            field = state.field
        else:
            field = torch.ones(state.n_cells, 1, dtype=state.pos.dtype, device=state.pos.device)

        derivatives = torch.zeros(state.n_cells, 1, dtype=state.pos.dtype, device=state.pos.device)

        pos = state.pos
        d_pos = state.vel / self.vnorm
        if self.rotation_augmentation & self.training:
            if self.dimension == 2:
                self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * np.pi * 2
                self.rotation_matrix = torch.stack([
                    torch.stack([torch.cos(self.phi), torch.sin(self.phi)]),
                    torch.stack([-torch.sin(self.phi), torch.cos(self.phi)])
                ])
                self.rotation_matrix = self.rotation_matrix.permute(*torch.arange(self.rotation_matrix.ndim - 1, -1, -1)).squeeze()
            else:
                # Random SO(3) rotation via QR decomposition
                rand_matrix = torch.randn(3, 3, dtype=torch.float32, device=self.device)
                q, r = torch.linalg.qr(rand_matrix)
                # Ensure proper rotation (det = +1)
                q = q * torch.sign(torch.diag(r))
                if torch.det(q) < 0:
                    q[:, 0] = -q[:, 0]
                self.rotation_matrix = q

            d = self.dimension
            d_pos[:, :d] = d_pos[:, :d] @ self.rotation_matrix

            for n in range(derivatives.shape[1] // d):
                derivatives[:, n*d:n*d+d] = derivatives[:, n*d:n*d+d] @ self.rotation_matrix

        cell_id = state.index.unsqueeze(-1)
        if self.state == 'sequence':
            embedding = self.get_interp_a(k, cell_id, self.data_id)
        else:
            embedding = self.a[self.data_id.clone().detach(), cell_id, :].squeeze()

        # Gather features for source (j) and target (i) nodes
        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = pos[dst], pos[src]
        d_pos_i, d_pos_j = d_pos[dst], d_pos[src]
        embedding_i, embedding_j = embedding[dst], embedding[src]
        field_j = field[src]
        derivatives_j = derivatives[src]

        messages = self.message(dst, src, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, field_j, derivatives_j)

        n_nodes = pos.shape[0]
        out = scatter_aggregate(messages, dst, n_nodes, self.aggr_type)

        if self.update_type == 'mlp':
            if has_field:
                out = self.lin_phi(torch.cat((out, embedding, d_pos, field), dim=-1))
            else:
                out = self.lin_phi(torch.cat((out, embedding, d_pos), dim=-1))
        if self.rotation_augmentation & self.training:
            d = self.dimension
            out[:, :d] = out[:, :d] @ self.rotation_matrix.T

        if self.reflection_augmentation & self.training:
            if group in [0, 1]:
                out[:, group] = -out[:, group]
            else:
                out = -out

        return out

    def message(self, dst, src, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i, embedding_j, field_j, derivatives_j):

        # distance normalized by the max radius
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        if self.rotation_augmentation & self.training:
            d = self.dimension
            delta_pos[:, :d] = delta_pos[:, :d] @ self.rotation_matrix

        match self.model:
            case 'arbitrary_ode' | 'particle_spring_force_ode' | 'particle_spring_force_prescribed_field':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
            case 'boids_ode':
                in_features = torch.cat((delta_pos, r[:, None], d_pos_i, d_pos_j, embedding_i), dim=-1)
            case 'gravity_ode':
                in_features = torch.cat((delta_pos, r[:, None], d_pos_i, d_pos_j, embedding_j), dim=-1)

        out = self.lin_edge(in_features)

        if self.training==False:
            if out.shape[0] == 0:
                self.msg = torch.zeros(1, out.shape[1], device=out.device) if out.dim() > 1 else torch.zeros(1, device=out.device)
            else:
                pos = torch.argwhere(dst == self.cell_of_interest)
                if pos.numel()>0:
                    self.msg = out[pos[:,0]]
                else:
                    self.msg = out[0]

        return out

    def psi(self, r, p1, p2=None):

        if self.model == 'arbitrary_ode':
            return r * (p1[0] * torch.exp(-torch.abs(r) ** (2 * p1[1]) / (2 * self.sigma ** 2)) - p1[2] * torch.exp(-torch.abs(r) ** (2 * p1[3]) / (2 * self.sigma ** 2)))
        if self.model == 'boids_ode':
            cohesion = p1[0] * 0.5E-5 * r
            separation = -p1[2] * 1E-8 / r
            return (cohesion + separation) * p1[1] / 500
        if self.model == 'gravity_ode':
            psi = p1 / r ** 2
            return psi[:, None]
        if self.model in ('particle_spring_force_ode', 'particle_spring_force_prescribed_field'):
            k_rep, r0, kadh, r_on, delta, mu_f = p1[0], p1[1], p1[2], p1[3], p1[4], p1[5]
            delta_safe = max(delta, 1e-8)
            F_rep = k_rep * torch.relu(r0 - r)
            g_on = torch.sigmoid((r - r0) / delta_safe)
            g_off = torch.sigmoid(-(r - r_on) / delta_safe)
            F_adh = -kadh * g_on * g_off * (r - r0)
            return mu_f * (F_rep + F_adh)
