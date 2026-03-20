import numpy as np
import torch
import torch.nn as nn
from cell_gnn.models.MLP import MLP
from cell_gnn.utils import to_numpy
from cell_gnn.models import Siren_Network
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_model


@register_model("arbitrary_field_ode", "boids_field_ode")
class CellFieldGNN(nn.Module):
    """GNN for cell dynamics with a continuous field (SIREN).

    Combines pairwise interactions (g_phi) with a learned spatiotemporal field (model_f).
    output = g_phi(delta_pos, delta_vel, a_i, a_j) * field_j
    """

    PARAMS_DOC = {
        "model_name": "CellFieldGNN",
        "description": "GNN for cell dynamics with a continuous field (SIREN). "
                       "Combines pairwise interactions (g_phi) with a learned spatiotemporal field (model_f). "
                       "output = g_phi(delta_pos, delta_vel, a_i) * field_j",
        "equations": {
            "message_arbitrary_field": "msg_j = g_phi(delta_pos / max_r, r / max_r, a_i) * field_j",
            "message_boids_field": "msg_j = g_phi(delta_pos / max_r, r / max_r, dpos_i, dpos_j, a_i) * field_j",
            "field": "field(t) = model_f(t / T)^2  (SIREN network, squared output)",
            "update_linear": "acceleration = lin_update(aggr(messages), dpos, a_i)",
        },
        "graph_model_config": {
            "g_phi (edge message MLP)": {
                "description": "Pairwise interaction function, multiplied by field value at source",
                "input_size": {
                    "arbitrary_field_ode": "dimension + 1 + embedding_dim  (delta_pos, r, a_i)",
                    "boids_field_ode": "dimension + 1 + 2*dimension + embedding_dim  (delta_pos, r, dpos_i, dpos_j, a_i)",
                },
                "output_size": "dimension",
                "hidden_dim": {"typical_range": [64, 256], "default": 128},
                "n_layers": {"typical_range": [3, 7], "default": 5},
            },
            "model_f (SIREN)": {
                "description": "Continuous field via SIREN network — output squared and multiplied with edge messages",
                "input_size_nnr": "1 (time) or 3 (t, x, y)",
                "hidden_dim_nnr": {"default": 128},
                "n_layers_nnr": {"default": 5},
                "omega": {"description": "SIREN frequency parameter", "default": 80.0},
            },
            "embedding a": {
                "shape": "(n_datasets, n_cells + n_ghosts, embedding_dim)",
                "description": "Learned per-cell embedding encoding cell type",
            },
        },
        "training_config": {
            "learning_rate_start": {"description": "LR for MLP params"},
            "learning_rate_embedding_start": {"description": "LR for embedding"},
            "learning_rate_nnr": {"description": "LR for SIREN field network"},
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
        self.n_cells = simulation_config.n_cells
        self.n_nodes = simulation_config.n_nodes
        self.n_nodes_per_axis = int(np.sqrt(self.n_nodes))
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.sigma = simulation_config.sigma
        self.model = model_config.cell_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension
        self.embedding_trial = config.training.embedding_trial

        # Auto-compute input_size from model type to stay consistent with embedding_dim
        match self.model:
            case 'arbitrary_field_ode':
                self.input_size = self.dimension + 1 + self.embedding_dim
            case 'boids_field_ode':
                self.input_size = self.dimension + 1 + 4 + self.embedding_dim
            case _:
                self.input_size = model_config.input_size

        self.g_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_cells) + self.n_ghosts, self.embedding_dim)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

        if self.update_type != 'none':
            self.lin_update = MLP(input_size=self.output_size + self.embedding_dim + self.dimension, output_size=self.output_size,
                                  nlayers=self.n_layers_update, hidden_size=self.hidden_dim_update, device=self.device)

    def forward(self, state: CellState, edge_index: torch.Tensor, data_id=[], training=[], phi=[], has_field=False):

        self.data_id = data_id
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.has_field = has_field
        self.training = training

        edge_index = remove_self_loops(edge_index)

        pos = state.pos
        d_pos = state.vel
        if has_field:
            field = state.field
        else:
            field = torch.ones_like(state.field)

        cell_id = state.index.unsqueeze(-1)

        # Gather features for source (j) and target (i) nodes
        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = pos[dst], pos[src]
        d_pos_i, d_pos_j = d_pos[dst], d_pos[src]
        cell_id_i, cell_id_j = cell_id[dst], cell_id[src]
        field_j = field[src]

        messages = self.message(pos_i, pos_j, d_pos_i, d_pos_j, cell_id_i, cell_id_j, field_j)

        n_nodes = pos.shape[0]
        pred = scatter_aggregate(messages, dst, n_nodes, self.aggr_type)

        if self.update_type == 'linear':
            embedding = self.a[self.data_id, cell_id, :]
            pred = self.lin_update(torch.cat((pred, d_pos, embedding), dim=-1))

        return pred

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, cell_id_i, cell_id_j, field_j):
        # squared distance
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm
        if self.dimension == 3:
            dpos_z_i = d_pos_i[:, 2] / self.vnorm
            dpos_z_j = d_pos_j[:, 2] / self.vnorm

        if self.rotation_augmentation & (self.training == True):
            new_delta_pos_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_delta_pos_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_delta_pos_x
            delta_pos[:, 1] = new_delta_pos_y
            new_dpos_x_i = self.cos_phi * dpos_x_i + self.sin_phi * dpos_y_i
            new_dpos_y_i = -self.sin_phi * dpos_x_i + self.cos_phi * dpos_y_i
            dpos_x_i = new_dpos_x_i
            dpos_y_i = new_dpos_y_i
            new_dpos_x_j = self.cos_phi * dpos_x_j + self.sin_phi * dpos_y_j
            new_dpos_y_j = -self.sin_phi * dpos_x_j + self.cos_phi * dpos_y_j
            dpos_x_j = new_dpos_x_j
            dpos_y_j = new_dpos_y_j


        embedding_i = self.a[self.data_id, to_numpy(cell_id_i), :].squeeze()

        match self.model:

            case 'arbitrary_field_ode':
                    in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
            case 'boids_field_ode':
                    in_features = torch.cat(
                        (delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                         dpos_y_j[:, None], embedding_i), dim=-1)

        out = self.g_phi(in_features) * field_j

        return out

    def psi(self, r, p1, p2):

        if self.model == 'arbitrary_field_ode':
            return r * (p1[0] * torch.exp(-r ** (2 * p1[1]) / (2 * self.sigma ** 2)) - p1[2] * torch.exp(-r ** (2 * p1[3]) / (2 * self.sigma ** 2)))
        if self.model == 'boids_field_ode':
            cohesion = p1[0] * 0.5E-5 * r
            separation = -p1[2] * 1E-8 / r
            return (cohesion + separation) * p1[1] / 500
