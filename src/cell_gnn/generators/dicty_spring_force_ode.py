
import torch
import torch.nn as nn
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_simulator


@register_simulator("dicty_spring_force_ode")
class DictySpringForceODE(nn.Module):
    """Overdamped cell dynamics with spring-based repulsion and sigmoid-gated adhesion.

    Each cell type has parameters p = (k_rep, r0, kadh, r_on, delta, mu_f):
      - k_rep : repulsion stiffness
      - r0    : equilibrium (rest) distance
      - kadh  : adhesion strength
      - r_on  : adhesion cutoff distance
      - delta : sigmoid steepness
      - mu_f  : global force scaling factor

    Force law:
      F_rep = k_rep * relu(r0 - r) * rhat
      g_on  = sigmoid((r - r0) / delta)
      g_off = sigmoid(-(r - r_on) / delta)
      F_adh = -kadh * g_on * g_off * (r - r0) * rhat
      F     = F_rep + F_adh + noise
    """

    def __init__(self, aggr_type=[], p=[], bc_dpos=[], dimension=3, noise_model_level=0):
        super().__init__()

        self.aggr_type = aggr_type
        self.p = p
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.noise_model_level = noise_model_level

    def forward(self, state: CellState, edge_index: torch.Tensor, has_field=False, k=0):
        if has_field:
            field = state.field
        else:
            field = torch.ones(state.n_cells, 1, dtype=state.pos.dtype, device=state.pos.device)

        edge_index = remove_self_loops(edge_index)
        p = self.p.unsqueeze(0) if self.p.dim() == 1 else self.p
        parameters = p[to_numpy(state.cell_type), :]

        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = state.pos[dst], state.pos[src]
        parameters_i = parameters[dst]
        field_j = field[src]

        messages = self.message(pos_i, pos_j, parameters_i, field_j)
        d_pos = scatter_aggregate(messages, dst, state.n_cells, self.aggr_type)

        self.last_clean = d_pos.clone()

        if self.noise_model_level > 0:
            noise = self.noise_model_level * torch.randn_like(d_pos)
            self.last_noise = noise
            d_pos = d_pos + noise
        else:
            self.last_noise = torch.zeros_like(d_pos)

        return d_pos

    def message(self, pos_i, pos_j, parameters_i, field_j):
        delta_pos = self.bc_dpos(pos_j - pos_i)
        r = torch.sqrt(torch.sum(delta_pos ** 2, dim=1))
        r_safe = torch.clamp(r, min=1e-8)
        rhat = delta_pos / r_safe[:, None]

        k_rep = parameters_i[:, 0]
        r0 = parameters_i[:, 1]
        kadh = parameters_i[:, 2]
        r_on = parameters_i[:, 3]
        delta = parameters_i[:, 4]
        mu_f = parameters_i[:, 5]

        delta_safe = torch.clamp(delta, min=1e-8)

        # Repulsion: linear spring for r < r0
        F_rep = k_rep * torch.relu(r0 - r)

        # Adhesion: sigmoid-gated attractive force
        g_on = torch.sigmoid((r - r0) / delta_safe)
        g_off = torch.sigmoid(-(r - r_on) / delta_safe)
        F_adh = -kadh * g_on * g_off * (r - r0)

        F_total = -mu_f[:, None] * (F_rep + F_adh)[:, None] * rhat * field_j

        return F_total

    def psi(self, r, p):
        """Scalar force profile for plotting. r is a 1-D tensor of distances, p = [k_rep, r0, kadh, r_on, delta, mu_f]."""
        k_rep, r0, kadh, r_on, delta, mu_f = p[0], p[1], p[2], p[3], p[4], p[5]
        delta_safe = max(delta, 1e-8)

        F_rep = k_rep * torch.relu(r0 - r)
        g_on = torch.sigmoid((r - r0) / delta_safe)
        g_off = torch.sigmoid(-(r - r_on) / delta_safe)
        F_adh = -kadh * g_on * g_off * (r - r0)

        return mu_f * (F_rep + F_adh)
