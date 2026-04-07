
import torch
import torch.nn as nn
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_simulator


def moving_gaussian_field(pos, t, field_params):
    """Evaluate a moving Gaussian chemical field and its gradient at particle positions.

    The Gaussian center moves along a prescribed trajectory:
        center(t) = center_0 + velocity * t

    Field value:
        C(x, t) = amplitude * exp(-|x - center(t)|^2 / (2 * sigma^2))

    Gradient:
        grad C = -C(x,t) * (x - center(t)) / sigma^2

    Args:
        pos:          (N, dim) particle positions
        t:            scalar, current time (frame * delta_t)
        field_params: dict with keys:
            center_0:   (dim,) initial center position
            velocity:   (dim,) center velocity
            amplitude:  scalar, peak concentration
            sigma:      scalar, Gaussian width

    Returns:
        field_value:    (N, 1) chemical concentration at each particle
        field_gradient: (N, dim) gradient of concentration at each particle
    """
    center = field_params['center_0'] + field_params['velocity'] * t  # (dim,)
    diff = pos - center.unsqueeze(0)  # (N, dim)

    # periodic boundary: wrap diff to [-0.5, 0.5] if using periodic BC
    if field_params.get('periodic', False):
        diff = diff - torch.round(diff)

    dist_sq = (diff ** 2).sum(dim=1, keepdim=True)  # (N, 1)
    sigma = field_params['sigma']
    C = field_params['amplitude'] * torch.exp(-dist_sq / (2 * sigma ** 2))  # (N, 1)
    grad_C = -C * diff / (sigma ** 2)  # (N, dim)

    return C, grad_C


@register_simulator("particle_spring_force_dynamic_field", "particle_spring_force_dynamic_field_siren")
class ParticleSpringForceDynamicField(nn.Module):
    """Overdamped cell dynamics with spring forces + chemotaxis from a moving Gaussian field.

    Total force on cell i:
        F_i = sum_j F_spring(i,j) + mu_chem * grad C(x_i, t) + noise

    Spring force law (same as ParticleSpringForceODE):
        F_rep = k_rep * relu(r0 - r) * rhat
        g_on  = sigmoid((r - r0) / delta)
        g_off = sigmoid(-(r - r_on) / delta)
        F_adh = -kadh * g_on * g_off * (r - r0) * rhat

    Chemotaxis:
        F_chem = mu_chem * grad C(x_i, t)
        where C is a moving Gaussian: C(x,t) = A * exp(-|x - center(t)|^2 / (2*sigma^2))

    Cell parameters p = (k_rep, r0, kadh, r_on, delta, mu_f):
        same as ParticleSpringForceODE

    Field parameters (separate from cell params):
        center_0:   initial Gaussian center
        velocity:   center drift velocity
        amplitude:  peak concentration
        sigma:      Gaussian width
        mu_chem:    chemotactic coupling strength
    """

    def __init__(self, aggr_type=[], p=[], bc_dpos=[], dimension=3,
                 noise_model_level=0, field_params=None):
        super().__init__()

        self.aggr_type = aggr_type
        self.p = p
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.noise_model_level = noise_model_level
        self.field_params = field_params or {}

    def forward(self, state: CellState, edge_index: torch.Tensor, has_field=False, k=0):
        edge_index = remove_self_loops(edge_index)
        p = self.p.unsqueeze(0) if self.p.dim() == 1 else self.p
        parameters = p[to_numpy(state.cell_type), :]

        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = state.pos[dst], state.pos[src]
        parameters_i = parameters[dst]

        # pairwise spring forces
        messages = self.message(pos_i, pos_j, parameters_i)
        d_pos = scatter_aggregate(messages, dst, state.n_cells, self.aggr_type)

        # chemotaxis from dynamic field
        t = k * self.field_params.get('delta_t', 1.0)
        field_value, field_gradient = moving_gaussian_field(state.pos, t, self.field_params)
        mu_chem = self.field_params.get('mu_chem', 0.5)
        d_pos = d_pos + mu_chem * field_gradient

        # store field value on state so it can be saved / used downstream
        state.field = field_value

        self.last_clean = d_pos.clone()

        if self.noise_model_level > 0:
            noise = self.noise_model_level * torch.randn_like(d_pos)
            self.last_noise = noise
            d_pos = d_pos + noise
        else:
            self.last_noise = torch.zeros_like(d_pos)

        return d_pos

    def message(self, pos_i, pos_j, parameters_i):
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

        F_total = -mu_f[:, None] * (F_rep + F_adh)[:, None] * rhat

        return F_total

    def psi(self, r, p):
        """Scalar force profile for plotting (spring part only)."""
        k_rep, r0, kadh, r_on, delta, mu_f = p[0], p[1], p[2], p[3], p[4], p[5]
        delta_safe = max(delta, 1e-8)

        F_rep = k_rep * torch.relu(r0 - r)
        g_on = torch.sigmoid((r - r0) / delta_safe)
        g_off = torch.sigmoid(-(r - r_on) / delta_safe)
        F_adh = -kadh * g_on * g_off * (r - r0)

        return mu_f * (F_rep + F_adh)
