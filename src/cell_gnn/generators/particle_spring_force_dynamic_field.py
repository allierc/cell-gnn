
import torch
import torch.nn as nn
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_simulator


def single_gaussian_field(pos, t, field_params):
    """Single moving Gaussian chemical field.

    C(x, t) = amplitude * exp(-|x - c(t)|^2 / (2*sigma^2))
    c(t)    = center_0 + velocity * t
    grad C  = -C * (x - c(t)) / sigma^2
    """
    center = field_params['center_0'] + field_params['velocity'] * t  # (dim,)
    diff = pos - center.unsqueeze(0)                                  # (N, dim)

    if field_params.get('periodic', False):
        diff = diff - torch.round(diff)

    dist_sq = (diff ** 2).sum(dim=1, keepdim=True)                    # (N, 1)
    sigma = field_params['sigma']
    C = field_params['amplitude'] * torch.exp(-dist_sq / (2 * sigma ** 2))  # (N, 1)
    grad_C = -C * diff / (sigma ** 2)                                 # (N, dim)
    return C, grad_C


def multi_gaussian_field(pos, t, field_params):
    """Sum of S moving Gaussian chemical sources. All sources share the same
    scalar amplitude and sigma — only their centers (and velocities) differ.

    C(x, t) = amplitude * sum_s exp(-|x - c_s(t)|^2 / (2*sigma^2))
    c_s(t)  = center_0[s] + velocity[s] * t
    grad C  = sum_s -C_s * (x - c_s(t)) / sigma^2
    """
    centers = field_params['center_0'] + field_params['velocity'] * t  # (S, dim)
    amplitude = field_params['amplitude']                              # scalar
    sigma = field_params['sigma']                                      # scalar

    diff = pos.unsqueeze(1) - centers.unsqueeze(0)                     # (N, S, dim)
    if field_params.get('periodic', False):
        diff = diff - torch.round(diff)

    dist_sq = (diff ** 2).sum(dim=-1)                                  # (N, S)
    C_s = amplitude * torch.exp(-dist_sq / (2.0 * sigma ** 2))         # (N, S)

    field_value = C_s.sum(dim=1, keepdim=True)                         # (N, 1)
    field_gradient = (-C_s.unsqueeze(-1) * diff / (sigma ** 2)).sum(dim=1)
    return field_value, field_gradient


def moving_gaussian_field(pos, t, field_params):
    """Dispatch entry point: picks single or multi based on ``field_params['field_type']``.

    Defaults to 'single' when unspecified (back-compatible with pre-v7 configs).
    """
    field_type = field_params.get('field_type', 'single')
    if field_type == 'multi':
        return multi_gaussian_field(pos, t, field_params)
    return single_gaussian_field(pos, t, field_params)


@register_simulator("particle_spring_force_dynamic_field", "particle_spring_force_dynamic_field_siren")
class ParticleSpringForceDynamicField(nn.Module):
    """Overdamped cell dynamics with spring forces + chemotaxis from a moving chemical field.

    Chemical field is either a single moving Gaussian or a sum of multiple moving Gaussians
    that share the same amplitude/sigma. Selected via ``field_params['field_type']``:

        field_type: 'single'   (default — back-compatible)
            center_0:  (dim,) initial center
            velocity:  (dim,) drift velocity
            amplitude: scalar peak
            sigma:     scalar Gaussian width
            mu_chem:   scalar coupling

        field_type: 'multi'
            center_0:  (S, dim) per-source initial centers
            velocity:  (S, dim) per-source drift velocities
            amplitude: scalar peak   (same for all S sources)
            sigma:     scalar width  (same for all S sources)
            mu_chem:   scalar coupling

    Cell parameters p = (k_rep, r0, kadh, r_on, delta, mu_f): same as ParticleSpringForceODE.
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

        # lazily move tensor field params to the right device
        for key in ('center_0', 'velocity'):
            v = self.field_params.get(key, None)
            if torch.is_tensor(v) and v.device != state.pos.device:
                self.field_params[key] = v.to(state.pos.device)

        field_value, field_gradient = moving_gaussian_field(state.pos, t, self.field_params)
        mu_chem = self.field_params['mu_chem']
        d_pos = d_pos + mu_chem * field_gradient

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

        F_rep = k_rep * torch.relu(r0 - r)
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
