
import torch
import torch.nn as nn
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_simulator


def init_gaussian_field(resolution, dimension, centers, amplitude, sigma, periodic, device):
    """Initialize concentration field on a uniform grid with Gaussian blobs.

    Returns tensor of shape (N,)*dimension.
    """
    coords = torch.linspace(0, 1 - 1.0 / resolution, resolution, device=device)

    if dimension == 3:
        gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
        grid = torch.stack([gx, gy, gz], dim=-1)          # (N, N, N, 3)
    else:
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1)               # (N, N, 2)

    if centers.dim() == 1:
        centers = centers.unsqueeze(0)                      # (1, dim)

    C = torch.zeros(grid.shape[:-1], device=device)
    for s in range(centers.shape[0]):
        diff = grid - centers[s]
        if periodic:
            diff = diff - torch.round(diff)
        dist_sq = (diff ** 2).sum(dim=-1)
        C = C + amplitude * torch.exp(-dist_sq / (2 * sigma ** 2))
    return C


def build_spectral_decay(resolution, dimension, D, lam, dt, device):
    r"""Precompute per-wavenumber decay factor for one timestep.

    For  :math:`\partial c / \partial t = D \nabla^2 c - \lambda c`
    the exact Fourier-space update is
    :math:`\hat c(k, t+dt) = \hat c(k, t) \exp\bigl(-(D |k|^2 + \lambda) dt\bigr)`.
    """
    freq_full = torch.fft.fftfreq(resolution, d=1.0 / resolution, device=device)
    freq_r = torch.fft.rfftfreq(resolution, d=1.0 / resolution, device=device)

    if dimension == 3:
        kz, ky, kx = torch.meshgrid(freq_full, freq_full, freq_r, indexing='ij')
        k_sq = (2 * torch.pi) ** 2 * (kx ** 2 + ky ** 2 + kz ** 2)
    else:
        ky, kx = torch.meshgrid(freq_full, freq_r, indexing='ij')
        k_sq = (2 * torch.pi) ** 2 * (kx ** 2 + ky ** 2)

    return torch.exp(-(D * k_sq + lam) * dt)


def spectral_gradient(C_hat, resolution, dimension, device):
    r"""Gradient of the field computed in Fourier space.

    :math:`\nabla_d c = \mathrm{ifft}(2\pi i\, k_d \,\hat c)`

    Returns tensor of shape ``(*grid_shape, dimension)``.
    """
    freq_full = torch.fft.fftfreq(resolution, d=1.0 / resolution, device=device)
    freq_r = torch.fft.rfftfreq(resolution, d=1.0 / resolution, device=device)
    s = [resolution] * dimension

    if dimension == 3:
        kz, ky, kx = torch.meshgrid(freq_full, freq_full, freq_r, indexing='ij')
        k_list = [kx, ky, kz]
    else:
        ky, kx = torch.meshgrid(freq_full, freq_r, indexing='ij')
        k_list = [kx, ky]

    grads = []
    for k_d in k_list:
        grad_hat = 2j * torch.pi * k_d * C_hat
        grads.append(torch.fft.irfftn(grad_hat, s=s))

    return torch.stack(grads, dim=-1)


def interp_periodic(field, pos, resolution):
    """Tri/bi-linear interpolation with periodic wrapping.

    Parameters
    ----------
    field : Tensor, shape ``(N,)*dim`` or ``(N,)*dim + (dim,)``
        Scalar or vector field on the grid.
    pos : Tensor, shape ``(P, dim)``
        Particle positions in [0, 1).
    resolution : int

    Returns
    -------
    Tensor, shape ``(P,)`` for scalar input or ``(P, dim)`` for vector input.
    """
    vector = field.dim() > len(pos[0]) + 0  # quick check: vector field has extra trailing dim
    dim = pos.shape[1]

    scaled = pos * resolution
    idx0 = scaled.long() % resolution
    idx1 = (idx0 + 1) % resolution
    frac = scaled - scaled.floor()

    if dim == 3:
        ix0, iy0, iz0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        ix1, iy1, iz1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        # Eight corners — index order is [z, y, x, ...]
        c000 = field[iz0, iy0, ix0]
        c001 = field[iz0, iy0, ix1]
        c010 = field[iz0, iy1, ix0]
        c011 = field[iz0, iy1, ix1]
        c100 = field[iz1, iy0, ix0]
        c101 = field[iz1, iy0, ix1]
        c110 = field[iz1, iy1, ix0]
        c111 = field[iz1, iy1, ix1]

        if vector:
            fx, fy, fz = fx.unsqueeze(-1), fy.unsqueeze(-1), fz.unsqueeze(-1)

        val = (c000 * (1 - fx) * (1 - fy) * (1 - fz)
             + c001 * fx       * (1 - fy) * (1 - fz)
             + c010 * (1 - fx) * fy       * (1 - fz)
             + c011 * fx       * fy       * (1 - fz)
             + c100 * (1 - fx) * (1 - fy) * fz
             + c101 * fx       * (1 - fy) * fz
             + c110 * (1 - fx) * fy       * fz
             + c111 * fx       * fy       * fz)
        return val

    else:  # 2D
        ix0, iy0 = idx0[:, 0], idx0[:, 1]
        ix1, iy1 = idx1[:, 0], idx1[:, 1]
        fx, fy = frac[:, 0], frac[:, 1]

        c00 = field[iy0, ix0]
        c01 = field[iy0, ix1]
        c10 = field[iy1, ix0]
        c11 = field[iy1, ix1]

        if vector:
            fx, fy = fx.unsqueeze(-1), fy.unsqueeze(-1)

        val = (c00 * (1 - fx) * (1 - fy)
             + c01 * fx       * (1 - fy)
             + c10 * (1 - fx) * fy
             + c11 * fx       * fy)
        return val


def deposit_to_grid(pos, strength, resolution, dimension, device):
    """Scatter particle sources onto the grid using tri/bi-linear weights.

    This is the transpose of ``interp_periodic``: each particle deposits
    ``strength`` to its surrounding grid corners, weighted by distance.

    Parameters
    ----------
    pos : (P, dim)  particle positions in [0, 1)
    strength : float  source strength per particle per timestep
    resolution : int
    dimension : int (2 or 3)
    device : torch.device

    Returns
    -------
    Tensor of shape ``(resolution,)*dimension`` — source field on the grid.
    """
    grid_shape = [resolution] * dimension
    S = torch.zeros(grid_shape, device=device)

    scaled = pos * resolution
    idx0 = scaled.long() % resolution
    idx1 = (idx0 + 1) % resolution
    frac = scaled - scaled.floor()

    if dimension == 3:
        ix0, iy0, iz0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        ix1, iy1, iz1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        # Eight corners with trilinear weights
        weights = [
            (iz0, iy0, ix0, (1 - fx) * (1 - fy) * (1 - fz)),
            (iz0, iy0, ix1, fx       * (1 - fy) * (1 - fz)),
            (iz0, iy1, ix0, (1 - fx) * fy       * (1 - fz)),
            (iz0, iy1, ix1, fx       * fy       * (1 - fz)),
            (iz1, iy0, ix0, (1 - fx) * (1 - fy) * fz),
            (iz1, iy0, ix1, fx       * (1 - fy) * fz),
            (iz1, iy1, ix0, (1 - fx) * fy       * fz),
            (iz1, iy1, ix1, fx       * fy       * fz),
        ]
        for iz, iy, ix, w in weights:
            S.index_put_((iz, iy, ix), strength * w, accumulate=True)

    else:  # 2D
        ix0, iy0 = idx0[:, 0], idx0[:, 1]
        ix1, iy1 = idx1[:, 0], idx1[:, 1]
        fx, fy = frac[:, 0], frac[:, 1]

        weights = [
            (iy0, ix0, (1 - fx) * (1 - fy)),
            (iy0, ix1, fx       * (1 - fy)),
            (iy1, ix0, (1 - fx) * fy),
            (iy1, ix1, fx       * fy),
        ]
        for iy, ix, w in weights:
            S.index_put_((iy, ix), strength * w, accumulate=True)

    return S


@register_simulator("particle_spring_force_diffusion_field",
                     "particle_spring_force_diffusion_field_siren",
                     "particle_spring_force_diffusion_field_siren_grad")
class ParticleSpringForceDiffusionField(nn.Module):
    r"""Overdamped cell dynamics with spring forces + chemotaxis from a
    diffusing chemical field.

    The chemical concentration satisfies the unsteady diffusion–decay PDE
    with a cell-sourced production term:

    .. math::
        \frac{\partial c}{\partial t} = D \nabla^2 c - \lambda c
            + \alpha \sum_i \delta(\mathbf{x} - \mathbf{x}_i)

    where :math:`\alpha` (``source_strength``) controls how much chemical
    each cell secretes per unit time.  The source is deposited onto the grid
    via trilinear interpolation (same stencil as the read-back).

    Solved spectrally (FFT) on a periodic [0,1)^dim grid.

    ``field_params`` keys
    ---------------------
    diffusion_coeff : float
        Diffusion coefficient *D*.
    lambda_decay : float
        Linear decay rate :math:`\lambda`.
    grid_resolution : int
        Number of grid points per axis (e.g. 64).
    source_strength : float
        Chemical secretion rate per cell (:math:`\alpha`).  Default 0 (no source).
    center_0 : (S, dim) or (dim,)
        Initial Gaussian blob centre(s).
    amplitude : float
        Peak of initial Gaussian(s).
    sigma : float
        Width of initial Gaussian(s).
    mu_chem : float
        Chemotactic coupling strength.

    Cell parameters p = (k_rep, r0, kadh, r_on, delta, mu_f).
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

        # Lazy-initialised grid state (set on first forward call)
        self._field_grid = None   # real-space concentration  (N,)*dim
        self._decay = None        # spectral decay factor
        self._last_step = -1      # track which timestep we're on

    # ------------------------------------------------------------------
    def _init_grid(self, device):
        fp = self.field_params
        res = fp['grid_resolution']
        dim = self.dimension
        periodic = fp.get('periodic', True)

        self._field_grid = init_gaussian_field(
            res, dim, fp['center_0'], fp['amplitude'], fp['sigma'],
            periodic, device)

        self._decay = build_spectral_decay(
            res, dim, fp['diffusion_coeff'], fp['lambda_decay'],
            fp['delta_t'], device)

    # ------------------------------------------------------------------
    def _advance_field(self, pos, n_steps=1):
        """Advance the PDE by *n_steps* timesteps in Fourier space.

        Each step: add cell sources to the grid, then apply the exact
        spectral decay/diffusion operator.
        """
        dim = self.dimension
        res = self.field_params['grid_resolution']
        dt = self.field_params['delta_t']
        alpha = self.field_params.get('source_strength', 0.0)
        s = [res] * dim

        for _ in range(n_steps):
            # Deposit cell sources onto grid
            if alpha > 0:
                source = deposit_to_grid(pos, alpha * dt, res, dim, pos.device)
                self._field_grid = self._field_grid + source

            # Diffusion + decay step in Fourier space
            C_hat = torch.fft.rfftn(self._field_grid, s=s)
            C_hat = C_hat * self._decay
            self._field_grid = torch.fft.irfftn(C_hat, s=s)

        # Return final Fourier coefficients for gradient computation
        C_hat = torch.fft.rfftn(self._field_grid, s=s)
        return C_hat

    # ------------------------------------------------------------------
    def forward(self, state: CellState, edge_index: torch.Tensor,
                has_field=False, k=0):
        edge_index = remove_self_loops(edge_index)
        p = self.p.unsqueeze(0) if self.p.dim() == 1 else self.p
        parameters = p[to_numpy(state.cell_type), :]

        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = state.pos[dst], state.pos[src]
        parameters_i = parameters[dst]

        # --- pairwise spring forces ---
        messages = self.message(pos_i, pos_j, parameters_i)
        d_pos = scatter_aggregate(messages, dst, state.n_cells, self.aggr_type)

        # --- diffusion field ---
        device = state.pos.device

        # Lazily move tensor field params to the right device
        for key in ('center_0',):
            v = self.field_params.get(key)
            if torch.is_tensor(v) and v.device != device:
                self.field_params[key] = v.to(device)

        # Initialise grid on first call
        if self._field_grid is None:
            self._init_grid(device)

        # Advance PDE from last step to current step k
        steps = k - self._last_step
        if steps > 0:
            C_hat = self._advance_field(state.pos, n_steps=steps)
        else:
            s = [self.field_params['grid_resolution']] * self.dimension
            C_hat = torch.fft.rfftn(self._field_grid, s=s)
        self._last_step = k

        # Interpolate field value & gradient to particle positions
        res = self.field_params['grid_resolution']
        grad_grid = spectral_gradient(C_hat, res, self.dimension, device)

        field_value = interp_periodic(self._field_grid, state.pos, res)  # (N,)
        field_gradient = interp_periodic(grad_grid, state.pos, res)      # (N, dim)

        mu_chem = self.field_params['mu_chem']
        d_pos = d_pos + mu_chem * field_gradient

        state.field = field_value.unsqueeze(-1)  # (N, 1) to match convention

        self.last_clean = d_pos.clone()

        if self.noise_model_level > 0:
            noise = self.noise_model_level * torch.randn_like(d_pos)
            self.last_noise = noise
            d_pos = d_pos + noise
        else:
            self.last_noise = torch.zeros_like(d_pos)

        return d_pos

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def psi(self, r, p):
        """Scalar force profile for plotting (spring part only)."""
        k_rep, r0, kadh, r_on, delta, mu_f = p[0], p[1], p[2], p[3], p[4], p[5]
        delta_safe = max(delta, 1e-8)

        F_rep = k_rep * torch.relu(r0 - r)
        g_on = torch.sigmoid((r - r0) / delta_safe)
        g_off = torch.sigmoid(-(r - r_on) / delta_safe)
        F_adh = -kadh * g_on * g_off * (r - r0)

        return mu_f * (F_rep + F_adh)
