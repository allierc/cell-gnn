
import torch
import torch.nn as nn
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_simulator


# ---------------------------------------------------------------------------
# Torch helpers: static field, spectral gradient, vectorized trilinear sampling
# ---------------------------------------------------------------------------

def _periodic_displacement(a, a0, L):
    """Minimum-image periodic displacement."""
    d = a - a0
    return d - L * torch.round(d / L)


def init_gaussian_blob_field(Nx, L, blob_amp, blob_sigma,
                             blob_center_x, blob_center_y, blob_center_z,
                             device=None):
    """Create a static periodic 3D Gaussian blob concentration field.

    Returns:
        c: (Nx, Nx, Nx) tensor — scalar concentration field.
    """
    dx = L / Nx
    coords = torch.arange(Nx, dtype=torch.float64, device=device) * dx
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing="ij")

    dxp = _periodic_displacement(X, blob_center_x, L)
    dyp = _periodic_displacement(Y, blob_center_y, L)
    dzp = _periodic_displacement(Z, blob_center_z, L)

    r2 = dxp * dxp + dyp * dyp + dzp * dzp
    c = blob_amp * torch.exp(-0.5 * r2 / (blob_sigma ** 2))
    return c


def compute_spectral_gradient(c, Nx, L):
    """Compute grad c on the grid using spectral (FFT) differentiation.

    Args:
        c: (Nx, Nx, Nx) tensor — scalar field.
        Nx: grid resolution.
        L: domain size.

    Returns:
        (gx, gy, gz): tuple of (Nx, Nx, Nx) tensors — gradient components.
    """
    dx = L / Nx
    k = 2.0 * torch.pi * torch.fft.fftfreq(Nx, d=dx, device=c.device, dtype=c.dtype)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing="ij")

    chat = torch.fft.fftn(c)
    gx = torch.real(torch.fft.ifftn(1j * kx * chat))
    gy = torch.real(torch.fft.ifftn(1j * ky * chat))
    gz = torch.real(torch.fft.ifftn(1j * kz * chat))
    return gx, gy, gz


def trilinear_sample_periodic(fields, pos, L, Nx):
    """Vectorised trilinear interpolation on a periodic 3D grid.

    Args:
        fields: list of (Nx, Nx, Nx) tensors to sample.
        pos: (N, 3) tensor — particle positions.
        L: domain size (scalar).
        Nx: grid resolution (int).

    Returns:
        list of (N,) tensors — sampled values, one per input field.
    """
    dx = L / Nx

    # grid coordinates (periodic)
    gx = (pos[:, 0] / dx) % Nx
    gy = (pos[:, 1] / dx) % Nx
    gz = (pos[:, 2] / dx) % Nx

    # integer lower-corner indices and fractional offsets
    ix = torch.floor(gx).long()
    iy = torch.floor(gy).long()
    iz = torch.floor(gz).long()

    fx = gx - ix.to(gx.dtype)
    fy = gy - iy.to(gy.dtype)
    fz = gz - iz.to(gz.dtype)

    ix1 = (ix + 1) % Nx
    iy1 = (iy + 1) % Nx
    iz1 = (iz + 1) % Nx

    # eight trilinear weights
    w000 = (1 - fx) * (1 - fy) * (1 - fz)
    w100 = fx * (1 - fy) * (1 - fz)
    w010 = (1 - fx) * fy * (1 - fz)
    w110 = fx * fy * (1 - fz)
    w001 = (1 - fx) * (1 - fy) * fz
    w101 = fx * (1 - fy) * fz
    w011 = (1 - fx) * fy * fz
    w111 = fx * fy * fz

    results = []
    for F in fields:
        val = (w000 * F[ix, iy, iz]
               + w100 * F[ix1, iy, iz]
               + w010 * F[ix, iy1, iz]
               + w110 * F[ix1, iy1, iz]
               + w001 * F[ix, iy, iz1]
               + w101 * F[ix1, iy, iz1]
               + w011 * F[ix, iy1, iz1]
               + w111 * F[ix1, iy1, iz1])
        results.append(val)

    return results


# ---------------------------------------------------------------------------
# Simulator module
# ---------------------------------------------------------------------------

@register_simulator("particle_spring_force_ode_static_field")
class ParticleSpringForceODEStaticField(nn.Module):
    """Overdamped cell dynamics with spring-based pair forces and chemotaxis.

    Combines spring repulsion/adhesion with chemotactic drift driven by a
    static concentration field, following dicty3d_minimal_grad_pair_noise.

    The concentration field is a periodic 3D Gaussian blob. Its gradient is
    computed spectrally (FFT) once at init. Each forward call trilinear-samples
    c and grad c at particle positions using pure PyTorch on the GPU.

    Each cell type has parameters p = (k_rep, r0, kadh, r_on, delta, mu_f, chi, beta_sig):
      - k_rep     : repulsion stiffness
      - r0        : equilibrium (rest) distance
      - kadh      : adhesion strength
      - r_on      : adhesion cutoff distance
      - delta     : sigmoid steepness for pair force gating
      - mu_f      : global force scaling factor
      - chi       : chemotaxis strength
      - beta_sig  : sigmoid steepness for gradient magnitude response

    Force law (pair):
      F_rep = k_rep * relu(r0 - r) * rhat
      g_on  = sigmoid((r - r0) / delta)
      g_off = sigmoid(-(r - r_on) / delta)
      F_adh = -kadh * g_on * g_off * (r - r0) * rhat
      F_pair = mu_f * (F_rep + F_adh)

    Chemotaxis:
      v_chem = chi * sigmoid(beta_sig * |grad c|) * grad_hat

    Total:
      d_pos = F_pair + v_chem + noise
    """

    def __init__(self, aggr_type=[], p=[], bc_dpos=[], dimension=3,
                 noise_model_level=0, field_config=None):
        super().__init__()

        self.aggr_type = aggr_type
        self.p = p
        self.bc_dpos = bc_dpos
        self.dimension = dimension
        self.noise_model_level = noise_model_level

        # --- build static concentration field via PyTorch ---
        fc = field_config or {}
        self.field_L = fc.get("L", 1.0)
        self.field_Nx = fc.get("Nx", 64)
        device = fc.get("device", None)

        c_grid = init_gaussian_blob_field(
            Nx=self.field_Nx,
            L=self.field_L,
            blob_amp=fc.get("blob_amp", 1.0),
            blob_sigma=fc.get("blob_sigma", 0.20),
            blob_center_x=fc.get("blob_center_x", 0.5),
            blob_center_y=fc.get("blob_center_y", 0.5),
            blob_center_z=fc.get("blob_center_z", 0.5),
            device=device,
        )

        # spectral gradient — computed once (field is static)
        gc_x, gc_y, gc_z = compute_spectral_gradient(
            c_grid, self.field_Nx, self.field_L,
        )

        # store as persistent buffers (move with .to(device), saved in state_dict)
        self.register_buffer("c_grid", c_grid)
        self.register_buffer("gc_x", gc_x)
        self.register_buffer("gc_y", gc_y)
        self.register_buffer("gc_z", gc_z)

    # ------------------------------------------------------------------
    # Field sampling
    # ------------------------------------------------------------------

    def sample_field(self, pos: torch.Tensor):
        """Sample c and grad c at particle positions.

        Args:
            pos: (N, 3) tensor — particle positions.

        Returns:
            c_i: (N, 1) tensor — concentration at particles.
            grad_c: (N, 3) tensor — concentration gradient at particles.
        """
        pos_d = pos.to(self.c_grid.dtype)

        c_val, gx_val, gy_val, gz_val = trilinear_sample_periodic(
            [self.c_grid, self.gc_x, self.gc_y, self.gc_z],
            pos_d, self.field_L, self.field_Nx,
        )

        dtype = pos.dtype
        c_i = c_val.to(dtype).unsqueeze(1)                  # (N, 1)
        grad_c = torch.stack([
            gx_val.to(dtype),
            gy_val.to(dtype),
            gz_val.to(dtype),
        ], dim=1)                                            # (N, 3)

        return c_i, grad_c

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, state: CellState, edge_index: torch.Tensor, has_field=False, k=0):
        # --- sample concentration field at particle positions ---
        c_i, grad_c = self.sample_field(state.pos)
        self.last_c = c_i
        self.last_grad_c = grad_c

        # --- pair forces via message passing ---
        edge_index = remove_self_loops(edge_index)
        p = self.p.unsqueeze(0) if self.p.dim() == 1 else self.p
        parameters = p[to_numpy(state.cell_type), :]

        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = state.pos[dst], state.pos[src]
        parameters_i = parameters[dst]

        messages = self.pair_message(pos_i, pos_j, parameters_i)
        d_pos_pair = scatter_aggregate(messages, dst, state.n_cells, self.aggr_type)

        # --- chemotactic drift ---
        chi = parameters[:, 6]           # (N,)
        beta_sig = parameters[:, 7]      # (N,)
        d_pos_chem = self.chemotaxis(grad_c, chi, beta_sig)

        # --- combine ---
        d_pos = d_pos_pair + d_pos_chem

        self.last_clean = d_pos.clone()
        self.last_pair = d_pos_pair.clone()
        self.last_chem = d_pos_chem.clone()

        if self.noise_model_level > 0:
            noise = self.noise_model_level * torch.randn_like(d_pos)
            self.last_noise = noise
            d_pos = d_pos + noise
        else:
            self.last_noise = torch.zeros_like(d_pos)

        return d_pos

    # ------------------------------------------------------------------
    # Pair forces
    # ------------------------------------------------------------------

    def pair_message(self, pos_i, pos_j, parameters_i):
        """Compute pair-wise spring force messages (repulsion + adhesion)."""
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

    # ------------------------------------------------------------------
    # Chemotaxis
    # ------------------------------------------------------------------

    def chemotaxis(self, grad_c, chi, beta_sig):
        """Compute chemotactic velocity: chi * sigmoid(beta_sig * |grad c|) * grad_hat.

        Args:
            grad_c: (N, dim) concentration gradient at each particle.
            chi: (N,) chemotaxis strength per particle.
            beta_sig: (N,) sigmoid steepness per particle.

        Returns:
            (N, dim) chemotactic velocity.
        """
        grad_mag = torch.norm(grad_c, dim=1, keepdim=True)  # (N, 1)
        grad_hat = grad_c / torch.clamp(grad_mag, min=1e-12)
        grad_hat = torch.where(grad_mag > 1e-12, grad_hat, torch.zeros_like(grad_hat))

        response = torch.sigmoid(beta_sig[:, None] * grad_mag)  # (N, 1)
        v_chem = chi[:, None] * response * grad_hat

        return v_chem

    # ------------------------------------------------------------------
    # Plotting utility
    # ------------------------------------------------------------------

    def psi(self, r, p):
        """Scalar pair-force profile for plotting.

        r is a 1-D tensor of distances,
        p = [k_rep, r0, kadh, r_on, delta, mu_f, chi, beta_sig].
        """
        k_rep, r0, kadh, r_on, delta, mu_f = p[0], p[1], p[2], p[3], p[4], p[5]
        delta_safe = max(delta, 1e-8)

        F_rep = k_rep * torch.relu(r0 - r)
        g_on = torch.sigmoid((r - r0) / delta_safe)
        g_off = torch.sigmoid(-(r - r_on) / delta_safe)
        F_adh = -kadh * g_on * g_off * (r - r0)

        return mu_f * (F_rep + F_adh)
