# communicating_active_matter_3d.py
# 3D extension of communicating active matter simulation with soft spring repulsion.
# Inspired by:
# Ziepke, Maryshev, Aranson & Frey,
# "Multi-scale organization in communicating active matter",
# Nature Communications 13, 6727 (2022).
#
# Hard-core position projection (resolve_hard_core_3d) is replaced by a
# continuous soft spring repulsion that enters the equation of motion as a
# pairwise force (mirroring communicating_activematter_spring.py):
#
#     dx_i/dt = v0 * n_x + mu * sum_j F_rep(r_ij) * r_hat_ji
#     F_rep(r) = k_rep * relu(2*rp - r)
#
# All other dynamics (polar alignment torque, chemotaxis, internal state s,
# chemical field c) are unchanged.
#
# Run:
#   python communicating_activematter_3d.py
#
# Output:
#   snapshots in ./frames_3d_spring/
#   simulation data in communicating_active_matter_3d_spring.npz

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------
# Parameters
# -----------------------------

class Params:
    # domain
    L = 80.0
    Nx = 64
    Ny = 64
    Nz = 64

    # particles
    N = 1000
    rp = 0.1                 # particle radius (contact distance = 2*rp)
    v0 = 0.15                 # self-propulsion speed
    rc = 2.5                 # polar alignment radius

    # soft repulsion (replaces hard-core projection)
    k_rep = 50.0             # spring stiffness
    mu = 1.0                 # mobility (force -> velocity)

    # orientation dynamics
    Gamma = 6              # neighbor polar-alignment strength
    omega = 6              # chemotactic alignment strength
    DR = 0.05                # rotational diffusion

    # chemical field
    Dc = 2.0                 # chemical diffusion
    alpha = 10              # chemical degradation
    beta = 3.0               # source strength
    source_width = 0.8       # Gaussian source width

    # Schmitt-trigger-like internal dynamics
    eps = 0.25               # relaxation rate of internal state s
    a = 4.0                  # threshold slope parameter
    b = 0.15                 # baseline threshold

    # numerics
    dt = 0.01
    T = 80.0
    save_every = 200
    seed = 1


P = Params()


# -----------------------------
# Helper functions
# -----------------------------

def periodic_displacement(dx, L):
    """Minimum-image displacement."""
    return dx - L * np.round(dx / L)


def deposit_trilinear(field, x, y, z, weight, L):
    """
    Deposit particle source onto 3D grid using trilinear interpolation.
    field has shape (Nx, Ny, Nz), periodic.
    """
    Nx, Ny, Nz = field.shape
    hx = L / Nx
    hy = L / Ny
    hz = L / Nz
    vol = hx * hy * hz

    gx = x / hx
    gy = y / hy
    gz = z / hz

    i0 = np.floor(gx).astype(int) % Nx
    j0 = np.floor(gy).astype(int) % Ny
    k0 = np.floor(gz).astype(int) % Nz

    tx = gx - np.floor(gx)
    ty = gy - np.floor(gy)
    tz = gz - np.floor(gz)

    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    k1 = (k0 + 1) % Nz

    w = weight / vol

    np.add.at(field, (i0, j0, k0), w * (1 - tx) * (1 - ty) * (1 - tz))
    np.add.at(field, (i1, j0, k0), w * tx       * (1 - ty) * (1 - tz))
    np.add.at(field, (i0, j1, k0), w * (1 - tx) * ty       * (1 - tz))
    np.add.at(field, (i1, j1, k0), w * tx       * ty       * (1 - tz))
    np.add.at(field, (i0, j0, k1), w * (1 - tx) * (1 - ty) * tz)
    np.add.at(field, (i1, j0, k1), w * tx       * (1 - ty) * tz)
    np.add.at(field, (i0, j1, k1), w * (1 - tx) * ty       * tz)
    np.add.at(field, (i1, j1, k1), w * tx       * ty       * tz)


def interp_trilinear(field, x, y, z, L):
    """Interpolate 3D grid field to particle positions."""
    Nx, Ny, Nz = field.shape
    hx = L / Nx
    hy = L / Ny
    hz = L / Nz

    gx = x / hx
    gy = y / hy
    gz = z / hz

    i0 = np.floor(gx).astype(int) % Nx
    j0 = np.floor(gy).astype(int) % Ny
    k0 = np.floor(gz).astype(int) % Nz

    tx = gx - np.floor(gx)
    ty = gy - np.floor(gy)
    tz = gz - np.floor(gz)

    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    k1 = (k0 + 1) % Nz

    return (
        field[i0, j0, k0] * (1 - tx) * (1 - ty) * (1 - tz)
      + field[i1, j0, k0] * tx       * (1 - ty) * (1 - tz)
      + field[i0, j1, k0] * (1 - tx) * ty       * (1 - tz)
      + field[i1, j1, k0] * tx       * ty       * (1 - tz)
      + field[i0, j0, k1] * (1 - tx) * (1 - ty) * tz
      + field[i1, j0, k1] * tx       * (1 - ty) * tz
      + field[i0, j1, k1] * (1 - tx) * ty       * tz
      + field[i1, j1, k1] * tx       * ty       * tz
    )


def make_neighbor_pairs_3d(x, y, z, cutoff, L):
    """
    3D cell-list neighbor search.
    Returns pairs (i, j, dx, dy, dz, r) with distance < cutoff.
    """
    N = len(x)
    ncell = max(3, int(L / cutoff))
    cell_size = L / ncell

    cells = [[[[] for _ in range(ncell)] for _ in range(ncell)] for _ in range(ncell)]

    cx = np.floor(x / cell_size).astype(int) % ncell
    cy = np.floor(y / cell_size).astype(int) % ncell
    cz = np.floor(z / cell_size).astype(int) % ncell

    for i in range(N):
        cells[cx[i]][cy[i]][cz[i]].append(i)

    pairs = []
    cutoff2 = cutoff * cutoff

    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                here = cells[ix][iy][iz]
                if not here:
                    continue

                for dix in [-1, 0, 1]:
                    for diy in [-1, 0, 1]:
                        for diz in [-1, 0, 1]:
                            jx = (ix + dix) % ncell
                            jy = (iy + diy) % ncell
                            jz = (iz + diz) % ncell

                            neigh = cells[jx][jy][jz]
                            if not neigh:
                                continue

                            for i in here:
                                for j in neigh:
                                    if j <= i:
                                        continue

                                    ddx = periodic_displacement(x[j] - x[i], L)
                                    ddy = periodic_displacement(y[j] - y[i], L)
                                    ddz = periodic_displacement(z[j] - z[i], L)
                                    r2 = ddx * ddx + ddy * ddy + ddz * ddz

                                    if r2 < cutoff2:
                                        pairs.append((i, j, ddx, ddy, ddz, np.sqrt(r2)))

    return pairs


def compute_pair_repulsion_3d(x, y, z, rp, k_rep, L):
    """Continuous soft-disk repulsion in 3D: F_ij = k_rep * relu(2*rp - r_ij).

    Returns (Fx, Fy, Fz) of shape (N,) — net repulsive force on each particle,
    with Newton's-third-law symmetry between pair endpoints.
    """
    N = len(x)
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    Fz = np.zeros(N)
    cutoff = 2.0 * rp
    pairs = make_neighbor_pairs_3d(x, y, z, cutoff, L)

    for i, j, ddx, ddy, ddz, r in pairs:
        if r < 1e-12:
            # random direction on the sphere
            theta_r = np.arccos(2 * np.random.rand() - 1)
            phi_r = 2 * np.pi * np.random.rand()
            ddx = 1e-3 * np.sin(theta_r) * np.cos(phi_r)
            ddy = 1e-3 * np.sin(theta_r) * np.sin(phi_r)
            ddz = 1e-3 * np.cos(theta_r)
            r = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)

        overlap = cutoff - r
        if overlap <= 0:
            continue

        f_mag = k_rep * overlap
        ux = ddx / r              # from i to j
        uy = ddy / r
        uz = ddz / r

        # repulsion pushes i away from j (along -u) and j away from i (along +u)
        Fx[i] -= f_mag * ux
        Fy[i] -= f_mag * uy
        Fz[i] -= f_mag * uz
        Fx[j] += f_mag * ux
        Fy[j] += f_mag * uy
        Fz[j] += f_mag * uz

    return Fx, Fy, Fz


def random_unit_vectors(n):
    """Sample n uniform random unit vectors on the sphere."""
    theta = np.arccos(2 * np.random.rand(n) - 1)
    phi_az = 2 * np.pi * np.random.rand(n)
    return theta, phi_az


def orientation_to_cartesian(theta, phi_az):
    """Convert (theta, phi_az) to unit vector (nx, ny, nz)."""
    nx = np.sin(theta) * np.cos(phi_az)
    ny = np.sin(theta) * np.sin(phi_az)
    nz = np.cos(theta)
    return nx, ny, nz


def rotate_orientation(theta, phi_az, torque_x, torque_y, torque_z, dt):
    """
    Update orientation angles given a torque vector.
    The torque is the angular velocity omega = d(n_hat)/dt cross-product form.

    We use Rodrigues' rotation: rotate the orientation unit vector by
    omega * dt about the omega axis.
    """
    nx, ny, nz = orientation_to_cartesian(theta, phi_az)

    # angular displacement
    wx = torque_x * dt
    wy = torque_y * dt
    wz = torque_z * dt

    angle = np.sqrt(wx**2 + wy**2 + wz**2)
    mask = angle > 1e-12

    # for particles with nonzero torque, apply Rodrigues rotation
    if np.any(mask):
        ax = np.where(mask, wx / np.where(mask, angle, 1.0), 0.0)
        ay = np.where(mask, wy / np.where(mask, angle, 1.0), 0.0)
        az = np.where(mask, wz / np.where(mask, angle, 1.0), 0.0)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rodrigues: n' = n cos(a) + (k x n) sin(a) + k (k . n)(1 - cos(a))
        dot = ax * nx + ay * ny + az * nz

        # cross product k x n
        cx = ay * nz - az * ny
        cy = az * nx - ax * nz
        cz = ax * ny - ay * nx

        nx_new = np.where(mask, nx * cos_a + cx * sin_a + ax * dot * (1 - cos_a), nx)
        ny_new = np.where(mask, ny * cos_a + cy * sin_a + ay * dot * (1 - cos_a), ny)
        nz_new = np.where(mask, nz * cos_a + cz * sin_a + az * dot * (1 - cos_a), nz)

        nx, ny, nz = nx_new, ny_new, nz_new

    # normalize
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    # back to angles
    nz_clipped = np.clip(nz, -1.0, 1.0)
    theta_new = np.arccos(nz_clipped)
    phi_az_new = np.arctan2(ny, nx) % (2 * np.pi)

    return theta_new, phi_az_new


def rotational_diffusion_3d(theta, phi_az, DR, dt, N):
    """
    Apply rotational diffusion on the sphere.
    Generate a random rotation perpendicular to the current orientation.
    """
    nx, ny, nz = orientation_to_cartesian(theta, phi_az)

    # random angular displacement magnitude
    sigma = np.sqrt(2 * DR * dt)

    # two random angles in the tangent plane
    dw1 = sigma * np.random.randn(N)
    dw2 = sigma * np.random.randn(N)

    # construct two orthogonal tangent vectors for each particle
    # e1 = z_hat x n / |z_hat x n|, with fallback for poles
    e1x = -ny.copy()
    e1y = nx.copy()
    e1z = np.zeros(N)
    e1_norm = np.sqrt(e1x**2 + e1y**2 + e1z**2)

    # for particles near poles, use x_hat x n instead
    pole_mask = e1_norm < 1e-6
    if np.any(pole_mask):
        e1y[pole_mask] = -nz[pole_mask]
        e1z[pole_mask] = ny[pole_mask]
        e1x[pole_mask] = 0.0
        e1_norm = np.sqrt(e1x**2 + e1y**2 + e1z**2)

    e1x /= e1_norm
    e1y /= e1_norm
    e1z /= e1_norm

    # e2 = n x e1
    e2x = ny * e1z - nz * e1y
    e2y = nz * e1x - nx * e1z
    e2z = nx * e1y - ny * e1x

    # random rotation vector in the tangent plane
    wx = dw1 * e1x + dw2 * e2x
    wy = dw1 * e1y + dw2 * e2y
    wz = dw1 * e1z + dw2 * e2z

    # apply as finite rotation (reuse rotate_orientation with dt=1)
    theta_new, phi_az_new = rotate_orientation(theta, phi_az, wx, wy, wz, dt=1.0)
    return theta_new, phi_az_new


# -----------------------------
# Main simulation
# -----------------------------

def run(out_dir=".", overrides=None, quiet=False):
    overrides = overrides or {}
    for key, val in overrides.items():
        if not hasattr(P, key):
            raise ValueError(f"Unknown parameter override: {key}")
        setattr(P, key, val)

    np.random.seed(P.seed)

    frames_dir = os.path.join(out_dir, "frames_3d_spring")
    os.makedirs(frames_dir, exist_ok=True)

    L = P.L
    Nx = P.Nx
    Ny = P.Ny
    Nz = P.Nz
    hx = L / Nx

    # particle states
    x = L * np.random.rand(P.N)
    y = L * np.random.rand(P.N)
    z = L * np.random.rand(P.N)
    theta, phi_az = random_unit_vectors(P.N)
    s = np.zeros(P.N)

    # chemical field
    c = np.zeros((Nx, Ny, Nz))

    # initial chemical perturbation
    Xg = np.linspace(0, L, Nx, endpoint=False)
    Yg = np.linspace(0, L, Ny, endpoint=False)
    Zg = np.linspace(0, L, Nz, endpoint=False)
    XX, YY, ZZ = np.meshgrid(Xg, Yg, Zg, indexing="ij")

    for _ in range(6):
        x0 = L * np.random.rand()
        y0 = L * np.random.rand()
        z0 = L * np.random.rand()
        ddx = periodic_displacement(XX - x0, L)
        ddy = periodic_displacement(YY - y0, L)
        ddz = periodic_displacement(ZZ - z0, L)
        c += 0.5 * np.exp(-(ddx**2 + ddy**2 + ddz**2) / (2 * 3.0**2))

    # Fourier wave numbers for chemical diffusion
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=hx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=hx)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=hx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2

    n_steps = int(P.T / P.dt)

    saved_x = []
    saved_y = []
    saved_z = []
    saved_theta = []
    saved_phi_az = []
    saved_s = []
    saved_c = []
    saved_t = []
    saved_fpair = []  # repulsive force per particle, useful as GNN target/diagnostic

    for step in range(n_steps + 1):
        t = step * P.dt

        # Soft pair repulsion — computed once per step, used in position update and saved.
        Fx_rep, Fy_rep, Fz_rep = compute_pair_repulsion_3d(x, y, z, P.rp, P.k_rep, L)

        # -----------------------------
        # Save / plot
        # -----------------------------
        if step % P.save_every == 0:
            if not quiet:
                # report worst overlap so the user can tune k_rep
                pairs = make_neighbor_pairs_3d(x, y, z, 2.0 * P.rp, L)
                max_overlap = 0.0
                for _, _, _, _, _, r in pairs:
                    overlap = 2.0 * P.rp - r
                    if overlap > max_overlap:
                        max_overlap = overlap
                print(
                    f"step {step:6d} / {n_steps}, t = {t:.2f}, "
                    f"max c = {c.max():.3f}, max overlap = {max_overlap:.3f} "
                    f"(rp = {P.rp})"
                )

            saved_x.append(x.copy())
            saved_y.append(y.copy())
            saved_z.append(z.copy())
            saved_theta.append(theta.copy())
            saved_phi_az.append(phi_az.copy())
            saved_s.append(s.copy())
            saved_c.append(c.copy())
            saved_t.append(t)
            saved_fpair.append(np.stack([Fx_rep, Fy_rep, Fz_rep], axis=-1))

            # plot: 3D scatter + midplane chemical slice
            fig = plt.figure(figsize=(14, 6))

            # 3D particle scatter
            ax1 = fig.add_subplot(121, projection='3d')
            nx_p, ny_p, nz_p = orientation_to_cartesian(theta, phi_az)
            # color by polar angle theta for orientation visualization
            colors = theta / np.pi
            ax1.scatter(x, y, z, c=colors, s=4, cmap='hsv',
                        vmin=0, vmax=1, alpha=0.6, depthshade=True)
            ax1.set_xlim(0, L)
            ax1.set_ylim(0, L)
            ax1.set_zlim(0, L)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title(f'Particles, t = {t:.2f}')

            # midplane chemical field slice
            ax2 = fig.add_subplot(122)
            mid_z = Nz // 2
            ax2.imshow(
                c[:, :, mid_z].T,
                origin="lower",
                extent=[0, L, 0, L],
                cmap="viridis",
                alpha=0.85,
            )
            ax2.set_xlim(0, L)
            ax2.set_ylim(0, L)
            ax2.set_aspect("equal")
            ax2.set_title(f'Chemical (z-midplane), t = {t:.2f}')
            ax2.set_xticks([])
            ax2.set_yticks([])

            plt.tight_layout()
            plt.savefig(os.path.join(frames_dir, f"frame_{step:06d}.png"), dpi=160)
            plt.close(fig)

        # -----------------------------
        # Chemical gradients (3D spectral)
        # -----------------------------
        chat = np.fft.fftn(c)
        dc_dx = np.real(np.fft.ifftn(1j * KX * chat))
        dc_dy = np.real(np.fft.ifftn(1j * KY * chat))
        dc_dz = np.real(np.fft.ifftn(1j * KZ * chat))

        c_i = interp_trilinear(c, x, y, z, L)
        gx_i = interp_trilinear(dc_dx, x, y, z, L)
        gy_i = interp_trilinear(dc_dy, x, y, z, L)
        gz_i = interp_trilinear(dc_dz, x, y, z, L)

        # chemical gradient direction as a unit vector
        grad_norm = np.sqrt(gx_i**2 + gy_i**2 + gz_i**2)
        grad_mask = grad_norm > 1e-12
        gc_x = np.where(grad_mask, gx_i / grad_norm, 0.0)
        gc_y = np.where(grad_mask, gy_i / grad_norm, 0.0)
        gc_z = np.where(grad_mask, gz_i / grad_norm, 0.0)

        # -----------------------------
        # Orientation dynamics
        # In 3D, torques are computed as omega vectors:
        #   polar alignment: -Gamma * (n_i x n_j) / r_ij
        #   chemotactic:      omega * (n_i x grad_c_hat)
        # Both produce rotation axes perpendicular to n_i.
        # -----------------------------
        nx_p, ny_p, nz_p = orientation_to_cartesian(theta, phi_az)

        torque_x = np.zeros(P.N)
        torque_y = np.zeros(P.N)
        torque_z = np.zeros(P.N)

        pairs = make_neighbor_pairs_3d(x, y, z, P.rc, L)

        for i, j, ddx, ddy, ddz, r in pairs:
            if r < 1e-12:
                continue

            # polar alignment torque: -Gamma * (n_i x n_j) / r
            # n_i x n_j
            cx_ij = ny_p[i] * nz_p[j] - nz_p[i] * ny_p[j]
            cy_ij = nz_p[i] * nx_p[j] - nx_p[i] * nz_p[j]
            cz_ij = nx_p[i] * ny_p[j] - ny_p[i] * nx_p[j]

            fac = -P.Gamma / r
            torque_x[i] += fac * cx_ij
            torque_y[i] += fac * cy_ij
            torque_z[i] += fac * cz_ij

            # Newton's third law for alignment: torque on j from i
            torque_x[j] -= fac * cx_ij
            torque_y[j] -= fac * cy_ij
            torque_z[j] -= fac * cz_ij

        # chemotactic alignment torque: omega * (n_i x grad_c_hat)
        chem_tx = P.omega * (ny_p * gc_z - nz_p * gc_y)
        chem_ty = P.omega * (nz_p * gc_x - nx_p * gc_z)
        chem_tz = P.omega * (nx_p * gc_y - ny_p * gc_x)

        torque_x += chem_tx
        torque_y += chem_ty
        torque_z += chem_tz

        # apply deterministic torque rotation
        theta, phi_az = rotate_orientation(theta, phi_az,
                                           torque_x, torque_y, torque_z, P.dt)

        # apply rotational diffusion
        theta, phi_az = rotational_diffusion_3d(theta, phi_az, P.DR, P.dt, P.N)

        # -----------------------------
        # Position update — soft repulsion ENTERS the equation of motion
        # (this is the only structural change vs. the hard-core 3D version)
        # -----------------------------
        nx_p, ny_p, nz_p = orientation_to_cartesian(theta, phi_az)
        x += P.dt * (P.v0 * nx_p + P.mu * Fx_rep)
        y += P.dt * (P.v0 * ny_p + P.mu * Fy_rep)
        z += P.dt * (P.v0 * nz_p + P.mu * Fz_rep)
        x %= L
        y %= L
        z %= L

        # -----------------------------
        # Source production (Schmitt trigger)
        # -----------------------------
        c_th = (s + P.b) / P.a
        active = c_i > c_th
        source_strength = P.beta * np.maximum(1.0 - s, 0.0) * active.astype(float)

        source = np.zeros_like(c)
        deposit_trilinear(source, x, y, z, source_strength, L)

        # smooth source with Gaussian kernel in Fourier space
        source_hat = np.fft.fftn(source)
        gaussian_filter = np.exp(-0.5 * P.source_width**2 * K2)
        source = np.real(np.fft.ifftn(source_hat * gaussian_filter))

        # -----------------------------
        # Chemical field update (semi-implicit spectral)
        # -----------------------------
        chat = np.fft.fftn(c)
        source_hat = np.fft.fftn(source)

        chat_new = (chat + P.dt * source_hat) / (1.0 + P.dt * (P.Dc * K2 + P.alpha))
        c = np.real(np.fft.ifftn(chat_new))
        c = np.maximum(c, 0.0)

        # -----------------------------
        # Internal state update
        # -----------------------------
        c_i_new = interp_trilinear(c, x, y, z, L)
        s += P.dt * P.eps * (c_i_new - s)
        s = np.clip(s, 0.0, 1.5)

    npz_path = os.path.join(out_dir, "communicating_active_matter_3d_spring.npz")
    np.savez_compressed(
        npz_path,
        t=np.array(saved_t),
        x=np.array(saved_x),
        y=np.array(saved_y),
        z=np.array(saved_z),
        theta=np.array(saved_theta),
        phi_az=np.array(saved_phi_az),
        s=np.array(saved_s),
        c=np.array(saved_c),
        fpair=np.array(saved_fpair),
        params={k: v for k, v in P.__dict__.items() if not k.startswith("_")},
    )

    if not quiet:
        print(f"Saved: {npz_path}")
        print(f"Saved frames in {frames_dir}")


if __name__ == "__main__":
    run()
