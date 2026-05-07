# communicating_activematter_spring.py
# Variant of communicating_activematter.py for GNN learning.
#
# The original script enforces non-overlap with a hard-core position
# projection (resolve_hard_core), which is a discontinuous correction and
# is not a smooth function of state — a GNN cannot learn it cleanly.
#
# Here, hard-core projection is replaced by a soft spring repulsion that
# enters the equation of motion as a continuous pairwise force:
#
#     dx_i/dt = v0 * cos(phi_i) + mu * sum_j F_rep(r_ij) * r_hat_ji
#     dy_i/dt = v0 * sin(phi_i) + mu * sum_j F_rep(r_ij) * r_hat_ji
#     F_rep(r) = k_rep * relu(2*rp - r)
#
# All other dynamics (polar alignment, chemotaxis, internal state s,
# chemical field c) are unchanged.
#
# Run:
#   python communicating_activematter_spring.py
#
# Output:
#   snapshots in ./frames_spring/
#   simulation data in communicating_active_matter_spring.npz

import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------

class Params:
    # domain
    L = 200.0
    Nx = 128
    Ny = 128

    # particles
    N = 8000
    rp = 0.4                 # particle radius (contact distance = 2*rp)
    v0 = 1.2                 # self-propulsion speed
    rc = 2.5                 # polar alignment radius

    # soft repulsion (replaces hard-core projection)
    k_rep = 50.0             # spring stiffness
    mu = 1.0                 # mobility (force -> velocity)

    # orientation dynamics
    Gamma = 0.8              # neighbor polar-alignment strength
    omega = 0.2              # chemotactic alignment strength
    DR = 0.05                # rotational diffusion

    # chemical field
    Dc = 8.0
    alpha = 0.8
    beta = 3.0
    source_width = 0.8

    # Schmitt-trigger-like internal dynamics
    eps = 0.25
    a = 4.0
    b = 0.15

    # numerics
    dt = 0.01
    T = 80.0
    save_every = 200
    seed = 1


P = Params()


# -----------------------------
# Helper functions (identical to communicating_activematter.py)
# -----------------------------

def periodic_displacement(dx, L):
    return dx - L * np.round(dx / L)


def deposit_bilinear(field, x, y, weight, L):
    Nx, Ny = field.shape
    dx = L / Nx
    dy = L / Ny
    gx = x / dx
    gy = y / dy
    i0 = np.floor(gx).astype(int) % Nx
    j0 = np.floor(gy).astype(int) % Ny
    tx = gx - np.floor(gx)
    ty = gy - np.floor(gy)
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    np.add.at(field, (i0, j0), weight * (1 - tx) * (1 - ty) / (dx * dy))
    np.add.at(field, (i1, j0), weight * tx * (1 - ty) / (dx * dy))
    np.add.at(field, (i0, j1), weight * (1 - tx) * ty / (dx * dy))
    np.add.at(field, (i1, j1), weight * tx * ty / (dx * dy))


def interp_bilinear(field, x, y, L):
    Nx, Ny = field.shape
    dx = L / Nx
    dy = L / Ny
    gx = x / dx
    gy = y / dy
    i0 = np.floor(gx).astype(int) % Nx
    j0 = np.floor(gy).astype(int) % Ny
    tx = gx - np.floor(gx)
    ty = gy - np.floor(gy)
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    return (
        field[i0, j0] * (1 - tx) * (1 - ty)
        + field[i1, j0] * tx * (1 - ty)
        + field[i0, j1] * (1 - tx) * ty
        + field[i1, j1] * tx * ty
    )


def make_neighbor_pairs(x, y, cutoff, L):
    N = len(x)
    ncell = max(3, int(L / cutoff))
    cell_size = L / ncell

    cells = [[[] for _ in range(ncell)] for _ in range(ncell)]
    cx = np.floor(x / cell_size).astype(int) % ncell
    cy = np.floor(y / cell_size).astype(int) % ncell
    for i in range(N):
        cells[cx[i]][cy[i]].append(i)

    pairs = []
    for ix in range(ncell):
        for iy in range(ncell):
            here = cells[ix][iy]
            if not here:
                continue
            for dix in [-1, 0, 1]:
                for diy in [-1, 0, 1]:
                    jx = (ix + dix) % ncell
                    jy = (iy + diy) % ncell
                    neigh = cells[jx][jy]
                    if not neigh:
                        continue
                    for i in here:
                        for j in neigh:
                            if j <= i:
                                continue
                            dx = periodic_displacement(x[j] - x[i], L)
                            dy = periodic_displacement(y[j] - y[i], L)
                            r2 = dx * dx + dy * dy
                            if r2 < cutoff * cutoff:
                                pairs.append((i, j, dx, dy, np.sqrt(r2)))
    return pairs


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# Soft pair repulsion (replaces resolve_hard_core)
# -----------------------------

def compute_pair_repulsion(x, y, rp, k_rep, L):
    """Continuous soft-disk repulsion: F_ij = k_rep * relu(2*rp - r_ij).

    Returns (Fx, Fy) of shape (N,) — net repulsive force on each particle,
    with Newton's-third-law symmetry between pair endpoints.
    """
    N = len(x)
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    cutoff = 2.0 * rp
    pairs = make_neighbor_pairs(x, y, cutoff, L)

    for i, j, dx, dy, r in pairs:
        if r < 1e-12:
            angle = 2 * np.pi * np.random.rand()
            dx = np.cos(angle) * 1e-3
            dy = np.sin(angle) * 1e-3
            r = np.sqrt(dx * dx + dy * dy)

        overlap = cutoff - r
        if overlap <= 0:
            continue

        f_mag = k_rep * overlap
        ux = dx / r              # from i to j
        uy = dy / r

        # repulsion pushes i away from j (along -u) and j away from i (along +u)
        Fx[i] -= f_mag * ux
        Fy[i] -= f_mag * uy
        Fx[j] += f_mag * ux
        Fy[j] += f_mag * uy

    return Fx, Fy


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

    frames_dir = os.path.join(out_dir, "frames_spring")
    os.makedirs(frames_dir, exist_ok=True)

    L = P.L
    Nx = P.Nx
    Ny = P.Ny
    dx = L / Nx

    # particle states
    x = L * np.random.rand(P.N)
    y = L * np.random.rand(P.N)
    phi = 2 * np.pi * np.random.rand(P.N)
    s = np.zeros(P.N)

    # chemical field
    c = np.zeros((Nx, Ny))

    # small initial chemical perturbation to seed wave activity
    Xg = np.linspace(0, L, Nx, endpoint=False)
    Yg = np.linspace(0, L, Ny, endpoint=False)
    XX, YY = np.meshgrid(Xg, Yg, indexing="ij")
    for _ in range(6):
        x0 = L * np.random.rand()
        y0 = L * np.random.rand()
        ddx = periodic_displacement(XX - x0, L)
        ddy = periodic_displacement(YY - y0, L)
        c += 0.5 * np.exp(-(ddx**2 + ddy**2) / (2 * 3.0**2))

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    n_steps = int(P.T / P.dt)

    saved_x = []
    saved_y = []
    saved_phi = []
    saved_s = []
    saved_c = []
    saved_t = []
    saved_fpair = []  # repulsive force per particle, useful as GNN target/diagnostic

    for step in range(n_steps + 1):
        t = step * P.dt

        # Soft pair repulsion — computed once per step, used in position update and saved.
        Fx_rep, Fy_rep = compute_pair_repulsion(x, y, P.rp, P.k_rep, L)

        # -----------------------------
        # Save / plot
        # -----------------------------
        if step % P.save_every == 0:
            if not quiet:
                # report worst overlap so the user can tune k_rep
                pairs = make_neighbor_pairs(x, y, 2.0 * P.rp, L)
                max_overlap = 0.0
                for _, _, _, _, r in pairs:
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
            saved_phi.append(phi.copy())
            saved_s.append(s.copy())
            saved_c.append(c.copy())
            saved_t.append(t)
            saved_fpair.append(np.stack([Fx_rep, Fy_rep], axis=-1))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(
                c.T,
                origin="lower",
                extent=[0, L, 0, L],
                cmap="viridis",
                alpha=0.85,
            )
            ax.scatter(
                x,
                y,
                c=phi,
                s=8,
                cmap="hsv",
                vmin=0,
                vmax=2 * np.pi,
                edgecolors="none",
            )
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_aspect("equal")
            ax.set_title(f"t = {t:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(frames_dir, f"frame_{step:06d}.png"), dpi=160)
            plt.close(fig)

        # -----------------------------
        # Chemical gradients at particle positions
        # -----------------------------
        chat = np.fft.fft2(c)
        dc_dx = np.real(np.fft.ifft2(1j * KX * chat))
        dc_dy = np.real(np.fft.ifft2(1j * KY * chat))

        c_i = interp_bilinear(c, x, y, L)
        gx_i = interp_bilinear(dc_dx, x, y, L)
        gy_i = interp_bilinear(dc_dy, x, y, L)
        phi_c = np.arctan2(gy_i, gx_i)

        # -----------------------------
        # Orientation dynamics
        # -----------------------------
        dphi = np.zeros(P.N)
        pairs = make_neighbor_pairs(x, y, P.rc, L)
        for i, j, ddx, ddy, r in pairs:
            if r < 1e-12:
                continue
            torque_i = -P.Gamma * np.sin(phi[i] - phi[j]) / r
            torque_j = -P.Gamma * np.sin(phi[j] - phi[i]) / r
            dphi[i] += torque_i
            dphi[j] += torque_j

        chem_torque = P.omega * np.sin(phi_c - phi)
        noise = np.sqrt(2 * P.DR * P.dt) * np.random.randn(P.N)

        phi += P.dt * (dphi + chem_torque) + noise
        phi %= 2 * np.pi

        # -----------------------------
        # Position update — soft repulsion ENTERS the equation of motion
        # (this is the only structural change vs. communicating_activematter.py)
        # -----------------------------
        x += P.dt * (P.v0 * np.cos(phi) + P.mu * Fx_rep)
        y += P.dt * (P.v0 * np.sin(phi) + P.mu * Fy_rep)
        x %= L
        y %= L

        # -----------------------------
        # Source production
        # -----------------------------
        c_th = (s + P.b) / P.a
        active = c_i > c_th
        source_strength = P.beta * np.maximum(1.0 - s, 0.0) * active.astype(float)

        source = np.zeros_like(c)
        deposit_bilinear(source, x, y, source_strength, L)

        source_hat = np.fft.fft2(source)
        gaussian_filter = np.exp(-0.5 * P.source_width**2 * K2)
        source = np.real(np.fft.ifft2(source_hat * gaussian_filter))

        # -----------------------------
        # Chemical field update
        # -----------------------------
        chat = np.fft.fft2(c)
        source_hat = np.fft.fft2(source)
        chat_new = (chat + P.dt * source_hat) / (1.0 + P.dt * (P.Dc * K2 + P.alpha))
        c = np.real(np.fft.ifft2(chat_new))
        c = np.maximum(c, 0.0)

        # -----------------------------
        # Internal state update
        # -----------------------------
        c_i_new = interp_bilinear(c, x, y, L)
        s += P.dt * P.eps * (c_i_new - s)
        s = np.clip(s, 0.0, 1.5)

    npz_path = os.path.join(out_dir, "communicating_active_matter_spring.npz")
    np.savez_compressed(
        npz_path,
        t=np.array(saved_t),
        x=np.array(saved_x),
        y=np.array(saved_y),
        phi=np.array(saved_phi),
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
