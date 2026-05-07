# visualize_communicating_active_matter.py
# Load 2D or 3D communicating_active_matter npz and produce an MP4 video.
# Auto-detects 2D vs 3D based on the presence of a "z" field.
#
# Usage:
#   python visualize_communicating_active_matter.py [path_to_npz]
#
# Requires: matplotlib, numpy, ffmpeg (for mp4 output)

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -----------------------------
# Load data
# -----------------------------
npz_path = sys.argv[1] if len(sys.argv) > 1 else "communicating_active_matter_spring.npz"
data = np.load(npz_path, allow_pickle=True)

t = data["t"]
x = data["x"]
y = data["y"]
s = data["s"]
c = data["c"]

is_3d = "z" in data

if is_3d:
    z = data["z"]
    theta = data["theta"]
    phi_az = data["phi_az"]
else:
    phi = data["phi"]

# try to recover L from params dict
try:
    params = data["params"].item()
    L = params.get("L", 80.0)
except Exception:
    L = 80.0

n_frames = len(t)

if is_3d:
    print(f"Loaded {npz_path} (3D): {n_frames} frames, {x.shape[1]} particles, "
          f"grid {c.shape[1]}x{c.shape[2]}x{c.shape[3]}, L={L}")
else:
    print(f"Loaded {npz_path} (2D): {n_frames} frames, {x.shape[1]} particles, "
          f"grid {c.shape[1]}x{c.shape[2]}, L={L}")


# -----------------------------
# 2D visualization
# -----------------------------
def make_anim_2d():
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        c[0].T,
        origin="lower",
        extent=[0, L, 0, L],
        cmap="viridis",
        alpha=0.85,
        vmin=0,
        vmax=c.max() * 0.8,
    )
    sc = ax.scatter(
        x[0], y[0],
        c=phi[0],
        s=8,
        cmap="hsv",
        vmin=0, vmax=2 * np.pi,
        edgecolors="none",
    )
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(f"t = {t[0]:.2f}")
    plt.tight_layout()

    def update(frame):
        im.set_data(c[frame].T)
        sc.set_offsets(np.column_stack([x[frame], y[frame]]))
        sc.set_array(phi[frame])
        title.set_text(f"t = {t[frame]:.2f}")
        return im, sc, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    return fig, anim


# -----------------------------
# 3D visualization
# Shows three panels:
#   1. XY midplane slice of chemical field + particle projection
#   2. XZ midplane slice + projection
#   3. YZ midplane slice + projection
# -----------------------------
def make_anim_3d():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_xy, ax_xz, ax_yz = axes

    Nz_grid = c.shape[3]
    Ny_grid = c.shape[2]
    Nx_grid = c.shape[1]
    mid_z = Nz_grid // 2
    mid_y = Ny_grid // 2
    mid_x = Nx_grid // 2

    # use theta for color (orientation polar angle, 0..pi)
    color_vals = theta[0] / np.pi
    c_max = c.max() * 0.8

    # XY midplane (z = L/2)
    im_xy = ax_xy.imshow(
        c[0][:, :, mid_z].T, origin="lower", extent=[0, L, 0, L],
        cmap="viridis", alpha=0.85, vmin=0, vmax=c_max,
    )
    sc_xy = ax_xy.scatter(x[0], y[0], c=color_vals, s=4, cmap="hsv",
                          vmin=0, vmax=1, edgecolors="none", alpha=0.5)
    ax_xy.set_xlim(0, L); ax_xy.set_ylim(0, L)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_title("XY midplane")

    # XZ midplane (y = L/2)
    im_xz = ax_xz.imshow(
        c[0][:, mid_y, :].T, origin="lower", extent=[0, L, 0, L],
        cmap="viridis", alpha=0.85, vmin=0, vmax=c_max,
    )
    sc_xz = ax_xz.scatter(x[0], z[0], c=color_vals, s=4, cmap="hsv",
                          vmin=0, vmax=1, edgecolors="none", alpha=0.5)
    ax_xz.set_xlim(0, L); ax_xz.set_ylim(0, L)
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.set_title("XZ midplane")

    # YZ midplane (x = L/2)
    im_yz = ax_yz.imshow(
        c[0][mid_x, :, :].T, origin="lower", extent=[0, L, 0, L],
        cmap="viridis", alpha=0.85, vmin=0, vmax=c_max,
    )
    sc_yz = ax_yz.scatter(y[0], z[0], c=color_vals, s=4, cmap="hsv",
                          vmin=0, vmax=1, edgecolors="none", alpha=0.5)
    ax_yz.set_xlim(0, L); ax_yz.set_ylim(0, L)
    ax_yz.set_aspect("equal")
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")
    ax_yz.set_title("YZ midplane")

    suptitle = fig.suptitle(f"t = {t[0]:.2f}", fontsize=14)
    plt.tight_layout()

    def update(frame):
        color_vals = theta[frame] / np.pi

        im_xy.set_data(c[frame][:, :, mid_z].T)
        sc_xy.set_offsets(np.column_stack([x[frame], y[frame]]))
        sc_xy.set_array(color_vals)

        im_xz.set_data(c[frame][:, mid_y, :].T)
        sc_xz.set_offsets(np.column_stack([x[frame], z[frame]]))
        sc_xz.set_array(color_vals)

        im_yz.set_data(c[frame][mid_x, :, :].T)
        sc_yz.set_offsets(np.column_stack([y[frame], z[frame]]))
        sc_yz.set_array(color_vals)

        suptitle.set_text(f"t = {t[frame]:.2f}")
        return im_xy, sc_xy, im_xz, sc_xz, im_yz, sc_yz, suptitle

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    return fig, anim


# -----------------------------
# Create and save
# -----------------------------
if is_3d:
    fig, anim = make_anim_3d()
else:
    fig, anim = make_anim_2d()

out_path = npz_path.replace(".npz", ".mp4")
try:
    writer = FFMpegWriter(fps=15, bitrate=3000)
    anim.save(out_path, writer=writer)
    print(f"Saved video: {out_path}")
except Exception as e:
    out_path_gif = npz_path.replace(".npz", ".gif")
    print(f"ffmpeg not available ({e}), saving as gif instead...")
    anim.save(out_path_gif, writer="pillow", fps=15)
    print(f"Saved video: {out_path_gif}")

plt.close(fig)
