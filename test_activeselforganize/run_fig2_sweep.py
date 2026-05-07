"""
Fig 2 phase-diagram-style sweep for communicating_activematter.py.

Sweeps v0 (self-propulsion speed) x omega (chemotactic coupling) on a 4x4 grid,
qualitatively mirroring the axes of Ziepke et al. 2022 Fig 2.
Writes per-run frames, npz, and mp4 into sweep_fig2/<tag>/.
"""

import os
import subprocess
import sys
import time
import traceback
from itertools import product
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import communicating_activematter as sim


V0_VALUES = [0.15, 0.35, 0.7, 1.2]
OMEGA_VALUES = [0.25, 0.8, 2.0, 4.0]

T_SIM = 200.0
SAVE_EVERY = 200
N_WORKERS = 4

SWEEP_ROOT = "sweep_fig2"


def tag_for(v0, omega):
    return f"v0_{v0:.2f}_omega_{omega:.2f}"


def run_one(args):
    v0, omega = args
    tag = tag_for(v0, omega)
    out_dir = os.path.join(SWEEP_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "run.log")
    t0 = time.time()
    try:
        with open(log_path, "w") as log:
            orig_stdout = sys.stdout
            sys.stdout = log
            try:
                sim.run(
                    out_dir=out_dir,
                    overrides={
                        "v0": v0,
                        "omega": omega,
                        "T": T_SIM,
                        "save_every": SAVE_EVERY,
                    },
                    quiet=False,
                )
            finally:
                sys.stdout = orig_stdout

        mp4_path = os.path.join(out_dir, "sim.mp4")
        frames_glob = os.path.join(out_dir, "frames", "frame_*.png")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "20",
            "-pattern_type", "glob", "-i", frames_glob,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            mp4_path,
        ]
        subprocess.run(cmd, check=True)

        dt = time.time() - t0
        return (tag, True, dt, None)
    except Exception as e:
        dt = time.time() - t0
        return (tag, False, dt, f"{e}\n{traceback.format_exc()}")


def make_overview():
    fig, axes = plt.subplots(
        len(OMEGA_VALUES), len(V0_VALUES),
        figsize=(3 * len(V0_VALUES), 3 * len(OMEGA_VALUES)),
        squeeze=False,
    )
    for iy, omega in enumerate(reversed(OMEGA_VALUES)):
        for ix, v0 in enumerate(V0_VALUES):
            ax = axes[iy, ix]
            tag = tag_for(v0, omega)
            frames_dir = os.path.join(SWEEP_ROOT, tag, "frames")
            if not os.path.isdir(frames_dir):
                ax.axis("off")
                continue
            pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
            if not pngs:
                ax.axis("off")
                continue
            img = plt.imread(os.path.join(frames_dir, pngs[-1]))
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if iy == len(OMEGA_VALUES) - 1:
                ax.set_xlabel(f"v0 = {v0}")
            if ix == 0:
                ax.set_ylabel(f"omega = {omega}")

    fig.suptitle("Fig2-style phase diagram sweep (last frame of each run)")
    fig.tight_layout()
    out = os.path.join(SWEEP_ROOT, "grid_overview.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    combos = list(product(V0_VALUES, OMEGA_VALUES))
    print(f"Launching {len(combos)} sims across {N_WORKERS} workers "
          f"(T={T_SIM}, N={sim.P.N}, L={sim.P.L})")

    t0 = time.time()
    with Pool(processes=N_WORKERS) as pool:
        for tag, ok, dt, err in pool.imap_unordered(run_one, combos):
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] {tag}  ({dt:6.1f} s)")
            if not ok:
                print(f"         {err.splitlines()[0] if err else ''}")

    print(f"All runs done in {time.time() - t0:.1f} s")
    make_overview()


if __name__ == "__main__":
    main()
