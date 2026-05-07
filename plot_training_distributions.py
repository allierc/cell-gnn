"""Show what data the trainer actually sees: distributions of c, s, ds/dt
over every cell at every frame from the saved zarr.

Outputs a 2x3 figure to
    ./graphs_data/{dataset}/distributions.png
plus the per-channel summary printed to stdout.

Top row    — marginal histograms of c, s, ds/dt (log-y).
Bottom row — joint distributions: (c, s) with the equilibrium s=c line,
             (c, ds/dt), (s, ds/dt).  In all three the analytic ground-truth
             relation (ds/dt = coeff_s * (c - s)) is overlaid in red so you
             can see whether your sampled trajectory actually covers the
             manifold the MLP is trying to learn.

Usage
-----
    python plot_training_distributions.py \
        --config dicty_spring_force_rk4_diffusion_field_internal_v1
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cell_gnn.config import CellGNNConfig
from cell_gnn.utils import add_pre_folder
from cell_gnn.zarr_io import load_simulation_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_points", type=int, default=200_000,
                    help="cap on points used for joint scatter plots")
    args = ap.parse_args()

    config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
    config_file, pre_folder = add_pre_folder(args.config)
    cfg = CellGNNConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    cfg.dataset = pre_folder + cfg.dataset

    # --- load the full saved time series (CPU is fine, no model) ---
    x_ts = load_simulation_data(
        f"graphs_data/{cfg.dataset}/x_list_0", cfg.simulation.dimension,
        fields=['field']
    )
    field = x_ts.field  # (T, N, F)
    T, N, F = field.shape
    if F < 3:
        raise SystemExit(f"state.field has {F} channels; need 3 for [c, s, ds/dt]")

    c    = field[:, :, 0].numpy().reshape(-1)
    s    = field[:, :, 1].numpy().reshape(-1)
    dsdt = field[:, :, 2].numpy().reshape(-1)
    print(f"loaded {T} frames x {N} cells = {c.size} samples")

    # --- summary stats ---
    def _stats(name, arr):
        return (f"  {name:>6}: mean={arr.mean():+.4f}  std={arr.std():.4f}  "
                f"min={arr.min():+.4f}  q25={np.percentile(arr, 25):+.4f}  "
                f"q50={np.percentile(arr, 50):+.4f}  q75={np.percentile(arr, 75):+.4f}  "
                f"max={arr.max():+.4f}")
    print("\n=== distribution stats ===")
    print(_stats('c',     c))
    print(_stats('s',     s))
    print(_stats('ds/dt', dsdt))

    # --- analytic ground truth: ds/dt = coeff_s * (c - s) ---
    fp = cfg.simulation.field_params
    coeff_s = float(getattr(fp, 'internal_coeff_s', 2.0))

    # --- subsample for joint plots ---
    rng = np.random.default_rng(0)
    n_pts = min(args.max_points, c.size)
    sel = rng.choice(c.size, size=n_pts, replace=False)
    c_s = c[sel]; s_s = s[sel]; d_s = dsdt[sel]

    # --- figure ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    bins = 120

    ax = axes[0, 0]
    ax.hist(c, bins=bins, color='tab:orange', alpha=0.85)
    ax.set_yscale('log'); ax.set_title('p(c)'); ax.set_xlabel('c')

    ax = axes[0, 1]
    ax.hist(s, bins=bins, color='tab:green', alpha=0.85)
    ax.set_yscale('log'); ax.set_title('p(s)'); ax.set_xlabel('s')

    ax = axes[0, 2]
    ax.hist(dsdt, bins=bins, color='tab:purple', alpha=0.85)
    ax.set_yscale('log'); ax.set_title('p(ds/dt)'); ax.set_xlabel('ds/dt')

    # --- joint (c, s) with equilibrium s = c overlay ---
    # Linear relaxation ds/dt = coeff_s * (c - s) → fixed point is s = c.
    ax = axes[1, 0]
    ax.scatter(c_s, s_s, s=1.5, alpha=0.05, color='C0', rasterized=True)
    c_line = np.linspace(c.min(), c.max(), 200)
    ax.plot(c_line, c_line, 'r-', lw=2, label='s = c (equilibrium)')
    ax.set_xlabel('c'); ax.set_ylabel('s'); ax.set_title('(c, s) joint')
    ax.legend(loc='lower right')

    # --- joint (c, ds/dt) with analytic relation at s = <s|c> overlay ---
    ax = axes[1, 1]
    ax.scatter(c_s, d_s, s=1.5, alpha=0.05, color='C2', rasterized=True)
    nb = 60
    c_bins = np.linspace(c.min(), c.max(), nb + 1)
    c_mid  = 0.5 * (c_bins[:-1] + c_bins[1:])
    s_cond = np.array([s[(c >= c_bins[i]) & (c < c_bins[i + 1])].mean()
                       if ((c >= c_bins[i]) & (c < c_bins[i + 1])).any() else np.nan
                       for i in range(nb)])
    dsdt_cond = coeff_s * (c_mid - s_cond)
    ax.plot(c_mid, dsdt_cond, 'r-', lw=2, label='analytic at $\\langle s|c\\rangle$')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('c'); ax.set_ylabel('ds/dt'); ax.set_title('(c, ds/dt) joint')
    ax.legend(loc='best')

    # --- joint (s, ds/dt) with analytic relation at c = <c|s> overlay ---
    ax = axes[1, 2]
    ax.scatter(s_s, d_s, s=1.5, alpha=0.05, color='C3', rasterized=True)
    s_bins = np.linspace(s.min(), s.max(), nb + 1)
    s_mid  = 0.5 * (s_bins[:-1] + s_bins[1:])
    c_cond = np.array([c[(s >= s_bins[i]) & (s < s_bins[i + 1])].mean()
                       if ((s >= s_bins[i]) & (s < s_bins[i + 1])).any() else np.nan
                       for i in range(nb)])
    dsdt_cond_s = coeff_s * (c_cond - s_mid)
    ax.plot(s_mid, dsdt_cond_s, 'r-', lw=2, label='analytic at $\\langle c|s\\rangle$')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel('s'); ax.set_ylabel('ds/dt'); ax.set_title('(s, ds/dt) joint')
    ax.legend(loc='best')

    plt.suptitle(
        f'{cfg.dataset}   T={T} frames × N={N} cells = {c.size:,} samples   '
        f'(coeff$_s$={coeff_s})',
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = f"./graphs_data/{cfg.dataset}/distributions.png"
    fig.savefig(out, dpi=120)
    # Also drop a copy alongside the other model-diagnostic plots so it's easy to find.
    log_out_dir = f"./log/{cfg.dataset}/tmp_training/internal"
    os.makedirs(log_out_dir, exist_ok=True)
    log_out = f"{log_out_dir}/distributions.png"
    fig.savefig(log_out, dpi=120)
    plt.close(fig)
    print(f"\n-> {out}")
    print(f"-> {log_out}")


if __name__ == "__main__":
    main()
