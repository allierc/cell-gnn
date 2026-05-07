"""Quantitative diagnostic for an ..._internal model after training.

Loads a checkpoint, replays the model on N_frames sampled from the saved
zarr data, and writes a multi-panel report to
    ./log/{cfg}/tmp_training/internal/diagnostic_{epoch}.png
plus a CSV of per-frame component losses.

Panels
------
  (1) Per-frame loss components — pos / pde / internal / total — vs frame
      → check whether the internal term is being out-weighted across the run.
  (2) c_pred (SIREN) vs c_true at every sampled cell.
      → tests hypothesis #1: bad SIREN c poisons the (s, c_pred) inputs.
  (3) ds/dt pred vs target at every sampled cell, with R^2.
      → on-manifold accuracy (the heatmap is mostly off-manifold).
  (4) (c, s) coverage of trajectory data overlaid on the
      |MLP(s, c) − analytic(s, c)| error map. Where the trajectory has
      density, the MLP gets gradient. White regions are extrapolation.

Usage
-----
    python analyze_internal_from_checkpoint.py \
        --config dicty_spring_force_rk4_diffusion_field_internal_v1
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from cell_gnn.config import CellGNNConfig
from cell_gnn.models.registry import get_model_class
from cell_gnn.utils import (
    add_pre_folder, set_device, edges_radius_blockwise, choose_boundary_values,
    to_numpy,
)
from cell_gnn.zarr_io import load_simulation_data, load_raw_array


def _latest_epoch_checkpoint(model_dir, n_runs):
    pat = os.path.join(model_dir, f"best_model_with_{n_runs - 1}_graphs_*.pt")
    files = [f for f in glob.glob(pat)
             if re.search(r"_graphs_(\d+)\.pt$", os.path.basename(f))]
    if not files:
        raise FileNotFoundError(f"no end-of-epoch checkpoints under {model_dir}")
    files.sort(key=lambda p: int(re.search(r"_graphs_(\d+)\.pt$", p).group(1)))
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--n_frames", type=int, default=80,
                    help="number of evenly-spaced frames to evaluate")
    ap.add_argument("--max_cells_scatter", type=int, default=50000,
                    help="cap on points used in scatter plots")
    args = ap.parse_args()

    config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
    config_file, pre_folder = add_pre_folder(args.config)
    cfg = CellGNNConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    cfg.dataset = pre_folder + cfg.dataset
    cfg.config_file = pre_folder + args.config
    device = set_device(cfg.training.device)
    log_dir = f"./log/{cfg.config_file}"
    model_dir = os.path.join(log_dir, "models")
    n_runs = cfg.training.n_runs

    # --- checkpoint ---
    if args.epoch is None:
        ckpt = _latest_epoch_checkpoint(model_dir, n_runs)
        epoch_label = int(re.search(r"_graphs_(\d+)\.pt$", ckpt).group(1))
    else:
        ckpt = os.path.join(model_dir, f"best_model_with_{n_runs - 1}_graphs_{args.epoch}.pt")
        epoch_label = args.epoch
    print(f"loading checkpoint: {ckpt}")

    # --- model ---
    bc_pos, bc_dpos = choose_boundary_values(cfg.simulation.boundary)
    model_cls = get_model_class(cfg.graph_model.cell_model_name)
    model = model_cls(cfg, device, aggr_type=cfg.graph_model.aggr_type,
                      bc_dpos=bc_dpos, dimension=cfg.simulation.dimension)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.vnorm = torch.load(os.path.join(log_dir, "vnorm.pt"), map_location=device)
    model.ynorm = torch.load(os.path.join(log_dir, "ynorm.pt"), map_location=device)
    ynorm = float(model.ynorm.item())

    # --- data ---
    dataset_name = cfg.dataset
    x_ts = load_simulation_data(
        f"graphs_data/{dataset_name}/x_list_0", cfg.simulation.dimension
    ).to(device)
    y_arr = load_raw_array(f"graphs_data/{dataset_name}/y_list_0")  # (T, N, dim)
    n_frames_ts = x_ts.n_frames
    frame_idxs = np.linspace(20, n_frames_ts - 2, args.n_frames, dtype=int)

    # --- knobs for analytic ground truth: ds/dt = coeff_s * (c - s) ---
    fp = cfg.simulation.field_params
    coeff_s = float(getattr(fp, 'internal_coeff_s', 2.0))

    # Match trainer's getattr fallbacks. Note: pde_weight / pde_reg_weight are
    # not in the Pydantic schema, so even if the YAML sets them they're silently
    # dropped — the actual training used these defaults. internal_weight IS in
    # the schema, so it follows the YAML.
    pde_w = float(getattr(cfg.training, 'pde_weight', 1.0))
    int_w = float(getattr(cfg.training, 'internal_weight', 0.0))
    print(f'effective weights used in training: pde={pde_w}, internal={int_w}')

    # --- per-frame replay ---
    rows = []
    s_all, c_obs_all, c_pred_all = [], [], []
    dsdt_pred_all, dsdt_true_all = [], []

    print(f"replaying {len(frame_idxs)} frames...")
    for k in frame_idxs:
        x_state = x_ts.frame(int(k)).to(device).clone().detach()
        edges = edges_radius_blockwise(
            x_state.pos, bc_dpos,
            cfg.simulation.min_radius, cfg.simulation.max_radius, block=4096
        )
        n_b = x_state.n_cells
        data_id = torch.zeros((n_b, 1), dtype=torch.int, device=device)
        k_batch = torch.full((n_b, 1), int(k), dtype=torch.int, device=device)

        with torch.no_grad():
            pred = model(x_state, edges, data_id=data_id, training=False,
                         has_field=False, k=k_batch)

        # --- losses (re-derive components without backprop) ---
        y_true_norm = torch.tensor(y_arr[int(k)], dtype=torch.float32, device=device) / ynorm
        pos_loss = (pred - y_true_norm).pow(2).sum().item()  # SE not norm

        pde_residual = getattr(model, 'last_pde_residual', None)
        pde_loss = 0.0 if pde_residual is None else \
            (pde_w * pde_residual.pow(2).sum()).item()

        internal_residual = getattr(model, 'last_internal_residual', None)
        int_loss = 0.0 if internal_residual is None else \
            (int_w * internal_residual.pow(2).sum()).item()

        # raw (unweighted) MSE for the internal ODE
        int_mse_raw = 0.0 if internal_residual is None else \
            internal_residual.pow(2).mean().item()

        # SIREN c at cells vs true c (state.field[:, 0] is the ground-truth observed c)
        s_obs = x_state.field[:, 1].detach()
        c_true = x_state.field[:, 0].detach()
        # Use the same SIREN-on-grid path as forward to recover c_pred at cells
        c_grid = model._reeval_c_grid_for_internal(k=int(k))
        c_pred = model._trilinear_interp(c_grid.unsqueeze(0), x_state.pos).squeeze(-1).detach()
        c_mse = (c_pred - c_true).pow(2).mean().item()

        # ds/dt on this frame
        dsdt_pred = getattr(model, 'last_internal_pred', None)
        dsdt_true = getattr(model, 'last_internal_target', None)

        rows.append({
            'frame': int(k),
            'pos_loss': pos_loss / n_b,
            'pde_loss': pde_loss / n_b,
            'internal_loss': int_loss / n_b,
            'internal_mse_raw': int_mse_raw,
            'c_mse_pred_vs_true': c_mse,
        })

        s_all.append(to_numpy(s_obs))
        c_obs_all.append(to_numpy(c_true))
        c_pred_all.append(to_numpy(c_pred))
        if dsdt_pred is not None and dsdt_true is not None:
            dsdt_pred_all.append(to_numpy(dsdt_pred))
            dsdt_true_all.append(to_numpy(dsdt_true))

    df = pd.DataFrame(rows)
    s_all       = np.concatenate(s_all)
    c_obs_all   = np.concatenate(c_obs_all)
    c_pred_all  = np.concatenate(c_pred_all)
    dsdt_pred_all = np.concatenate(dsdt_pred_all) if dsdt_pred_all else None
    dsdt_true_all = np.concatenate(dsdt_true_all) if dsdt_true_all else None

    # subsample for scatter
    rng = np.random.default_rng(0)
    n_pts = min(args.max_cells_scatter, s_all.shape[0])
    sel = rng.choice(s_all.shape[0], size=n_pts, replace=False)

    # --- aggregate stats ---
    print("\n=== mean per-cell loss components (sampled frames) ===")
    print(df.describe()[['pos_loss', 'pde_loss', 'internal_loss',
                         'internal_mse_raw', 'c_mse_pred_vs_true']].T)

    out_csv = f"./{log_dir}/tmp_training/internal/diagnostic_{epoch_label}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\ncsv -> {out_csv}")

    # --- figure: 4 panels ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (1) per-frame loss components on log scale
    ax = axes[0, 0]
    ax.plot(df['frame'], df['pos_loss'], label='pos', color='tab:blue', lw=1.2)
    if (df['pde_loss'] > 0).any():
        ax.plot(df['frame'], df['pde_loss'], label='pde (weighted)', color='tab:green', lw=1.2)
    if (df['internal_loss'] > 0).any():
        ax.plot(df['frame'], df['internal_loss'], label='internal (weighted)', color='tab:purple', lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('frame'); ax.set_ylabel('per-cell loss contribution')
    ax.set_title('(1) Per-frame loss components')
    ax.legend()

    # (2) c_pred vs c_true
    ax = axes[0, 1]
    ax.scatter(c_obs_all[sel], c_pred_all[sel], s=2, alpha=0.2, color='C0')
    cmin = min(c_obs_all.min(), c_pred_all.min())
    cmax = max(c_obs_all.max(), c_pred_all.max())
    ax.plot([cmin, cmax], [cmin, cmax], 'r--', lw=1)
    ss_res = float(np.mean((c_pred_all - c_obs_all) ** 2))
    ss_tot = float(np.var(c_obs_all)) + 1e-12
    r2_c = 1.0 - ss_res / ss_tot
    rmse_c = float(np.sqrt(ss_res))
    ax.text(0.02, 0.98, f'R²={r2_c:.3f}\nRMSE={rmse_c:.4g}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.6))
    ax.set_xlabel('c_true'); ax.set_ylabel('c_pred (SIREN)')
    ax.set_title('(2) SIREN c-field accuracy at cells')

    # (3) ds/dt scatter
    ax = axes[1, 0]
    if dsdt_pred_all is not None:
        ax.scatter(dsdt_true_all[sel], dsdt_pred_all[sel], s=2, alpha=0.2, color='C2')
        m = max(np.abs(dsdt_true_all).max(), np.abs(dsdt_pred_all).max())
        ax.plot([-m, m], [-m, m], 'r--', lw=1)
        ss_res = float(np.mean((dsdt_pred_all - dsdt_true_all) ** 2))
        ss_tot = float(np.var(dsdt_true_all)) + 1e-12
        r2_d = 1.0 - ss_res / ss_tot
        rmse_d = float(np.sqrt(ss_res))
        ax.text(0.02, 0.98, f'R²={r2_d:.3f}\nRMSE={rmse_d:.4g}',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.6))
    ax.set_xlabel('ds/dt target (analytic)'); ax.set_ylabel('ds/dt predicted (MLP)')
    ax.set_title('(3) ds/dt accuracy on actual trajectory')

    # (4) coverage: trajectory (c, s) points on top of MLP-vs-analytic error map
    ax = axes[1, 1]
    n_grid = 64
    s_axis = torch.linspace(0, 1, n_grid, device=device)
    c_axis = torch.linspace(0.0, max(float(c_obs_all.max()) * 1.05, 1e-3), n_grid, device=device)
    S, C = torch.meshgrid(s_axis, c_axis, indexing='ij')
    pts_sc = torch.stack([S.reshape(-1), C.reshape(-1)], dim=-1)
    with torch.no_grad():
        mean_embed = model.a[0, :, :].mean(dim=0, keepdim=True)
        embed_grid = mean_embed.expand(pts_sc.shape[0], -1)
        pts = torch.cat([pts_sc, embed_grid], dim=-1)
        pred_grid = model.lin_internal(pts).reshape(n_grid, n_grid)
    true_grid = coeff_s * (C - S)
    err = to_numpy(pred_grid - true_grid)
    elim = max(np.abs(err).max(), 1e-8)
    extent = [c_axis.min().item(), c_axis.max().item(), 0, 1]
    im = ax.imshow(err, origin='lower', extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=-elim, vmax=elim)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='pred - true')
    ax.scatter(c_obs_all[sel], s_all[sel], s=1.5, alpha=0.05, color='black')
    # overlay equilibrium s = c (where ds/dt = 0)
    c_line = np.linspace(0, c_axis.max().item(), 200)
    s_eq = np.clip(c_line, 0, 1)
    ax.plot(c_line, s_eq, 'k--', lw=1.5, alpha=0.8, label='s = c (equilibrium)')
    ax.set_xlabel('c'); ax.set_ylabel('s')
    ax.set_title('(4) MLP error vs (c, s) trajectory coverage')
    ax.legend(loc='lower right')

    plt.suptitle(
        f'{cfg.config_file}   epoch={epoch_label}   '
        f'mean(pos)={df.pos_loss.mean():.3e}  mean(pde)={df.pde_loss.mean():.3e}  '
        f'mean(int)={df.internal_loss.mean():.3e}',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = f"./{log_dir}/tmp_training/internal/diagnostic_{epoch_label}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"png -> {out_png}")


if __name__ == "__main__":
    main()
