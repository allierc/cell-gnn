"""Render the internal-state diagnostic for a trained ..._internal model.

Usage
-----
    python plot_internal_from_checkpoint.py \
        --config dicty_spring_force_rk4_diffusion_field_internal_v1 \
        --epoch 9                                          # default = latest

Loads the checkpoint from
    ./log/{pre_folder}{config_name}/models/best_model_with_{n_runs-1}_graphs_{epoch}.pt
re-evaluates the model on a single batch from the saved zarr data so
``last_internal_pred`` / ``last_internal_target`` are populated, then writes
    ./log/.../tmp_training/internal/post_{epoch}.png
via ``cell_gnn.plot.plot_internal_state``.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from cell_gnn.config import CellGNNConfig
from cell_gnn.cell_state import CellState
from cell_gnn.models.registry import get_model_class
from cell_gnn.plot import plot_internal_state
from cell_gnn.utils import (
    add_pre_folder, set_device, edges_radius_blockwise, choose_boundary_values,
)
from cell_gnn.zarr_io import load_simulation_data


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
    ap.add_argument("--config", required=True,
                    help="config name (without .yaml), as you'd pass to GNN_Main.py")
    ap.add_argument("--epoch", type=int, default=None,
                    help="epoch checkpoint to load (default: latest)")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index to evaluate (default: middle of timeseries)")
    args = ap.parse_args()

    config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
    config_file, pre_folder = add_pre_folder(args.config)
    cfg = CellGNNConfig.from_yaml(f"{config_root}/{config_file}.yaml")
    cfg.dataset = pre_folder + cfg.dataset
    cfg.config_file = pre_folder + args.config
    device = set_device(cfg.training.device)
    log_dir = f"./log/{cfg.config_file}"
    model_dir = os.path.join(log_dir, "models")

    # --- pick checkpoint ---
    n_runs = cfg.training.n_runs
    if args.epoch is None:
        ckpt = _latest_epoch_checkpoint(model_dir, n_runs)
    else:
        ckpt = os.path.join(model_dir, f"best_model_with_{n_runs - 1}_graphs_{args.epoch}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
    print(f"loading checkpoint: {ckpt}")

    # --- build model + load weights ---
    bc_pos, bc_dpos = choose_boundary_values(cfg.simulation.boundary)
    model_cls = get_model_class(cfg.graph_model.cell_model_name)
    model = model_cls(cfg, device, aggr_type=cfg.graph_model.aggr_type,
                      bc_dpos=bc_dpos, dimension=cfg.simulation.dimension)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # vnorm / ynorm — model.forward divides by self.vnorm so they must be set
    model.vnorm = torch.load(os.path.join(log_dir, "vnorm.pt"), map_location=device)
    model.ynorm = torch.load(os.path.join(log_dir, "ynorm.pt"), map_location=device)

    # --- pull one frame from the saved zarr to populate last_internal_{pred,target} ---
    dataset_name = cfg.dataset
    x_ts = load_simulation_data(
        f"graphs_data/{dataset_name}/x_list_0", cfg.simulation.dimension
    ).to(device)
    n_frames_ts = x_ts.n_frames
    k = args.frame if args.frame is not None else n_frames_ts // 2
    print(f"evaluating on frame k={k} (out of {n_frames_ts})")

    x_state = x_ts.frame(k).to(device).clone().detach()
    edges = edges_radius_blockwise(
        x_state.pos, bc_dpos, cfg.simulation.min_radius, cfg.simulation.max_radius, block=4096
    )
    n_b = x_state.n_cells
    data_id = torch.zeros((n_b, 1), dtype=torch.int, device=device)
    k_batch = torch.full((n_b, 1), k, dtype=torch.int, device=device)

    with torch.no_grad():
        _ = model(x_state, edges, data_id=data_id, training=False, has_field=False, k=k_batch)

    epoch_label = args.epoch if args.epoch is not None else \
        int(re.search(r"_graphs_(\d+)\.pt$", ckpt).group(1))
    plot_internal_state(model, cfg, log_dir, epoch=f"post_{epoch_label}", N=k, device=device)
    out = f"./{log_dir}/tmp_training/internal/post_{epoch_label}_{k}.png"
    print(f"-> {out}")


if __name__ == "__main__":
    main()
