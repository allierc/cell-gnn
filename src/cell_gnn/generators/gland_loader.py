"""Load Gland (salivary gland) cell-tracking CSV data and write to cell-gnn zarr V3 format.

The raw data consists of per-timepoint CSV files with cell properties:
  - centroid_x/y/z: cell positions (already in microns)
  - volume, surface_area, elongation, sphericity: morphology features
  - label: cell segmentation ID

Plus tracking data linking cells across frames for velocity computation.

Source pipeline: GNN_analysis_Claude_Code (decomp-GNN tissue analysis)
"""

import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import trange

from cell_gnn.cell_state import CellState
from cell_gnn.figure_style import default_style
from cell_gnn.utils import choose_boundary_values, edges_radius_blockwise
from cell_gnn.zarr_io import ZarrSimulationWriterV3, ZarrArrayWriter, save_edge_index


# ---------------------------------------------------------------------------
# Auto-detect data root
# ---------------------------------------------------------------------------

_GLAND_CANDIDATES = [
    '/groups/wang/wanglab/GNN/240408-SMG2',
]

def _find_gland_root():
    """Find the Gland dataset root directory.

    Checks (in order):
      1. GLAND_DATA_ROOT environment variable
      2. Known paths (Janelia cluster)
    """
    env_root = os.environ.get('GLAND_DATA_ROOT')
    if env_root and os.path.isdir(env_root):
        return env_root

    for path in _GLAND_CANDIDATES:
        if os.path.isdir(path):
            return path

    return _GLAND_CANDIDATES[0]


GLAND_DATA_ROOT = _find_gland_root()

# CSV subdirectory and file pattern
LABEL_PROPS_DIR = 'masks_smooth2_label_props_alignment'
CSV_PATTERN = re.compile(
    r'.*-t(\d{3})_cp_masks_label_props_alignment\.csv$'
)

# tracking subdirectory candidates (tried in order)
TRACKS_DIR_CANDIDATES = ['tracking_motile', 'tracks']

# field feature columns (morphology only)
FIELD_COLS = ['volume', 'surface_area', 'elongation', 'sphericity']

DT_MINUTES = 5.0  # time step between frames


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_timepoints(root_dir):
    """Load all per-timepoint CSV files.

    Returns:
        dict[int, pd.DataFrame]: timepoint -> DataFrame with cell properties
        sorted_timepoints: sorted list of available timepoints
    """
    props_dir = Path(root_dir) / LABEL_PROPS_DIR
    if not props_dir.exists():
        raise FileNotFoundError(f"Label props directory not found: {props_dir}")

    frame_data = {}
    for csv_path in sorted(props_dir.glob('*.csv')):
        m = CSV_PATTERN.match(csv_path.name)
        if m:
            t = int(m.group(1))
            frame_data[t] = pd.read_csv(csv_path)

    if not frame_data:
        raise FileNotFoundError(f"No matching CSV files in {props_dir}")

    sorted_timepoints = sorted(frame_data.keys())
    print(f"  loaded {len(sorted_timepoints)} timepoints "
          f"(t={sorted_timepoints[0]}..{sorted_timepoints[-1]}), "
          f"cells/frame: {frame_data[sorted_timepoints[0]].shape[0]}")
    return frame_data, sorted_timepoints


def load_tracks(root_dir):
    """Load tracking data from CSV.

    Returns:
        pd.DataFrame with (track_id, timepoint, label), or None if not found
    """
    tracks_dir = None
    for candidate in TRACKS_DIR_CANDIDATES:
        d = Path(root_dir) / candidate
        if d.exists():
            tracks_dir = d
            break

    if tracks_dir is None:
        print("  WARNING: tracks directory not found, velocities will be zero")
        return None

    csv_files = list(tracks_dir.glob('*.csv'))
    if not csv_files:
        print("  WARNING: no track CSV files found, velocities will be zero")
        return None

    tracks = pd.read_csv(csv_files[0])

    # try to normalize column names
    col_map = {}
    for col in tracks.columns:
        cl = col.lower().strip()
        if cl in ('t', 'frame', 'time') and 'timepoint' not in tracks.columns:
            col_map[col] = 'timepoint'
        elif cl in ('cell_id', 'segmentation_id') and 'label' not in tracks.columns:
            col_map[col] = 'label'
    if col_map:
        tracks = tracks.rename(columns=col_map)

    # shift zero-indexed timepoints to match CSV naming (t starts at 1)
    if 'timepoint' in tracks.columns and tracks['timepoint'].min() == 0:
        tracks['timepoint'] = tracks['timepoint'] + 1

    required = {'track_id', 'timepoint', 'label'}
    if not required.issubset(tracks.columns):
        print(f"  WARNING: tracks CSV missing columns {required - set(tracks.columns)}")
        return None

    print(f"  loaded tracks: {len(tracks)} rows, "
          f"{tracks['track_id'].nunique()} unique tracks")
    return tracks


# ---------------------------------------------------------------------------
# Velocity computation
# ---------------------------------------------------------------------------

def compute_velocities(frame_data, sorted_timepoints, tracks, dt=DT_MINUTES):
    """Compute per-cell velocities via vectorized pandas groupby diff.

    Returns:
        dict[int, dict[int, np.ndarray]]:
            timepoint -> {label -> (vx, vy, vz)} in microns/minute
    """
    vel_data = {t: {} for t in sorted_timepoints}

    if tracks is None:
        return vel_data

    # concatenate all frames into one DataFrame with timepoint column
    frames = []
    for t in sorted_timepoints:
        df = frame_data[t][['label', 'centroid_x', 'centroid_y', 'centroid_z']].copy()
        df['timepoint'] = t
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)

    # merge track_id into position data
    all_data = all_data.merge(
        tracks[['track_id', 'timepoint', 'label']],
        on=['timepoint', 'label'], how='left',
    )

    n_tracked = all_data['track_id'].notna().sum()
    print(f"  tracking coverage: {n_tracked}/{len(all_data)} "
          f"({100 * n_tracked / len(all_data):.1f}%)")

    # sort by track and time, then vectorized forward diff
    all_data = all_data.sort_values(['track_id', 'timepoint']).reset_index(drop=True)
    pos_cols = ['centroid_x', 'centroid_y', 'centroid_z']
    for pc in pos_cols:
        all_data['v_' + pc] = all_data.groupby('track_id')[pc].diff() / dt

    # backfill first timepoint of each track, fill untracked with 0
    vel_cols = ['v_centroid_x', 'v_centroid_y', 'v_centroid_z']
    for vc in vel_cols:
        all_data[vc] = all_data.groupby('track_id')[vc].bfill()
        all_data[vc] = all_data[vc].fillna(0.0)

    # convert to dict for fast per-frame lookup (vectorized groupby)
    vel_arr = all_data[vel_cols].values
    tp_arr = all_data['timepoint'].values.astype(int)
    lab_arr = all_data['label'].values.astype(int)
    for t in sorted_timepoints:
        mask = tp_arr == t
        labels_t = lab_arr[mask]
        vels_t = vel_arr[mask]
        vel_data[t] = {int(lab): vels_t[i] for i, lab in enumerate(labels_t)}

    return vel_data


# ---------------------------------------------------------------------------
# Radius graph construction (blockwise, consistent with graph_trainer)
# ---------------------------------------------------------------------------

def build_radius_edge_index(positions, max_radius, min_radius=0.0,
                            bc_dpos=None, device='cpu'):
    """Build bidirectional radius graph using blockwise method from utils.

    Args:
        positions: (N, 3) numpy array (normalized coordinates)
        max_radius: maximum connection radius
        min_radius: minimum connection radius
        bc_dpos: boundary condition function (identity for 'no' boundary)
        device: torch device

    Returns:
        (2, E) long tensor, bidirectional
    """
    if bc_dpos is None:
        bc_dpos = lambda x: x

    pos_t = torch.from_numpy(positions).float().to(device)
    edge_index = edges_radius_blockwise(
        pos_t, bc_dpos=bc_dpos,
        min_radius=min_radius, max_radius=max_radius, block=4096,
    )
    return edge_index.cpu()


# ---------------------------------------------------------------------------
# Visualization — dot + edge scatter plot (fixed bounding box)
# ---------------------------------------------------------------------------

def _plot_gland_frame(pos, edge_index, t_idx, run, dataset_name, plot_bounds,
                      max_plot_edges=50_000):
    """3D scatter plot of cell positions with radius edges, fixed bounding box."""
    fig = plt.figure(figsize=(16, 14), facecolor=default_style.background)
    ax = fig.add_subplot(111, projection='3d')

    valid = ~np.isnan(pos[:, 0])
    pos_valid = pos[valid]
    n_cells = len(pos_valid)

    # dots
    ax.scatter(
        pos_valid[:, 0], pos_valid[:, 1], pos_valid[:, 2],
        s=2, color='#1f77b4', alpha=0.5, edgecolors='none',
        depthshade=True,
    )

    # edges (subsample if too many for matplotlib)
    if edge_index is not None and edge_index.shape[1] > 0:
        ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        n_edges = ei.shape[1]
        # only draw forward edges (src < dst) to avoid double-drawing
        fwd = ei[0] < ei[1]
        ei = ei[:, fwd]
        # subsample if still too many
        if ei.shape[1] > max_plot_edges:
            idx = np.random.choice(ei.shape[1], max_plot_edges, replace=False)
            ei = ei[:, idx]
        # vectorized segment building
        src_pos = pos[ei[0]]
        dst_pos = pos[ei[1]]
        valid_edges = ~(np.isnan(src_pos).any(axis=1) | np.isnan(dst_pos).any(axis=1))
        segments = np.stack([src_pos[valid_edges], dst_pos[valid_edges]], axis=1)
        if len(segments) > 0:
            lc = Line3DCollection(
                segments, colors='#888888', linewidths=0.3, alpha=0.3,
            )
            ax.add_collection3d(lc)

    # fixed bounding box from first frame
    center = plot_bounds['center']
    hr = plot_bounds['half_range']
    ax.set_xlim(center[0] - hr, center[0] + hr)
    ax.set_ylim(center[1] - hr, center[1] + hr)
    ax.set_zlim(center[2] - hr, center[2] + hr)
    default_style.xlabel(ax, 'X')
    default_style.ylabel(ax, 'Y')
    ax.set_zlabel('Z', fontsize=default_style.label_font_size, color=default_style.foreground)
    n_total_edges = edge_index.shape[1] if edge_index is not None else 0
    ax.set_title(f'Gland frame {t_idx} ({n_cells} cells, {n_total_edges} edges)',
                 fontsize=default_style.font_size, color=default_style.foreground)

    default_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Fig_{run}_{t_idx:06d}.png")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_from_gland(
    config,
    visualize=True,
    step=100,
    device=None,
    save=True,
):
    """Load Gland data and write to zarr V3 format."""

    dataset_name = config.dataset
    dimension = config.simulation.dimension

    print(f"\n=== Loading Gland data into {dataset_name} ===")
    print(f"  data root: {GLAND_DATA_ROOT}")

    folder = f"./graphs_data/{dataset_name}/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/Fig/", exist_ok=True)

    if not os.path.isdir(GLAND_DATA_ROOT):
        print(f"  ERROR: data root not found: {GLAND_DATA_ROOT}")
        return

    # ---- load centroid CSV data ----
    frame_data, sorted_timepoints = load_csv_timepoints(GLAND_DATA_ROOT)
    n_frames = len(sorted_timepoints)

    # load tracking + compute velocities
    tracks = load_tracks(GLAND_DATA_ROOT)
    vel_data = compute_velocities(frame_data, sorted_timepoints, tracks)

    # determine max_N across all frames
    max_n = max(len(frame_data[t]) for t in sorted_timepoints)
    print(f"  frames: {n_frames}, max cells: {max_n}")

    # collect all positions for global bounding box
    all_pos = []
    for t in sorted_timepoints:
        df = frame_data[t]
        pos = df[['centroid_x', 'centroid_y', 'centroid_z']].values
        all_pos.append(pos)
    all_pos = np.concatenate(all_pos, axis=0)

    # longest-axis normalization
    bbox_min = all_pos.min(axis=0)
    bbox_max = all_pos.max(axis=0)
    bbox_extent = bbox_max - bbox_min
    max_extent = bbox_extent.max()
    print(f"  bounding box: {bbox_min} -> {bbox_max}")
    print(f"  max extent: {max_extent:.2f} um, normalizing to [0, 1]")

    # determine field feature count
    sample_df = frame_data[sorted_timepoints[0]]
    available_field_cols = [c for c in FIELD_COLS if c in sample_df.columns]
    n_field_features = len(available_field_cols)
    print(f"  field features ({n_field_features}): {available_field_cols}")

    # radius graph parameters
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    _, bc_dpos = choose_boundary_values(config.simulation.boundary)
    print(f"  radius graph: max_radius={max_radius}, min_radius={min_radius}",
          flush=True)
    sys.stdout.flush()
    time.sleep(1)

    # ---- build padded arrays ----
    pos_frames = np.full((n_frames, max_n, dimension), np.nan, dtype=np.float32)
    vel_frames = np.full((n_frames, max_n, dimension), np.nan, dtype=np.float32)
    field_frames = np.zeros((n_frames, max_n, n_field_features), dtype=np.float32)
    edge_index_list = []

    # compute plot bounding box from first frame (fixed for all frames)
    first_df = frame_data[sorted_timepoints[0]]
    first_pos = (first_df[['centroid_x', 'centroid_y', 'centroid_z']].values
                 - bbox_min) / max_extent
    plot_bounds = {
        'center': (first_pos.min(axis=0) + first_pos.max(axis=0)) / 2,
        'half_range': float((first_pos.max(axis=0)
                             - first_pos.min(axis=0)).max() / 2 * 1.05),
    }

    # cell_type: all zeros (no predefined types)
    cell_type = torch.zeros(max_n, dtype=torch.long)

    run = 0
    run_path = f"graphs_data/{dataset_name}/x_list_{run}"

    trange_obj = trange(n_frames, ncols=150, desc="  processing Gland")
    for t_idx in trange_obj:
        t = sorted_timepoints[t_idx]
        df = frame_data[t]
        n_cells = len(df)
        labels = df['label'].values.astype(int)

        # positions (normalized)
        raw_pos = df[['centroid_x', 'centroid_y', 'centroid_z']].values
        norm_pos = (raw_pos - bbox_min) / max_extent
        pos_frames[t_idx, :n_cells] = norm_pos.astype(np.float32)

        # velocities (normalized by same scale factor)
        for i, label in enumerate(labels):
            if label in vel_data.get(t, {}):
                vel_frames[t_idx, i] = vel_data[t][label] / max_extent

        # field features
        for f_idx, col in enumerate(available_field_cols):
            if col in df.columns:
                field_frames[t_idx, :n_cells, f_idx] = (
                    df[col].values.astype(np.float32)
                )

        # radius edge index (blockwise, same method as graph_trainer)
        ei = build_radius_edge_index(
            norm_pos, max_radius=max_radius, min_radius=min_radius,
            bc_dpos=bc_dpos,
        )
        edge_index_list.append(ei)
        if t_idx == 0:
            trange_obj.set_postfix_str(
                f"{n_cells} cells, {ei.shape[1]} edges "
                f"({ei.shape[1] // max(n_cells, 1)} edges/cell)"
            )

        # visualization (every `step` frames)
        if visualize and t_idx % step == 0:
            _plot_gland_frame(
                pos_frames[t_idx], ei, t_idx, run, dataset_name, plot_bounds,
            )

    if not save:
        print("  save=False, skipping zarr write")
        return

    # write zarr V3
    x_writer = ZarrSimulationWriterV3(
        path=run_path,
        n_cells=max_n,
        dimension=dimension,
        time_chunks=min(2000, n_frames),
    )
    y_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_list_{run}",
        n_cells=max_n,
        n_features=dimension,
        time_chunks=min(2000, n_frames),
    )

    for t_idx in trange(n_frames, ncols=100, desc="  writing zarr"):
        pos_t = pos_frames[t_idx]
        vel_t = vel_frames[t_idx]
        field_t = field_frames[t_idx]

        # replace NaN with 0 for zarr storage
        pos_t_clean = np.nan_to_num(pos_t, nan=0.0)
        vel_t_clean = np.nan_to_num(vel_t, nan=0.0)

        state = CellState(
            index=torch.arange(max_n, dtype=torch.long),
            pos=torch.from_numpy(pos_t_clean),
            vel=torch.from_numpy(vel_t_clean),
            cell_type=cell_type,
            field=torch.from_numpy(field_t),
        )
        x_writer.append_state(state)

        # training target: velocity (first_derivative prediction)
        y_writer.append(vel_t_clean)

    n_written = x_writer.finalize()
    y_writer.finalize()
    print(f"  wrote {n_written} frames to zarr (x_list + y_list)")

    # save edge_index
    save_edge_index(run_path, edge_index_list)
    print(f"  saved edge_index.pt ({len(edge_index_list)} frames, radius={max_radius})")

    print(f"\n=== Done loading Gland data ===")
