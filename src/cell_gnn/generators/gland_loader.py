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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree
from tqdm import trange

from cell_gnn.cell_state import CellState
from cell_gnn.figure_style import default_style
from cell_gnn.zarr_io import ZarrSimulationWriterV3, save_edge_index


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
    """Compute per-cell velocities via finite difference of tracked positions.

    Returns:
        dict[int, dict[int, np.ndarray]]:
            timepoint -> {label -> (vx, vy, vz)} in microns/minute
    """
    vel_data = {t: {} for t in sorted_timepoints}

    if tracks is None:
        return vel_data

    # build lookup: (timepoint, label) -> (x, y, z)
    pos_lookup = {}
    for t in sorted_timepoints:
        df = frame_data[t]
        for _, row in df.iterrows():
            pos_lookup[(t, int(row['label']))] = np.array([
                row['centroid_x'], row['centroid_y'], row['centroid_z']
            ])

    # group tracks by track_id
    track_groups = tracks.groupby('track_id')
    for track_id, group in track_groups:
        group_sorted = group.sort_values('timepoint')
        tps = group_sorted['timepoint'].values
        labels = group_sorted['label'].values

        for i in range(len(tps)):
            t = int(tps[i])
            label = int(labels[i])

            # forward difference
            if i + 1 < len(tps):
                t_next = int(tps[i + 1])
                label_next = int(labels[i + 1])
                key_now = (t, label)
                key_next = (t_next, label_next)
                if key_now in pos_lookup and key_next in pos_lookup:
                    dp = pos_lookup[key_next] - pos_lookup[key_now]
                    dt_actual = (t_next - t) * dt
                    if dt_actual > 0:
                        vel_data[t][label] = dp / dt_actual
                        continue

            # backward difference fallback
            if i - 1 >= 0:
                t_prev = int(tps[i - 1])
                label_prev = int(labels[i - 1])
                key_now = (t, label)
                key_prev = (t_prev, label_prev)
                if key_now in pos_lookup and key_prev in pos_lookup:
                    dp = pos_lookup[key_now] - pos_lookup[key_prev]
                    dt_actual = (t - t_prev) * dt
                    if dt_actual > 0:
                        vel_data[t][label] = dp / dt_actual

    return vel_data


# ---------------------------------------------------------------------------
# kNN graph construction
# ---------------------------------------------------------------------------

def build_knn_edge_index(positions, k=15):
    """Build bidirectional kNN graph using scipy cKDTree.

    Args:
        positions: (N, 3) numpy array
        k: number of nearest neighbors

    Returns:
        (2, E) long tensor, bidirectional
    """
    tree = cKDTree(positions)
    k_query = min(k + 1, len(positions))  # +1 because query includes self
    _, indices = tree.query(positions, k=k_query)

    src_list = []
    dst_list = []
    for i in range(len(positions)):
        for j in range(k_query):
            neighbor = indices[i, j]
            if neighbor != i:
                src_list.append(i)
                dst_list.append(neighbor)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


# ---------------------------------------------------------------------------
# Visualization — dot + edge scatter plot (fixed bounding box)
# ---------------------------------------------------------------------------

def _plot_gland_frame(pos, edge_index, t_idx, run, dataset_name, plot_bounds):
    """3D scatter plot of cell positions with kNN edges, fixed bounding box."""
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

    # kNN edges
    if edge_index is not None and edge_index.shape[1] > 0:
        ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        segments = []
        for e in range(ei.shape[1]):
            src, dst = ei[0, e], ei[1, e]
            if src < len(pos) and dst < len(pos):
                p0 = pos[src]
                p1 = pos[dst]
                if not (np.isnan(p0).any() or np.isnan(p1).any()):
                    segments.append([p0, p1])
        if segments:
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
    ax.set_title(f'Gland frame {t_idx} ({n_cells} cells)',
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

    for t_idx in trange(n_frames, ncols=100, desc="  processing Gland"):
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

        # kNN edge index
        k = 15
        ei = build_knn_edge_index(norm_pos, k=min(k, n_cells - 1))
        edge_index_list.append(ei)

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

    n_written = x_writer.finalize()
    print(f"  wrote {n_written} frames to zarr")

    # save edge_index
    save_edge_index(run_path, edge_index_list)
    print(f"  saved edge_index.pt ({len(edge_index_list)} frames, kNN)")

    print(f"\n=== Done loading Gland data ===")
