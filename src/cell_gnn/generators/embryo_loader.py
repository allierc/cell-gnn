"""Load raw MultiCell embryo mesh .mat files and write to cell-gnn zarr V3 format.

The raw .mat files contain Drosophila embryo segmentation meshes:
  C_cells.mat:    [frame, cell_id, neighbor_cell_id] — Delaunay adjacency
  C_vertices.mat: [frame, cell_id, vertex_id] — cell-to-vertex mapping
  V_coords.mat:   [frame, vertex_id, x, y, z] — vertex 3D coordinates
  V_vertices.mat: [frame, vertex_id, neighbor_vertex_id] — vertex mesh edges

Phase 1: positions + cell IDs + cell types -> zarr V3
Phase 2: Delaunay adjacency -> edge_index.pt
Phase 3: vertex-level mesh data -> vertex_pos.pt, cell_vertex_index.pt, vertex_edge_index.pt
"""

import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from cell_gnn.cell_state import CellState, VertexTimeSeries
from cell_gnn.figure_style import default_style
from cell_gnn.zarr_io import (
    ZarrSimulationWriterV3, save_edge_index,
    save_vertex_indices, save_vertex_timeseries,
)


# ---------------------------------------------------------------------------
# Embryo registry — maps short keys to paths + metadata
# ---------------------------------------------------------------------------
EMBRYO_REGISTRY = {
    '1830': {
        'data_dir': 'Deconstructing Gastrulation - Data',
        'image_data_dirname': '1830',
        'frame_rate': 1.0,       # minutes per frame
        'lag': 1,
        'VFF_frame': 5,
    },
    '1620': {
        'data_dir': 'Deconstructing Gastrulation - Data',
        'image_data_dirname': 'Img_1620 (intercalations)',
        'frame_rate': 0.25,
        'lag': 4,
        'VFF_frame': 50,
    },
    'seq2': {
        'data_dir': 'Data 2025',
        'image_data_dirname': '7-2-2022 stg two channels (Sequence 2)',
        'frame_rate': 40.0 / 60.0,
        'lag': 2,
        'VFF_frame': 30,
    },
    'seq3': {
        'data_dir': 'Data 2025',
        'image_data_dirname': '7-2-2022 stg two channels (Sequence 3)',
        'frame_rate': 40.0 / 60.0,
        'lag': 2,
        'VFF_frame': 28,
    },
}

# path to the MultiCell dataset — auto-detect from known locations or env var
def _find_multicell_root():
    """Find the MultiCell dataset root directory.

    Checks (in order):
      1. MULTICELL_DATA_ROOT environment variable
      2. Known paths (devcontainer, Janelia cluster)
    """
    # 1. Environment variable override
    env_root = os.environ.get('MULTICELL_DATA_ROOT')
    if env_root and os.path.isdir(env_root):
        return env_root

    # 2. Known candidate paths
    candidates = [
        '/workspace/tmp_repo/MultiCell/preprocess/MultiCell_dataset',          # devcontainer
        '/groups/saalfeld/home/allierc/Graph/tmp_repo/MultiCell/preprocess/MultiCell_dataset',  # Janelia cluster
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path

    # Fall back to first candidate (will produce clear error messages downstream)
    return candidates[0]


MULTICELL_DATA_ROOT = _find_multicell_root()


# ---------------------------------------------------------------------------
# Dual-format .mat reader (v5 via scipy, v7.3 via h5py)
# ---------------------------------------------------------------------------

def load_mat(filepath, key):
    """Load a variable from a .mat file, handling both v5 and v7.3 formats.

    Returns numpy array. For v7.3 (HDF5), transposes to match MATLAB convention.
    """
    try:
        import scipy.io
        data = scipy.io.loadmat(filepath)
        return data[key]
    except NotImplementedError:
        # MATLAB v7.3 — use h5py
        import h5py
        with h5py.File(filepath, 'r') as f:
            arr = f[key][:]
            # HDF5 stores MATLAB arrays transposed (column-major → row-major)
            return arr.T


# ---------------------------------------------------------------------------
# Phase 1: Build cell centroids per frame from mesh data
# ---------------------------------------------------------------------------

def build_frame_data(mesh_dir):
    """Extract cell centroids and per-cell vertex lists from raw mesh .mat files.

    Returns:
        frame_data: dict[frame_num -> dict[cell_id -> np.array([x, y, z])]]
        frame_vertex_coords: dict[frame_num -> dict[vertex_id -> np.array([x, y, z])]]
        frame_cell_vertices: dict[frame_num -> dict[cell_id -> list[vertex_id]]]
        sorted_frames: sorted list of frame numbers
    """
    mesh_dir = str(mesh_dir)

    # Load cell-to-vertex mapping: (M, 3) with [frame, cell_id, vertex_id]
    cv = load_mat(os.path.join(mesh_dir, 'C_vertices.mat'), 'C_vertices')
    # Load vertex coordinates: (M, 5) with [frame, vertex_id, x, y, z]
    vc = load_mat(os.path.join(mesh_dir, 'V_coords.mat'), 'V_coords')

    # Group vertex coords by frame
    frames = np.unique(cv[:, 0]).astype(int)
    sorted_frames = sorted(frames)

    frame_data = {}
    frame_vertex_coords = {}
    frame_cell_vertices = {}

    for frame in sorted_frames:
        # vertex coords for this frame
        vc_f = vc[vc[:, 0] == frame]
        vertex_coords = {}
        for row in vc_f:
            vid = int(row[1])
            # axis swap [x, z, y] matching MATLAB convention (GetNodeInfo line 764)
            vertex_coords[vid] = row[2:5][[0, 2, 1]]

        # cell -> vertex_ids for this frame
        cv_f = cv[cv[:, 0] == frame]
        cell_vertices = defaultdict(list)
        for row in cv_f:
            cell_vertices[int(row[1])].append(int(row[2]))

        # compute centroids
        centroids = {}
        for cell_id, vert_ids in cell_vertices.items():
            verts = [vertex_coords[v] for v in vert_ids if v in vertex_coords]
            if verts:
                centroids[cell_id] = np.mean(verts, axis=0)

        frame_data[frame] = centroids
        frame_vertex_coords[frame] = vertex_coords
        frame_cell_vertices[frame] = dict(cell_vertices)

    print(f"  built centroids for {len(sorted_frames)} frames")
    return frame_data, frame_vertex_coords, frame_cell_vertices, sorted_frames


# ---------------------------------------------------------------------------
# Phase 2: Build Delaunay adjacency per frame
# ---------------------------------------------------------------------------

def build_adjacency(mesh_dir):
    """Extract cell adjacency (Delaunay) from C_cells.mat.

    Mirrors MATLAB GetCellAdj: for each cell, get neighbor cells, build sorted unique pairs.

    Returns:
        adj_data: dict[frame_num -> np.array of shape (E, 2)]
    """
    mesh_dir = str(mesh_dir)
    cc = load_mat(os.path.join(mesh_dir, 'C_cells.mat'), 'C_cells')

    frames = np.unique(cc[:, 0]).astype(int)
    adj_data = {}

    for frame in sorted(frames):
        cc_f = cc[cc[:, 0] == frame]
        # each row is [frame, cell_id, neighbor_cell_id]
        pairs = cc_f[:, 1:3].astype(int)
        # sort each pair and unique
        pairs_sorted = np.sort(pairs, axis=1)
        unique_pairs = np.unique(pairs_sorted, axis=0)
        adj_data[frame] = unique_pairs

    print(f"  built adjacency for {len(adj_data)} frames")
    return adj_data


# ---------------------------------------------------------------------------
# Phase 3: Build vertex-level mesh data per frame
# ---------------------------------------------------------------------------

def build_vertex_edge_index(mesh_dir):
    """Extract vertex-vertex connectivity from V_vertices.mat.

    Returns:
        vv_data: dict[frame_num -> np.array of shape (E, 2)] — unique undirected pairs
    """
    mesh_dir = str(mesh_dir)
    vv_path = os.path.join(mesh_dir, 'V_vertices.mat')
    if not os.path.exists(vv_path):
        print("  WARNING: V_vertices.mat not found, skipping vertex edge index")
        return {}

    vv = load_mat(vv_path, 'V_vertices')
    frames = np.unique(vv[:, 0]).astype(int)
    vv_data = {}

    for frame in sorted(frames):
        vv_f = vv[vv[:, 0] == frame]
        pairs = vv_f[:, 1:3].astype(int)
        pairs_sorted = np.sort(pairs, axis=1)
        unique_pairs = np.unique(pairs_sorted, axis=0)
        vv_data[frame] = unique_pairs

    print(f"  built vertex edges for {len(vv_data)} frames")
    return vv_data


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_from_embryo(
    config,
    visualize=True,
    step=100,
    device=None,
    save=True,
):
    """Load embryo mesh data and write to zarr V3 format.

    Processes all embryos in EMBRYO_REGISTRY, each as a separate run.
    """

    dataset_name = config.dataset
    dimension = config.simulation.dimension
    # embryo data has only 60-198 frames; use step=5 for plots
    step = 5

    print(f"\n=== Loading embryo data into {dataset_name} ===")

    folder = f"./graphs_data/{dataset_name}/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/Fig/", exist_ok=True)

    for run, (embryo_key, meta) in enumerate(EMBRYO_REGISTRY.items()):
        print(f"\n--- Embryo {embryo_key} (run {run}) ---")

        embryo_dir = os.path.join(
            MULTICELL_DATA_ROOT,
            meta['data_dir'],
            meta['image_data_dirname'],
        )
        mesh_dir = os.path.join(embryo_dir, 'Mesh')

        if not os.path.exists(mesh_dir):
            print(f"  WARNING: mesh dir not found: {mesh_dir}, skipping")
            continue

        # Phase 1: collect xyz + cell IDs + vertex data
        frame_data, frame_vertex_coords, frame_cell_vertices, sorted_frames = \
            build_frame_data(mesh_dir)
        n_frames = len(sorted_frames)

        # determine max_N across all frames
        max_n = max(len(centroids) for centroids in frame_data.values())
        print(f"  frames: {n_frames}, max cells: {max_n}")

        # build sorted cell_id lists per frame for consistent ordering
        frame_cell_ids = {}
        for frame in sorted_frames:
            frame_cell_ids[frame] = sorted(frame_data[frame].keys())

        # collect all positions to compute global bounding box
        all_pos = []
        for frame in sorted_frames:
            centroids = frame_data[frame]
            for pos in centroids.values():
                all_pos.append(pos)
            # include vertex positions in bbox calculation
            for pos in frame_vertex_coords[frame].values():
                all_pos.append(pos)
        all_pos = np.array(all_pos)
        bbox_min = all_pos.min(axis=0)
        bbox_max = all_pos.max(axis=0)
        bbox_extent = bbox_max - bbox_min
        # normalize by longest axis only (preserve aspect ratio)
        max_extent = bbox_extent.max()
        if max_extent < 1e-8:
            max_extent = 1.0
        print(f"  bbox min: {bbox_min}, max: {bbox_max}, extent: {bbox_extent}")
        print(f"  normalizing by longest axis: {max_extent:.4f}")

        # normalize cell centroid positions to [0, ~1] (longest axis = 1)
        pos_frames = np.full((n_frames, max_n, 3), np.nan, dtype=np.float32)
        for t_idx, frame in enumerate(sorted_frames):
            cell_ids = frame_cell_ids[frame]
            for c_idx, cell_id in enumerate(cell_ids):
                raw_pos = frame_data[frame][cell_id]
                pos_frames[t_idx, c_idx] = (raw_pos - bbox_min) / max_extent

        # compute velocity from finite differences (no periodic wrapping)
        vel_frames = np.zeros_like(pos_frames)
        delta_t = meta['frame_rate']
        for t_idx in range(n_frames - 1):
            dp = pos_frames[t_idx + 1] - pos_frames[t_idx]
            # NaN - anything = NaN, which is correct for padded cells
            vel_frames[t_idx] = dp / delta_t
        if n_frames > 1:
            vel_frames[-1] = vel_frames[-2]  # repeat last velocity

        # cell_type: use 0 for all (domain loading is phase 2 extension)
        cell_type = torch.zeros(max_n, dtype=torch.long)

        # Phase 2: collect Delaunay adjacency
        adj_data = build_adjacency(mesh_dir)

        # build edge_index list (remapped to padded slot indices, 0-based)
        edge_index_list = []
        for t_idx, frame in enumerate(sorted_frames):
            cell_ids = frame_cell_ids[frame]
            # remap: matlab_cell_id -> slot_index (0-based)
            id_to_slot = {cid: slot for slot, cid in enumerate(cell_ids)}

            if frame in adj_data:
                pairs = adj_data[frame]
                # filter pairs to only include cells present in this frame
                remapped = []
                for i, j in pairs:
                    if i in id_to_slot and j in id_to_slot:
                        remapped.append([id_to_slot[i], id_to_slot[j]])
                if remapped:
                    edges = np.array(remapped, dtype=np.int64).T  # (2, E)
                    # make bidirectional
                    edges_bi = np.concatenate([edges, edges[[1, 0]]], axis=1)
                    edge_index_list.append(torch.from_numpy(edges_bi))
                else:
                    edge_index_list.append(torch.zeros(2, 0, dtype=torch.long))
            else:
                edge_index_list.append(torch.zeros(2, 0, dtype=torch.long))

        # Phase 3: vertex-level mesh data → VertexTimeSeries + vertex_indices
        vv_data = build_vertex_edge_index(mesh_dir)

        vertex_pos_list = []           # list of (V_t, 3) float tensors
        vertex_edge_index_list = []    # list of (2, E_v_t) long tensors
        vertex_indices_list = []       # list of (N_t, max_V_per_cell_t) long tensors, -1 padded

        for t_idx, frame in enumerate(sorted_frames):
            cell_ids = frame_cell_ids[frame]
            id_to_slot = {cid: slot for slot, cid in enumerate(cell_ids)}
            vert_coords = frame_vertex_coords[frame]
            cell_verts = frame_cell_vertices[frame]

            # build sorted vertex id list for this frame
            all_vids = sorted(vert_coords.keys())
            vid_to_local = {vid: idx for idx, vid in enumerate(all_vids)}

            # vertex positions (normalized same way as centroids)
            vpos = np.array([(vert_coords[vid] - bbox_min) / max_extent
                             for vid in all_vids], dtype=np.float32)
            vertex_pos_list.append(torch.from_numpy(vpos))

            # vertex-vertex edges (bidirectional) + build vv_adj for cycle ordering
            vv_adj = {}  # local_vid -> list of local_vid neighbors
            if frame in vv_data:
                vv_pairs = vv_data[frame]
                remapped_vv = []
                for vi, vj in vv_pairs:
                    if vi in vid_to_local and vj in vid_to_local:
                        li, lj = vid_to_local[vi], vid_to_local[vj]
                        remapped_vv.append([li, lj])
                        vv_adj.setdefault(li, []).append(lj)
                        vv_adj.setdefault(lj, []).append(li)
                if remapped_vv:
                    vv_edges = np.array(remapped_vv, dtype=np.int64).T
                    vv_edges_bi = np.concatenate([vv_edges, vv_edges[[1, 0]]], axis=1)
                    vertex_edge_index_list.append(torch.from_numpy(vv_edges_bi))
                else:
                    vertex_edge_index_list.append(torch.zeros(2, 0, dtype=torch.long))
            else:
                vertex_edge_index_list.append(torch.zeros(2, 0, dtype=torch.long))

            # vertex_indices: ordered vertex ring per cell, padded to max_V_per_cell
            n_cells_frame = len(cell_ids)
            per_cell_rings = []   # list of lists of local vertex indices (ordered)
            max_v = 0
            for cell_id in cell_ids:
                if cell_id in cell_verts:
                    raw_vids = [vid_to_local[vid] for vid in cell_verts[cell_id]
                                if vid in vid_to_local]
                    ordered = _order_cell_vertices(raw_vids, vv_adj)
                    per_cell_rings.append(ordered)
                    if len(ordered) > max_v:
                        max_v = len(ordered)
                else:
                    per_cell_rings.append([])

            # pad to (n_cells_frame, max_v) with -1
            if max_v == 0:
                max_v = 1  # avoid zero-width tensor
            vi_arr = torch.full((n_cells_frame, max_v), -1, dtype=torch.long)
            for slot, ring in enumerate(per_cell_rings):
                for k, vid_local in enumerate(ring):
                    vi_arr[slot, k] = vid_local

            # pad to max_n (padded cells get all -1)
            if n_cells_frame < max_n:
                pad = torch.full((max_n - n_cells_frame, max_v), -1, dtype=torch.long)
                vi_arr = torch.cat([vi_arr, pad], dim=0)
            vertex_indices_list.append(vi_arr)

        # print vertex data stats for first frame
        vi0 = vertex_indices_list[0]
        valid_counts = (vi0 >= 0).sum(dim=1)
        print(f"  vertex data frame 0: {vertex_pos_list[0].shape[0]} vertices, "
              f"{vertex_edge_index_list[0].shape[1] // 2} vertex edges, "
              f"verts/cell: {valid_counts[valid_counts > 0].float().mean():.1f} mean "
              f"({valid_counts[valid_counts > 0].min()}-{valid_counts[valid_counts > 0].max()} range)")

        if not save:
            print("  save=False, skipping zarr write")
            continue

        run_path = f"graphs_data/{dataset_name}/x_list_{run}"

        # write zarr V3
        x_writer = ZarrSimulationWriterV3(
            path=run_path,
            n_cells=max_n,
            dimension=dimension,
            time_chunks=min(2000, n_frames),
        )

        for t_idx in trange(n_frames, ncols=100, desc=f"  writing {embryo_key}"):
            pos_t = pos_frames[t_idx]
            vel_t = vel_frames[t_idx]

            # replace NaN with 0 for zarr storage (padded cells)
            pos_t_clean = np.nan_to_num(pos_t, nan=0.0)
            vel_t_clean = np.nan_to_num(vel_t, nan=0.0)

            state = CellState(
                index=torch.arange(max_n, dtype=torch.long),
                pos=torch.from_numpy(pos_t_clean),
                vel=torch.from_numpy(vel_t_clean),
                cell_type=cell_type,
            )
            x_writer.append_state(state)

            # visualization
            if visualize:
                _plot_embryo_frame(
                    pos_t, edge_index_list[t_idx],
                    vertex_pos_list[t_idx].numpy(),
                    vertex_indices_list[t_idx],
                    t_idx, run, dataset_name, embryo_key,
                )

        n_written = x_writer.finalize()
        print(f"  wrote {n_written} frames to zarr")

        # save edge_index (Delaunay cell-cell)
        save_edge_index(run_path, edge_index_list)
        print(f"  saved edge_index.pt ({len(edge_index_list)} frames)")

        # save vertex_indices (ordered vertex ring per cell)
        save_vertex_indices(run_path, vertex_indices_list)
        print(f"  saved vertex_indices.pt ({len(vertex_indices_list)} frames)")

        # save VertexTimeSeries (vertex positions + vertex-vertex edges)
        vts = VertexTimeSeries(pos=vertex_pos_list, edge_index=vertex_edge_index_list)
        save_vertex_timeseries(run_path, vts)
        print(f"  saved VertexTimeSeries ({len(vertex_pos_list)} frames)")

    print(f"\n=== Done loading embryo data ===")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _order_cell_vertices(vert_ids, vv_adj):
    """Order a cell's vertices into a cycle using vertex-vertex adjacency.

    Given the set of vertex IDs belonging to one cell and the global vertex
    adjacency dict, walk around the cell boundary to produce an ordered cycle.

    Returns ordered list, or the original list if cycle walk fails.
    """
    vert_set = set(vert_ids)
    if len(vert_set) < 3:
        return list(vert_ids)

    # build local adjacency within this cell
    local_adj = {v: [] for v in vert_set}
    for v in vert_set:
        for n in vv_adj.get(v, []):
            if n in vert_set:
                local_adj[v].append(n)
    # deduplicate
    local_adj = {v: list(set(ns)) for v, ns in local_adj.items()}

    # walk the cycle
    ordered = [vert_ids[0]]
    visited = {vert_ids[0]}
    current = vert_ids[0]
    for _ in range(len(vert_set) - 1):
        neighbors = [n for n in local_adj.get(current, []) if n not in visited]
        if not neighbors:
            break
        current = neighbors[0]
        ordered.append(current)
        visited.add(current)

    return ordered


def _plot_embryo_frame(pos, edge_index, vertex_pos, vertex_indices,
                       t_idx, run, dataset_name, embryo_key):
    """3D plot with filled Voronoi polygons.

    Args:
        pos: (max_N, 3) numpy array — cell centroids (NaN for padded)
        edge_index: (2, E) tensor — cell-cell edges (unused, kept for API)
        vertex_pos: (V, 3) numpy array — vertex positions
        vertex_indices: (max_N, max_V_per_cell) long tensor — ordered vertex ring per cell, -1 padded
        t_idx: frame index
        run: run index
        dataset_name: dataset name for output path
        embryo_key: embryo key for title
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 14), facecolor=default_style.background)
    ax = fig.add_subplot(111, projection='3d')

    valid = ~np.isnan(pos[:, 0])
    pos_valid = pos[valid]

    # --- Filled Voronoi cell polygons from vertex_indices ---
    polygons = []
    vi = vertex_indices.numpy() if hasattr(vertex_indices, 'numpy') else vertex_indices
    n_cells = vi.shape[0]
    n_verts = len(vertex_pos)

    for i in range(n_cells):
        ring = vi[i]
        ring_valid = ring[ring >= 0]
        if len(ring_valid) < 3:
            continue
        # filter out-of-range indices
        ring_valid = ring_valid[ring_valid < n_verts]
        if len(ring_valid) < 3:
            continue
        polygons.append(vertex_pos[ring_valid])

    if polygons:
        pc = Poly3DCollection(
            polygons,
            facecolors='#F5E6D0',    # light warm fill
            edgecolors='#E07020',     # orange outlines
            linewidths=0.3,
            alpha=0.95,
        )
        pc.set_sort_zpos(0)
        ax.add_collection3d(pc)

    # equal aspect ratio bounding box
    data_min = pos_valid.min(axis=0)
    data_max = pos_valid.max(axis=0)
    data_center = (data_min + data_max) / 2
    half_range = (data_max - data_min).max() / 2 * 1.05
    ax.set_xlim(data_center[0] - half_range, data_center[0] + half_range)
    ax.set_ylim(data_center[1] - half_range, data_center[1] + half_range)
    ax.set_zlim(data_center[2] - half_range, data_center[2] + half_range)
    default_style.xlabel(ax, 'X')
    default_style.ylabel(ax, 'Y')
    ax.set_zlabel('Z', fontsize=default_style.label_font_size, color=default_style.foreground)
    ax.set_title(f'{embryo_key} frame {t_idx}  '
                 f'({len(pos_valid)} cells, {len(polygons)} polygons)',
                 fontsize=default_style.font_size, color=default_style.foreground)

    num = f"{t_idx:06}"
    default_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png")
