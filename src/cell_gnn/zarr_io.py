"""zarr/tensorstore I/O utilities for simulation data.

provides:
- ZarrSimulationWriterV3: per-field writer for CellState/FieldState data (static + dynamic fields)
- ZarrArrayWriter: incremental writer for raw (T, N, F) arrays (e.g. derivative targets)
- detect_format: check if V3 zarr or .npy exists at path
- load_simulation_data: load as CellTimeSeries with optional field selection
- load_field_data: load as FieldTimeSeries (for mesh/field nodes)
- load_raw_array: load raw numpy array from zarr or npy (for derivative targets etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tensorstore as ts

if TYPE_CHECKING:
    from cell_gnn.cell_state import CellState, CellTimeSeries, FieldState, FieldTimeSeries, VertexTimeSeries


class ZarrArrayWriter:
    """incremental writer for raw (T, N, F) zarr arrays.

    used for derivative targets (y_list) and other non-CellState data.

    usage:
        writer = ZarrArrayWriter(path, n_cells=1000, n_features=2)
        for frame in simulation:
            writer.append(frame)  # frame is (N, F)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_cells: int,
        n_features: int,
        time_chunks: int = 2000,
        dtype: np.dtype = np.float32,
    ):
        self.path = Path(path)
        if not str(self.path).endswith('.zarr'):
            self.path = Path(str(self.path) + '.zarr')

        self.n_cells = n_cells
        self.n_features = n_features
        self.time_chunks = time_chunks
        self.dtype = dtype

        self._buffer: list[np.ndarray] = []
        self._total_frames = 0
        self._store: ts.TensorStore | None = None
        self._initialized = False

    def _initialize_store(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            import shutil
            shutil.rmtree(self.path, ignore_errors=True)

        initial_cap = max(self.time_chunks * 10, 1000)
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(self.path)},
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': [initial_cap, self.n_cells, self.n_features],
                'chunks': [self.time_chunks, self.n_cells, self.n_features],
                'compressor': {
                    'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                },
            },
            'create': True,
            'delete_existing': True,
        }
        self._store = ts.open(spec).result()
        self._initialized = True

    def append(self, frame: np.ndarray):
        if frame.shape != (self.n_cells, self.n_features):
            raise ValueError(
                f"frame shape {frame.shape} doesn't match expected "
                f"({self.n_cells}, {self.n_features})"
            )
        self._buffer.append(frame.astype(self.dtype, copy=False))
        if len(self._buffer) >= self.time_chunks:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        if not self._initialized:
            self._initialize_store()

        data = np.stack(self._buffer, axis=0)
        n_frames = data.shape[0]

        needed = self._total_frames + n_frames
        if needed > self._store.shape[0]:
            new_size = max(needed, self._store.shape[0] * 2)
            self._store = self._store.resize(
                exclusive_max=[new_size, self.n_cells, self.n_features]
            ).result()

        self._store[self._total_frames:self._total_frames + n_frames].write(data).result()
        self._total_frames += n_frames
        self._buffer.clear()

    def finalize(self):
        self._flush()
        if self._store is not None and self._total_frames > 0:
            self._store = self._store.resize(
                exclusive_max=[self._total_frames, self.n_cells, self.n_features]
            ).result()
        return self._total_frames


# cell-gnn field classification (mirrors cell_state.py)
_DYNAMIC_FIELDS = ['pos', 'vel', 'field']
_STATIC_FIELDS = ['cell_type']
# note: index is NOT saved — it is arange(n_cells) and constructed at load time


class ZarrSimulationWriterV3:
    """per-field zarr writer — each CellState/FieldState field gets its own zarr array.

    storage structure:
        path/
            cell_type.zarr  # (N,) int32 — static
            pos.zarr            # (T, N, dim) float32 — dynamic
            vel.zarr            # (T, N, dim) float32 — dynamic
            field.zarr          # (T, N, F) float32 — dynamic (optional)

    note: index is NOT saved — it is arange(n) and constructed at load time.

    usage:
        writer = ZarrSimulationWriterV3(path, n_cells=1000, dimension=2)
        for state in simulation:
            writer.append_state(state)  # accepts CellState or FieldState
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_cells: int,
        dimension: int,
        time_chunks: int = 2000,
    ):
        self.path = Path(path)
        self.n_cells = n_cells
        self.dimension = dimension
        self.time_chunks = time_chunks

        self._static_saved = False
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._stores: dict[str, ts.TensorStore] = {}
        self._total_frames = 0
        self._dynamic_initialized = False
        self._field_shapes: dict[str, tuple] = {}

    def _save_static(self, state: CellState | FieldState):
        """save static fields from first frame."""
        from cell_gnn.utils import to_numpy

        self.path.mkdir(parents=True, exist_ok=True)

        static_data = {
            'cell_type': to_numpy(state.cell_type).astype(np.int32),
        }

        for name, data in static_data.items():
            zarr_path = self.path / f'{name}.zarr'
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path, ignore_errors=True)

            dtype_str = '<i4' if data.dtype in (np.int32, np.int64) else '<f4'
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
                'metadata': {
                    'dtype': dtype_str,
                    'shape': list(data.shape),
                    'chunks': list(data.shape),
                    'compressor': {
                        'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                    },
                },
                'create': True,
                'delete_existing': True,
            }
            store = ts.open(spec).result()
            store.write(data).result()

        self._static_saved = True

    def _get_dynamic_fields(self, state: CellState | FieldState) -> dict[str, np.ndarray]:
        """extract dynamic field arrays from a CellState or FieldState."""
        from cell_gnn.utils import to_numpy

        fields = {}

        # pos: (N, dim)
        if state.pos is not None:
            fields['pos'] = to_numpy(state.pos).astype(np.float32)

        # vel: (N, dim)
        if state.vel is not None:
            fields['vel'] = to_numpy(state.vel).astype(np.float32)

        # field: (N, F) or (N,)
        if state.field is not None:
            f = to_numpy(state.field).astype(np.float32)
            if f.ndim == 1:
                f = f[:, np.newaxis]
            fields['field'] = f

        return fields

    def _initialize_dynamic_stores(self):
        """create zarr stores for dynamic fields."""
        initial_cap = max(self.time_chunks * 10, 1000)

        for name, shape in self._field_shapes.items():
            zarr_path = self.path / f'{name}.zarr'
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path, ignore_errors=True)

            # shape is per-frame shape, e.g. (N, dim) or (N, 1)
            full_shape = [initial_cap] + list(shape)
            chunk_shape = [self.time_chunks] + list(shape)

            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
                'metadata': {
                    'dtype': '<f4',
                    'shape': full_shape,
                    'chunks': chunk_shape,
                    'compressor': {
                        'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                    },
                },
                'create': True,
                'delete_existing': True,
            }
            self._stores[name] = ts.open(spec).result()

        self._dynamic_initialized = True

    def append_state(self, state: CellState | FieldState):
        """append one frame from CellState or FieldState."""
        if not self._static_saved:
            self._save_static(state)

        fields = self._get_dynamic_fields(state)

        # on first frame, record field shapes and init buffers
        if not self._field_shapes:
            for name, data in fields.items():
                self._field_shapes[name] = data.shape
                self._buffers[name] = []

        for name, data in fields.items():
            self._buffers[name].append(data)

        # flush when buffer is full (check any field)
        first_field = next(iter(self._buffers))
        if len(self._buffers[first_field]) >= self.time_chunks:
            self._flush_buffer()

    def _flush_buffer(self):
        """write buffered dynamic data to zarr stores."""
        first_field = next(iter(self._buffers), None)
        if first_field is None or not self._buffers[first_field]:
            return

        if not self._dynamic_initialized:
            self._initialize_dynamic_stores()

        n_frames = len(self._buffers[first_field])

        for name in self._buffers:
            data = np.stack(self._buffers[name], axis=0)  # (chunk, N, ...)

            # resize if needed
            current_shape = self._stores[name].shape
            needed = self._total_frames + n_frames
            if needed > current_shape[0]:
                new_size = max(needed, current_shape[0] * 2)
                new_max = [new_size] + list(current_shape[1:])
                self._stores[name] = self._stores[name].resize(
                    exclusive_max=new_max
                ).result()

            self._stores[name][self._total_frames:self._total_frames + n_frames].write(data).result()
            self._buffers[name].clear()

        self._total_frames += n_frames

    def finalize(self):
        """flush remaining buffer and resize stores to exact size."""
        self._flush_buffer()

        for name in self._stores:
            if self._total_frames > 0:
                current_shape = self._stores[name].shape
                final_max = [self._total_frames] + list(current_shape[1:])
                self._stores[name] = self._stores[name].resize(
                    exclusive_max=final_max
                ).result()

        return self._total_frames


def detect_format(path: str | Path) -> Literal['npy', 'zarr_v3', 'none']:
    """check what format exists at path.

    args:
        path: base path without extension

    returns:
        'zarr_v3' if V3 zarr directory exists (per-field .zarr arrays)
        'npy' if .npy file exists
        'none' if nothing exists
    """
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    # check for V3 zarr format (directory with per-field .zarr arrays)
    if base_path.exists() and base_path.is_dir():
        if (base_path / 'pos.zarr').exists():
            return 'zarr_v3'

    # check for npy
    npy_path = Path(str(base_path) + '.npy')
    if npy_path.exists():
        return 'npy'

    return 'none'


def load_simulation_data(path: str | Path, dimension: int, fields=None) -> CellTimeSeries:
    """load simulation data as CellTimeSeries (V3 zarr or npy).

    args:
        path: base path (with or without extension)
        dimension: spatial dimension (2 or 3) — needed for npy fallback
        fields: list of field names to load (V3 only, e.g. ['pos', 'vel']).
                None = all fields.

    returns:
        CellTimeSeries with requested fields (others are None)

    raises:
        FileNotFoundError: if no data found at path
    """
    from cell_gnn.cell_state import CellTimeSeries
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    fmt = detect_format(base_path)

    if fmt == 'zarr_v3':
        return _load_zarr_v3(base_path, fields)
    elif fmt == 'npy':
        npy_path = Path(str(base_path) + '.npy')
        return CellTimeSeries.from_packed(np.load(npy_path), dimension)

    raise FileNotFoundError(f"no .zarr or .npy found at {base_path}")


def _load_zarr_v3(path: Path, fields=None) -> CellTimeSeries:
    """load per-field zarr arrays into CellTimeSeries."""
    import torch
    from cell_gnn.cell_state import CellTimeSeries

    def _read_zarr(zarr_path):
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        }
        return ts.open(spec).result().read().result()

    all_dynamic = {'pos', 'vel', 'field'}
    all_static = {'cell_type'}
    load_fields = set(fields) if fields else (all_dynamic | all_static)

    kwargs = {}

    # static: cell_type
    pt_path = path / 'cell_type.zarr'
    if pt_path.exists() and 'cell_type' in load_fields:
        kwargs['cell_type'] = torch.from_numpy(np.array(_read_zarr(pt_path))).long()

    # index: reconstructed
    if 'cell_type' in kwargs:
        n = kwargs['cell_type'].shape[0]
    else:
        # infer n from first dynamic field
        for name in all_dynamic:
            zp = path / f'{name}.zarr'
            if zp.exists():
                arr = np.array(_read_zarr(zp))
                n = arr.shape[1]
                break
        else:
            raise FileNotFoundError(f"no fields found at {path}")
    kwargs['index'] = torch.arange(n, dtype=torch.long)

    # dynamic fields
    for name in all_dynamic:
        if name not in load_fields:
            continue
        zp = path / f'{name}.zarr'
        if not zp.exists():
            continue
        arr = np.array(_read_zarr(zp))
        t = torch.from_numpy(arr).float()
        kwargs[name] = t

    # ragged field: edge_index (list of (2, E_t) tensors)
    edge_list = load_edge_index(path)
    kwargs['edge_index'] = edge_list

    # ragged field: vertex_indices (list of (N_t, max_V_per_cell_t) tensors)
    vi_list = load_vertex_indices(path)
    kwargs['vertex_indices'] = vi_list

    return CellTimeSeries(**kwargs)


def load_field_data(path: str | Path, dimension: int, fields=None) -> FieldTimeSeries:
    """load mesh/field data as FieldTimeSeries (V3 zarr or legacy .pt).

    args:
        path: base path (with or without extension)
        dimension: spatial dimension (2 or 3)
        fields: list of field names to load (V3 only, e.g. ['pos', 'field']).
                None = all fields.

    returns:
        FieldTimeSeries with requested fields (others are None)

    raises:
        FileNotFoundError: if no data found at path
    """
    import torch
    from cell_gnn.cell_state import FieldTimeSeries

    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    fmt = detect_format(base_path)

    if fmt == 'zarr_v3':
        return _load_field_zarr_v3(base_path, fields)

    # legacy .pt fallback (list of packed tensors)
    pt_path = Path(str(base_path) + '.pt')
    if pt_path.exists():
        frames = torch.load(pt_path, map_location='cpu', weights_only=False)
        return FieldTimeSeries.from_packed_list(frames, dimension)

    raise FileNotFoundError(f"no .zarr or .pt found at {base_path}")


def _load_field_zarr_v3(path: Path, fields=None) -> FieldTimeSeries:
    """load per-field zarr arrays into FieldTimeSeries.

    FieldTimeSeries has static pos (N, dim), so pos is loaded from frame 0.
    """
    import torch
    from cell_gnn.cell_state import FieldTimeSeries

    def _read_zarr(zarr_path):
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        }
        return ts.open(spec).result().read().result()

    all_dynamic = {'vel', 'field'}
    all_static = {'cell_type', 'pos'}
    load_fields = set(fields) if fields else (all_dynamic | all_static)

    kwargs = {}

    # static: cell_type
    pt_path = path / 'cell_type.zarr'
    if pt_path.exists() and 'cell_type' in load_fields:
        kwargs['cell_type'] = torch.from_numpy(np.array(_read_zarr(pt_path))).long()

    # static: pos — saved as (T, N, dim), take frame 0 for FieldTimeSeries
    pos_path = path / 'pos.zarr'
    if pos_path.exists() and 'pos' in load_fields:
        arr = np.array(_read_zarr(pos_path))
        if arr.ndim == 3:
            kwargs['pos'] = torch.from_numpy(arr[0]).float()
        else:
            kwargs['pos'] = torch.from_numpy(arr).float()

    # infer n for index reconstruction
    if 'cell_type' in kwargs:
        n = kwargs['cell_type'].shape[0]
    elif 'pos' in kwargs:
        n = kwargs['pos'].shape[0]
    else:
        for name in all_dynamic:
            zp = path / f'{name}.zarr'
            if zp.exists():
                arr = np.array(_read_zarr(zp))
                n = arr.shape[1]
                break
        else:
            raise FileNotFoundError(f"no fields found at {path}")
    kwargs['index'] = torch.arange(n, dtype=torch.long)

    # dynamic fields: vel, field
    for name in all_dynamic:
        if name not in load_fields:
            continue
        zp = path / f'{name}.zarr'
        if not zp.exists():
            continue
        arr = np.array(_read_zarr(zp))
        kwargs[name] = torch.from_numpy(arr).float()

    return FieldTimeSeries(**kwargs)


def save_edge_index(path: str | Path, edge_list: list) -> None:
    """save per-frame edge_index list as .pt file.

    args:
        path: directory path (e.g. graphs_data/embryo/x_list_0)
        edge_list: list of (2, E_t) tensors, one per frame
    """
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(edge_list, path / 'edge_index.pt')


def load_edge_index(path: str | Path) -> list | None:
    """load per-frame edge_index list from .pt file, if it exists.

    args:
        path: directory path (e.g. graphs_data/embryo/x_list_0)

    returns:
        list of (2, E_t) tensors, or None if file doesn't exist
    """
    import torch
    pt_path = Path(path) / 'edge_index.pt'
    if pt_path.exists():
        return torch.load(pt_path, map_location='cpu', weights_only=False)
    return None


def save_vertex_indices(path: str | Path, vi_list: list) -> None:
    """save per-frame vertex_indices list as .pt file.

    args:
        path: directory path (e.g. graphs_data/embryo/x_list_0)
        vi_list: list of (N_t, max_V_per_cell_t) long tensors, -1 padded
    """
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(vi_list, path / 'vertex_indices.pt')


def load_vertex_indices(path: str | Path) -> list | None:
    """load per-frame vertex_indices list from .pt file, if it exists.

    returns:
        list of (N_t, max_V_per_cell_t) long tensors, or None
    """
    import torch
    pt_path = Path(path) / 'vertex_indices.pt'
    if pt_path.exists():
        return torch.load(pt_path, map_location='cpu', weights_only=False)
    return None


def save_vertex_timeseries(path: str | Path, vts: VertexTimeSeries) -> None:
    """save VertexTimeSeries as .pt files.

    saves:
        vertex_pos.pt — list of (V_t, dim) float tensors
        vertex_edge_index.pt — list of (2, E_v_t) long tensors
    """
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if vts.pos is not None:
        torch.save(vts.pos, path / 'vertex_pos.pt')
    if vts.edge_index is not None:
        torch.save(vts.edge_index, path / 'vertex_edge_index.pt')


def load_vertex_timeseries(path: str | Path) -> VertexTimeSeries | None:
    """load VertexTimeSeries from .pt files, if they exist.

    returns:
        VertexTimeSeries or None if no vertex data found
    """
    import torch
    from cell_gnn.cell_state import VertexTimeSeries

    path = Path(path)
    vp_path = path / 'vertex_pos.pt'
    ve_path = path / 'vertex_edge_index.pt'

    if not vp_path.exists():
        return None

    pos = torch.load(vp_path, map_location='cpu', weights_only=False)
    edge_index = None
    if ve_path.exists():
        edge_index = torch.load(ve_path, map_location='cpu', weights_only=False)

    return VertexTimeSeries(pos=pos, edge_index=edge_index)


def save_mesh_frame(
    path: str | Path,
    t_idx: int,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_labels: torch.Tensor,
) -> None:
    """save a single mesh frame to mesh/ subdirectory.

    saves as graphs_data/.../mesh/frame_XXXX.pt containing a dict with
    'vertices' (V,3) float32, 'faces' (F,3) int32, 'face_labels' (F,2) int32.
    """
    import torch
    mesh_dir = Path(path) / 'mesh'
    mesh_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {'vertices': vertices, 'faces': faces, 'face_labels': face_labels},
        mesh_dir / f'frame_{t_idx:04d}.pt',
    )


def load_mesh_timeseries(path: str | Path):
    """load MeshTimeSeries from per-frame .pt files in mesh/ subdirectory.

    returns:
        MeshTimeSeries or None if no mesh data found
    """
    import torch
    from cell_gnn.cell_state import MeshTimeSeries

    mesh_dir = Path(path) / 'mesh'
    if not mesh_dir.exists():
        return None

    frame_files = sorted(mesh_dir.glob('frame_*.pt'))
    if not frame_files:
        return None

    vertices_list = []
    faces_list = []
    face_labels_list = []

    for ff in frame_files:
        data = torch.load(ff, map_location='cpu', weights_only=False)
        vertices_list.append(data['vertices'])
        faces_list.append(data['faces'])
        face_labels_list.append(data['face_labels'])

    return MeshTimeSeries(
        vertices=vertices_list,
        faces=faces_list,
        face_labels=face_labels_list,
    )


def load_raw_array(path: str | Path) -> np.ndarray:
    """load a raw numpy array from .zarr or .npy (for y_list derivative targets etc.).

    args:
        path: base path (with or without extension)

    returns:
        numpy array

    raises:
        FileNotFoundError: if no data found at path
    """
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    # try zarr (single array)
    zarr_path = Path(str(base_path) + '.zarr')
    if zarr_path.exists() and zarr_path.is_dir():
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        }
        return np.array(ts.open(spec).result().read().result())

    # try npy
    npy_path = Path(str(base_path) + '.npy')
    if npy_path.exists():
        return np.load(npy_path)

    raise FileNotFoundError(f"no .zarr or .npy found at {base_path}")
