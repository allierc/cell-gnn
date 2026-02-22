"""cell state and field state dataclasses for cell-gnn simulation.

replaces the packed (N, 8) [2D] or (N, 10) [3D] tensor with named fields.
the dimension-dependent column layout is encapsulated in from_packed() / to_packed().

packed tensor layout:
  column 0:             index (cell ID)
  columns 1:1+dim:      pos (position, 2D or 3D)
  columns 1+dim:1+2*dim: vel (velocity, 2D or 3D)
  column 1+2*dim:       cell_type
  column 2+2*dim:       field (H1, variable width)

classes:
  CellState       — single-frame cell state (N, C) -> named fields
  CellTimeSeries  — full simulation timeseries, static metadata + dynamic per-frame data
  FieldState          — single-frame mesh/field node state (same packed layout as cells)
  FieldTimeSeries     — full simulation timeseries for mesh/field nodes
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


# field classification for timeseries classes
# unlike flyvis-gnn (neurons don't move), cells move so pos is dynamic
STATIC_FIELDS = {'index', 'cell_type'}
DYNAMIC_FIELDS = {'pos', 'vel', 'field'}
ALL_FIELDS = STATIC_FIELDS | DYNAMIC_FIELDS

FIELD_STATIC_FIELDS = {'index', 'pos', 'cell_type'}
FIELD_DYNAMIC_FIELDS = {'vel', 'field'}
FIELD_ALL_FIELDS = FIELD_STATIC_FIELDS | FIELD_DYNAMIC_FIELDS


def _apply(tensor, fn):
    """apply fn to tensor if not None, else return None."""
    return fn(tensor) if tensor is not None else None


def _subset_edges(edge_index: torch.Tensor, ids) -> torch.Tensor:
    """filter edge_index (2, E) to keep only edges between selected cell ids,
    then remap to the new [0, len(ids)) index space."""
    if not isinstance(ids, torch.Tensor):
        ids = torch.as_tensor(ids, dtype=torch.long, device=edge_index.device)
    id_set = set(ids.tolist())
    src, dst = edge_index[0], edge_index[1]
    mask = torch.tensor(
        [(s.item() in id_set) and (d.item() in id_set) for s, d in zip(src, dst)],
        dtype=torch.bool, device=edge_index.device,
    )
    filtered = edge_index[:, mask]
    # remap: old_id -> new_index
    remap = torch.full((ids.max().item() + 1,), -1, dtype=torch.long, device=edge_index.device)
    remap[ids] = torch.arange(len(ids), dtype=torch.long, device=edge_index.device)
    return remap[filtered]


def _unpack(x, dimension):
    """unpack a (N, C) tensor into named fields dict."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).float()
    else:
        t = x.float() if x.dtype != torch.float32 else x

    p1 = 1                      # pos start
    p2 = 1 + dimension           # pos end = vel start
    v2 = 1 + 2 * dimension       # vel end = type index
    type_idx = v2
    field_start = type_idx + 1

    result = dict(
        index=t[:, 0].long(),
        pos=t[:, p1:p2],
        vel=t[:, p2:v2],
        cell_type=t[:, type_idx].long(),
        field=None,
    )

    remaining = t.shape[1] - field_start
    if remaining >= 1:
        result['field'] = t[:, field_start:]

    return result


def _pack(index, pos, vel, cell_type, field, device):
    """pack named fields back into (N, C) tensor."""
    dim = pos.shape[1]
    n = pos.shape[0]
    n_cols = 1 + dim + dim + 1  # index + pos + vel + type
    if field is not None:
        field_cols = field.shape[1] if field.dim() > 1 else 1
        n_cols += field_cols

    x = torch.zeros(n, n_cols, dtype=torch.float32, device=device)
    col = 0

    if index is not None:
        x[:, col] = index.float()
    col += 1

    x[:, col:col + dim] = pos
    col += dim

    if vel is not None:
        x[:, col:col + dim] = vel
    col += dim

    if cell_type is not None:
        x[:, col] = cell_type.float()
    col += 1

    if field is not None:
        if field.dim() > 1:
            fc = field.shape[1]
            x[:, col:col + fc] = field
            col += fc
        else:
            x[:, col] = field
            col += 1

    return x


@dataclass
class CellState:
    """single-frame cell state for N cells.

    all fields default to None — only populated fields are used.
    """

    index: torch.Tensor | None = None           # (N,) long — cell IDs
    pos: torch.Tensor | None = None             # (N, dim) float32 — position
    vel: torch.Tensor | None = None             # (N, dim) float32 — velocity
    cell_type: torch.Tensor | None = None   # (N,) long — type label
    field: torch.Tensor | None = None           # (N, F) float32 — field / features
    edge_index: torch.Tensor | None = None      # (2, E) long — Delaunay adjacency
    vertex_indices: torch.Tensor | None = None  # (N, max_V_per_cell) long — ordered vertex ring, -1 padded

    # fields that participate in per-cell indexing (edge_index, vertex_indices excluded — variable/special size)
    _NODE_FIELDS = {'index', 'pos', 'vel', 'cell_type', 'field'}

    @property
    def n_cells(self) -> int:
        """infer N from the first non-None per-cell field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None and f.name in self._NODE_FIELDS:
                return val.shape[0]
        raise ValueError("CellState has no populated fields")

    @property
    def dimension(self) -> int:
        """infer spatial dimension from pos shape."""
        if self.pos is not None:
            return self.pos.shape[1]
        if self.vel is not None:
            return self.vel.shape[1]
        raise ValueError("cannot infer dimension: pos and vel are both None")

    @property
    def device(self) -> torch.device:
        """infer device from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("CellState has no populated fields")

    @classmethod
    def from_packed(cls, x: torch.Tensor | np.ndarray, dimension: int) -> CellState:
        """create from packed (N, C) tensor.

        2D layout (C=7+F): [index, x, y, vx, vy, type, field...]
        3D layout (C=8+F): [index, x, y, z, vx, vy, vz, type, field...]
        """
        d = _unpack(x, dimension)
        return cls(**d)

    def to_packed(self) -> torch.Tensor:
        """pack back into (N, C) tensor for legacy compatibility."""
        return _pack(self.index, self.pos, self.vel, self.cell_type,
                     self.field, self.device)

    def to(self, device: torch.device) -> CellState:
        """move all non-None tensors to device."""
        return CellState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def clone(self) -> CellState:
        """deep clone all non-None tensors."""
        return CellState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.clone())
            for f in dc_fields(self)
        })

    def detach(self) -> CellState:
        """detach all non-None tensors from computation graph."""
        return CellState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.detach())
            for f in dc_fields(self)
        })

    def subset(self, ids) -> CellState:
        """select a subset of cells by index.

        per-cell fields are indexed directly.
        edge_index is filtered to keep only edges between selected cells,
        then remapped to the new [0, len(ids)) index space.
        vertex_indices is indexed per-cell (no remapping — vertex IDs are global).
        """
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name == 'edge_index':
                kwargs[f.name] = _subset_edges(val, ids)
            elif f.name in self._NODE_FIELDS or f.name == 'vertex_indices':
                kwargs[f.name] = val[ids]
            else:
                kwargs[f.name] = val[ids]
        return CellState(**kwargs)

    @classmethod
    def zeros(cls, n_cells: int, dimension: int = 2,
              device: torch.device = None) -> CellState:
        """create zero-initialized CellState."""
        return cls(
            index=torch.arange(n_cells, dtype=torch.long, device=device),
            pos=torch.zeros(n_cells, dimension, dtype=torch.float32, device=device),
            vel=torch.zeros(n_cells, dimension, dtype=torch.float32, device=device),
            cell_type=torch.zeros(n_cells, dtype=torch.long, device=device),
            field=torch.zeros(n_cells, 1, dtype=torch.float32, device=device),
        )


@dataclass
class CellTimeSeries:
    """full simulation timeseries — static metadata + dynamic per-frame data.

    static fields are stored once (same for all frames): index, cell_type.
    dynamic fields have a leading time dimension (T, N, ...): pos, vel, field.

    follows the NeuronTimeSeries pattern from flyvis-gnn, but pos is dynamic
    because cells move (neurons don't).
    """

    # static (stored once)
    index: torch.Tensor | None = None           # (N,) long
    cell_type: torch.Tensor | None = None   # (N,) long

    # dynamic (per frame) — unlike flyvis-gnn, cells move so pos is dynamic
    pos: torch.Tensor | None = None             # (T, N, dim) float32
    vel: torch.Tensor | None = None             # (T, N, dim) float32
    field: torch.Tensor | None = None           # (T, N, F) float32

    # ragged dynamic — edge count varies per frame, stored as list of (2, E_t) tensors
    edge_index: list[torch.Tensor] | None = None

    # ragged dynamic — vertex ring per cell varies per frame
    # list of (N_t, max_V_per_cell_t) long tensors, -1 padded
    vertex_indices: list[torch.Tensor] | None = None

    @property
    def n_frames(self) -> int:
        """infer T from the first non-None dynamic field."""
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name)
            if val is not None:
                return val.shape[0]
        if self.edge_index is not None:
            return len(self.edge_index)
        raise ValueError("CellTimeSeries has no dynamic fields")

    @property
    def n_cells(self) -> int:
        """infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[-2] if f.name in DYNAMIC_FIELDS else val.shape[0]
        raise ValueError("CellTimeSeries has no populated fields")

    @property
    def dimension(self) -> int:
        """infer spatial dimension from pos (T, N, dim)."""
        if self.pos is not None:
            return self.pos.shape[2]
        if self.vel is not None:
            return self.vel.shape[2]
        raise ValueError("cannot infer dimension")

    @property
    def device(self) -> torch.device:
        """infer device from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("CellTimeSeries has no populated fields")

    # ragged list fields — stored as list of tensors, not regular (T, N, ...) arrays
    _RAGGED_FIELDS = {'edge_index', 'vertex_indices'}

    def frame(self, t: int) -> CellState:
        """extract single-frame CellState at time t.

        static fields are shared (not cloned).
        dynamic fields are cloned so the caller can modify them.
        ragged list fields (edge_index, vertex_indices) are extracted by list index.
        """
        kwargs = {}
        for f in dc_fields(self):
            if f.name in self._RAGGED_FIELDS:
                val = getattr(self, f.name)
                kwargs[f.name] = val[t] if val is not None else None
                continue
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in DYNAMIC_FIELDS:
                kwargs[f.name] = val[t].clone()
            else:
                kwargs[f.name] = val
        return CellState(**kwargs)

    def to(self, device: torch.device) -> CellTimeSeries:
        """move all non-None tensors to device."""
        kwargs = {}
        for f in dc_fields(self):
            if f.name in self._RAGGED_FIELDS:
                val = getattr(self, f.name)
                kwargs[f.name] = [x.to(device) for x in val] if val is not None else None
                continue
            kwargs[f.name] = _apply(getattr(self, f.name), lambda t: t.to(device))
        return CellTimeSeries(**kwargs)

    def subset_cells(self, ids) -> CellTimeSeries:
        """select a subset of cells by index.

        edge_index lists are filtered and remapped per frame.
        vertex_indices lists are indexed per cell (vertex IDs are global).
        """
        kwargs = {}
        for f in dc_fields(self):
            if f.name in self._RAGGED_FIELDS:
                continue
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in DYNAMIC_FIELDS:
                kwargs[f.name] = val[:, ids]
            else:
                kwargs[f.name] = val[ids]
        if self.edge_index is not None:
            kwargs['edge_index'] = [_subset_edges(ei, ids) for ei in self.edge_index]
        else:
            kwargs['edge_index'] = None
        if self.vertex_indices is not None:
            kwargs['vertex_indices'] = [vi[ids] for vi in self.vertex_indices]
        else:
            kwargs['vertex_indices'] = None
        return CellTimeSeries(**kwargs)

    @classmethod
    def from_packed(cls, arr: torch.Tensor | np.ndarray, dimension: int) -> CellTimeSeries:
        """create from packed (T, N, C) tensor or numpy array.

        2D layout (C=7+F): [index, x, y, vx, vy, type, field...]
        3D layout (C=8+F): [index, x, y, z, vx, vy, vz, type, field...]
        """
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr).float()
        else:
            t = arr.float() if arr.dtype != torch.float32 else arr

        p1 = 1
        p2 = 1 + dimension
        v2 = 1 + 2 * dimension
        type_idx = v2
        field_start = type_idx + 1

        result = cls(
            # static — take from first frame
            index=t[0, :, 0].long(),
            cell_type=t[0, :, type_idx].long(),
            # dynamic — all frames (cells move, unlike flyvis-gnn neurons)
            pos=t[:, :, p1:p2],
            vel=t[:, :, p2:v2],
        )

        remaining = t.shape[2] - field_start
        if remaining >= 1:
            result.field = t[:, :, field_start:]

        return result

    @classmethod
    def load(cls, path: str | Path, dimension: int) -> CellTimeSeries:
        """load from .npy file (legacy format)."""
        path = Path(path)
        if path.suffix == '.npy' or path.with_suffix('.npy').exists():
            npy_path = path if path.suffix == '.npy' else path.with_suffix('.npy')
            return cls.from_packed(np.load(npy_path), dimension)
        raise FileNotFoundError(f"no .npy found at {path}")


@dataclass
class FieldState:
    """single-frame mesh/field node state.

    same packed layout as CellState — used for mesh nodes in cell-field coupling.
    """

    index: torch.Tensor | None = None           # (N,) long — node IDs
    pos: torch.Tensor | None = None             # (N, dim) float32 — position
    vel: torch.Tensor | None = None             # (N, dim) float32 — velocity (often zero for mesh)
    cell_type: torch.Tensor | None = None   # (N,) long — type label
    field: torch.Tensor | None = None           # (N, F) float32 — field values

    @property
    def n_nodes(self) -> int:
        """infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[0]
        raise ValueError("FieldState has no populated fields")

    @property
    def dimension(self) -> int:
        if self.pos is not None:
            return self.pos.shape[1]
        if self.vel is not None:
            return self.vel.shape[1]
        raise ValueError("cannot infer dimension")

    @property
    def device(self) -> torch.device:
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("FieldState has no populated fields")

    @classmethod
    def from_packed(cls, x: torch.Tensor | np.ndarray, dimension: int) -> FieldState:
        """create from packed (N, C) tensor. same layout as CellState."""
        d = _unpack(x, dimension)
        return cls(**d)

    def to_packed(self) -> torch.Tensor:
        """pack back into (N, C) tensor for legacy compatibility."""
        return _pack(self.index, self.pos, self.vel, self.cell_type,
                     self.field, self.device)

    def to(self, device: torch.device) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def clone(self) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.clone())
            for f in dc_fields(self)
        })

    def detach(self) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.detach())
            for f in dc_fields(self)
        })

    def subset(self, ids) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t[ids])
            for f in dc_fields(self)
        })


@dataclass
class FieldTimeSeries:
    """full simulation timeseries for mesh/field nodes.

    static fields stored once: index, pos, cell_type.
    dynamic fields per frame: vel, field.
    """

    # static
    index: torch.Tensor | None = None           # (N,)
    pos: torch.Tensor | None = None             # (N, dim)
    cell_type: torch.Tensor | None = None   # (N,)

    # dynamic
    vel: torch.Tensor | None = None             # (T, N, dim)
    field: torch.Tensor | None = None           # (T, N, F)

    @property
    def n_frames(self) -> int:
        for name in FIELD_DYNAMIC_FIELDS:
            val = getattr(self, name)
            if val is not None:
                return val.shape[0]
        raise ValueError("FieldTimeSeries has no dynamic fields")

    @property
    def n_nodes(self) -> int:
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[-2] if f.name in FIELD_DYNAMIC_FIELDS else val.shape[0]
        raise ValueError("FieldTimeSeries has no populated fields")

    @property
    def dimension(self) -> int:
        if self.pos is not None:
            return self.pos.shape[1]
        if self.vel is not None:
            return self.vel.shape[2]
        raise ValueError("cannot infer dimension")

    @property
    def device(self) -> torch.device:
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("FieldTimeSeries has no populated fields")

    def frame(self, t: int) -> FieldState:
        """extract single-frame FieldState at time t."""
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in FIELD_DYNAMIC_FIELDS:
                kwargs[f.name] = val[t].clone()
            else:
                kwargs[f.name] = val
        return FieldState(**kwargs)

    def to(self, device: torch.device) -> FieldTimeSeries:
        return FieldTimeSeries(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def subset_nodes(self, ids) -> FieldTimeSeries:
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in FIELD_DYNAMIC_FIELDS:
                kwargs[f.name] = val[:, ids]
            else:
                kwargs[f.name] = val[ids]
        return FieldTimeSeries(**kwargs)

    @classmethod
    def from_packed_list(cls, frames: list[torch.Tensor], dimension: int) -> FieldTimeSeries:
        """create from list of (N, C) packed tensors (one per frame).

        this matches the legacy x_mesh_list format.
        """
        stacked = torch.stack(frames)  # (T, N, C)
        p1 = 1
        p2 = 1 + dimension
        v2 = 1 + 2 * dimension
        type_idx = v2
        field_start = type_idx + 1

        result = cls(
            index=stacked[0, :, 0].long(),
            pos=stacked[0, :, p1:p2],
            cell_type=stacked[0, :, type_idx].long(),
            vel=stacked[:, :, p2:v2],
        )

        remaining = stacked.shape[2] - field_start
        if remaining >= 1:
            result.field = stacked[:, :, field_start:]

        return result


# ---------------------------------------------------------------------------
# Vertex state — single frame and timeseries for the Voronoi vertex graph
# ---------------------------------------------------------------------------

@dataclass
class VertexState:
    """single-frame vertex state for V vertices.

    represents the Voronoi vertex graph (graph V in the MultiCell dual-graph
    formulation).  vertices are the learnable degrees of freedom in a vertex
    model — cell centroids are derived from vertex positions.
    """

    pos: torch.Tensor | None = None             # (V, dim) float32 — vertex positions
    edge_index: torch.Tensor | None = None      # (2, E_v) long — vertex-vertex mesh edges

    @property
    def n_vertices(self) -> int:
        if self.pos is not None:
            return self.pos.shape[0]
        raise ValueError("VertexState has no populated fields")

    @property
    def dimension(self) -> int:
        if self.pos is not None:
            return self.pos.shape[1]
        raise ValueError("cannot infer dimension: pos is None")

    @property
    def device(self) -> torch.device:
        if self.pos is not None:
            return self.pos.device
        if self.edge_index is not None:
            return self.edge_index.device
        raise ValueError("VertexState has no populated fields")

    def to(self, device: torch.device) -> VertexState:
        return VertexState(
            pos=_apply(self.pos, lambda t: t.to(device)),
            edge_index=_apply(self.edge_index, lambda t: t.to(device)),
        )

    def clone(self) -> VertexState:
        return VertexState(
            pos=_apply(self.pos, lambda t: t.clone()),
            edge_index=_apply(self.edge_index, lambda t: t.clone()),
        )

    def detach(self) -> VertexState:
        return VertexState(
            pos=_apply(self.pos, lambda t: t.detach()),
            edge_index=_apply(self.edge_index, lambda t: t.detach()),
        )


@dataclass
class VertexTimeSeries:
    """full simulation timeseries for Voronoi vertices.

    both pos and edge_index are ragged (vertex count and edge count change
    per frame as cells divide / invaginate), so they are stored as lists.
    """

    # ragged per-frame lists
    pos: list[torch.Tensor] | None = None             # list of (V_t, dim) float32
    edge_index: list[torch.Tensor] | None = None      # list of (2, E_v_t) long

    @property
    def n_frames(self) -> int:
        if self.pos is not None:
            return len(self.pos)
        if self.edge_index is not None:
            return len(self.edge_index)
        raise ValueError("VertexTimeSeries has no populated fields")

    def frame(self, t: int) -> VertexState:
        """extract single-frame VertexState at time t."""
        return VertexState(
            pos=self.pos[t] if self.pos is not None else None,
            edge_index=self.edge_index[t] if self.edge_index is not None else None,
        )

    def to(self, device: torch.device) -> VertexTimeSeries:
        return VertexTimeSeries(
            pos=[p.to(device) for p in self.pos] if self.pos is not None else None,
            edge_index=[e.to(device) for e in self.edge_index] if self.edge_index is not None else None,
        )


@dataclass
class MeshTimeSeries:
    """triangulated surface mesh timeseries (3D tissue segmentation meshes).

    each frame has a multi-cell mesh where faces have boundary labels
    indicating which two cells share that face.  vertex/face counts can
    change per frame (stored as ragged lists).
    """

    # ragged per-frame lists
    vertices: list[torch.Tensor] | None = None      # list of (V_t, 3) float32
    faces: list[torch.Tensor] | None = None         # list of (F_t, 3) int32
    face_labels: list[torch.Tensor] | None = None   # list of (F_t, 2) int32

    @property
    def n_frames(self) -> int:
        if self.vertices is not None:
            return len(self.vertices)
        if self.faces is not None:
            return len(self.faces)
        raise ValueError("MeshTimeSeries has no populated fields")

    def frame(self, t: int) -> tuple:
        """extract single-frame mesh data at time t.

        returns:
            (vertices, faces, face_labels) tuple of tensors (or None)
        """
        return (
            self.vertices[t] if self.vertices is not None else None,
            self.faces[t] if self.faces is not None else None,
            self.face_labels[t] if self.face_labels is not None else None,
        )

    def to(self, device: torch.device) -> MeshTimeSeries:
        return MeshTimeSeries(
            vertices=[v.to(device) for v in self.vertices] if self.vertices is not None else None,
            faces=[f.to(device) for f in self.faces] if self.faces is not None else None,
            face_labels=[l.to(device) for l in self.face_labels] if self.face_labels is not None else None,
        )
