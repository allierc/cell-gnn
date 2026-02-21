"""
Native PyTorch replacements for torch_geometric utilities.

Provides remove_self_loops, scatter_aggregate, GraphData, collate_graph_batch,
and compute_mesh_laplacian — removing the torch-geometric dependency entirely.
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  Edge utilities
# ------------------------------------------------------------------ #

def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove self-loops from edge_index.

    Args:
        edge_index: (2, E) tensor of source/target node indices.

    Returns:
        Filtered edge_index of shape (2, E') where E' <= E.
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


# ------------------------------------------------------------------ #
#  Scatter aggregation
# ------------------------------------------------------------------ #

def scatter_aggregate(
    messages: torch.Tensor,
    target_index: torch.Tensor,
    num_nodes: int,
    aggr: str = 'add',
) -> torch.Tensor:
    """Aggregate per-edge messages to target nodes.

    Args:
        messages:     (E, D) per-edge message tensor.
        target_index: (E,)   target node index for each edge.
        num_nodes:    total number of nodes N.
        aggr:         'add' or 'mean'.

    Returns:
        (N, D) aggregated node features.
    """
    D = messages.shape[1]
    out = torch.zeros(num_nodes, D, dtype=messages.dtype, device=messages.device)
    idx = target_index.unsqueeze(1).expand_as(messages)
    out.scatter_add_(0, idx, messages)

    if aggr == 'mean':
        count = torch.zeros(num_nodes, dtype=messages.dtype, device=messages.device)
        count.scatter_add_(0, target_index, torch.ones_like(target_index, dtype=messages.dtype))
        count = count.clamp(min=1)
        out = out / count.unsqueeze(1)

    return out


# ------------------------------------------------------------------ #
#  Graph data container  (replaces torch_geometric.data.Data)
# ------------------------------------------------------------------ #

@dataclass
class GraphData:
    """Minimal graph data container.

    Attributes:
        x:          (N, C) node features, or a list of tensors for time-window.
        edge_index: (2, E) edge connectivity.
        num_nodes:  explicit node count (inferred from x if omitted).
    """
    x: object = None            # torch.Tensor or List[torch.Tensor]
    edge_index: torch.Tensor = None
    num_nodes: int = 0

    def __post_init__(self):
        if self.num_nodes == 0 and self.x is not None:
            if isinstance(self.x, list):
                self.num_nodes = self.x[0].shape[0]
            else:
                self.num_nodes = self.x.shape[0]


def collate_graph_batch(graphs: List[GraphData]) -> GraphData:
    """Batch multiple GraphData objects into one (offsets edge_index).

    Equivalent to torch_geometric DataLoader with batch_size=len(graphs).

    Args:
        graphs: list of GraphData.

    Returns:
        Single GraphData with concatenated x and offset edge_index.
    """
    if len(graphs) == 1:
        return graphs[0]

    is_list_x = isinstance(graphs[0].x, list)
    edge_list = []
    offset = 0

    if is_list_x:
        n_time = len(graphs[0].x)
        x_lists = [[] for _ in range(n_time)]
        for g in graphs:
            for t, xt in enumerate(g.x):
                x_lists[t].append(xt)
            edge_list.append(g.edge_index + offset)
            offset += g.num_nodes
        batched_x = [torch.cat(t_list, dim=0) for t_list in x_lists]
    else:
        x_list = []
        for g in graphs:
            x_list.append(g.x)
            edge_list.append(g.edge_index + offset)
            offset += g.num_nodes
        batched_x = torch.cat(x_list, dim=0)

    batched_edges = torch.cat(edge_list, dim=1)
    return GraphData(x=batched_x, edge_index=batched_edges, num_nodes=offset)


# ------------------------------------------------------------------ #
#  Mesh Laplacian  (replaces torch_geometric.utils.get_mesh_laplacian)
# ------------------------------------------------------------------ #

def compute_mesh_laplacian(
    pos: torch.Tensor,
    face: torch.Tensor,
) -> tuple:
    """Cotangent-weighted mesh Laplacian (unnormalized).

    Equivalent to ``get_mesh_laplacian(pos, face, normalization="None")``.

    Args:
        pos:  (N, 3) vertex positions.
        face: (3, F) triangle face indices.

    Returns:
        edge_index: (2, E) sparse indices (includes diagonal).
        edge_weight: (E,)  Laplacian weights.
    """
    N = pos.shape[0]
    i, j, k = face[0], face[1], face[2]

    def _cot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dot = (a * b).sum(dim=1)
        cross_norm = torch.linalg.norm(torch.cross(a, b, dim=1), dim=1).clamp(min=1e-8)
        return dot / cross_norm

    cot_k = _cot(pos[i] - pos[k], pos[j] - pos[k])   # edge (i,j)
    cot_i = _cot(pos[j] - pos[i], pos[k] - pos[i])   # edge (j,k)
    cot_j = _cot(pos[i] - pos[j], pos[k] - pos[j])   # edge (i,k)

    # Build symmetric off-diagonal entries (both directions)
    row = torch.cat([i, j, j, k, i, k])
    col = torch.cat([j, i, k, j, k, i])
    w   = torch.cat([cot_k, cot_k, cot_i, cot_i, cot_j, cot_j]) * 0.5

    # Coalesce duplicate edges
    edge_index = torch.stack([row, col], dim=0)
    sparse = torch.sparse_coo_tensor(edge_index, w, (N, N)).coalesce()
    off_idx = sparse.indices()
    off_val = sparse.values()

    # Diagonal: L_ii = sum of off-diagonal weights in row i
    diag_val = torch.zeros(N, dtype=w.dtype, device=w.device)
    diag_val.scatter_add_(0, off_idx[0], off_val)

    diag_idx = torch.arange(N, device=pos.device)
    diag_edge = torch.stack([diag_idx, diag_idx], dim=0)

    final_edge_index = torch.cat([off_idx, diag_edge], dim=1)
    final_weight = torch.cat([-off_val, diag_val])  # negative off-diagonal, positive diagonal

    return final_edge_index, final_weight
