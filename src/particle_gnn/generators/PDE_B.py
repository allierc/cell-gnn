
import torch
import torch.nn as nn
from particle_gnn.utils import to_numpy
from particle_gnn.particle_state import ParticleState
from particle_gnn.graph_utils import remove_self_loops, scatter_aggregate

class PDE_B(nn.Module):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of Boids as a function of their relative positions and relative velocities.
    The interaction function is defined by three parameters p = (p1, p2, p3)

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    dd_pos : float
        the acceleration of the Boids (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], bc_dpos=[], dimension=2):
        super().__init__()

        self.aggr_type = aggr_type
        self.p = p
        self.bc_dpos = bc_dpos
        self.dimension = dimension

        self.a1 = 0.5E-5
        self.a2 = 5E-4
        self.a3 = 1E-8
        self.a4 = 0.5E-5
        self.a5 = 1E-8

    def forward(self, state: ParticleState, edge_index: torch.Tensor, has_field=False):
        if has_field:
            field = state.field
        else:
            field = torch.ones(state.n_particles, 1, dtype=state.pos.dtype, device=state.pos.device)

        edge_index = remove_self_loops(edge_index)
        parameters = self.p[to_numpy(state.particle_type), :]
        d_pos = state.vel.clone().detach()

        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = state.pos[dst], state.pos[src]
        parameters_i = parameters[dst]
        d_pos_i, d_pos_j = d_pos[dst], d_pos[src]
        field_j = field[src]

        messages = self.message(pos_i, pos_j, parameters_i, d_pos_i, d_pos_j, field_j)
        dd_pos = scatter_aggregate(messages, dst, state.n_particles, self.aggr_type)
        return dd_pos

    def message(self, pos_i, pos_j, parameters_i, d_pos_i, d_pos_j, field_j):
        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # distance squared

        cohesion = parameters_i[:,0,None] * self.a1 * self.bc_dpos(pos_j - pos_i)
        alignment = parameters_i[:,1,None] * self.a2 * self.bc_dpos(d_pos_j - d_pos_i)
        separation = - parameters_i[:,2,None] * self.a3 * self.bc_dpos(pos_j - pos_i) / distance_squared[:, None]

        return (separation + alignment + cohesion) * field_j


    def psi(self, r, p):
        cohesion = p[0] * self.a4 * r
        separation = -p[2] * self.a5 / r
        return (cohesion + separation)  # 5E-4 alignement
