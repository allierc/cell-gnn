import numpy as np
import torch
from scipy.spatial import Delaunay
from tifffile import imread
from time import sleep
from particle_gnn.graph_utils import compute_mesh_laplacian

from particle_gnn.generators import PDE_A, PDE_B, PDE_G
from particle_gnn.particle_state import ParticleState, FieldState
from particle_gnn.utils import choose_boundary_values, to_numpy, get_equidistant_points


def choose_model(config=[], W=[], device=[]):
    particle_model_name = config.graph_model.particle_model_name
    aggr_type = config.graph_model.aggr_type
    n_particles = config.simulation.n_particles
    n_particle_types = config.simulation.n_particle_types

    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)

    dimension = config.simulation.dimension

    params = config.simulation.params
    p = torch.tensor(params, dtype=torch.float32, device=device).squeeze()

    # create GNN depending on type specified in config file
    match particle_model_name:
        case 'PDE_A' | 'PDE_ParticleField_A':
            if config.simulation.non_discrete_level > 0:
                p = torch.ones(n_particle_types, 4, device=device) + torch.rand(n_particle_types, 4, device=device)
                pp = []
                n_particle_types = len(params)
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
                for n in range(n_particle_types):
                    if n == 0:
                        pp = p[n].repeat(n_particles // n_particle_types, 1)
                    else:
                        pp = torch.cat((pp, p[n].repeat(n_particles // n_particle_types, 1)), 0)
                p = pp.clone().detach()
                p = p + torch.randn(n_particles, 4, device=device) * config.simulation.non_discrete_level
            sigma = config.simulation.sigma
            p = p if n_particle_types == 1 else torch.squeeze(p)
            func_p = config.simulation.func_params
            embedding_step = config.simulation.n_frames // 100
            model = PDE_A(aggr_type=aggr_type, p=p, func_p=func_p, sigma=sigma, bc_dpos=bc_dpos,
                          dimension=dimension, embedding_step=embedding_step)
        case 'PDE_B':
            model = PDE_B(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_G':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            model = PDE_G(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos, dimension=dimension)
        case _:
            raise ValueError(f'Unknown particle model: {particle_model_name}')

    return model, bc_pos, bc_dpos


def init_particlestate(config=[], scenario='none', ratio=1, device=[]):
    """initialize particle state as a ParticleState dataclass."""
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles * ratio
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if simulation_config.boundary == 'periodic':
        pos = torch.rand(n_particles, dimension, device=device)
        if n_particles <= 10:
            pos = pos * 0.1 + 0.45
        elif n_particles <= 100:
            pos = pos * 0.2 + 0.4
        elif n_particles <= 500:
            pos = pos * 0.5 + 0.25
    else:
        pos = torch.randn(n_particles, dimension, device=device) * 0.5

    vel = dpos_init * torch.randn((n_particles, dimension), device=device)
    vel = torch.clamp(vel, min=-torch.std(vel), max=+torch.std(vel))

    particle_type = torch.zeros(int(n_particles / n_particle_types), device=device)
    for n in range(1, n_particle_types):
        particle_type = torch.cat((particle_type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
    if particle_type.shape[0] < n_particles:
        particle_type = torch.cat((particle_type, n * torch.ones(n_particles - particle_type.shape[0], device=device)), 0)
    if config.simulation.non_discrete_level > 0:
        particle_type = torch.tensor(np.arange(n_particles), device=device)

    field = torch.cat((torch.randn((n_particles, 1), device=device) * 5,
                        0.1 * torch.randn((n_particles, 1), device=device)), 1)

    if 'uniform' in scenario:
        particle_type = torch.ones(n_particles, device=device) * int(scenario.split()[-1])
    if 'stripes' in scenario:
        l = n_particles // n_particle_types
        for n in range(n_particle_types):
            index = np.arange(n * l, (n + 1) * l)
            pos[index, 1:2] = torch.rand(l, 1, device=device) * (1 / n_particle_types) + n / n_particle_types

    return ParticleState(
        index=torch.arange(n_particles, device=device),
        pos=pos,
        vel=vel,
        particle_type=particle_type.long(),
        field=field,
    )


def random_rotation_matrix(device='cpu'):
    # Random Euler angles
    roll = torch.rand(1, device=device) * 2 * torch.pi
    pitch = torch.rand(1, device=device) * 2 * torch.pi
    yaw = torch.rand(1, device=device) * 2 * torch.pi

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices around each axis
    R_x = torch.tensor([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ], device=device).squeeze()

    R_y = torch.tensor([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], device=device).squeeze()

    R_z = torch.tensor([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ], device=device).squeeze()

    # Combined rotation matrix: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R


def init_mesh(config, device):
    """initialize mesh state as a FieldState dataclass."""

    simulation_config = config.simulation
    model_config = config.graph_model

    n_nodes = simulation_config.n_nodes
    node_value_map = simulation_config.node_value_map
    field_grid = model_config.field_grid

    n_nodes_per_axis = int(np.sqrt(n_nodes))
    xs = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    ys = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing='xy')
    x_grid = torch.reshape(x_grid, (n_nodes_per_axis ** 2, 1))
    y_grid = torch.reshape(y_grid, (n_nodes_per_axis ** 2, 1))
    mesh_size = 1 / n_nodes_per_axis
    pos = torch.zeros((n_nodes, 2), device=device)
    pos[0:n_nodes, 0:1] = x_grid[0:n_nodes]
    pos[0:n_nodes, 1:2] = y_grid[0:n_nodes]

    i0 = imread(f'graphs_data/{node_value_map}')
    if len(i0.shape) == 2:
        i0 = np.flipud(i0)
        values = i0[(to_numpy(pos[:, 1]) * 255).astype(int), (to_numpy(pos[:, 0]) * 255).astype(int)]

    mask_mesh = (x_grid > torch.min(x_grid) + 0.02) & (x_grid < torch.max(x_grid) - 0.02) & (y_grid > torch.min(y_grid) + 0.02) & (y_grid < torch.max(y_grid) - 0.02)

    if 'grid' not in field_grid:
        if 'pattern_Null.tif' in simulation_config.node_value_map:
            pos = pos + torch.randn(n_nodes, 2, device=device) * mesh_size / 24
        else:
            pos = pos + torch.randn(n_nodes, 2, device=device) * mesh_size / 8

    mesh_state = FieldState(
        index=torch.arange(n_nodes, device=device),
        pos=pos,
        vel=torch.zeros((n_nodes, 2), device=device),
        particle_type=torch.zeros(n_nodes, device=device).long(),
        field=torch.zeros((n_nodes, 2), device=device),
    )

    # Delaunay triangulation
    pos_np = to_numpy(mesh_state.pos)
    tri = Delaunay(pos_np, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    # removal of skinny faces
    sleep(0.5)
    for k in range(face.shape[0]):
        x1 = pos_np[face[k, 0], :]
        x2 = pos_np[face[k, 1], :]
        x3 = pos_np[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((mesh_state.pos, torch.ones((n_nodes, 1), device=device)), dim=1)
    edge_index_mesh, edge_weight_mesh = compute_mesh_laplacian(pos=pos_3d, face=face)
    edge_weight_mesh = edge_weight_mesh.to(dtype=torch.float32)

    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}

    return mesh_state, mesh_data
