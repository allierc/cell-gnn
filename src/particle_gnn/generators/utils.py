import numpy as np
import torch

from particle_gnn.generators import PDE_A, PDE_B, PDE_G
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
        case 'PDE_A':
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
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case _:
            raise ValueError(f'Unknown particle model: {particle_model_name}')

    return model, bc_pos, bc_dpos


def init_particles(config=[], scenario='none', ratio=1, device=[]):
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

    dpos = dpos_init * torch.randn((n_particles, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    type = torch.zeros(int(n_particles / n_particle_types), device=device)
    for n in range(1, n_particle_types):
        type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
    if type.shape[0] < n_particles:
        type = torch.cat((type, n * torch.ones(n_particles - type.shape[0], device=device)), 0)
    if config.simulation.non_discrete_level > 0:
        type = torch.tensor(np.arange(n_particles), device=device)

    features = torch.cat((torch.randn((n_particles, 1), device=device) * 5,
                           0.1 * torch.randn((n_particles, 1), device=device)), 1)

    type = type[:, None]
    particle_id = torch.arange(n_particles, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_particles, 1), device=device)

    if 'uniform' in scenario:
        type = torch.ones(n_particles, device=device) * int(scenario.split()[-1])
        type = type[:, None]
    if 'stripes' in scenario:
        l = n_particles // n_particle_types
        for n in range(n_particle_types):
            index = np.arange(n * l, (n + 1) * l)
            pos[index, 1:2] = torch.rand(l, 1, device=device) * (1 / n_particle_types) + n / n_particle_types

    return pos, dpos, type, features, age, particle_id


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
