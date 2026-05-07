import numpy as np
import torch
from scipy.spatial import Delaunay
from tifffile import imread
from time import sleep
from cell_gnn.graph_utils import compute_mesh_laplacian

from cell_gnn.models.registry import get_simulator_class
from cell_gnn.cell_state import CellState, FieldState
from cell_gnn.utils import choose_boundary_values, to_numpy, get_equidistant_points


def choose_model(config=[], W=[], device=[]):
    cell_model_name = config.graph_model.cell_model_name
    aggr_type = config.graph_model.aggr_type
    n_cells = config.simulation.n_cells
    n_cell_types = config.simulation.n_cell_types

    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)

    dimension = config.simulation.dimension

    params = config.simulation.cell_params
    p = torch.tensor(params, dtype=torch.float32, device=device).squeeze()

    sim_cls = get_simulator_class(cell_model_name)

    # Per-model parameter setup
    match cell_model_name:
        case 'arbitrary_ode' | 'arbitrary_field_ode':
            if config.simulation.non_discrete_level > 0:
                p = torch.ones(n_cell_types, 4, device=device) + torch.rand(n_cell_types, 4, device=device)
                pp = []
                n_cell_types = len(params)
                for n in range(n_cell_types):
                    p[n] = torch.tensor(params[n])
                for n in range(n_cell_types):
                    if n == 0:
                        pp = p[n].repeat(n_cells // n_cell_types, 1)
                    else:
                        pp = torch.cat((pp, p[n].repeat(n_cells // n_cell_types, 1)), 0)
                p = pp.clone().detach()
                p = p + torch.randn(n_cells, 4, device=device) * config.simulation.non_discrete_level
            sigma = config.simulation.sigma
            p = p if n_cell_types == 1 else torch.squeeze(p)
            func_p = config.simulation.func_params
            embedding_step = config.simulation.n_frames // 100
            model = sim_cls(aggr_type=aggr_type, p=p, func_p=func_p, sigma=sigma, bc_dpos=bc_dpos,
                            dimension=dimension, embedding_step=embedding_step)
        case 'boids_ode' | 'boids_field_ode':
            model = sim_cls(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension)
        case 'gravity_ode':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_cell_types)
                p = torch.tensor(p, device=device)
            model = sim_cls(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                            pred_limit=config.training.pred_limit, bc_dpos=bc_dpos, dimension=dimension)
        case 'particle_spring_force_ode':
            noise_model_level = config.simulation.noise_model_level if hasattr(config.simulation, 'noise_model_level') else 0
            model = sim_cls(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension,
                            noise_model_level=noise_model_level)
        case 'particle_spring_force_prescribed_field' | 'particle_spring_force_prescribed_field_siren' | 'particle_spring_force_prescribed_field_siren_grad':
            noise_model_level = config.simulation.noise_model_level if hasattr(config.simulation, 'noise_model_level') else 0
            fp = config.simulation.field_params
            field_type = getattr(fp, 'field_type', None) or 'single'
            # Single mode: center_0/velocity are (dim,).
            # Multi mode:  center_0/velocity are (S, dim).
            # amplitude, sigma, mu_chem are scalars in both modes (all S blobs share them).
            field_params = {
                'field_type': field_type,
                'center_0':   torch.tensor(fp.center_0, dtype=torch.float32, device=device),
                'velocity':   torch.tensor(fp.velocity, dtype=torch.float32, device=device),
                'amplitude':  fp.amplitude,
                'sigma':      fp.sigma,
                'mu_chem':    fp.mu_chem,
                'delta_t':    config.simulation.delta_t,
                'periodic':   config.simulation.boundary == 'periodic',
            }
            model = sim_cls(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension,
                            noise_model_level=noise_model_level, field_params=field_params)
        case 'particle_spring_force_diffusion_field' | 'particle_spring_force_diffusion_field_siren' | 'particle_spring_force_diffusion_field_siren_pde' | 'particle_spring_force_diffusion_field_siren_grad' | 'particle_spring_force_diffusion_field_siren_grad_pde' | 'particle_spring_force_diffusion_field_siren_grid_pde' | 'particle_spring_force_diffusion_field_internal' | 'particle_spring_force_diffusion_field_siren_grid_pde_internal':
            noise_model_level = config.simulation.noise_model_level if hasattr(config.simulation, 'noise_model_level') else 0
            fp = config.simulation.field_params
            field_params = {
                'center_0':        torch.tensor(fp.center_0, dtype=torch.float32, device=device),
                'amplitude':       fp.amplitude,
                'sigma':           fp.sigma,
                'mu_chem':         fp.mu_chem,
                'diffusion_coeff': fp.diffusion_coeff,
                'lambda_decay':    fp.lambda_decay,
                'grid_resolution': fp.grid_resolution,
                'source_strength': getattr(fp, 'source_strength', 0.0),
                'source_fraction': getattr(fp, 'source_fraction', 1.0),
                'pulse_period':    getattr(fp, 'pulse_period', 0),
                'pulse_duty':      getattr(fp, 'pulse_duty', 1.0),
                'chem_saturation_scale': getattr(fp, 'chem_saturation_scale', None),
                'chem_saturation_type':  getattr(fp, 'chem_saturation_type', 'log'),
                'internal_coeff_s':     getattr(fp, 'internal_coeff_s', 2.0),
                'internal_coeff_emit':  getattr(fp, 'internal_coeff_emit', 20.0),
                'internal_c_threshold': getattr(fp, 'internal_c_threshold', 0.01),
                'internal_c_sharpness': getattr(fp, 'internal_c_sharpness', 200.0),
                'internal_init_range':  tuple(getattr(fp, 'internal_init_range', (0.0, 1.0))),
                'delta_t':         config.simulation.delta_t,
                'periodic':        config.simulation.boundary == 'periodic',
            }
            model = sim_cls(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension,
                            noise_model_level=noise_model_level, field_params=field_params)
        case _:
            raise ValueError(f'Unknown cell model: {cell_model_name}')

    return model, bc_pos, bc_dpos


def init_cellstate(config=[], scenario='none', ratio=1, device=[]):
    """initialize cell state as a CellState dataclass."""
    simulation_config = config.simulation
    n_cells = simulation_config.n_cells * ratio
    n_cell_types = simulation_config.n_cell_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if simulation_config.boundary == 'periodic':
        pos = torch.rand(n_cells, dimension, device=device)
        if n_cells <= 10:
            pos = pos * 0.1 + 0.45
        elif n_cells <= 100:
            pos = pos * 0.2 + 0.4
        elif n_cells <= 500:
            pos = pos * 0.5 + 0.25
    else:
        pos = torch.randn(n_cells, dimension, device=device) * 0.5

    vel = dpos_init * torch.randn((n_cells, dimension), device=device)
    vel = torch.clamp(vel, min=-torch.std(vel), max=+torch.std(vel))

    cell_type = torch.zeros(int(n_cells / n_cell_types), device=device)
    for n in range(1, n_cell_types):
        cell_type = torch.cat((cell_type, n * torch.ones(int(n_cells / n_cell_types), device=device)), 0)
    if cell_type.shape[0] < n_cells:
        cell_type = torch.cat((cell_type, n * torch.ones(n_cells - cell_type.shape[0], device=device)), 0)
    if config.simulation.non_discrete_level > 0:
        cell_type = torch.tensor(np.arange(n_cells), device=device)

    field = torch.cat((torch.randn((n_cells, 1), device=device) * 5,
                        0.1 * torch.randn((n_cells, 1), device=device)), 1)

    if 'uniform' in scenario:
        cell_type = torch.ones(n_cells, device=device) * int(scenario.split()[-1])
    if 'stripes' in scenario:
        l = n_cells // n_cell_types
        for n in range(n_cell_types):
            index = np.arange(n * l, (n + 1) * l)
            pos[index, 1:2] = torch.rand(l, 1, device=device) * (1 / n_cell_types) + n / n_cell_types

    return CellState(
        index=torch.arange(n_cells, device=device),
        pos=pos,
        vel=vel,
        cell_type=cell_type.long(),
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
        cell_type=torch.zeros(n_nodes, device=device).long(),
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
