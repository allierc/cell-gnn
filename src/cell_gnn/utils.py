import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
import gc
import subprocess
from dataclasses import dataclass

import warnings

warnings.filterwarnings('ignore')


def sort_key(filename):
            # Extract the numeric parts using regular expressions
            if filename.split('_')[-2] == 'graphs':
                return 0
            else:
                return 1E7 * int(filename.split('_')[-2]) + int(filename.split('_')[-1][:-3])


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    return tensor.detach().cpu().numpy()


def set_device(device: str = 'auto'):
    """
    Set the device to use for computations. If 'auto' is specified, the device is chosen automatically:
     * if GPUs are available, the GPU with the most free memory is chosen
     * if MPS is available, MPS is used
     * otherwise, the CPU is used
    :param device: The device to use for computations. Automatically chosen if 'auto' is specified (default).
    :return: The torch.device object that is used for computations.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # Unset CUDA_VISIBLE_DEVICES

    if device == 'auto':
        if torch.cuda.is_available():
            try:
                # Use nvidia-smi to get free memory of each GPU
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                # Parse the output
                free_mem_list = []
                for line in result.strip().split('\n'):
                    index_str, mem_str = line.strip().split(',')
                    index = int(index_str)
                    free_mem = float(mem_str) * 1024 * 1024  # Convert MiB to bytes
                    free_mem_list.append((index, free_mem))
                # Ensure the device count matches
                num_gpus = torch.cuda.device_count()
                if num_gpus != len(free_mem_list):
                    print(f"Mismatch in GPU count between PyTorch ({num_gpus}) and nvidia-smi ({len(free_mem_list)})")
                    device = 'cpu'
                    print(f"using device: {device}")
                else:
                    # Find the GPU with the most free memory
                    max_free_memory = -1
                    best_device_id = -1
                    for index, free_mem in free_mem_list:
                        if free_mem > max_free_memory:
                            max_free_memory = free_mem
                            best_device_id = index
                    if best_device_id == -1:
                        raise ValueError("Could not determine the GPU with the most free memory.")

                    device = f'cuda:{best_device_id}'
                    torch.cuda.set_device(best_device_id)  # Set the chosen device globally
                    total_memory_gb = torch.cuda.get_device_properties(best_device_id).total_memory / 1024 ** 3
                    free_memory_gb = max_free_memory / 1024 ** 3
                    print(
                        f"using device: {device}, name: {torch.cuda.get_device_name(best_device_id)}, "
                        f"total memory: {total_memory_gb:.2f} GB, free memory: {free_memory_gb:.2f} GB")
            except Exception as e:
                print(f"Failed to get GPU information: {e}")
                device = 'cpu'
                print(f"using device: {device}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"using device: {device}")
        else:
            device = 'cpu'
            print(f"using device: {device}")
    return device


def set_size(x, cells, mass_distrib_index):
    # cells = index_cells[n]

    #size = 5 * np.power(3, ((to_numpy(x[index_cells[n] , -2]) - 200)/100)) + 10
    size = np.power((to_numpy(x[cells, mass_distrib_index])), 1.2) / 1.5

    return size


def get_gpu_memory_map(device=None):
    t = np.round(torch.cuda.get_device_properties(device).total_memory / 1E9, 2)
    r = np.round(torch.cuda.memory_reserved(device) / 1E9, 2)
    a = np.round(torch.cuda.memory_allocated(device) / 1E9, 2)

    return t, r, a


def symmetric_cutoff(x, percent=1):
    """
    Minimum and maximum values if a certain percentage of the data is cut off from both ends.
    """
    x_lower = np.percentile(x, percent)
    x_upper = np.percentile(x, 100 - percent)
    return x_lower, x_upper


def norm_area(x, device):

    pos = torch.argwhere(x[:, -1]<1.0)
    ax = torch.std(x[pos, -1])

    return torch.tensor([ax], device=device)


def norm_velocity(x, dimension, device):
    from cell_gnn.cell_state import CellState
    if isinstance(x, CellState):
        vel = x.vel
    else:
        vel = CellState.from_packed(x, dimension).vel

    vx = torch.std(vel[:, 0])

    return torch.tensor([vx], device=device)


def get_2d_bounding_box(pos):

    x_min, y_min = torch.min(pos, dim=0).values
    x_max, y_max = torch.max(pos, dim=0).values

    bounding_box = {
        'x_min': x_min.item(),
        'x_max': x_max.item(),
        'y_min': y_min.item(),
        'y_max': y_max.item()
    }

    return bounding_box


def get_3d_bounding_box(pos):

    x_min, y_min, z_min = torch.min(pos, dim=0).values
    x_max, y_max, z_max = torch.max(pos, dim=0).values

    bounding_box = {
        'x_min': x_min.item(),
        'x_max': x_max.item(),
        'y_min': y_min.item(),
        'y_max': y_max.item(),
        'z_min': z_min.item(),
        'z_max': z_max.item()
    }

    return bounding_box


def norm_position(x, dimension, device):
    from cell_gnn.cell_state import CellState
    if isinstance(x, CellState):
        pos = x.pos
    else:
        pos = CellState.from_packed(x, dimension).pos

    if dimension == 2:
        bounding_box = get_2d_bounding_box(pos * 1.1)
        posnorm = max(bounding_box.values())
        return torch.tensor(posnorm, dtype=torch.float32, device=device), torch.tensor([bounding_box['x_max']/posnorm, bounding_box['y_max']/posnorm], dtype=torch.float32, device=device)
    else:
        bounding_box = get_3d_bounding_box(pos * 1.1)
        posnorm = max(bounding_box.values())
        return torch.tensor(posnorm, dtype=torch.float32, device=device), torch.tensor([bounding_box['x_max']/posnorm, bounding_box['y_max']/posnorm, bounding_box['z_max']/posnorm], dtype=torch.float32, device=device)


def norm_acceleration(yy, device):
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = symmetric_cutoff(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = symmetric_cutoff(nay)

    # return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)

    return torch.tensor([ax], device=device)


def choose_boundary_values(bc_name):
    def identity(x):
        return x

    def periodic(x):
        return torch.remainder(x, 1.0)

    def periodic_wall(x):
        y = torch.remainder(x[:,0:1], 1.0)
        return torch.cat((y,x[:,1:2]), 1)

    def shifted_periodic(x):
        return torch.remainder(x - 0.5, 1.0) - 0.5

    def shifted_periodic_wall(x):
        y = torch.remainder(x[:,0:1] - 0.5, 1.0) - 0.5
        return torch.cat((y,x[:,1:2]), 1)


    match bc_name:
        case 'no':
            return identity, identity
        case 'periodic':
            return periodic, shifted_periodic
        case 'wall':
            return periodic_wall, shifted_periodic_wall
        case _:
            raise ValueError(f'unknown boundary condition {bc_name}')


def get_r2_numpy_corrcoef(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2


class CustomColorMap:
    def __init__(self, config):
        self.cmap_name = config.plotting.colormap
        self.model_name = config.graph_model.cell_model_name

        if self.cmap_name == 'tab10':
            self.nmap = 8
        else:
            self.nmap = config.simulation.n_cells

    def color(self, index):
        color_map = plt.colormaps.get_cmap(self.cmap_name)
        if self.cmap_name == 'tab20':
            color = color_map(index % 20)
        else:
            color = color_map(index)

        return color


def add_pre_folder(config_file_):

    if 'arbitrary' in config_file_:
        config_file = os.path.join('arbitrary', config_file_)
        pre_folder = 'arbitrary/'
    elif 'boids' in config_file_:
        config_file = os.path.join('boids', config_file_)
        pre_folder = 'boids/'
    elif 'gravity' in config_file_:
        config_file = os.path.join('gravity', config_file_)
        pre_folder = 'gravity/'
    elif 'dicty' in config_file_:
        config_file = os.path.join('misc', config_file_)
        pre_folder = 'misc/'
    elif 'embryo' in config_file_:
        config_file = os.path.join('misc', config_file_)
        pre_folder = 'misc/'
    elif 'gland' in config_file_:
        config_file = os.path.join('misc', config_file_)
        pre_folder = 'misc/'
    else:
        raise ValueError(f'unknown config file type: {config_file_}')

    return config_file, pre_folder


def get_log_dir(config=[]):

    if 'arbitrary_ode' in config.graph_model.cell_model_name:
        l_dir = os.path.join('./log/arbitrary/')
    elif 'boids_ode' in config.graph_model.cell_model_name:
        l_dir = os.path.join('./log/boids/')
    elif 'gravity_ode' in config.graph_model.cell_model_name:
        l_dir = os.path.join('./log/gravity/')
    else:
        raise ValueError(f'unknown cell model name: {config.graph_model.cell_model_name}')

    return l_dir


def create_log_dir(config=[], erase=True):

    log_dir = os.path.join('.', 'log', config.config_file)
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/cell'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/external_input'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/matrix'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/prediction'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/MLP0'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/MLP1'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/edges_embedding'), exist_ok=True)
    if config.training.n_ghosts > 0:
        os.makedirs(os.path.join(log_dir, 'tmp_training/ghost'), exist_ok=True)

    if erase:
        files = glob.glob(f"{log_dir}/results/*")
        for f in files:
            if ('all' not in f) & ('field' not in f):
                os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/cell/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/external_input/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/matrix/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/function/MLP1/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/function/MLP0s/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/embedding/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/ghost/*")
        for f in files:
            os.remove(f)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)

    logger = logging.getLogger()
    logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return log_dir, logger


def fig_init(fontsize=None, formatx='%.2f', formaty='%.2f'):
    from cell_gnn.figure_style import default_style
    fig, ax = default_style.figure(height=12, formatx=formatx, formaty=formaty)
    fs = fontsize if fontsize is not None else default_style.frame_tick_font_size
    plt.xticks([])
    plt.yticks([])
    ax.tick_params(axis='both', which='major', pad=15)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    return fig, ax


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def check_and_clear_memory(
        device: str = None,
        iteration_number: int = None,
        every_n_iterations: int = 100,
        memory_percentage_threshold: float = 0.6
):
    """
    Check the memory usage of a GPU and clear the cache every n iterations or if it exceeds a certain threshold.
    :param device: The device to check the memory usage for.
    :param iteration_number: The current iteration number.
    :param every_n_iterations: Clear the cache every n iterations.
    :param memory_percentage_threshold: Percentage of memory usage that triggers a clearing.
    """

    if device and 'cuda' in device:
        logger = logging.getLogger(__name__)

        if (iteration_number % every_n_iterations == 0):

            torch.cuda.memory_allocated(device)
            gc.collect()
            torch.cuda.empty_cache()

            if (iteration_number==0):
                logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
                logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")


        if torch.cuda.memory_allocated(device) > memory_percentage_threshold * torch.cuda.get_device_properties(device).total_memory:
            print ("Memory usage is high. Calling garbage collector and clearing cache.")
            gc.collect()
            torch.cuda.empty_cache()


def get_index_cells(x, n_cell_types, dimension):
    from cell_gnn.cell_state import CellState
    if isinstance(x, CellState):
        ptype = x.cell_type
    else:
        ptype = CellState.from_packed(x, dimension).cell_type

    index_cells = []
    for n in range(n_cell_types):
        index = np.argwhere(ptype.detach().cpu().numpy() == n)
        index_cells.append(index.squeeze())
    return index_cells


def get_equidistant_points(n_points=1024):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    r = np.sqrt(indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y = r * np.cos(theta), r * np.sin(theta)

    return x, y


def edges_radius_blockwise(
    x,
    dimension,
    bc_dpos,
    min_radius,
    max_radius,
    block=1024,
    include_self=False,
):
    """
    Builds a bidirectional edge_index (both i->j and j->i) for pairs with
    min_radius < dist(i,j) < max_radius, using blockwise computation to avoid OOM.
    Uses j>i to compute each pair once, then mirrors.

    Returns:
        edge_index: LongTensor of shape [2, E] on the same device as x
    """
    from cell_gnn.cell_state import CellState
    if isinstance(x, CellState):
        pos = x.pos
    else:
        pos = CellState.from_packed(x, dimension).pos
    device = pos.device
    N = pos.shape[0]
    min2 = float(min_radius * min_radius)
    max2 = float(max_radius * max_radius)

    rows = []
    cols = []

    # Precompute global j indices once (saves a little overhead)
    global_j = torch.arange(N, device=device)[None, :]  # [1, N]

    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        pi = pos[i0:i1]  # [B, D]

        d = bc_dpos(pi[:, None, :] - pos[None, :, :])    # [B, N, D]
        dist2 = (d * d).sum(dim=-1)                      # [B, N]

        # radius rule
        mask = (dist2 > min2) & (dist2 < max2)           # [B, N] bool

        # keep only upper triangle: j > i
        global_i = torch.arange(i0, i1, device=device)[:, None]  # [B, 1]
        mask &= (global_j > global_i)

        if include_self:
            # (usually you don't want self-edges; if you do, enable this)
            mask |= (global_j == global_i)

        ii, jj = mask.nonzero(as_tuple=True)             # ii in [0,B), jj in [0,N)
        rows.append(ii + i0)                             # global i
        cols.append(jj)                                  # global j

        # free big temporaries before next block
        del d, dist2, mask, ii, jj

    # unique (i<j) edges
    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    edge_ij = torch.stack([row, col], dim=0).long()      # [2, E_half]

    # mirror to get both directions
    edge_index = torch.cat([edge_ij, edge_ij.flip(0)], dim=1).contiguous()  # [2, 2*E_half]
    return edge_index


@dataclass
class NeighborCache:
    pos_ref: torch.Tensor = None        # [N,D] positions at last rebuild (for displacement tracking)
    edge_list: torch.Tensor = None      # [2,Elist] pairs within r_list (bidirectional if you want)
    r_cut: float = None
    r_skin: float = None
    r_list: float = None


def build_edge_list_blockwise(pos, bc_dpos, r_list, min_radius=0.0, block=2048):
    # Use edges_radius_blockwise with max_radius=r_list
    edge_list = edges_radius_blockwise(
        x=torch.cat([torch.zeros((pos.shape[0],1), device=pos.device), pos], dim=1), # dummy to match signature
        dimension=pos.shape[1],
        bc_dpos=bc_dpos,
        min_radius=min_radius,
        max_radius=r_list,
        block=block
    )
    return edge_list


def filter_edges_by_cutoff(pos, edge_list, bc_dpos, r_cut, min_radius=0.0):
    src = edge_list[0]
    dst = edge_list[1]
    d = bc_dpos(pos[src] - pos[dst])
    dist2 = (d*d).sum(dim=-1)
    keep = (dist2 < r_cut*r_cut) & (dist2 > min_radius*min_radius)
    return edge_list[:, keep].contiguous()


def get_edges_with_cache(
    pos: torch.Tensor,
    bc_dpos,
    cache: NeighborCache,
    r_cut: float,
    r_skin: float,
    min_radius: float = 0.0,
    block: int = 2048,
):
    # Initialize cache if needed or radii changed
    if (cache.edge_list is None) or (cache.pos_ref is None) or (cache.r_cut != r_cut) or (cache.r_skin != r_skin):
        cache.r_cut = float(r_cut)
        cache.r_skin = float(r_skin)
        cache.r_list = float(r_cut + r_skin)
        cache.pos_ref = pos.detach().clone()
        cache.edge_list = build_edge_list_blockwise(pos, bc_dpos, cache.r_list, min_radius=min_radius, block=block)

    # Check max displacement since last rebuild
    disp = bc_dpos(pos - cache.pos_ref)                  # [N,D]
    max_disp = torch.sqrt((disp*disp).sum(dim=-1)).max() # scalar

    if max_disp > (cache.r_skin * 0.5):
        # Rebuild list
        cache.pos_ref = pos.detach().clone()
        cache.edge_list = build_edge_list_blockwise(pos, bc_dpos, cache.r_list, min_radius=min_radius, block=block)

    # Always filter list to current cutoff
    edge_index = filter_edges_by_cutoff(pos, cache.edge_list, bc_dpos, r_cut, min_radius=min_radius)
    return edge_index
