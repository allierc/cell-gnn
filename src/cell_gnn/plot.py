"""Centralized plot functions and vectorized helpers for cell-gnn.

All plot functions that were previously scattered across models/utils.py,
models/graph_trainer.py, and generators/graph_data_generator.py are
consolidated here.
"""
from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import torch
import torch.nn as nn
import umap

from cell_gnn.cell_state import CellState
from cell_gnn.figure_style import default_style, dark_style, FigureStyle
from cell_gnn.utils import to_numpy


# --------------------------------------------------------------------------- #
#  Diffusion field ground-truth replay
# --------------------------------------------------------------------------- #

def _replay_diffusion_field(config, k_frame, grid_pos, device, return_scalar=False):
    """Replay the diffusion PDE up to frame *k_frame* and return the true
    chemotactic force ``F_chem`` evaluated at *grid_pos*.

    Honors pulsed sources (``source_fraction``, ``pulse_period``, ``pulse_duty``)
    and the saturating-log chemotaxis (``chem_saturation_scale``) when those are
    set in ``field_params``.

    Returns
    -------
    v_true : (M, dim) tensor — chemotactic force at *grid_pos*
    c_true : (M, 1) tensor (only if return_scalar=True) — scalar c at grid_pos
    """
    from cell_gnn.zarr_io import load_simulation_data
    from cell_gnn.generators.particle_spring_force_diffusion_field import (
        init_gaussian_field, build_spectral_decay, deposit_to_grid,
        deposit_to_grid_weighted, spectral_gradient, interp_periodic,
    )

    sim = config.simulation
    fp = sim.field_params
    dataset_name = config.dataset
    dim = sim.dimension
    res = fp.grid_resolution
    dt = sim.delta_t
    D = fp.diffusion_coeff
    lam = fp.lambda_decay
    alpha = fp.source_strength
    mu_chem = fp.mu_chem

    # Detect the ..._internal model family — emission rule there is
    #   alpha_i = coeff_emit * (1 - s_i) * sigmoid(beta * (c_i - c_threshold))
    # NOT the constant-rate / pulsed-rate scheme used by the older models.
    model_name = getattr(getattr(config, 'graph_model', None), 'cell_model_name', '') or ''
    is_internal = '_internal' in model_name

    # Optional pulsed-source params (defaults match the generator's no-pulse path)
    source_fraction = getattr(fp, 'source_fraction', 1.0)
    pulse_period = getattr(fp, 'pulse_period', 0)
    pulse_duty = getattr(fp, 'pulse_duty', 1.0)
    g0 = getattr(fp, 'chem_saturation_scale', None)

    # Internal-state emission knobs (used only on the is_internal branch).
    coeff_emit_int  = float(getattr(fp, 'internal_coeff_emit', 20.0))
    c_threshold_int = float(getattr(fp, 'internal_c_threshold', 0.01))
    c_sharpness_int = float(getattr(fp, 'internal_c_sharpness', 200.0))

    center_0 = torch.tensor(fp.center_0, dtype=torch.float32, device=device)

    # initialise field grid + decay operator
    field_grid = init_gaussian_field(
        res, dim, center_0, fp.amplitude, fp.sigma,
        sim.boundary == 'periodic', device)
    decay = build_spectral_decay(res, dim, D, lam, dt, device)

    # load saved positions (and `field` channels for the internal-model branch)
    x_ts = load_simulation_data(f'graphs_data/{dataset_name}/x_list_0', dim)

    # Lazily build pulsed source mask + phases using the same seed as data gen
    source_mask = None
    source_phase = None
    seed = getattr(sim, 'seed', 0)

    s = [res] * dim
    for k in range(k_frame + 1):
        pos_k = x_ts.frame(k).pos.to(device)

        if is_internal:
            # Match the simulator's per-step emission rule. The simulator uses
            #   c_at_pos_old: c sampled from the running grid at the *current*
            #                 cell positions (BEFORE this step's spectral update),
            #   s_in        : s entering this step  =  s at end of previous step,
            #                 saved as state.field[:, 1] at frame max(0, k-1).
            if coeff_emit_int > 0:
                k_in = max(0, k - 1)
                fld_in = x_ts.frame(k_in).field
                if fld_in is None or fld_in.shape[-1] < 2:
                    # No saved s — fall back to s=0.5 (mid of init range).
                    s_in = torch.full((pos_k.shape[0],), 0.5, device=device, dtype=field_grid.dtype)
                else:
                    s_in = fld_in[:, 1].to(device).to(field_grid.dtype)

                c_at_pos = interp_periodic(field_grid, pos_k, res)
                gate = torch.sigmoid(c_sharpness_int * (c_at_pos - c_threshold_int))
                emit = coeff_emit_int * (1.0 - s_in) * gate
                src = deposit_to_grid_weighted(pos_k, emit * dt, res, dim, device)
                field_grid = field_grid + src

        elif alpha > 0:
            if source_mask is None:
                n = pos_k.shape[0]
                gen = torch.Generator(device=pos_k.device)
                gen.manual_seed(int(seed))
                n_src = max(1, int(round(source_fraction * n)))
                perm = torch.randperm(n, generator=gen, device=pos_k.device)
                source_mask = torch.zeros(n, dtype=torch.bool, device=pos_k.device)
                source_mask[perm[:n_src]] = True
                source_phase = torch.rand(n, generator=gen, device=pos_k.device)

            if pulse_period and pulse_period > 0:
                phase = (k / float(pulse_period) + source_phase) % 1.0
                d_ = torch.minimum(phase, 1.0 - phase)
                sigma_p = max(pulse_duty, 1e-3) / 2.0
                pulse = torch.exp(-0.5 * (d_ / sigma_p) ** 2) * source_mask.float()
                gain = 1.0 / max(pulse_duty, 1e-3)
                active = pulse > 1e-4
                if active.any():
                    strength = alpha * dt * gain * pulse[active]
                    src = deposit_to_grid_weighted(pos_k[active], strength, res, dim, device)
                    field_grid = field_grid + src
            else:
                emit_pos = pos_k[source_mask] if source_fraction < 1.0 else pos_k
                src = deposit_to_grid(emit_pos, alpha * dt, res, dim, device)
                field_grid = field_grid + src

        C_hat = torch.fft.rfftn(field_grid, s=s)
        C_hat = C_hat * decay
        field_grid = torch.fft.irfftn(C_hat, s=s)

    # gradient + interpolation
    C_hat = torch.fft.rfftn(field_grid, s=s)
    grad_grid = spectral_gradient(C_hat, res, dim, device)

    grid_pos_dev = grid_pos.to(device)
    field_gradient = interp_periodic(grad_grid, grid_pos_dev, res)  # (M, dim)

    # Saturating-log chemotaxis if g0 is set, else linear.
    if g0 is not None and g0 > 0:
        g_norm = field_gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
        v_true = mu_chem * torch.log1p(g_norm / g0) * (field_gradient / g_norm)
    else:
        v_true = mu_chem * field_gradient

    if return_scalar:
        c_true = interp_periodic(field_grid, grid_pos_dev, res).unsqueeze(-1)  # (M, 1)
        return v_true, c_true
    return v_true


def _get_plot_grid(model, dimension, device, z_target=0.5, fallback_n=24):
    """Pick the spatial sampling grid for field plots.

    If the model exposes ``grid_coords`` (e.g. CellGNNSirenGridPDE), use that exact
    grid so the plot matches the surface the model was trained on. For 3D models,
    take a z-slice closest to ``z_target``. Otherwise build a uniform fallback grid.

    Returns
    -------
    pos      : (M, D) tensor of query points (flat)
    n_x, n_y : ints, used to reshape (M,) → (n_y, n_x) for imshow
    extent   : (xmin, xmax, ymin, ymax) for imshow
    z_slice  : float, the actual z used (for plot titles)
    """
    if hasattr(model, 'grid_coords') and hasattr(model, 'grid_shape'):
        gc = model.grid_coords        # (G, D), built with indexing='ij' (x outermost)
        gs = model.grid_shape         # tuple of length D
        if dimension == 3 and len(gs) == 3:
            n_x, n_y, n_z = gs
            coords3 = gc.view(n_x, n_y, n_z, 3)             # (Nx, Ny, Nz, 3)
            zs = coords3[0, 0, :, 2]                         # z values along k-axis
            z_idx = int(torch.argmin(torch.abs(zs - z_target)).item())
            z_slice = float(zs[z_idx].item())
            slab = coords3[:, :, z_idx, :]                   # (Nx, Ny, 3) — i=x, j=y
            # Reorder to (Ny, Nx, 3) so flatten gives (y outer, x inner) = same as fallback
            slab = slab.permute(1, 0, 2).contiguous()        # (Ny, Nx, 3)
            pos = slab.reshape(-1, 3).to(device)
            x_lo = float(coords3[0, 0, 0, 0].item()); x_hi = float(coords3[-1, 0, 0, 0].item())
            y_lo = float(coords3[0, 0, 0, 1].item()); y_hi = float(coords3[0, -1, 0, 1].item())
            return pos, n_x, n_y, (x_lo, x_hi, y_lo, y_hi), z_slice
        elif dimension == 2 and len(gs) == 2:
            n_x, n_y = gs
            coords2 = gc.view(n_x, n_y, 2).permute(1, 0, 2).contiguous()   # (Ny, Nx, 2)
            pos = coords2.reshape(-1, 2).to(device)
            x_lo = float(coords2[0, 0, 0].item()); x_hi = float(coords2[0, -1, 0].item())
            y_lo = float(coords2[0, 0, 1].item()); y_hi = float(coords2[-1, 0, 1].item())
            return pos, n_x, n_y, (x_lo, x_hi, y_lo, y_hi), None

    # Fallback: ad-hoc uniform grid in [0, 1]
    n = fallback_n
    xs = torch.linspace(0.0, 1.0, n, device=device)
    ys = torch.linspace(0.0, 1.0, n, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    if dimension == 3:
        pos = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            torch.full_like(grid_x.reshape(-1), float(z_target)),
        ], dim=1)
        return pos, n, n, (0.0, 1.0, 0.0, 1.0), float(z_target)
    else:
        pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        return pos, n, n, (0.0, 1.0, 0.0, 1.0), None


def _eval_siren_field(model, pos, k_frame, ynorm, device, return_scalar=False):
    """Evaluate the SIREN field branch, handling both vector and scalar-grad models.

    For vector SIREN (out_features == dim): direct output.
    For scalar SIREN (out_features == 1): compute grad_x(c) via autograd.
    If the model has learnable ``log_mu_chem``/``log_chem_g0`` (PDE model), apply
    the saturating-log chemotaxis to match the generator.

    Returns
    -------
    v_learned : (M, dim) np.ndarray
    c_learned : (M, 1) np.ndarray (only if return_scalar=True, else None for vector SIREN)
    """
    is_grad_model = ('siren_grad' in model.model) or ('siren_grid' in model.model)
    k_tensor = torch.full((pos.shape[0], 1), float(k_frame), dtype=pos.dtype, device=device)

    if is_grad_model:
        pos_field = pos.detach().requires_grad_(True)
        # Use physical time if the model was trained with it (PDE model), else normalized.
        if hasattr(model, 'log_mu_chem'):
            t_in = (k_tensor * model.delta_t).detach()
        else:
            t_in = k_tensor / model.n_frames
        siren_input = torch.cat([pos_field, t_in], dim=1)
        c = model.siren_field(siren_input)  # (M, 1)
        grad_c = torch.autograd.grad(
            outputs=c, inputs=pos_field,
            grad_outputs=torch.ones_like(c),
            create_graph=False, retain_graph=False,
        )[0]  # (M, dim)

        # Match model.forward: linear or saturating depending on the model.
        if hasattr(model, 'log_mu_chem'):
            mu_chem = torch.exp(model.log_mu_chem)
            if getattr(model, 'linear_chemotaxis', False):
                v_learned = mu_chem * grad_c
            elif hasattr(model, 'log_chem_g0'):
                g0 = torch.exp(model.log_chem_g0)
                g_norm = grad_c.norm(dim=1, keepdim=True).clamp_min(1e-12)
                v_learned = mu_chem * torch.log1p(g_norm / g0) * (grad_c / g_norm)
            else:
                v_learned = mu_chem * grad_c
        else:
            v_learned = grad_c * float(ynorm)

        v_np = to_numpy(v_learned.detach())
        if return_scalar:
            return v_np, to_numpy(c.detach())
        return v_np
    else:
        with torch.no_grad():
            siren_input = torch.cat([pos, k_tensor / model.n_frames], dim=1)
            v_learned = model.siren_field(siren_input) * float(ynorm)
        v_np = to_numpy(v_learned)
        if return_scalar:
            return v_np, None
        return v_np


# --------------------------------------------------------------------------- #
#  Vectorized helpers
# --------------------------------------------------------------------------- #

def build_edge_features(rr, embedding, model_name, max_radius, dimension=2, batched=None):
    """Build input features for the edge MLP, supporting batched embeddings.

    Args:
        rr: (n_pts,) tensor of radial distances
        embedding: (N, embed_dim) or (n_pts, embed_dim) tensor
        model_name: one of arbitrary_ode, boids_ode, gravity_ode, arbitrary_field_ode, boids_field_ode
        max_radius: float
        dimension: int, spatial dimension (2 or 3)
        batched: if True, force batched path (embedding is (N, embed_dim)); if False, force
            non-batched path (embedding is (n_pts, embed_dim)); if None, auto-detect by
            shape (ambiguous when N == n_pts).

    Returns:
        (N, n_pts, input_dim) or (n_pts, input_dim) tensor of features
    """
    if batched is None:
        batched = embedding.dim() == 2 and rr.dim() == 1 and embedding.shape[0] != rr.shape[0]
    if batched:
        N, embed_dim = embedding.shape
        n_pts = rr.shape[0]
        rr_exp = rr[None, :].expand(N, n_pts)  # (N, n_pts)
        emb_exp = embedding[:, None, :].expand(N, n_pts, embed_dim)  # (N, n_pts, embed_dim)

        # delta_pos: (N, n_pts, dimension) — first component is rr, rest zeros
        delta_pos = torch.zeros(N, n_pts, dimension, dtype=rr.dtype, device=rr.device)
        delta_pos[:, :, 0] = rr_exp / max_radius
        r = rr_exp.unsqueeze(-1) / max_radius  # (N, n_pts, 1)

        match model_name:
            case 'arbitrary_ode' | 'arbitrary_field_ode' | 'particle_spring_force_ode' | 'particle_spring_force_prescribed_field' | 'particle_spring_force_prescribed_field_siren' | 'particle_spring_force_prescribed_field_siren_grad' | 'particle_spring_force_diffusion_field' | 'particle_spring_force_diffusion_field_siren' | 'particle_spring_force_diffusion_field_siren_pde' | 'particle_spring_force_diffusion_field_siren_grad' | 'particle_spring_force_diffusion_field_siren_grad_pde' | 'particle_spring_force_diffusion_field_siren_grid_pde' | 'particle_spring_force_diffusion_field_internal' | 'particle_spring_force_diffusion_field_siren_grid_pde_internal':
                return torch.cat((delta_pos, r, emb_exp), dim=-1)
            case 'boids_ode' | 'boids_field_ode':
                r_abs = torch.abs(rr_exp).unsqueeze(-1) / max_radius
                vel_zeros = torch.zeros(N, n_pts, dimension * 2, dtype=rr.dtype, device=rr.device)
                return torch.cat((delta_pos, r_abs, vel_zeros, emb_exp), dim=-1)
            case 'gravity_ode':
                vel_zeros = torch.zeros(N, n_pts, dimension * 2, dtype=rr.dtype, device=rr.device)
                return torch.cat((delta_pos, r, vel_zeros, emb_exp), dim=-1)
            case _:
                raise ValueError(f'Unknown model name in build_edge_features: {model_name}')
    else:
        # Original non-batched path (embedding is (n_pts, embed_dim))
        n_pts = rr.shape[0]
        delta_pos = torch.zeros(n_pts, dimension, dtype=rr.dtype, device=rr.device)
        delta_pos[:, 0] = rr / max_radius
        r = rr[:, None] / max_radius

        match model_name:
            case 'arbitrary_ode' | 'arbitrary_field_ode' | 'particle_spring_force_ode' | 'particle_spring_force_prescribed_field' | 'particle_spring_force_prescribed_field_siren' | 'particle_spring_force_prescribed_field_siren_grad' | 'particle_spring_force_diffusion_field' | 'particle_spring_force_diffusion_field_siren' | 'particle_spring_force_diffusion_field_siren_pde' | 'particle_spring_force_diffusion_field_siren_grad' | 'particle_spring_force_diffusion_field_siren_grad_pde' | 'particle_spring_force_diffusion_field_siren_grid_pde' | 'particle_spring_force_diffusion_field_internal' | 'particle_spring_force_diffusion_field_siren_grid_pde_internal':
                return torch.cat((delta_pos, r, embedding), dim=1)
            case 'boids_ode' | 'boids_field_ode':
                r_abs = torch.abs(rr[:, None]) / max_radius
                vel_zeros = torch.zeros(n_pts, dimension * 2, dtype=rr.dtype, device=rr.device)
                return torch.cat((delta_pos, r_abs, vel_zeros, embedding), dim=1)
            case 'gravity_ode':
                vel_zeros = torch.zeros(n_pts, dimension * 2, dtype=rr.dtype, device=rr.device)
                return torch.cat((delta_pos, r, vel_zeros, embedding), dim=1)
            case _:
                raise ValueError(f'Unknown model name in build_edge_features: {model_name}')


def _batched_mlp_eval(mlp, embeddings, rr, model_name, max_radius, device, dimension=2, chunk_size=512):
    """Evaluate an MLP for all cells in batched mode.

    Args:
        mlp: nn.Module — the edge MLP
        embeddings: (N, embed_dim) tensor of cell embeddings
        rr: (n_pts,) tensor of radial sample points
        model_name: str — model name for feature construction
        max_radius: float
        device: torch device
        dimension: int, spatial dimension (2 or 3)
        chunk_size: number of cells per chunk to avoid OOM

    Returns:
        (N, n_pts) tensor of MLP output (first output dim)
    """
    N = embeddings.shape[0]
    n_pts = rr.shape[0]
    results = []

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            emb_chunk = embeddings[start:end]  # (chunk, embed_dim)

            # Build features: (chunk, n_pts, input_dim)
            features = build_edge_features(rr, emb_chunk, model_name, max_radius, dimension=dimension)
            chunk_n = features.shape[0]

            # Flatten to (chunk * n_pts, input_dim), run MLP, reshape back
            features_flat = features.reshape(chunk_n * n_pts, -1)
            out = mlp(features_flat.float())[:, 0]  # (chunk * n_pts,)
            results.append(out.reshape(chunk_n, n_pts))

    return torch.cat(results, dim=0)  # (N, n_pts)


def _plot_curves_fast(ax, rr, func_matrix, type_list, cmap, ynorm=1.0, subsample=None, alpha=0.25, linewidth=1):
    """Plot N curves using a single LineCollection.

    Args:
        ax: matplotlib Axes
        rr: (n_pts,) numpy array of x values
        func_matrix: (N, n_pts) numpy array of y values
        type_list: (N,) numpy array of int type labels for coloring
        cmap: CustomColorMap instance
        ynorm: scalar or numpy array to multiply y values by
        subsample: int or None — plot every `subsample`-th curve. None plots all.
        alpha: float
        linewidth: float
    """
    N = func_matrix.shape[0]
    if subsample is not None:
        indices = np.arange(0, N, subsample)
    else:
        indices = np.arange(N)

    if len(indices) == 0:
        return

    # Build line segments for LineCollection
    rr_np = np.asarray(rr)
    ynorm_val = float(ynorm) if np.isscalar(ynorm) else np.asarray(ynorm)
    segments = []
    colors = []
    for i in indices:
        y_vals = func_matrix[i] * ynorm_val
        pts = np.column_stack([rr_np, y_vals])
        segments.append(pts)
        colors.append(cmap.color(int(type_list[i])))

    lc = mcoll.LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)
    ax.autoscale_view()


def _vectorized_linear_fit(x, y):
    """Vectorized closed-form least-squares linear fit.

    Args:
        x: (N,) tensor
        y: (N,) tensor

    Returns:
        slope, intercept as scalars
    """
    N = x.shape[0]
    sx = x.sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sx2 = (x * x).sum()
    denom = N * sx2 - sx * sx
    slope = (N * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / N
    return slope, intercept


# --------------------------------------------------------------------------- #
#  Embedding helpers
# --------------------------------------------------------------------------- #

def get_embedding(model_a=None, dataset_number=0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())
    return embedding


def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_cells=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())
    indexes = np.arange(n_frames) * n_cells + cell_id
    return embedding[indexes]


def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_cells=None, n_frames=None, has_cell_division=None):
    indexes = np.arange(n_frames) * n_cells + cell_id
    return new_labels[indexes]


# --------------------------------------------------------------------------- #
#  analyze_edge_function — vectorized
# --------------------------------------------------------------------------- #

def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model=None, n_nodes=0, n_cells=None, ynorm=None, type_list=None, cmap=None, update_type=None, device=None):

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    dimension = config.simulation.dimension
    config_model = config.graph_model.cell_model_name

    if rr == []:
        if config_model == 'gravity_ode':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    print('interaction functions ...')

    # Build all embeddings at once: (N, embed_dim)
    if len(model.a.shape) == 3:
        all_embeddings = model.a[0, :n_cells, :]  # (N, embed_dim)
    else:
        all_embeddings = model.a[:n_cells, :]  # (N, embed_dim)

    if config.training.do_tracking:
        pass  # embeddings used directly
    elif (update_type != 'NA') & model.embedding_trial:
        b_rep = model.b[0].clone().detach().repeat(1, 1).expand(n_cells, -1)
        all_embeddings = torch.cat((all_embeddings, b_rep), dim=1)

    # Batched MLP evaluation: (N, 1000)
    func_list = _batched_mlp_eval(model_MLP, all_embeddings, rr, config_model, max_radius, device,
                                    dimension=dimension)

    func_list_ = to_numpy(func_list)

    if vizualize:
        fig = plt.gcf()
        ax = plt.gca()

        # Determine subsampling
        if n_cells <= 200:
            subsample = 1
        else:
            subsample = max(1, n_cells // 200)

        _plot_curves_fast(
            ax, to_numpy(rr), func_list_,
            type_list.flatten() if type_list is not None else np.zeros(n_cells),
            cmap, ynorm=to_numpy(ynorm),
            subsample=subsample, alpha=0.25, linewidth=1,
        )

        if config.graph_model.cell_model_name == 'gravity_ode':
            plt.xlim([1E-3, 0.02])
        if config.plotting.ylim is not None:
            plt.ylim(config.plotting.ylim)

    print('UMAP reduction ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if func_list_.shape[0] > 1000:
            new_index = np.random.permutation(func_list_.shape[0])
            new_index = new_index[0:min(1000, func_list_.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(func_list_[new_index])
            proj_interaction = trans.transform(func_list_)
        else:
            trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0).fit(func_list_)
            proj_interaction = trans.transform(func_list_)

    return func_list, proj_interaction


# --------------------------------------------------------------------------- #
#  plot_training — vectorized
# --------------------------------------------------------------------------- #

def plot_siren_field_fixed(model, config, log_dir, epoch, N, ynorm, device, k_frame=None):
    """Plot the learned SIREN field vs ground truth at a SINGLE fixed time.

    Companion to `plot_siren_field_slice`. Uses a fixed `k_frame` (default = n_frames/2)
    so that flipping through `tmp_training/field_fixed/*.png` across checkpoints shows
    convergence at a fixed reference point — easier to compare epoch-to-epoch progress
    because the target doesn't move.
    """
    if not hasattr(model, 'siren_field'):
        return

    import os
    style = default_style
    sim = config.simulation
    fp = sim.field_params
    dimension = sim.dimension

    if k_frame is None:
        k_frame = sim.n_frames // 2

    # Use the model's training grid (siren_grid_pde) when available; otherwise
    # fall back to a uniform 24×24 z=0.5 slab.
    pos, grid_n_x, grid_n_y, extent, z_slice = _get_plot_grid(
        model, dimension, device, z_target=0.5, fallback_n=24)
    pos_np = to_numpy(pos)
    grid_n = grid_n_x  # back-compat alias used in the existing reshape calls

    # --- learned ---
    cell_model = config.graph_model.cell_model_name
    is_diffusion = 'diffusion_field' in cell_model
    t_phys = float(k_frame) * sim.delta_t

    if is_diffusion:
        # --- learned: force vector + scalar c (saturating if the model has it) ---
        v_learned_np, c_learned_np = _eval_siren_field(
            model, pos, k_frame, ynorm, device, return_scalar=True)

        # --- true: PDE replay (honors pulsed sources + chem_saturation_scale) ---
        with torch.no_grad():
            v_true, c_true = _replay_diffusion_field(
                config, k_frame, pos, device, return_scalar=True)
        v_true_np = to_numpy(v_true)
        c_true_np = to_numpy(c_true)

        common_scale = max(np.linalg.norm(v_true_np, axis=1).max() * 10.0, 1e-6)

        # Grid-reshape for colormap plots — flatten order is (y outer, x inner)
        c_true_img = c_true_np.reshape(grid_n_y, grid_n_x)
        if c_learned_np is not None:
            c_learned_img = c_learned_np.reshape(grid_n_y, grid_n_x)
        else:
            c_learned_img = None

        # Common vmin/vmax so colormaps are comparable
        c_vmin = float(min(c_true_img.min(),
                           c_learned_img.min() if c_learned_img is not None else c_true_img.min()))
        c_vmax = float(max(c_true_img.max(),
                           c_learned_img.max() if c_learned_img is not None else c_true_img.max()))

        # 3-row layout: forces | scalar c | errors (force + c)
        fig, axes = plt.subplots(3, 2, figsize=(10, 15), facecolor=style.background)

        # --- Row 1: force vectors (learned | true) ---
        ax = axes[0, 0]
        ax.quiver(pos_np[:, 0], pos_np[:, 1], v_learned_np[:, 0], v_learned_np[:, 1],
                  angles='xy', scale_units='xy', scale=common_scale,
                  color=style.foreground, width=0.004)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
        z_str = f', z={z_slice:.3f}' if z_slice is not None else ''
        ax.set_title(f'learned F_chem  (k={k_frame}, t={t_phys:.2f}{z_str})', color=style.foreground)

        ax = axes[0, 1]
        ax.quiver(pos_np[:, 0], pos_np[:, 1], v_true_np[:, 0], v_true_np[:, 1],
                  angles='xy', scale_units='xy', scale=common_scale,
                  color=style.foreground, width=0.004)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
        ax.set_title('true F_chem  (PDE replay)', color=style.foreground)

        # --- Row 2: scalar c colormaps (learned | true) ---
        ax = axes[1, 0]
        if c_learned_img is not None:
            im = ax.imshow(c_learned_img, origin='lower', extent=list(extent),
                           cmap='viridis', vmin=c_vmin, vmax=c_vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'scalar c N/A\n(vector SIREN)', ha='center', va='center')
        ax.set_aspect('equal')
        ax.set_title('learned c(x, t)', color=style.foreground)

        ax = axes[1, 1]
        im = ax.imshow(c_true_img, origin='lower', extent=list(extent),
                       cmap='viridis', vmin=c_vmin, vmax=c_vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')
        ax.set_title('true c(x, t)  (PDE replay)', color=style.foreground)

        # --- Row 3: error colormaps (|F| error | c error) ---
        ax = axes[2, 0]
        err_F = np.linalg.norm(v_learned_np - v_true_np, axis=1).reshape(grid_n_y, grid_n_x)
        im = ax.imshow(err_F, origin='lower', extent=list(extent), cmap='magma')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')
        ax.set_title(f'|F_learned - F_true|  (mean={err_F.mean():.4f})', color=style.foreground)

        ax = axes[2, 1]
        if c_learned_img is not None:
            err_c = np.abs(c_learned_img - c_true_img)
            im = ax.imshow(err_c, origin='lower', extent=list(extent), cmap='magma')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'|c_learned - c_true|  (mean={err_c.mean():.4f})', color=style.foreground)
        else:
            ax.text(0.5, 0.5, 'c error N/A', ha='center', va='center')
            ax.set_title('c error', color=style.foreground)
        ax.set_aspect('equal')
    else:
        v_learned_np = _eval_siren_field(model, pos, k_frame, ynorm, device)
        # Analytic moving-Gaussian ground truth
        # Supports both single-source and multi-source field.
        import math
        center_0 = torch.tensor(fp.center_0, dtype=pos.dtype, device=device)
        velocity = torch.tensor(fp.velocity, dtype=pos.dtype, device=device)
        is_multi = getattr(fp, 'field_type', 'single') == 'multi'

        if is_multi:
            centers_t = center_0 + velocity * t_phys                       # (S, dim)
            amplitude = float(fp.amplitude)
            sigma_f = float(fp.sigma)
            mu_chem = float(fp.mu_chem)

            diff = pos.unsqueeze(1) - centers_t.unsqueeze(0)               # (N, S, dim)
            if sim.boundary == 'periodic':
                diff = diff - torch.round(diff)
            dist_sq = (diff ** 2).sum(dim=-1)                              # (N, S)
            C_s = amplitude * torch.exp(-dist_sq / (2.0 * sigma_f ** 2))   # (N, S)
            grad_C = (-C_s.unsqueeze(-1) * diff / (sigma_f ** 2)).sum(dim=1)
            v_true = mu_chem * grad_C
            center_t = centers_t[0]
        else:
            sigma_f = float(fp.sigma)
            amplitude = float(fp.amplitude)
            mu_chem = float(fp.mu_chem)
            center_t = center_0 + velocity * t_phys
            if center_t.dim() > 1:
                center_t = center_t.squeeze(0)
            diff = pos - center_t.unsqueeze(0)
            if sim.boundary == 'periodic':
                diff = diff - torch.round(diff)
            r2 = (diff ** 2).sum(dim=1, keepdim=True)
            C = amplitude * torch.exp(-r2 / (2.0 * sigma_f ** 2))
            grad_C = -C * diff / (sigma_f ** 2)
            v_true = mu_chem * grad_C
        v_true_np = to_numpy(v_true)

        common_scale = max(np.linalg.norm(v_true_np, axis=1).max() * 10.0, 1e-6)
        cx, cy = float(center_t[0].item()), float(center_t[1].item())
        if sim.boundary == 'periodic':
            cx_w = cx - np.floor(cx); cy_w = cy - np.floor(cy)
        else:
            cx_w, cy_w = cx, cy

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=style.background)

        # learned
        ax = axes[0]
        ax.quiver(pos_np[:, 0], pos_np[:, 1], v_learned_np[:, 0], v_learned_np[:, 1],
                  angles='xy', scale_units='xy', scale=common_scale,
                  color=style.foreground, width=0.004)
        ax.plot(cx_w, cy_w, '*', markersize=15, color='red')
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
        ax.set_title(f'learned siren_field  (k={k_frame}, t={t_phys:.2f})', color=style.foreground)

        # true
        ax = axes[1]
        ax.quiver(pos_np[:, 0], pos_np[:, 1], v_true_np[:, 0], v_true_np[:, 1],
                  angles='xy', scale_units='xy', scale=common_scale,
                  color=style.foreground, width=0.004)
        ax.plot(cx_w, cy_w, '*', markersize=15, color='red')
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
        ax.set_title(f'true mu_chem*grad C  (mu={mu_chem})', color=style.foreground)

        # error magnitude
        ax = axes[2]
        err = np.linalg.norm(v_learned_np - v_true_np, axis=1).reshape(grid_n_y, grid_n_x)
        im = ax.imshow(err, origin='lower', extent=list(extent), cmap='magma')
        ax.set_aspect('equal')
        ax.set_title(f'|learned - true|  (mean={err.mean():.4f})', color=style.foreground)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    os.makedirs(f"./{log_dir}/tmp_training/field_fixed", exist_ok=True)
    plt.tight_layout()
    style.savefig(fig, f"./{log_dir}/tmp_training/field_fixed/{epoch}_{N}.png")
    plt.close(fig)


def plot_siren_field_time_evolve(model, config, log_dir, epoch, ynorm, device, frame_step=10):
    """Save a sequence of (learned vs true) field snapshots over time.

    For diffusion-field models, runs ONE pass of the PDE replay, collects c_true and
    c_learned at the model's z-slice grid every `frame_step` frames, then writes one
    PNG per frame to ``tmp_training/field_time_evolve/epoch_{epoch}/frame_{k:05d}.png``
    with shared color scale across all frames. The PNGs can be turned into a movie:

        ffmpeg -framerate 30 -pattern_type glob -i 'frame_*.png' \
               -c:v libx264 -pix_fmt yuv420p evolve.mp4

    Skips silently for non-diffusion models (no analytic GT replay implemented here).
    """
    if not hasattr(model, 'siren_field'):
        return
    cell_model = config.graph_model.cell_model_name
    if 'diffusion_field' not in cell_model:
        return  # only diffusion case has the PDE replay

    import os
    from cell_gnn.zarr_io import load_simulation_data
    from cell_gnn.generators.particle_spring_force_diffusion_field import (
        init_gaussian_field, build_spectral_decay,
        deposit_to_grid, deposit_to_grid_weighted, interp_periodic,
    )

    style = default_style
    sim = config.simulation
    fp = sim.field_params
    dimension = sim.dimension

    # Plotting grid (matches the model's training grid when available)
    pos, grid_n_x, grid_n_y, extent, z_slice = _get_plot_grid(
        model, dimension, device, z_target=0.5, fallback_n=24)
    pos_np = to_numpy(pos)

    out_dir = os.path.join(log_dir, 'tmp_training', 'field_time_evolve', f'epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)

    # ── PDE state setup (mirrors _replay_diffusion_field but stepped once over n_frames) ──
    res = fp.grid_resolution
    dt = sim.delta_t
    D = fp.diffusion_coeff
    lam = fp.lambda_decay
    alpha = fp.source_strength
    source_fraction = getattr(fp, 'source_fraction', 1.0)
    pulse_period = getattr(fp, 'pulse_period', 0)
    pulse_duty = getattr(fp, 'pulse_duty', 1.0)
    seed = getattr(sim, 'seed', 0)
    center_0 = torch.tensor(fp.center_0, dtype=torch.float32, device=device)

    # ..._internal models replace the constant/pulsed source with a sigmoid-gated,
    # (1-s)-scaled per-cell emission. We pull s_i from the saved time series.
    is_internal = '_internal' in cell_model
    coeff_emit_int  = float(getattr(fp, 'internal_coeff_emit', 20.0))
    c_threshold_int = float(getattr(fp, 'internal_c_threshold', 0.01))
    c_sharpness_int = float(getattr(fp, 'internal_c_sharpness', 200.0))

    field_grid = init_gaussian_field(
        res, dimension, center_0, fp.amplitude, fp.sigma,
        sim.boundary == 'periodic', device)
    decay = build_spectral_decay(res, dimension, D, lam, dt, device)
    s = [res] * dimension

    dataset_name = config.dataset
    x_ts = load_simulation_data(f'graphs_data/{dataset_name}/x_list_0', dimension)

    source_mask = None
    source_phase = None

    # Pass 1: simulate forward, collect c_true and c_learned at grid every frame_step
    snapshot_ks = []
    c_true_list = []
    c_learned_list = []

    for k in range(sim.n_frames):
        pos_k = x_ts.frame(k).pos.to(device)

        # emit
        if is_internal:
            # Sigmoid-gated, (1-s)-scaled emission. s entering step k is the
            # saved s at frame max(0, k-1).
            if coeff_emit_int > 0:
                k_in = max(0, k - 1)
                fld_in = x_ts.frame(k_in).field
                if fld_in is None or fld_in.shape[-1] < 2:
                    s_in = torch.full((pos_k.shape[0],), 0.5, device=device, dtype=field_grid.dtype)
                else:
                    s_in = fld_in[:, 1].to(device).to(field_grid.dtype)
                c_at_pos = interp_periodic(field_grid, pos_k, res)
                gate = torch.sigmoid(c_sharpness_int * (c_at_pos - c_threshold_int))
                emit = coeff_emit_int * (1.0 - s_in) * gate
                src = deposit_to_grid_weighted(pos_k, emit * dt, res, dimension, device)
                field_grid = field_grid + src
        elif alpha > 0:
            if source_mask is None:
                n = pos_k.shape[0]
                gen = torch.Generator(device=pos_k.device)
                gen.manual_seed(int(seed))
                n_src = max(1, int(round(source_fraction * n)))
                perm = torch.randperm(n, generator=gen, device=pos_k.device)
                source_mask = torch.zeros(n, dtype=torch.bool, device=pos_k.device)
                source_mask[perm[:n_src]] = True
                source_phase = torch.rand(n, generator=gen, device=pos_k.device)

            if pulse_period and pulse_period > 0:
                phase = (k / float(pulse_period) + source_phase) % 1.0
                d_ = torch.minimum(phase, 1.0 - phase)
                sigma_p = max(pulse_duty, 1e-3) / 2.0
                pulse = torch.exp(-0.5 * (d_ / sigma_p) ** 2) * source_mask.float()
                gain = 1.0 / max(pulse_duty, 1e-3)
                active = pulse > 1e-4
                if active.any():
                    strength = alpha * dt * gain * pulse[active]
                    src = deposit_to_grid_weighted(pos_k[active], strength, res, dimension, device)
                    field_grid = field_grid + src
            else:
                emit_pos = pos_k[source_mask] if source_fraction < 1.0 else pos_k
                src = deposit_to_grid(emit_pos, alpha * dt, res, dimension, device)
                field_grid = field_grid + src

        # diffusion + decay step (spectral)
        C_hat = torch.fft.rfftn(field_grid, s=s)
        C_hat = C_hat * decay
        field_grid = torch.fft.irfftn(C_hat, s=s)

        if k % frame_step == 0:
            # true c at plotting grid
            c_t = interp_periodic(field_grid, pos.to(device), res)  # (M,)
            c_true_list.append(to_numpy(c_t))
            # learned c at plotting grid
            _, c_l = _eval_siren_field(model, pos, k, ynorm, device, return_scalar=True)
            c_learned_list.append(c_l.squeeze(-1) if c_l is not None and c_l.ndim > 1 else c_l)
            snapshot_ks.append(k)

    # Shared color scale across all frames
    c_concat = np.concatenate(
        [c.ravel() for c in c_true_list]
        + [c.ravel() for c in c_learned_list if c is not None]
    )
    c_vmin = float(c_concat.min())
    c_vmax = float(c_concat.max())

    # Pass 2: render PNGs with shared scale
    for k, c_t, c_l in zip(snapshot_ks, c_true_list, c_learned_list):
        c_true_img = c_t.reshape(grid_n_y, grid_n_x)
        c_learned_img = c_l.reshape(grid_n_y, grid_n_x) if c_l is not None else None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=style.background)

        ax = axes[0]
        if c_learned_img is not None:
            im = ax.imshow(c_learned_img, origin='lower', extent=list(extent),
                           cmap='viridis', vmin=c_vmin, vmax=c_vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')
        ax.set_title(f'learned c  (k={k}, t={k*dt:.2f})', color=style.foreground)

        ax = axes[1]
        im = ax.imshow(c_true_img, origin='lower', extent=list(extent),
                       cmap='viridis', vmin=c_vmin, vmax=c_vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')
        ax.set_title('true c (PDE replay)', color=style.foreground)

        ax = axes[2]
        if c_learned_img is not None:
            err = np.abs(c_learned_img - c_true_img)
            im = ax.imshow(err, origin='lower', extent=list(extent), cmap='magma')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'|err|  mean={err.mean():.4f}', color=style.foreground)
        ax.set_aspect('equal')

        plt.tight_layout()
        style.savefig(fig, os.path.join(out_dir, f'frame_{k:05d}.png'))
        plt.close(fig)

    print(f'[field_time_evolve] saved {len(snapshot_ks)} frames to {out_dir}')


def plot_siren_field_slice(model, config, log_dir, epoch, N, ynorm, device, n_time_snapshots=4):
    """Plot the learned SIREN field vs ground-truth chemotactic field on a z-slice.

    For SIREN models that include a `siren_field` branch, sample the predicted velocity
    field on a 2D grid (z=0.5) at multiple time snapshots, alongside the analytic ground
    truth `mu_chem * grad C(x, t)`. Layout: 2 rows × n_time_snapshots cols.
    Top row = learned, bottom row = true. Each column is a different time.
    """
    if not hasattr(model, 'siren_field'):
        return

    import os
    style = default_style
    sim = config.simulation
    fp = sim.field_params
    dimension = sim.dimension

    # Use the model's training grid (siren_grid_pde) when available; otherwise
    # fall back to a uniform 24×24 z=0.5 slab.
    pos, grid_n_x, grid_n_y, extent, z_slice = _get_plot_grid(
        model, dimension, device, z_target=0.5, fallback_n=24)
    pos_np = to_numpy(pos)
    grid_n = grid_n_x  # back-compat alias

    # Time snapshots evenly spread across the trajectory
    k_frames = np.linspace(0, sim.n_frames - 1, n_time_snapshots).astype(int)

    cell_model = config.graph_model.cell_model_name
    is_diffusion = 'diffusion_field' in cell_model

    if is_diffusion:
        # 5 rows: learned F | true F | learned c | true c | |F error|
        # Columns = time snapshots.
        fig, axes = plt.subplots(5, n_time_snapshots,
                                  figsize=(4 * n_time_snapshots, 18),
                                  facecolor=style.background)
        if n_time_snapshots == 1:
            axes = axes.reshape(5, 1)

        # First pass: compute everything so we can use shared color limits
        v_learned_all = []
        v_true_all = []
        c_learned_all = []
        c_true_all = []
        t_phys_all = []
        for k_frame in k_frames:
            v_l, c_l = _eval_siren_field(
                model, pos, k_frame, ynorm, device, return_scalar=True)
            with torch.no_grad():
                v_t, c_t = _replay_diffusion_field(
                    config, k_frame, pos, device, return_scalar=True)
            v_learned_all.append(v_l)
            v_true_all.append(to_numpy(v_t))
            c_learned_all.append(c_l)
            c_true_all.append(to_numpy(c_t))
            t_phys_all.append(float(k_frame) * sim.delta_t)

        # Shared scales
        all_mags = np.concatenate([np.linalg.norm(v, axis=1) for v in v_true_all])
        common_scale = max(all_mags.max() * 10.0, 1e-6)

        c_concat = np.concatenate(
            [c.ravel() for c in c_true_all]
            + [c.ravel() for c in c_learned_all if c is not None]
        )
        c_vmin = float(c_concat.min())
        c_vmax = float(c_concat.max())

        for col, k_frame in enumerate(k_frames):
            v_learned_np = v_learned_all[col]
            v_true_np = v_true_all[col]
            c_learned_np = c_learned_all[col]
            c_true_np = c_true_all[col]
            t_phys = t_phys_all[col]

            # Row 0: learned F quiver
            ax = axes[0, col]
            ax.quiver(pos_np[:, 0], pos_np[:, 1], v_learned_np[:, 0], v_learned_np[:, 1],
                      angles='xy', scale_units='xy', scale=common_scale,
                      color=style.foreground, width=0.004)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
            ax.set_title(f'learned F  k={k_frame}  t={t_phys:.2f}', color=style.foreground, fontsize=10)

            # Row 1: true F quiver
            ax = axes[1, col]
            ax.quiver(pos_np[:, 0], pos_np[:, 1], v_true_np[:, 0], v_true_np[:, 1],
                      angles='xy', scale_units='xy', scale=common_scale,
                      color=style.foreground, width=0.004)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
            ax.set_title(f'true F  k={k_frame}', color=style.foreground, fontsize=10)

            # Row 2: learned c
            ax = axes[2, col]
            if c_learned_np is not None:
                img = c_learned_np.reshape(grid_n_y, grid_n_x)
                im = ax.imshow(img, origin='lower', extent=list(extent),
                               cmap='viridis', vmin=c_vmin, vmax=c_vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'c N/A', ha='center', va='center')
            ax.set_aspect('equal')
            ax.set_title('learned c', color=style.foreground, fontsize=10)

            # Row 3: true c
            ax = axes[3, col]
            img_t = c_true_np.reshape(grid_n_y, grid_n_x)
            im = ax.imshow(img_t, origin='lower', extent=list(extent),
                           cmap='viridis', vmin=c_vmin, vmax=c_vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect('equal')
            ax.set_title('true c', color=style.foreground, fontsize=10)

            # Row 4: |F error|
            ax = axes[4, col]
            err = np.linalg.norm(v_learned_np - v_true_np, axis=1).reshape(grid_n_y, grid_n_x)
            im = ax.imshow(err, origin='lower', extent=list(extent), cmap='magma')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect('equal')
            ax.set_title(f'|F err|  mean={err.mean():.4f}', color=style.foreground, fontsize=10)

        axes[0, 0].set_ylabel('learned F_chem', color=style.foreground)
        axes[1, 0].set_ylabel('true F_chem (PDE)', color=style.foreground)
        axes[2, 0].set_ylabel('learned c', color=style.foreground)
        axes[3, 0].set_ylabel('true c', color=style.foreground)
        axes[4, 0].set_ylabel('|F error|', color=style.foreground)
    else:
        center_0 = torch.tensor(fp.center_0, dtype=pos.dtype, device=device)
        velocity = torch.tensor(fp.velocity, dtype=pos.dtype, device=device)
        is_multi = getattr(fp, 'field_type', 'single') == 'multi'
        sigma_f = float(fp.sigma)
        amplitude = float(fp.amplitude)
        mu_chem = float(fp.mu_chem)

        fig, axes = plt.subplots(2, n_time_snapshots,
                                  figsize=(4 * n_time_snapshots, 8),
                                  facecolor=style.background)

        common_scale = None

        for col, k_frame in enumerate(k_frames):
            # --- learned ---
            v_learned_np = _eval_siren_field(model, pos, k_frame, ynorm, device)

            # --- true ---
            t_phys = float(k_frame) * sim.delta_t
            if is_multi:
                centers_t = center_0 + velocity * t_phys
                diff_multi = pos.unsqueeze(1) - centers_t.unsqueeze(0)
                if sim.boundary == 'periodic':
                    diff_multi = diff_multi - torch.round(diff_multi)
                dist_sq = (diff_multi ** 2).sum(dim=-1)
                C_s = amplitude * torch.exp(-dist_sq / (2.0 * sigma_f ** 2))
                grad_C = (-C_s.unsqueeze(-1) * diff_multi / (sigma_f ** 2)).sum(dim=1)
                v_true = mu_chem * grad_C
                center_t = centers_t[0]
            else:
                center_t = center_0 + velocity * t_phys
                if center_t.dim() > 1:
                    center_t = center_t.squeeze(0)
                diff = pos - center_t.unsqueeze(0)
                if sim.boundary == 'periodic':
                    diff = diff - torch.round(diff)
                r2 = (diff ** 2).sum(dim=1, keepdim=True)
                C = amplitude * torch.exp(-r2 / (2.0 * sigma_f ** 2))
                grad_C = -C * diff / (sigma_f ** 2)
                v_true = mu_chem * grad_C
            v_true_np = to_numpy(v_true)

            if common_scale is None:
                mag = np.linalg.norm(v_true_np, axis=1).max()
                common_scale = max(mag * 10.0, 1e-6)

            # learned (top row)
            ax = axes[0, col]
            ax.quiver(pos_np[:, 0], pos_np[:, 1], v_learned_np[:, 0], v_learned_np[:, 1],
                      angles='xy', scale_units='xy', scale=common_scale,
                      color=style.foreground, width=0.004)
            cx, cy = float(center_t[0].item()), float(center_t[1].item())
            if sim.boundary == 'periodic':
                cx_w = cx - np.floor(cx); cy_w = cy - np.floor(cy)
            else:
                cx_w, cy_w = cx, cy
            ax.plot(cx_w, cy_w, '*', markersize=12, color='red')
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
            ax.set_title(f'learned  k={k_frame}  t={t_phys:.2f}', color=style.foreground, fontsize=10)

            # true (bottom row)
            ax = axes[1, col]
            ax.quiver(pos_np[:, 0], pos_np[:, 1], v_true_np[:, 0], v_true_np[:, 1],
                      angles='xy', scale_units='xy', scale=common_scale,
                      color=style.foreground, width=0.004)
            ax.plot(cx_w, cy_w, '*', markersize=12, color='red')
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal')
            err = np.linalg.norm(v_learned_np - v_true_np, axis=1).mean()
            ax.set_title(f'true  k={k_frame}  err={err:.4f}', color=style.foreground, fontsize=10)

        axes[0, 0].set_ylabel('learned siren_field', color=style.foreground)
        axes[1, 0].set_ylabel(f'true mu_chem*grad C  (mu={mu_chem})', color=style.foreground)

    os.makedirs(f"./{log_dir}/tmp_training/field", exist_ok=True)
    plt.tight_layout()
    style.savefig(fig, f"./{log_dir}/tmp_training/field/{epoch}_{N}.png")
    plt.close(fig)


def plot_training(config, pred, gt, log_dir, epoch, N, x, index_cells, n_cells, n_cell_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):
    """Plot training diagnostics. Returns lin_edge R² mean or None if unavailable."""

    lin_edge_r2 = None
    style = default_style
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting
    sign_flip = -1.0 if getattr(plot_config, 'invert_mlp1_sign', False) else 1.0
    do_tracking = train_config.do_tracking
    max_radius = simulation_config.max_radius
    n_runs = train_config.n_runs
    dimension = simulation_config.dimension

    # --- Embedding scatter plot ---
    # All montage panels use style.montage_figure() for consistent sizing/fonts.
    if n_runs == 3:
        fig, axes = style.montage_figure(ncols=3, width=24)
        ax = axes[0]
        plt.sca(ax)
        embedding = get_embedding(model.a, 1)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        embedding = get_embedding(model.a, 2)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[2]
        plt.sca(ax)
        embedding = get_embedding(model.a, 1)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0)
        embedding = get_embedding(model.a, 2)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1]
        plt.sca(ax)
        embedding = get_embedding(model.a, 1)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        embedding = get_embedding(model.a, 2)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0)
    elif n_runs > 10:
        fig, ax = style.montage_figure()
        for m in range(1, n_runs):
            embedding = get_embedding(model.a, m)
            ax.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=1)
    else:
        fig, ax = style.montage_figure()
        if do_tracking:
            embedding = to_numpy(model.a)
            for n in range(n_cell_types):
                ax.scatter(embedding[index_cells[n], 0], embedding[index_cells[n], 1], color=cmap.color(n), s=1)
        elif simulation_config.state_type == 'sequence':
            embedding = to_numpy(model.a[0].squeeze())
            ax.scatter(embedding[:-200, 0], embedding[:-200, 1], color=style.foreground, s=0.1)
        else:
            embedding = get_embedding(model.a, plot_config.data_embedding)
            for n in range(n_cell_types):
                ax.scatter(embedding[index_cells[n], 0], embedding[index_cells[n], 1], color=cmap.color(n), s=1)

    if n_runs == 3:
        ax = axes[0]
    style.montage_xlabel(ax, r'$a_0$')
    style.montage_ylabel(ax, r'$a_1$')
    plt.tight_layout()
    style.savefig(fig, f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.png")

    # --- Pred vs true scatter ---
    fig, ax = style.figure()
    ax.scatter(to_numpy(gt[:, 0]), to_numpy(pred[:, 0]), c='r', s=1)
    ax.scatter(to_numpy(gt[:, 1]), to_numpy(pred[:, 1]), c='g', s=1)
    style.xlabel(ax, 'true value')
    style.ylabel(ax, 'pred value')
    plt.tight_layout()
    style.savefig(fig, f"./{log_dir}/tmp_training/prediction/{epoch}_{N}.png")

    # --- Interaction function curves (vectorized) ---
    if n_runs > 10:
        fig, ax = style.figure()
        rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 1000)).to(device)

        # Build all (n, k) pair features: n_runs-1 * n_cells^2 combinations
        all_funcs = []
        for m in range(1, n_runs):
            # Batched: for each m, build (n_cells * n_cells, n_pts) via pairs
            emb_n = model.a[m, :n_cells, :]  # (N, embed_dim)
            emb_k = model.a[m, :n_cells, :]  # (N, embed_dim)
            # Expand to all pairs (N*N, embed_dim)
            emb_n_rep = emb_n.unsqueeze(1).expand(-1, n_cells, -1).reshape(-1, emb_n.shape[-1])
            emb_k_rep = emb_k.unsqueeze(0).expand(n_cells, -1, -1).reshape(-1, emb_k.shape[-1])

            n_pts = rr.shape[0]
            n_pairs = emb_n_rep.shape[0]

            # Build features for all pairs
            rr_exp = rr[None, :].expand(n_pairs, n_pts)
            z = torch.zeros_like(rr_exp)
            emb_n_exp = emb_n_rep[:, None, :].expand(-1, n_pts, -1)
            emb_k_exp = emb_k_rep[:, None, :].expand(-1, n_pts, -1)

            features = torch.cat((
                rr_exp.unsqueeze(-1),
                z.unsqueeze(-1),
                z.unsqueeze(-1),
                emb_n_exp,
                emb_k_exp,
            ), dim=-1)

            # Chunk MLP evaluation
            chunk_size = 512
            funcs = []
            with torch.no_grad():
                for start in range(0, n_pairs, chunk_size):
                    end = min(start + chunk_size, n_pairs)
                    feat_flat = features[start:end].reshape(-1, features.shape[-1])
                    out = model.lin_edge(feat_flat.float())[:, 0]
                    funcs.append(out.reshape(end - start, n_pts))
            funcs = torch.cat(funcs, dim=0)  # (n_pairs, n_pts)
            all_funcs.append(funcs)

        all_funcs = torch.cat(all_funcs, dim=0)
        rr_np = to_numpy(rr)
        ynorm_np = to_numpy(ynorm)

        # Plot with LineCollection
        segments = []
        for i in range(all_funcs.shape[0]):
            y_vals = to_numpy(all_funcs[i]) * ynorm_np * sign_flip
            pts = np.column_stack([rr_np, y_vals])
            segments.append(pts)
        colors = ['b'] * len(segments)
        lc = mcoll.LineCollection(segments, colors=colors, linewidths=2, alpha=0.1)
        ax.add_collection(lc)
        ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')
        ax.autoscale_view()
        if sign_flip < 0:
            conv_text = '+: repulsive, −: attractive'
        else:
            conv_text = '+: attractive, −: repulsive'
        ax.text(0.02, 0.98, conv_text,
                transform=ax.transAxes, verticalalignment='top',
                fontsize=style.font_size * 0.8,
                color=style.foreground, alpha=0.7)

        plt.tight_layout()
        style.savefig(fig, f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.png")
    else:
        match model_config.cell_model_name:

            case 'arbitrary_ode' | 'arbitrary_field_ode' | 'gravity_ode' | 'particle_spring_force_ode' | 'particle_spring_force_prescribed_field' | 'particle_spring_force_prescribed_field_siren' | 'particle_spring_force_prescribed_field_siren_grad' | 'particle_spring_force_diffusion_field' | 'particle_spring_force_diffusion_field_siren' | 'particle_spring_force_diffusion_field_siren_pde' | 'particle_spring_force_diffusion_field_siren_grad' | 'particle_spring_force_diffusion_field_siren_grad_pde' | 'particle_spring_force_diffusion_field_siren_grid_pde' | 'particle_spring_force_diffusion_field_internal' | 'particle_spring_force_diffusion_field_siren_grid_pde_internal':
                fig, ax = style.montage_figure()
                if axis:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

                rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 1000)).to(device)

                # Vectorized: build all embeddings and eval MLP in batch
                if do_tracking:
                    all_embeddings = model.a[:n_cells, :]
                else:
                    all_embeddings = model.a[0, :n_cells, :]

                func_list = _batched_mlp_eval(model.lin_edge, all_embeddings, rr,
                                              config.graph_model.cell_model_name,
                                              simulation_config.max_radius, device,
                                              dimension=simulation_config.dimension)

                rr_np = to_numpy(rr)
                ynorm_np = to_numpy(ynorm)
                if isinstance(x, CellState):
                    type_arr = to_numpy(x.cell_type[:n_cells]).astype(int)
                else:
                    type_arr = to_numpy(CellState.from_packed(x, dimension).cell_type[:n_cells]).astype(int)

                # Plot predicted curves first (behind)
                subsample = 5 if n_runs <= 5 else 1
                _plot_curves_fast(ax, rr_np, to_numpy(func_list), type_arr, cmap,
                                  ynorm=ynorm_np * sign_flip, subsample=subsample, alpha=0.25, linewidth=2)

                # Plot true psi curves on top (thick)
                true_curves = _get_true_psi(rr, config, n_cell_types, device)
                _plot_true_lin_edge(ax, rr_np, true_curves, cmap, sign_flip=sign_flip)

                # Per-curve R² metric
                r2_values = _compute_curve_r2(to_numpy(func_list), type_arr, true_curves, ynorm=ynorm_np)
                if r2_values is not None:
                    valid = r2_values[~np.isnan(r2_values)]
                    if len(valid) > 0:
                        lin_edge_r2 = float(valid.mean())
                        ax.text(0.02, 0.98, f'R²={lin_edge_r2:.3f}±{valid.std():.3f}',
                                transform=ax.transAxes, verticalalignment='top',
                                fontsize=style.font_size,
                                color=style.foreground)

                if plot_config.xlim is not None:
                    ax.set_xlim(plot_config.xlim)
                else:
                    ax.set_xlim([0, simulation_config.max_radius])
                if plot_config.ylim is not None:
                    ax.set_ylim(plot_config.ylim)
                ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')
                if sign_flip < 0:
                    conv_text = '+: repulsive, −: attractive'
                else:
                    conv_text = '+: attractive, −: repulsive'
                ax.text(0.02, 0.90, conv_text,
                        transform=ax.transAxes, verticalalignment='top',
                        fontsize=style.font_size * 0.8,
                        color=style.foreground, alpha=0.7)
                style.montage_xlabel(ax, r'$r$')
                style.montage_ylabel(ax, r'learned $\mathrm{MLP}_1$')
                plt.tight_layout()
                style.savefig(fig, f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.png")

            case 'boids_ode' | 'boids_field_ode':
                max_radius_plot = 0.04
                fig, ax = style.montage_figure()
                rr = torch.tensor(np.linspace(-max_radius_plot, max_radius_plot, 1000)).to(device)

                # Vectorized MLP evaluation
                if do_tracking:
                    all_embeddings = model.a[:n_cells, :]
                else:
                    all_embeddings = model.a[0, :n_cells, :]

                func_list = _batched_mlp_eval(model.lin_edge, all_embeddings, rr,
                                              config.graph_model.cell_model_name,
                                              max_radius_plot, device,
                                              dimension=simulation_config.dimension)

                rr_np = to_numpy(rr)
                ynorm_np = to_numpy(ynorm)
                type_arr = np.array([int(n // (n_cells / n_cell_types)) for n in range(n_cells)])

                # Plot predicted curves first (behind)
                _plot_curves_fast(ax, rr_np, to_numpy(func_list), type_arr, cmap,
                                  ynorm=ynorm_np * sign_flip, subsample=5, alpha=1.0, linewidth=2)

                # Plot true psi curves on top (thick)
                true_curves = _get_true_psi(rr, config, n_cell_types, device)
                _plot_true_lin_edge(ax, rr_np, true_curves, cmap, sign_flip=sign_flip)

                # Per-curve R² metric
                r2_values = _compute_curve_r2(to_numpy(func_list), type_arr, true_curves, ynorm=ynorm_np)
                if r2_values is not None:
                    valid = r2_values[~np.isnan(r2_values)]
                    if len(valid) > 0:
                        lin_edge_r2 = float(valid.mean())
                        ax.text(0.02, 0.98, f'R²={lin_edge_r2:.3f}±{valid.std():.3f}',
                                transform=ax.transAxes, verticalalignment='top',
                                fontsize=style.font_size,
                                color=style.foreground)

                if plot_config.xlim is not None:
                    ax.set_xlim(plot_config.xlim)
                else:
                    ax.set_xlim([0, simulation_config.max_radius])
                if plot_config.ylim is not None:
                    ax.set_ylim(plot_config.ylim)
                ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                from matplotlib.ticker import FormatStrFormatter
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
                if sign_flip < 0:
                    conv_text = '+: repulsive, −: attractive'
                else:
                    conv_text = '+: attractive, −: repulsive'
                ax.text(0.02, 0.90, conv_text,
                        transform=ax.transAxes, verticalalignment='top',
                        fontsize=style.font_size * 0.8,
                        color=style.foreground, alpha=0.7)
                style.montage_xlabel(ax, r'$r$')
                style.montage_ylabel(ax, r'learned $\mathrm{MLP}_1$')
                plt.tight_layout()
                style.savefig(fig, f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.png")

    # --- SIREN field branch diagnostics (only for models that have one) ---
    if hasattr(model, 'siren_field'):
        # Fixed-time plot: same k every checkpoint → easy convergence comparison
        plot_siren_field_fixed(model, config, log_dir, epoch, N, ynorm, device)
        # Multi-time plot: 4 time snapshots → temporal sanity check
        plot_siren_field_slice(model, config, log_dir, epoch, N, ynorm, device, n_time_snapshots=4)

    # --- Internal-state ODE MLP diagnostics (only for ..._internal models) ---
    if hasattr(model, 'lin_internal'):
        plot_internal_state(model, config, log_dir, epoch, N, device)

    return lin_edge_r2


def plot_internal_state(model, config, log_dir, epoch, N, device):
    """Diagnostics for the internal-state MLP (s, c_local) -> ds/dt.

    Saves a 4-panel figure to tmp_training/internal/{epoch}_{N}.png:
      [predicted ds/dt heatmap]  [true ds/dt heatmap]  [signed error]  [scatter]

    The true ds/dt comes from the analytic ground truth used by the generator:
        ds/dt = coeff_s * (c - s)
    pulled from ``config.simulation.field_params``.  The scatter shows the
    last batch's predicted vs. target values stashed by ``forward()``.
    """
    import os
    style = default_style
    fp = config.simulation.field_params
    coeff_s = float(getattr(fp, 'internal_coeff_s', 2.0))

    # Choose c-axis from observed last batch if available, else fall back to a sensible default.
    last_pred   = getattr(model, 'last_internal_pred', None)
    last_target = getattr(model, 'last_internal_target', None)

    c_max = 0.5
    if hasattr(model, 'last_scalar_field') and model.last_scalar_field is not None:
        c_max = max(float(model.last_scalar_field.abs().max()) * 1.1, 1e-3)
    s_lo, s_hi = 0.0, 1.0

    n_grid = 64
    s_axis = torch.linspace(s_lo, s_hi, n_grid, device=device)
    c_axis = torch.linspace(0.0, c_max, n_grid, device=device)
    S, C = torch.meshgrid(s_axis, c_axis, indexing='ij')
    pts_sc = torch.stack([S.reshape(-1), C.reshape(-1)], dim=-1)        # (G, 2)

    # lin_internal also takes the per-cell embedding now. Use the mean
    # embedding across cells (run 0) as a representative — for a single-type
    # config this is the sole cluster centroid.
    with torch.no_grad():
        mean_embed = model.a[0, :, :].mean(dim=0, keepdim=True)         # (1, E)
        embed_grid = mean_embed.expand(pts_sc.shape[0], -1)             # (G, E)
        pts = torch.cat([pts_sc, embed_grid], dim=-1)                   # (G, 2+E)
        pred_grid = model.lin_internal(pts).reshape(n_grid, n_grid)
    true_grid = coeff_s * (C - S)
    err_grid = (pred_grid - true_grid)

    pred_np = to_numpy(pred_grid)
    true_np = to_numpy(true_grid)
    err_np  = to_numpy(err_grid)

    vlim = max(abs(true_np).max(), abs(pred_np).max(), 1e-8)
    elim = max(abs(err_np).max(), 1e-8)

    fig, axes = style.montage_figure(ncols=4, width=24)
    extent = [0.0, c_max, s_lo, s_hi]

    im0 = axes[0].imshow(pred_np, origin='lower', extent=extent, aspect='auto',
                         cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    axes[0].set_title('predicted ds/dt = MLP(s, c)', color=style.foreground)
    style.xlabel(axes[0], 'c'); style.ylabel(axes[0], 's')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(true_np, origin='lower', extent=extent, aspect='auto',
                         cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    axes[1].set_title(r'true ds/dt = coeff$_s$ * (c - s)', color=style.foreground)
    style.xlabel(axes[1], 'c'); style.ylabel(axes[1], 's')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(err_np, origin='lower', extent=extent, aspect='auto',
                         cmap='RdBu_r', vmin=-elim, vmax=elim)
    axes[2].set_title('error (pred - true)', color=style.foreground)
    style.xlabel(axes[2], 'c'); style.ylabel(axes[2], 's')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    ax3 = axes[3]
    if last_pred is not None and last_target is not None:
        p_np = to_numpy(last_pred.detach())
        t_np = to_numpy(last_target.detach())
        ax3.scatter(t_np, p_np, s=2, alpha=0.3, color='C0')
        lim_lo = float(min(p_np.min(), t_np.min()))
        lim_hi = float(max(p_np.max(), t_np.max()))
        pad = 0.05 * max(abs(lim_lo), abs(lim_hi), 1e-8)
        ax3.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad], 'r--', lw=1)
        # R^2
        ss_res = float(((p_np - t_np) ** 2).sum())
        ss_tot = float(((t_np - t_np.mean()) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        ax3.text(0.02, 0.98, f'R²={r2:.3f}', transform=ax3.transAxes,
                 verticalalignment='top', fontsize=style.font_size * 0.9, color=style.foreground)
        ax3.set_title('last batch: pred vs true', color=style.foreground)
        style.xlabel(ax3, 'true ds/dt'); style.ylabel(ax3, 'pred ds/dt')
    else:
        ax3.text(0.5, 0.5, 'no last batch data', ha='center', va='center',
                 transform=ax3.transAxes, color=style.foreground)
        ax3.set_xticks([]); ax3.set_yticks([])

    plt.tight_layout()
    os.makedirs(f"./{log_dir}/tmp_training/internal", exist_ok=True)
    style.savefig(fig, f"./{log_dir}/tmp_training/internal/{epoch}_{N}.png")


# --------------------------------------------------------------------------- #
#  plot_training_cell_field — vectorized
# --------------------------------------------------------------------------- #

def plot_training_cell_field(config, has_siren, has_siren_time, model_f, n_frames, model_name, log_dir, epoch, N, x, x_mesh, index_cells, n_neurons, n_neuron_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    style = default_style
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting
    sign_flip = -1.0 if getattr(plot_config, 'invert_mlp1_sign', False) else 1.0
    dimension = simulation_config.dimension

    max_radius = simulation_config.max_radius
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))

    # extract cell_type from x (packed tensor or CellState)
    if isinstance(x, CellState):
        x_cell_type = x.cell_type
    else:
        x_cell_type = CellState.from_packed(x, dimension).cell_type

    # --- Embedding scatter ---
    fig, ax = style.figure(height=12)
    if axis:
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xticks(fontsize=style.frame_tick_font_size)
        plt.yticks(fontsize=style.frame_tick_font_size)
    else:
        plt.axis('off')
    embedding = get_embedding(model.a, dataset_num)
    if n_neuron_types > 1000:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x_cell_type) / n_neurons, s=1, cmap='viridis')
    else:
        for n in range(n_neuron_types):
            ax.scatter(embedding[index_cells[n], 0],
                       embedding[index_cells[n], 1], color=cmap.color(n), s=1)

    plt.tight_layout()
    style.savefig(fig, f"./{log_dir}/tmp_training/embedding/{model_name}_embedding_{epoch}_{N}.png")

    # --- Interaction function curves (vectorized) ---
    fig, ax = style.figure(height=12)
    if axis:
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.xticks(fontsize=style.frame_tick_font_size)
        plt.yticks(fontsize=style.frame_tick_font_size)
        plt.xlim([0, simulation_config.max_radius])
        plt.tight_layout()

    match model_config.cell_model_name:
        case 'arbitrary_field_ode':
            rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 200)).to(device)
        case 'boids_field_ode':
            rr = torch.tensor(np.linspace(-max_radius, max_radius, 200)).to(device)

    # Vectorized: all neurons at once
    all_embeddings = model.a[dataset_num, :n_neurons, :]  # (N, embed_dim)
    func_list = _batched_mlp_eval(model.lin_edge, all_embeddings, rr,
                                  model_config.cell_model_name,
                                  max_radius, device,
                                  dimension=simulation_config.dimension)

    # Plot with LineCollection
    rr_np = to_numpy(rr)
    ynorm_np = to_numpy(ynorm)
    type_arr = to_numpy(x_cell_type[:n_neurons]).astype(int)

    _plot_curves_fast(ax, rr_np, to_numpy(func_list), type_arr, cmap,
                      ynorm=ynorm_np * sign_flip, subsample=5, alpha=0.25, linewidth=8)

    if sign_flip < 0:
        conv_text = '+: repulsive, −: attractive'
    else:
        conv_text = '+: attractive, −: repulsive'
    ax.text(0.02, 0.98, conv_text,
            transform=ax.transAxes, verticalalignment='top',
            fontsize=style.font_size * 0.8,
            color=style.foreground, alpha=0.7)

    plt.tight_layout()
    style.savefig(fig, f"./{log_dir}/tmp_training/function/MLP1/{model_name}_function_{epoch}_{N}.png")

    # --- Siren field visualization ---
    if has_siren:
        if has_siren_time:
            frame_list = [54, 58, 62, 66]
        else:
            frame_list = [0]

        for frame in frame_list:
            if has_siren_time:
                with torch.no_grad():
                    tmp = model_f(time=frame / n_frames) ** 2
            else:
                with torch.no_grad():
                    tmp = model_f() ** 2
            tmp = torch.reshape(tmp, (n_nodes_per_axis, n_nodes_per_axis))
            tmp = to_numpy(torch.sqrt(tmp))
            if has_siren_time:
                tmp = np.rot90(tmp, k=1)
            fig, axf = style.figure(width=14, height=12)
            axf.imshow(tmp, cmap='grey')
            plt.colorbar(axf.images[0], ax=axf)
            axf.set_xticks([])
            axf.set_yticks([])
            plt.tight_layout()
            style.savefig(fig, f"./{log_dir}/tmp_training/external_input/{model_name}_{epoch}_{N}_{frame}.png")


# --------------------------------------------------------------------------- #
#  Vectorized sparsity MLP evaluation
# --------------------------------------------------------------------------- #

def batched_sparsity_mlp_eval(model, rr, n_cells, config, device):
    """Evaluate the edge MLP for all cells in batch mode, for sparsity fitting.

    Returns:
        pred: (N, n_pts, output_dim) tensor
    """
    mc = config.graph_model
    sim = config.simulation
    all_embeddings = model.a[0, :n_cells, :].clone().detach()  # (N, embed_dim)

    # Build features: (N, n_pts, input_dim)
    features = build_edge_features(rr, all_embeddings, mc.cell_model_name, sim.max_radius,
                                    dimension=sim.dimension)
    N, n_pts, input_dim = features.shape

    # Flatten, run MLP, reshape
    features_flat = features.reshape(N * n_pts, input_dim)
    pred_flat = model.lin_edge(features_flat.float())  # (N * n_pts, output_dim)
    output_dim = pred_flat.shape[1]
    pred = pred_flat.reshape(N, n_pts, output_dim)

    return pred


# --------------------------------------------------------------------------- #
#  True interaction function overlay
# --------------------------------------------------------------------------- #

def _get_true_psi(rr, config, n_cell_types, device):
    """Compute true psi interaction curves for each cell type.

    Returns:
        dict mapping type_index -> (n_pts,) numpy array, or empty dict if
        no ground truth is available (external data).
    """
    if config.data_folder_name != 'none':
        return {}

    from cell_gnn.generators.utils import choose_model

    try:
        true_model, _, _ = choose_model(config, device=device)
    except Exception:
        return {}
    config_model = config.graph_model.cell_model_name
    p = true_model.p
    # For models where p is (n_params,) for a single type, unsqueeze so p[0] gives the params vector.
    # For gravity_ode, p is (n_cell_types,) — one scalar mass per type — so don't unsqueeze.
    if p.dim() == 1 and 'gravity' not in config_model:
        p = p.unsqueeze(0)

    true_curves = {}
    for n in range(n_cell_types):
        with torch.no_grad():
            if 'arbitrary_ode' in config_model:
                func_type = 'arbitrary'
                if hasattr(config.simulation, 'func_params') and config.simulation.func_params:
                    func_type = config.simulation.func_params[n][0]
                psi_n = true_model.psi(rr, p[n], func=func_type)
            else:
                psi_n = true_model.psi(rr, p[n])
        true_curves[n] = to_numpy(psi_n).flatten()

    return true_curves


def _plot_true_lin_edge(ax, rr_np, true_curves, cmap, sign_flip=1.0):
    """Plot true psi curves on top of predicted (thick, per-type color).

    The true model's psi(r) uses the convention positive=repulsion,
    but lin_edge learns F = -psi * rhat, so lin_edge_x ≈ -psi.
    We negate psi here so the true curve matches the learned convention,
    then apply sign_flip (from invert_mlp1_sign) for display preference.
    """
    for n, psi_np in true_curves.items():
        ax.plot(rr_np, -psi_np * sign_flip, color=cmap.color(n), linewidth=6, alpha=0.5)


def _compute_curve_r2(func_list_np, type_arr, true_curves, ynorm=1.0):
    """Compute per-curve R² between predicted and ground-truth psi.

    Args:
        func_list_np: (N, n_pts) predicted curves (before ynorm scaling)
        type_arr: (N,) int array of cell type labels
        true_curves: dict type_index -> (n_pts,) true curve
        ynorm: scalar or array to multiply predicted curves by

    Returns:
        r2_values: (N,) numpy array of R² per curve, or None if no true curves
    """
    if not true_curves:
        return None

    N = func_list_np.shape[0]
    ynorm_val = float(ynorm) if np.isscalar(ynorm) else np.asarray(ynorm)
    r2_values = np.full(N, np.nan)

    for i in range(N):
        t = int(type_arr[i])
        if t not in true_curves:
            continue
        y_true = -true_curves[t]  # negate: psi convention is +repulsion, lin_edge learns -psi
        y_pred = func_list_np[i] * ynorm_val
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot > 0:
            r2_values[i] = 1.0 - ss_res / ss_tot

    return r2_values


# --------------------------------------------------------------------------- #
#  Training summary panels
# --------------------------------------------------------------------------- #

def plot_training_summary_panels(fig, log_dir, model, config, n_cells, n_cell_types,
                                 index_cells, type_list, ynorm, cmap,
                                 embedding_cluster, epoch, logger, device,
                                 loss_dict=None, regul_history=None):
    """Assemble epoch summary by loading saved plots and adding a UMAP panel.

    Panels 1-3 load images from tmp_training (embedding, MLP1, loss).
    Panel 4 draws UMAP of interaction functions directly.

    Args:
        fig: matplotlib Figure (2x2 subplots will be added).
        log_dir: path to the training log directory.
        model: trained GNN model (must have ``model.a`` and ``model.lin_edge``).
        config: CellGNNConfig.
        n_cells: int.
        n_cell_types: int.
        index_cells: list of index arrays per type.
        type_list: (N,) tensor of ground-truth type labels.
        ynorm: normalization tensor.
        cmap: CustomColorMap instance.
        embedding_cluster: EmbeddingCluster instance.
        epoch: current epoch number.
        logger: logging.Logger for accuracy reporting.
        device: torch device.
        loss_dict: dict with key ``'loss'`` (unused, kept for API compat).
        regul_history: dict from ``LossRegularizer.get_history()`` (unused).

    Returns:
        (labels, n_clusters, new_labels, func_list, model_a_, accuracy)
        where ``model_a_`` is the embedding with cluster medians applied.
    """
    import glob
    import os
    import imageio
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import accuracy_score
    from scipy.optimize import linear_sum_assignment

    style = default_style
    tc = config.training
    mc = config.graph_model
    sim = config.simulation

    def _load_panel(fig, pos, filepath):
        """Load an image file into a subplot, or leave blank if missing."""
        ax = fig.add_subplot(2, 2, pos)
        if os.path.exists(filepath):
            img = imageio.imread(filepath)
            ax.imshow(img)
        ax.axis('off')
        return ax

    # --- Find the last saved iteration snapshot ---
    embedding_files = glob.glob(f"./{log_dir}/tmp_training/embedding/*.png")
    if embedding_files:
        last_file = max(embedding_files, key=os.path.getctime)
        filename = os.path.basename(last_file)
        last_epoch_N = filename.replace('.png', '')
    else:
        last_epoch_N = f"{epoch}_0"

    # --- Panels 1-3: load saved images ---
    _load_panel(fig, 1, f"./{log_dir}/tmp_training/embedding/{last_epoch_N}.png")
    _load_panel(fig, 2, f"./{log_dir}/tmp_training/function/MLP1/function_{last_epoch_N}.png")
    _load_panel(fig, 3, f"./{log_dir}/tmp_training/loss.png")

    # --- Compute func_list for UMAP and sparsity ---
    embedding = get_embedding(model.a, 0)

    config_model = mc.cell_model_name
    if 'boids_ode' in config_model:
        max_radius_plot = 0.04
        rr = torch.tensor(np.linspace(-max_radius_plot, max_radius_plot, 1000)).to(device)
    elif config_model == 'gravity_ode':
        rr = torch.tensor(np.linspace(0, sim.max_radius * 1.3, 1000)).to(device)
    else:
        rr = torch.tensor(np.linspace(0, sim.max_radius, 1000)).to(device)

    func_list, _ = analyze_edge_function(
        rr=rr, vizualize=False, config=config,
        model_MLP=model.lin_edge, model=model,
        n_nodes=0, n_cells=n_cells, ynorm=ynorm,
        type_list=to_numpy(type_list), cmap=cmap,
        update_type='NA', device=device)

    # --- Clustering: UMAP on embedding + DBSCAN ---
    n_neighbors = 100
    min_dist = 0.3
    dbscan_eps = 0.3

    print('UMAP on embedding ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trans = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2,
                          random_state=tc.seed).fit(embedding)
        proj_embedding = trans.transform(embedding)

    db = DBSCAN(eps=dbscan_eps, min_samples=5)
    labels = db.fit_predict(proj_embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if -1 in labels:
        labels[labels == -1] = n_clusters
        n_clusters += 1

    # Hungarian algorithm for optimal label mapping
    type_np_flat = to_numpy(type_list).flatten().astype(int)
    size = max(n_cell_types, n_clusters)
    confusion = np.zeros((size, size))
    for t, c in zip(type_np_flat, labels):
        confusion[int(t), int(c)] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    new_labels = np.array([mapping.get(int(l), -1) for l in labels])

    accuracy = accuracy_score(type_np_flat, new_labels)
    print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')
    logger.info(f'accuracy: {np.round(accuracy, 3)}    n_clusters: {n_clusters}')

    # --- Save UMAP plot as standalone montage panel ---
    os.makedirs(f'./{log_dir}/tmp_training/umap', exist_ok=True)
    fig_umap, ax_umap = style.montage_figure()
    for n in np.unique(new_labels):
        pos = np.array(np.argwhere(new_labels == n).squeeze().astype(int))
        if pos.size > 0:
            ax_umap.scatter(proj_embedding[pos, 0], proj_embedding[pos, 1], s=5)
    style.montage_xlabel(ax_umap, 'UMAP 0')
    style.montage_ylabel(ax_umap, 'UMAP 1')
    style.montage_annotate(ax_umap,
                           'UMAP of learned $\\mathrm{MLP}_1$ curves\n'
                           f'input: {func_list.shape[1]} radial samples per cell\n'
                           f'n_neighbors={n_neighbors}  min_dist={min_dist}',
                           (0.02, 0.98), verticalalignment='top')
    plt.tight_layout()
    umap_path = f'./{log_dir}/tmp_training/umap/{epoch}.png'
    style.savefig(fig_umap, umap_path)

    # --- Panel 4: load UMAP as raster (consistent with panels 1-3) ---
    _load_panel(fig, 4, umap_path)

    # --- Compute sparsified embedding for return ---
    model_a_ = model.a[0].clone().detach()
    for n in range(n_clusters):
        pos = np.argwhere(labels == n).squeeze().astype(int)
        pos = np.array(pos)
        if pos.size > 0:
            median_center = model_a_[pos, :]
            median_center = torch.median(median_center, dim=0).values
            model_a_[pos, :] = median_center

    return labels, n_clusters, new_labels, func_list, model_a_, accuracy


# --------------------------------------------------------------------------- #
#  Loss component figure (loss.tif)
# --------------------------------------------------------------------------- #

def plot_loss_components(loss_dict, regul_history, log_dir, epoch=None, Niter=None):
    """Save a single-panel log-scale loss figure to ``{log_dir}/tmp_training/loss.tif``.

    Args:
        loss_dict: dict with key ``'loss'`` — list of per-epoch prediction loss.
        regul_history: dict from ``LossRegularizer.get_history()`` with keys
            ``'regul_total'``, ``'edge_weight'``, ``'edge_diff'``,
            ``'edge_norm'``, ``'continuous'``.
        log_dir: directory to save the figure.
        epoch: current epoch (for annotation).
        Niter: iterations per epoch (for annotation).
    """
    if len(loss_dict['loss']) == 0:
        return

    import os
    from matplotlib.ticker import AutoLocator, ScalarFormatter

    style = default_style

    # Use montage_figure for consistent sizing with other montage panels.
    # Then override locators — montage_figure doesn't set MaxNLocator/
    # FormatStrFormatter, so log-scale works correctly.
    fig_loss, ax = style.montage_figure()

    # Use proper locators for a loss plot (integer x, log y)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())

    info_text = ""
    if epoch is not None:
        info_text += f"epoch: {epoch}"
    if Niter is not None:
        if info_text:
            info_text += " | "
        info_text += f"iter/epoch: {Niter}"
    if info_text:
        style.montage_annotate(ax, info_text, (0.02, 0.98), verticalalignment='top')

    # --- curves to plot ---
    # Replace zeros/negatives with NaN so log scale doesn't break
    def _safe_log_data(data):
        arr = np.array(data, dtype=float)
        arr[arr <= 0] = np.nan
        return arr

    loss_data = _safe_log_data(loss_dict['loss'])
    ax.plot(loss_data, color='b', linewidth=style.line_width,
            label='loss', alpha=0.8)

    # Per-component curves (only plotted if populated by the trainer).
    for key, color, label in [
        ('pos',      'tab:blue',   'pos (pred vs y_batch)'),
        ('pde',      'tab:green',  'pde residual'),
        ('pde_reg',  'tab:olive',  'pde reg'),
        ('internal', 'tab:purple', 'internal ODE'),
    ]:
        data = loss_dict.get(key, []) if hasattr(loss_dict, 'get') else []
        if len(data) > 0 and any(v > 0 for v in data):
            ax.plot(_safe_log_data(data), color=color, linewidth=1.2, label=label, alpha=0.9)

    if regul_history:
        for key, color, label in [
            ('regul_total', 'red', 'total regul'),
            ('edge_weight', 'pink', 'edge weight'),
            ('edge_diff', 'orange', 'edge monotonicity'),
            ('edge_norm', 'brown', 'edge norm'),
            ('continuous', 'cyan', 'continuous'),
        ]:
            data = regul_history.get(key, [])
            if len(data) > 0 and any(v > 0 for v in data):
                ax.plot(_safe_log_data(data), color=color, linewidth=1, label=label, alpha=0.8)

    ax.set_yscale('log')

    # Force y-axis to include actual loss data
    pos_vals = [v for v in loss_dict['loss'] if v > 0]
    if pos_vals:
        ymin = min(pos_vals)
        ymax = max(pos_vals)
        ax.set_ylim([ymin * 0.5, ymax * 2.0])

    # Ensure x-axis shows full data range
    n_points = len(loss_dict['loss'])
    if n_points > 1:
        ax.set_xlim([0, n_points - 1])

    style.montage_xlabel(ax, 'iteration')
    style.montage_ylabel(ax, 'loss')
    ax.legend(fontsize=style.montage_legend_font_size, loc='best')

    os.makedirs(f'./{log_dir}/tmp_training', exist_ok=True)
    plt.tight_layout()
    style.savefig(fig_loss, f'./{log_dir}/tmp_training/loss.png')


# --------------------------------------------------------------------------- #
#  Residual field visualization
# --------------------------------------------------------------------------- #

def plot_residual_field_3d(pos, residual, frame, dimension, log_dir, cmap, sim):
    """Visualize the residual field (y_true - y_pred) as quiver arrows.

    For 3D: left panel is a 3D scatter+quiver, right panel is a Z cross-section.
    For 2D: single panel with 2D quiver.

    Args:
        pos: (N, dim) numpy array of cell positions.
        residual: (N, dim) numpy array of residual vectors.
        frame: int, frame number.
        dimension: int, 2 or 3.
        log_dir: str, output directory.
        cmap: CustomColorMap instance.
        sim: simulation config (needs max_radius).
    """
    import os

    style = default_style
    mag = np.sqrt((residual ** 2).sum(axis=-1))  # (N,)

    out_dir = f'./{log_dir}/results/residual'
    os.makedirs(out_dir, exist_ok=True)

    if dimension == 3:
        fig, (ax1, ax2) = style.figure(ncols=2, width=20, height=10,
                                        subplot_kw={'projection': None})
        # Replace left panel with 3D projection
        ax1.remove()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    s=10, c=mag, cmap='hot', alpha=0.5, edgecolors='none')
        n = pos.shape[0]
        step_q = max(1, n // 300)
        idx = np.arange(0, n, step_q)
        scale = sim.max_radius * 0.05
        ax1.quiver(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                   residual[idx, 0] * scale, residual[idx, 1] * scale, residual[idx, 2] * scale,
                   color='blue', alpha=0.6, arrow_length_ratio=0.3, linewidth=0.8)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_zlim([0, 1])
        style.xlabel(ax1, 'X')
        style.ylabel(ax1, 'Y')
        ax1.set_zlabel('Z', fontsize=style.label_font_size)
        ax1.set_title(f'residual field (3D) — frame {frame}', fontsize=style.font_size)

        # --- Right: Z cross-section ---
        style.clean_ax(ax2)
        z_center, z_thickness = 0.5, 0.1
        mask = np.abs(pos[:, 2] - z_center) < z_thickness
        if mask.sum() > 0:
            pos_s = pos[mask, :2]
            res_s = residual[mask, :2]
            mag_s = mag[mask]
            ax2.scatter(pos_s[:, 0], pos_s[:, 1], s=15, c=mag_s, cmap='hot',
                        alpha=0.6, edgecolors='none')
            ax2.quiver(pos_s[:, 0], pos_s[:, 1], res_s[:, 0], res_s[:, 1],
                       color='blue', alpha=0.6, scale=mag.max() * 10 + 1e-12,
                       width=0.003)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        style.xlabel(ax2, 'X')
        style.ylabel(ax2, 'Y')
        ax2.set_title(f'Z slice ({z_center - z_thickness:.1f} < z < {z_center + z_thickness:.1f})',
                      fontsize=style.font_size)
        ax2.set_aspect('equal')

    else:
        fig, ax = style.figure(width=10, height=10)
        ax.scatter(pos[:, 0], pos[:, 1], s=10, c=mag, cmap='hot',
                   alpha=0.5, edgecolors='none')
        n = pos.shape[0]
        step_q = max(1, n // 500)
        idx = np.arange(0, n, step_q)
        ax.quiver(pos[idx, 0], pos[idx, 1], residual[idx, 0], residual[idx, 1],
                  color='blue', alpha=0.6, scale=mag.max() * 10 + 1e-12,
                  width=0.003)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        style.xlabel(ax, 'X')
        style.ylabel(ax, 'Y')
        ax.set_title(f'residual field (2D) — frame {frame}', fontsize=style.font_size)
        ax.set_aspect('equal')

    plt.tight_layout()
    style.savefig(fig, f'{out_dir}/residual_{frame:06d}.png')


def plot_noise_vs_error_snapshot(pos, noise_t, pred_vs_clean_t, residual_t,
                                 frame, dimension, log_dir):
    """Per-frame comparison: noise, pred-vs-clean error, and residual.

    Three panels:
      Left:   scatter colored by |noise|
      Center: scatter colored by |pred - clean_force|
      Right:  scatter colored by |residual| (y_true - y_pred)
    """
    import os
    style = default_style
    out_dir = f'./{log_dir}/results/noise_vs_error'
    os.makedirs(out_dir, exist_ok=True)

    noise_mag = np.sqrt((noise_t ** 2).sum(axis=-1))
    pvc_mag = np.sqrt((pred_vs_clean_t ** 2).sum(axis=-1))
    res_mag = np.sqrt((residual_t ** 2).sum(axis=-1))

    # Common color scale
    vmax = max(noise_mag.max(), pvc_mag.max(), res_mag.max()) + 1e-12

    if dimension == 3:
        # Use Z cross-section for 3D
        z_center, z_thick = 0.5, 0.1
        mask = np.abs(pos[:, 2] - z_center) < z_thick
        if mask.sum() < 10:
            mask = np.ones(pos.shape[0], dtype=bool)
        p = pos[mask, :2]
        n_m, pvc_m, r_m = noise_mag[mask], pvc_mag[mask], res_mag[mask]
        noise_vec = noise_t[mask, :2]
        pvc_vec = pred_vs_clean_t[mask, :2]
        res_vec = residual_t[mask, :2]
    else:
        p = pos[:, :2]
        n_m, pvc_m, r_m = noise_mag, pvc_mag, res_mag
        noise_vec = noise_t[:, :2]
        pvc_vec = pred_vs_clean_t[:, :2]
        res_vec = residual_t[:, :2]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    # --- Top row: scatter colored by magnitude ---
    titles_top = ['|noise|', '|pred - clean_force|', '|residual|']
    data_top = [n_m, pvc_m, r_m]

    for ax, d, title in zip(axes[0], data_top, titles_top):
        style.clean_ax(ax)
        sc = ax.scatter(p[:, 0], p[:, 1], s=5, c=d, cmap='hot',
                        vmin=0, vmax=vmax, alpha=0.7, edgecolors='none')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.set_title(f'{title}  (mean={d.mean():.5f})', fontsize=style.font_size)
        style.xlabel(ax, 'X')
        style.ylabel(ax, 'Y')

    fig.colorbar(sc, ax=axes[0, -1], shrink=0.8)

    # --- Bottom row: quiver plots showing vectors ---
    titles_bot = ['noise vector', 'pred - clean_force vector', 'residual vector']
    vecs = [noise_vec, pvc_vec, res_vec]
    mags = [n_m, pvc_m, r_m]

    n = p.shape[0]
    step_q = max(1, n // 500)
    idx = np.arange(0, n, step_q)
    scale_val = vmax * 10 + 1e-12

    for ax, v, m, title in zip(axes[1], vecs, mags, titles_bot):
        style.clean_ax(ax)
        ax.scatter(p[:, 0], p[:, 1], s=3, c='gray', alpha=0.3, edgecolors='none')
        ax.quiver(p[idx, 0], p[idx, 1], v[idx, 0], v[idx, 1],
                  m[idx], cmap='hot', clim=(0, vmax),
                  alpha=0.7, scale=scale_val, width=0.003)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.set_title(f'{title}  (mean={m.mean():.5f})', fontsize=style.font_size)
        style.xlabel(ax, 'X')
        style.ylabel(ax, 'Y')

    fig.suptitle(f'frame {frame}', fontsize=style.font_size + 2)
    plt.tight_layout()
    style.savefig(fig, f'{out_dir}/noise_vs_error_{frame:06d}.png')


def plot_residual_vs_noise(residual_arr, noise_arr, pred_vs_clean_arr, log_dir):
    """Compare residuals, noise, and prediction-vs-clean-force over time.

    Args:
        residual_arr: (T, N, dim) — y_true - y_pred (includes noise).
        noise_arr: (T, N, dim) — injected noise during data generation.
        pred_vs_clean_arr: (T, N, dim) — predicted velocity - clean pair force.
        log_dir: output directory.
    """
    import os
    style = default_style

    T = residual_arr.shape[0]
    ts = np.arange(T)

    # Per-cell magnitudes, then mean/std across cells at each timestep
    res_mag = np.sqrt((residual_arr ** 2).sum(axis=-1))       # (T, N)
    noise_mag = np.sqrt((noise_arr ** 2).sum(axis=-1))         # (T, N)
    pvc_mag = np.sqrt((pred_vs_clean_arr ** 2).sum(axis=-1))   # (T, N)

    res_mean, res_std = res_mag.mean(axis=1), res_mag.std(axis=1)
    noise_mean, noise_std = noise_mag.mean(axis=1), noise_mag.std(axis=1)
    pvc_mean, pvc_std = pvc_mag.mean(axis=1), pvc_mag.std(axis=1)

    out_dir = f'./{log_dir}/results/residual'
    os.makedirs(out_dir, exist_ok=True)

    # --- Panel 1: Time series of mean magnitudes ---
    fig, axes = style.figure(nrows=2, ncols=1, width=14, height=10)
    ax1, ax2 = axes[0], axes[1]

    style.clean_ax(ax1)
    ax1.plot(ts, noise_mean, color='C0', label='noise |n|', linewidth=1.5)
    ax1.fill_between(ts, noise_mean - noise_std, noise_mean + noise_std,
                     color='C0', alpha=0.15)
    ax1.plot(ts, pvc_mean, color='C1', label='|pred - clean_force|', linewidth=1.5)
    ax1.fill_between(ts, pvc_mean - pvc_std, pvc_mean + pvc_std,
                     color='C1', alpha=0.15)
    ax1.plot(ts, res_mean, color='C2', label='|residual| (y_true - y_pred)', linewidth=1.5,
             linestyle='--')
    style.xlabel(ax1, 'frame')
    style.ylabel(ax1, 'mean magnitude')
    ax1.legend(fontsize=style.font_size - 2)
    ax1.set_title('Noise vs prediction error vs residual', fontsize=style.font_size)

    # --- Panel 2: Scatter of per-cell noise mag vs pred-vs-clean mag ---
    style.clean_ax(ax2)
    # Subsample for speed
    n_pts = min(50000, noise_mag.size)
    idx = np.random.choice(noise_mag.size, size=n_pts, replace=False)
    noise_flat = noise_mag.ravel()[idx]
    pvc_flat = pvc_mag.ravel()[idx]
    ax2.scatter(noise_flat, pvc_flat, s=1, alpha=0.15, color='C3', rasterized=True)
    lim = max(noise_flat.max(), pvc_flat.max()) * 1.05
    ax2.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5, label='y=x')
    ax2.set_xlim([0, lim])
    ax2.set_ylim([0, lim])
    style.xlabel(ax2, '|noise|')
    style.ylabel(ax2, '|pred - clean_force|')
    ax2.legend(fontsize=style.font_size - 2)
    ax2.set_title('Per-cell: noise magnitude vs prediction error', fontsize=style.font_size)
    ax2.set_aspect('equal')

    plt.tight_layout()
    style.savefig(fig, f'{out_dir}/residual_vs_noise.png')

    # Print summary statistics
    print(f'  Noise:          mean={noise_mean.mean():.6f}  std={noise_mean.std():.6f}')
    print(f'  Pred-vs-clean:  mean={pvc_mean.mean():.6f}  std={pvc_mean.std():.6f}')
    print(f'  Residual:       mean={res_mean.mean():.6f}  std={res_mean.std():.6f}')
    ratio = pvc_mean.mean() / (noise_mean.mean() + 1e-12)
    print(f'  |pred-clean|/|noise| ratio: {ratio:.3f}  (1.0 = prediction error ~ noise level)')
    print(f'  Saved: {out_dir}/residual_vs_noise.png')
