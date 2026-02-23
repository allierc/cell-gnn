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
#  Vectorized helpers
# --------------------------------------------------------------------------- #

def build_edge_features(rr, embedding, model_name, max_radius, dimension=2):
    """Build input features for the edge MLP, supporting batched embeddings.

    Args:
        rr: (n_pts,) tensor of radial distances
        embedding: (N, embed_dim) or (n_pts, embed_dim) tensor
        model_name: one of arbitrary_ode, boids_ode, gravity_ode, arbitrary_field_ode, boids_field_ode
        max_radius: float
        dimension: int, spatial dimension (2 or 3)

    Returns:
        (N, n_pts, input_dim) or (n_pts, input_dim) tensor of features
    """
    # Handle batched case: embedding is (N, embed_dim), rr is (n_pts,)
    if embedding.dim() == 2 and rr.dim() == 1 and embedding.shape[0] != rr.shape[0]:
        N, embed_dim = embedding.shape
        n_pts = rr.shape[0]
        rr_exp = rr[None, :].expand(N, n_pts)  # (N, n_pts)
        emb_exp = embedding[:, None, :].expand(N, n_pts, embed_dim)  # (N, n_pts, embed_dim)

        # delta_pos: (N, n_pts, dimension) — first component is rr, rest zeros
        delta_pos = torch.zeros(N, n_pts, dimension, dtype=rr.dtype, device=rr.device)
        delta_pos[:, :, 0] = rr_exp / max_radius
        r = rr_exp.unsqueeze(-1) / max_radius  # (N, n_pts, 1)

        match model_name:
            case 'arbitrary_ode' | 'arbitrary_field_ode':
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
            case 'arbitrary_ode' | 'arbitrary_field_ode':
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

        plt.tight_layout()
        style.savefig(fig, f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.png")
    else:
        match model_config.cell_model_name:

            case 'arbitrary_ode' | 'arbitrary_field_ode' | 'gravity_ode':
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
                style.montage_xlabel(ax, r'$r$')
                style.montage_ylabel(ax, r'learned $\mathrm{MLP}_1$')
                plt.tight_layout()
                style.savefig(fig, f"./{log_dir}/tmp_training/function/MLP1/function_{epoch}_{N}.png")

    return lin_edge_r2


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
    """Plot true psi curves on top of predicted (thick, per-type color)."""
    for n, psi_np in true_curves.items():
        ax.plot(rr_np, psi_np * sign_flip, color=cmap.color(n), linewidth=6, alpha=0.5)


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
        y_true = true_curves[t]
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
