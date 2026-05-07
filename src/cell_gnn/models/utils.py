import glob
import os
import shutil

import torch
import numpy as np
import torch.nn as nn

from cell_gnn.models.MLP import MLP
from cell_gnn.models.registry import get_model_class
from cell_gnn.utils import to_numpy, fig_init, choose_boundary_values


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the Apache License, Version 2.0
    # found in the LICENSE file in the root directory of this source tree.

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):

            # student_output = F.normalize(student_output, eps=eps, p=2, dim=0)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()

        return loss


def choose_training_model(model_config=None, device=None):
    """Create and return a model based on the configuration.

    Uses the model registry to look up the appropriate class.

    Args:
        model_config: Configuration object containing simulation and graph model parameters.
        device: Torch device to place the model on.

    Returns:
        Tuple of (model, bc_pos, bc_dpos).
    """

    aggr_type = model_config.graph_model.aggr_type
    dimension = model_config.simulation.dimension
    name = model_config.graph_model.cell_model_name

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model_cls = get_model_class(name)
    model = model_cls(
        aggr_type=aggr_type,
        config=model_config,
        device=device,
        bc_dpos=bc_dpos,
        dimension=dimension,
    )
    model.edges = []

    return model, bc_pos, bc_dpos


def get_type_list(x, dimension):
    from cell_gnn.cell_state import CellState
    if isinstance(x, CellState):
        return x.cell_type.clone().detach().unsqueeze(-1).float()
    return CellState.from_packed(x, dimension).cell_type.clone().detach().unsqueeze(-1).float()


def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size
    return get_batch_size


def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size
    return get_batch_size


def set_trainable_parameters(model=[], lr_embedding=[], lr=[], lr_update=[], lr_W=[], lr_modulation=[], learning_rate_nnr=[], learning_rate_edge_embedding=[]):

    # Unwrap torch.compile wrapper to avoid duplicate parameters
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    trainable_params = [param for _, param in raw_model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(raw_model.a)

    if lr_update == []:
        lr_update = lr

    # Use fused Adam when parameters are on CUDA — avoids launching many small kernels
    _use_fused = raw_model.a.is_cuda
    optimizer = torch.optim.Adam([raw_model.a], lr=lr_embedding, fused=_use_fused)
    for name, parameter in raw_model.named_parameters():
        if (parameter.requires_grad) & (name != 'a'):
            if (name == 'b') or ('lin_modulation' in name):
                optimizer.add_param_group({'params': parameter, 'lr': lr_modulation})
            elif 'lin_phi' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_update})
            elif 'W' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_W})
            elif 'NNR' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_nnr})
            elif 'edges_embedding' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_edge_embedding})
            else:
                optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params


class LossRegularizer:
    """Handles regularization terms, history tracking, and per-component loss recording.

    Adapted from flyvis-gnn's LossRegularizer for cell-gnn models that have
    ``model.a`` (embedding) and ``model.lin_edge`` (edge MLP).

    Components tracked: edge_weight, edge_diff, edge_norm, continuous.
    """

    COMPONENTS = ['edge_weight', 'edge_diff', 'edge_norm', 'continuous']

    def __init__(self, train_config, model_config, sim_config, n_cells, plot_frequency):
        self.tc = train_config
        self.mc = model_config
        self.sim = sim_config
        self.n_cells = n_cells
        self.plot_frequency = plot_frequency
        self.epoch = 0
        self.iter_count = 0
        self._iter_total = 0.0
        self._iter_tracker = {}
        self._history = {comp: [] for comp in self.COMPONENTS}
        self._history['regul_total'] = []

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iter_count = 0

    def reset_iteration(self):
        self.iter_count += 1
        self._iter_total = 0.0
        self._iter_tracker = {comp: 0.0 for comp in self.COMPONENTS}

    def _add(self, name, term):
        if term is None:
            return
        val = term.item() if hasattr(term, 'item') else float(term)
        self._iter_total += val
        if name in self._iter_tracker:
            self._iter_tracker[name] += val

    def compute(self, model, device):
        """Compute all regularization terms.

        Returns:
            Total regularization loss tensor.
        """
        tc = self.tc
        mc = self.mc
        sim = self.sim
        total_regul = torch.tensor(0.0, device=device)

        # --- edge_weight: L1 on lin_edge parameters ---
        if tc.coeff_edge_weight > 0:
            for param in model.lin_edge.parameters():
                regul_term = param.norm(1) * tc.coeff_edge_weight
                total_regul = total_regul + regul_term
                self._add('edge_weight', regul_term)

        # --- edge_diff: monotonicity constraint on edge function ---
        if tc.coeff_edge_diff > 0:
            n_sample = max(1, self.n_cells // 100)
            rr = torch.linspace(0, sim.max_radius, 1000, dtype=torch.float32, device=device)
            dr = sim.max_radius / 200
            from cell_gnn.plot import build_edge_features
            for n in np.random.permutation(self.n_cells)[:n_sample]:
                embedding_ = model.a[0, n, :] * torch.ones((1000, mc.embedding_dim), device=device)
                feat0 = build_edge_features(rr=rr, embedding=embedding_,
                                            model_name=mc.cell_model_name,
                                            max_radius=sim.max_radius,
                                            dimension=sim.dimension)
                feat1 = build_edge_features(rr=rr + dr, embedding=embedding_,
                                            model_name=mc.cell_model_name,
                                            max_radius=sim.max_radius,
                                            dimension=sim.dimension)
                msg0 = model.lin_edge(feat0)
                msg1 = model.lin_edge(feat1)
                regul_term = torch.relu(msg0 - msg1).norm(2) * tc.coeff_edge_diff
                total_regul = total_regul + regul_term
                self._add('edge_diff', regul_term)

        # --- edge_norm: edge function at max_radius should be near zero ---
        if tc.coeff_edge_norm > 0:
            n_sample = max(1, self.n_cells // 100)
            rr_max = torch.tensor([sim.max_radius], dtype=torch.float32, device=device)
            from cell_gnn.plot import build_edge_features
            for n in np.random.permutation(self.n_cells)[:n_sample]:
                embedding_ = model.a[0, n, :].unsqueeze(0)
                feat = build_edge_features(rr=rr_max, embedding=embedding_,
                                           model_name=mc.cell_model_name,
                                           max_radius=sim.max_radius,
                                           dimension=sim.dimension)
                msg_norm = model.lin_edge(feat)
                regul_term = msg_norm.norm(2) * tc.coeff_edge_norm
                total_regul = total_regul + regul_term
                self._add('edge_norm', regul_term)

        # --- continuous: edge function gradient smoothness ---
        if (tc.coeff_continuous > 0) and (self.epoch > 0):
            n_sample = max(1, self.n_cells // 100)
            rr = torch.linspace(0, sim.max_radius, 1000, dtype=torch.float32, device=device)
            dr = sim.max_radius / 200
            from cell_gnn.plot import build_edge_features
            for n in np.random.permutation(self.n_cells)[:n_sample]:
                embedding_ = model.a[0, n, :] * torch.ones((1000, mc.embedding_dim), device=device)
                feat1 = build_edge_features(rr=rr + dr, embedding=embedding_,
                                            model_name=mc.cell_model_name,
                                            max_radius=sim.max_radius,
                                            dimension=sim.dimension)
                func1 = model.lin_edge(feat1)
                feat0 = build_edge_features(rr=rr, embedding=embedding_,
                                            model_name=mc.cell_model_name,
                                            max_radius=sim.max_radius,
                                            dimension=sim.dimension)
                func0 = model.lin_edge(feat0)
                grad = func1 - func0
                regul_term = tc.coeff_continuous * grad.norm(2)
                total_regul = total_regul + regul_term
                self._add('continuous', regul_term)

        return total_regul

    def finalize_iteration(self):
        if (self.iter_count % self.plot_frequency == 0) or (self.iter_count == 1):
            n = max(self.n_cells, 1)
            self._history['regul_total'].append(self._iter_total / n)
            for comp in self.COMPONENTS:
                self._history[comp].append(self._iter_tracker.get(comp, 0) / n)

    def get_iteration_total(self):
        return self._iter_total

    def get_history(self):
        return self._history


def save_exploration_artifacts(root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                               iter_in_block=1, block_number=1):
    """Save exploration artifacts for Claude analysis.

    Adapted from flyvis-gnn save_exploration_artifacts(). Saves montage, MLP, embedding,
    and config snapshots to the exploration directory.

    Args:
        root_dir: Root directory of the project
        exploration_dir: Base directory for exploration artifacts
        config: Configuration object
        config_file_: Config file name (without extension)
        pre_folder: Prefix folder for config
        iteration: Current iteration number
        iter_in_block: Iteration number within current block (1-indexed)
        block_number: Current block number (1-indexed)

    Returns:
        dict with paths to saved directories
    """
    config_save_dir = f"{exploration_dir}/config"
    montage_save_dir = f"{exploration_dir}/montage"
    mlp_save_dir = f"{exploration_dir}/mlp"
    embedding_save_dir = f"{exploration_dir}/embedding"
    tree_save_dir = f"{exploration_dir}/exploration_tree"
    protocol_save_dir = f"{exploration_dir}/protocol"
    memory_save_dir = f"{exploration_dir}/memory"

    # create directories at start of experiment (clear only on iteration 1)
    if iteration == 1:
        if os.path.exists(exploration_dir):
            shutil.rmtree(exploration_dir)
    # always ensure directories exist (for resume support)
    for d in [config_save_dir, montage_save_dir, mlp_save_dir, embedding_save_dir,
              tree_save_dir, protocol_save_dir, memory_save_dir]:
        os.makedirs(d, exist_ok=True)

    is_block_start = (iter_in_block == 1)

    # save config file at first iteration of each block
    if is_block_start:
        src_config = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
        dst_config = f"{config_save_dir}/block_{block_number:03d}.yaml"
        if os.path.exists(src_config):
            shutil.copy2(src_config, dst_config)

    tmp_training_dir = f"{root_dir}/log/{pre_folder}{config_file_}/tmp_training"

    # save montage (most recent Fig_montage_*.tif or Fig_montage_*.png)
    montage_files = glob.glob(f"{tmp_training_dir}/Fig_montage_*.*")
    if montage_files:
        latest = max(montage_files, key=os.path.getmtime)
        ext = os.path.splitext(latest)[1]
        shutil.copy2(latest, f"{montage_save_dir}/iter_{iteration:03d}{ext}")

    # save MLP interaction function plot (from tmp_training)
    mlp_files = glob.glob(f"{tmp_training_dir}/Fig_*_MLP*.png") + glob.glob(f"{tmp_training_dir}/Fig_*_function*.png")
    if mlp_files:
        latest = max(mlp_files, key=os.path.getmtime)
        shutil.copy2(latest, f"{mlp_save_dir}/iter_{iteration:03d}.png")

    # save embedding UMAP plot
    embed_files = glob.glob(f"{tmp_training_dir}/Fig_*_embedding*.png") + glob.glob(f"{tmp_training_dir}/Fig_*_UMAP*.png")
    if not embed_files:
        # also check results dir
        results_dir = f"{root_dir}/log/{pre_folder}{config_file_}/results"
        embed_files = glob.glob(f"{results_dir}/embedding*.png") + glob.glob(f"{results_dir}/UMAP*.png")
    if embed_files:
        latest = max(embed_files, key=os.path.getmtime)
        shutil.copy2(latest, f"{embedding_save_dir}/iter_{iteration:03d}.png")

    # activity path (for passing to Claude prompt)
    activity_path = f"{montage_save_dir}/iter_{iteration:03d}.tif"
    if not os.path.exists(activity_path):
        activity_path = f"{montage_save_dir}/iter_{iteration:03d}.png"

    return {
        'config_save_dir': config_save_dir,
        'montage_save_dir': montage_save_dir,
        'mlp_save_dir': mlp_save_dir,
        'embedding_save_dir': embedding_save_dir,
        'tree_save_dir': tree_save_dir,
        'protocol_save_dir': protocol_save_dir,
        'memory_save_dir': memory_save_dir,
        'activity_path': activity_path,
    }
