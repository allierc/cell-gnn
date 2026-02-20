import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch
import numpy as np
import torch.nn as nn
import umap

from particle_gnn.models.Interaction_Particle import Interaction_Particle
from particle_gnn.models.Interaction_Particle_Field import Interaction_Particle_Field
from particle_gnn.models.MLP import MLP
from particle_gnn.utils import to_numpy, fig_init, choose_boundary_values


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


def get_embedding(model_a=None, dataset_number=0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    return embedding


def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    indexes = np.arange(n_frames) * n_particles + cell_id

    return embedding[indexes]


def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):

    indexes = np.arange(n_frames) * n_particles + cell_id

    return new_labels[indexes]


def get_in_features(rr=None, embedding=None, model=[], model_name=[], max_radius=[]):

    match model_name:
        case 'PDE_A' | 'PDE_ParticleField_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_A_bis':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_GS':
            in_features = torch.cat(
                (rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, 10 ** embedding), dim=1)
        case 'PDE_G':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)

    return in_features


def choose_training_model(model_config=None, device=None):
    """Create and return a model based on the configuration.

    Args:
        model_config: Configuration object containing simulation and graph model parameters.
        device: Torch device to place the model on.

    Returns:
        Tuple of (model, bc_pos, bc_dpos).
    """

    aggr_type = model_config.graph_model.aggr_type
    dimension = model_config.simulation.dimension

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    match model_config.graph_model.particle_model_name:
        case 'PDE_ParticleField_A' | 'PDE_ParticleField_B':
            model = Interaction_Particle_Field(
                aggr_type=aggr_type,
                config=model_config,
                device=device,
                bc_dpos=bc_dpos,
                dimension=dimension,
            )
        case _:
            model = Interaction_Particle(
                aggr_type=aggr_type,
                config=model_config,
                device=device,
                bc_dpos=bc_dpos,
                dimension=dimension,
            )
    model.edges = []

    return model, bc_pos, bc_dpos


def get_type_list(x, dimension):
    type_list = x[:, 1 + 2 * dimension:2 + 2 * dimension].clone().detach()
    return type_list


def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size
    return get_batch_size


def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size
    return get_batch_size


def set_trainable_parameters(model=[], lr_embedding=[], lr=[], lr_update=[], lr_W=[], lr_modulation=[], learning_rate_NNR=[], learning_rate_edge_embedding=[]):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.a)

    if lr_update == []:
        lr_update = lr

    optimizer = torch.optim.Adam([model.a], lr=lr_embedding)
    for name, parameter in model.named_parameters():
        if (parameter.requires_grad) & (name != 'a'):
            if (name == 'b') or ('lin_modulation' in name):
                optimizer.add_param_group({'params': parameter, 'lr': lr_modulation})
            elif 'lin_phi' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_update})
            elif 'W' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_W})
            elif 'NNR' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_NNR})
            elif 'edges_embedding' in name:
                optimizer.add_param_group({'params': parameter, 'lr': learning_rate_edge_embedding})
            else:
                optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params


def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model=None, n_nodes=0, n_particles=None, ynorm=None, type_list=None, cmap=None, update_type=None, device=None):

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    dimension = config.simulation.dimension

    config_model = config.graph_model.particle_model_name

    if rr == []:
        if config_model == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif config_model == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    print('interaction functions ...')
    func_list = []
    for n in range(n_particles):

        if len(model.a.shape) == 3:
            model_a = model.a[1, n, :]
        else:
            model_a = model.a[n, :]

        if config.training.do_tracking:
            embedding_ = model_a * torch.ones((1000, dimension), device=device)
        else:
            if (update_type != 'NA') & model.embedding_trial:
                embedding_ = torch.cat((model_a, model.b[0].clone().detach().repeat(n_particles, 1)), dim=1) * torch.ones((1000, 2 * dimension), device=device)
            else:
                embedding_ = model_a * torch.ones((1000, dimension), device=device)

        if update_type == 'NA':
            in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config_model, max_radius=max_radius)
        else:
            in_features = get_in_features_update(rr=rr[:, None], embedding=embedding_, model=model, device=device)
        with torch.no_grad():
            func = model_MLP(in_features.float())[:, 0]

        func_list.append(func)

        should_plot = vizualize and (
                n_particles <= 200 or
                (n % (n_particles // 200) == 0) or
                (config.graph_model.particle_model_name == 'PDE_GS')
        )

        if should_plot:
            plt.plot(
                to_numpy(rr),
                to_numpy(func) * to_numpy(ynorm),
                2,
                color=cmap.color(type_list[n].astype(int)),
                linewidth=1,
                alpha=0.25
            )

    func_list = torch.stack(func_list)
    func_list_ = to_numpy(func_list)

    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_GS':
            plt.xscale('log')
            plt.yscale('log')
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xlim([1E-3, 0.02])
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


def plot_training_particle_field(config, has_siren, has_siren_time, model_f, n_frames, model_name, log_dir, epoch, N, x, x_mesh, index_particles, n_neurons, n_neuron_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = simulation_config.max_radius

    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))

    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
    else:
        plt.axis('off')
    embedding = get_embedding(model.a, dataset_num)
    if n_neuron_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_neurons, s=1, cmap='viridis')
    else:
        for n in range(n_neuron_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=1)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{model_name}_embedding_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, simulation_config.max_radius])
        plt.tight_layout()

    match model_config.particle_model_name:
        case 'PDE_ParticleField_A':
            rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 200)).to(device)
        case 'PDE_ParticleField_B':
            rr = torch.tensor(np.linspace(-max_radius, max_radius, 200)).to(device)
    for n in range(n_neurons):
        embedding_ = model.a[dataset_num, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
        match model_config.particle_model_name:
            case 'PDE_ParticleField_A':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_ParticleField_B':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        if n % 5 == 0:
            plt.plot(to_numpy(rr),
                     to_numpy(func * ynorm),
                     linewidth=8,
                     color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/MLP1/{model_name}_function_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

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
            fig_ = plt.figure(figsize=(14, 12))
            axf = fig_.add_subplot(1, 1, 1)
            plt.imshow(tmp, cmap='grey')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/external_input/{model_name}_{epoch}_{N}_{frame}.tif", dpi=80)
            plt.close()
