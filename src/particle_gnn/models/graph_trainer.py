import glob
import os
import re
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from particle_gnn.models.utils import *
from particle_gnn.utils import *
from particle_gnn.models.Siren_Network import *
from particle_gnn.models.Ghost_Particles import Ghost_Particles
from particle_gnn.sparsify import EmbeddingCluster, sparsify_cluster, clustering_evaluation
from particle_gnn.generators.utils import choose_model
from particle_gnn.fitting_models import linear_model

from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
from torch_geometric.utils import dense_to_sparse
from sklearn import neighbors, metrics
from tqdm import trange
from prettytable import PrettyTable


def data_train(config=None, erase=False, best_model=None, device=None):
    """Route training to the particle training function.

    This simplified version only supports particle-based training.
    """

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    data_train_particle(config, erase, best_model, device)


def data_train_particle(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting

    print(f'training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    time_window = train_config.time_window
    time_step = train_config.time_step
    field_type = model_config.field_type
    omega = model_config.omega

    # noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    recursive_loop = train_config.recursive_loop
    coeff_continuous = train_config.coeff_continuous
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    has_ghost_particles = train_config.n_ghosts > 0
    has_bounding_box = 'PDE_F' in model_config.particle_model_name
    n_ghosts = train_config.n_ghosts
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_ratio = train_config.batch_ratio
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs

    log_dir, logger = create_log_dir(config, erase)
    print(f'graph files N: {n_runs}')
    logger.info(f'graph files N: {n_runs}')
    time.sleep(0.5)
    print('load data ...')
    x_list = []
    y_list = []

    run_lengths = list()
    time.sleep(0.5)
    n_particles_max = 0

    for run in trange(n_runs, ncols=80):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        if np.isnan(x).any() | np.isnan(y).any():
            print('Pb isnan')
        if x[0].shape[0] > n_particles_max:
            n_particles_max = x[0].shape[0]
        x_list.append(x)
        y_list.append(y)
        run_lengths.append(len(x))
    x = torch.tensor(x_list[0][0], dtype=torch.float32, device=device)
    y = torch.tensor(y_list[0][0], dtype=torch.float32, device=device)
    time.sleep(0.5)
    for run in trange(0, n_runs, max(n_runs // 10, 1), ncols=80):
        for k in range(run_lengths[run] - 5):
            if (k % 10 == 0) | (n_frames < 1000):
                try:
                    x = torch.cat((x, torch.tensor(x_list[run][k], dtype=torch.float32, device=device)), 0)
                except:
                    print(f'Error in run {run} frame {k}')
                y = torch.cat((y, torch.tensor(y_list[run][k], dtype=torch.float32, device=device)), 0)
        time.sleep(0.5)
    if torch.isnan(x).any() | torch.isnan(y).any():
        print('Pb isnan')
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)

    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'N particles: {n_particles}')
    logger.info(f'N particles: {n_particles}')
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    x = []
    y = []

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    if (best_model != None) & (best_model != ''):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    if 'PDE_K' in model_config.particle_model_name:
        model.connection_matrix = torch.load(f'graphs_data/{dataset_name}/connection_matrix_list.pt', map_location=device)

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    logger.info(f"total Trainable Params: {n_total_params}")
    logger.info(f'learning rates: {lr}, {lr_embedding}')
    model.train()

    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    x = torch.tensor(x_list[plot_config.data_embedding][0], dtype=torch.float32, device=device)
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N particles: {n_particles} {len(torch.unique(type_list))} types')
    logger.info(f'N particles:  {n_particles} {len(torch.unique(type_list))} types')

    if has_ghost_particles:
        ghosts_particles = Ghost_Particles(config, n_particles, vnorm, device)
        optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    if simulation_config.state_type == 'sequence':
        ind_a = torch.tensor(np.arange(1, n_particles * 100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]

    if field_type != '':
        print('create Siren network')
        has_field = True
        n_nodes_per_axis = int(np.sqrt(n_particles))
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                out_features=model_config.output_size_nnr, hidden_features=model_config.hidden_dim_nnr,
                                hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        optimizer_f = torch.optim.Adam(lr=train_config.learning_rate_NNR, params=model_f.parameters())
        model_f.train()
    else:
        has_field = False

    print("start training particles ...")
    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []

    time.sleep(1)
    for epoch in range(start_epoch, n_epochs + 1):

        logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        batch_size = int(get_batch_size(epoch))
        logger.info(f'batch_size: {batch_size}')

        if (epoch == 1) & (has_ghost_particles):
            mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
            mask_ghost = np.tile(mask_ghost, batch_size)
            mask_ghost = np.argwhere(mask_ghost == 1)
            mask_ghost = mask_ghost[:, 0].astype(int)

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio)
        else:
            Niter = n_frames * data_augmentation_loop // batch_size * (n_runs - 1)
        plot_frequency = int(Niter // 20)

        if epoch == 0:
            print(f'{Niter} iterations per epoch')
            logger.info(f'{Niter} iterations per epoch')
            print(f'plot every {plot_frequency} iterations')

        time.sleep(1)
        total_loss = 0

        for N in trange(Niter, ncols=80):

            if has_field:
                optimizer_f.zero_grad()

            dataset_batch = []
            ids_batch = []
            ids_index = 0
            loss = 0
            for batch in range(batch_size):

                run = 1 + np.random.randint(n_runs - 1)
                k = time_window + np.random.randint(run_lengths[run] - 1 - time_window - time_step - recursive_loop)
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
                if has_field:
                    field = model_f(time=k / n_frames) ** 2
                    x[:, 6:7] = field

                if has_ghost_particles:
                    x_ghost = ghosts_particles.get_pos(dataset_id=run, frame=k, bc_pos=bc_pos)
                    if ghosts_particles.boids:
                        distance = torch.sum(
                            bc_dpos(x_ghost[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                        dist_np = to_numpy(distance)
                        ind_np = torch.min(distance, axis=1)[1]
                        x_ghost[:, 3:5] = x[ind_np, 3:5].clone().detach()
                    x = torch.cat((x, x_ghost), 0)
                    with torch.no_grad():
                        model.a[run, n_particles:n_particles + n_ghosts] = model.a[
                            run, ghosts_particles.embedding_index].clone().detach()  # sample ghost embedding

                edge_index = edges_radius_blockwise(x, dimension, bc_dpos, min_radius, max_radius, block=4096)

                if batch_ratio < 1:
                    ids = np.random.permutation(x.shape[0])[:int(x.shape[0] * batch_ratio)]
                    ids = np.sort(ids)
                    mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                    edges = edges[:, mask]

                if time_window == 0:
                    dataset = data.Data(x=x[:, :], edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)
                else:
                    xt = []
                    for t in range(time_window):
                        x_ = torch.tensor(x_list[run][k - t], dtype=torch.float32, device=device)
                        xt.append(x_[:, :])
                    dataset = data.Data(x=xt, edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)

                if recursive_loop > 0:
                    y = torch.tensor(x_list[run][k + recursive_loop, :, 1:dimension + 1], dtype=torch.float32, device=device).clone().detach()
                elif time_step == 1:
                    y = torch.tensor(y_list[run][k], dtype=torch.float32, device=device).clone().detach() / ynorm
                elif time_step > 1:
                    y = torch.tensor(x_list[run][k + time_step, :, 1:dimension + 1], dtype=torch.float32,
                                     device=device).clone().detach()

                if train_config.shared_embedding:
                    run = 1
                if batch == 0:
                    data_id = torch.ones((y.shape[0], 1), dtype=torch.int) * run
                    x_batch = x
                    y_batch = y
                    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                    if batch_ratio < 1:
                        ids_batch = ids
                else:
                    data_id = torch.cat((data_id, torch.ones((y.shape[0], 1), dtype=torch.int) * run), dim=0)
                    x_batch = torch.cat((x_batch, x), dim=0)
                    y_batch = torch.cat((y_batch, y), dim=0)
                    k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)
                    if batch_ratio < 1:
                        ids_batch = np.concatenate((ids_batch, ids + ids_index), axis=0)

                ids_index += x.shape[0]

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            if has_ghost_particles:
                optimizer_ghost_particles.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=data_id, training=True, k=k_batch, has_field=has_field)

            if recursive_loop > 0:
                for loop in range(recursive_loop):
                    ids_index = 0
                    for batch in range(batch_size):
                        x = dataset_batch[batch].x.clone().detach()

                        X1 = x[:, 1:dimension + 1]
                        V1 = x[:, dimension + 1:2 * dimension + 1]
                        if model_config.prediction == '2nd_derivative':
                            V1 += pred[ids_index:ids_index + x.shape[0]] * ynorm * delta_t
                        else:
                            V1 = pred[ids_index:ids_index + x.shape[0]] * ynorm
                        x[:, 1:dimension + 1] = bc_pos(X1 + V1 * delta_t)
                        x[:, dimension + 1:2 * dimension + 1] = V1
                        dataset_batch[batch].x = x

                        ids_index += x.shape[0]

                    batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                    for batch in batch_loader:
                        pred = model(batch, data_id=data_id, training=True, k=k_batch)

            if has_ghost_particles:
                loss = ((pred[mask_ghost] - y_batch)).norm(2)
            if simulation_config.state_type == 'sequence':
                loss = (pred - y_batch).norm(2)
                loss = loss + train_config.coeff_model_a * (model.a[run, ind_a + 1] - model.a[run, ind_a]).norm(2)
            if (coeff_continuous > 0) & (epoch > 0):
                rr = torch.linspace(0, max_radius, 1000, dtype=torch.float32, device=device)
                for n in np.random.permutation(n_particles)[:n_particles // 100]:
                    embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                    in_features = get_in_features(rr=rr + simulation_config.max_radius / 200, embedding=embedding_,
                                                  model=model, model_name=config.graph_model.particle_model_name,
                                                  max_radius=simulation_config.max_radius)
                    func1 = model.lin_edge(in_features)
                    in_features = get_in_features(rr=rr, embedding=embedding_, model=model,
                                                  model_name=config.graph_model.particle_model_name,
                                                  max_radius=simulation_config.max_radius)
                    func0 = model.lin_edge(in_features)
                    grad = func1 - func0
                    loss = loss + coeff_continuous * grad.norm(2)

            if recursive_loop > 1:
                if batch_ratio < 1:
                    loss = (pred[ids_batch] - y_batch[ids_batch]).norm(2)
                else:
                    loss = (pred - y_batch).norm(2)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            elif time_step == 1:
                if batch_ratio < 1:
                    loss = (pred[ids_batch] - y_batch[ids_batch]).norm(2)
                else:
                    loss = (pred - y_batch).norm(2)
            elif time_step > 1:
                if model_config.prediction == '2nd_derivative':
                    x_pos_pred = x_batch[:, 1:dimension + 1] + delta_t * time_step * (
                                x_batch[:, dimension + 1:2 * dimension + 1] + delta_t * time_step * pred * ynorm)
                else:
                    x_pos_pred = x_batch[:, 1:dimension + 1] + delta_t * time_step * pred * ynorm

                if batch_ratio < 1:
                    loss = loss + (x_pos_pred[ids_batch] - y_batch[ids_batch]).norm(2)
                else:
                    loss = loss + (x_pos_pred - y_batch).norm(2)


            loss.backward()
            optimizer.step()

            if has_field:
                optimizer_f.step()

            if has_ghost_particles:
                optimizer_ghost_particles.step()

            total_loss += loss.item()

            if ((epoch < 30) & (N % plot_frequency == 0)) | (N == 0):
                plot_training(config=config, pred=pred, gt=y_batch, log_dir=log_dir,
                              epoch=epoch, N=N, x=x, model=model, n_nodes=0, n_node_types=0, index_nodes=0,
                              dataset_num=1,
                              index_particles=index_particles, n_particles=n_particles,
                              n_particle_types=n_particle_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                if has_field:
                    torch.save({'model_state_dict': model_f.state_dict(),
                                'optimizer_state_dict': optimizer_f.state_dict()}, os.path.join(log_dir,
                                                                                                'models',
                                                                                                f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

                check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50,
                                       memory_percentage_threshold=0.6)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_particles))
        logger.info("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_particles))
        list_loss.append(total_loss / n_particles)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        scheduler.step()
        print(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}')
        logger.info(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}')

        if has_ghost_particles:
            torch.save({'model_state_dict': ghosts_particles.state_dict(),
                        'optimizer_state_dict': optimizer_ghost_particles.state_dict()},
                       os.path.join(log_dir, 'models', f'best_ghost_particles_with_{n_runs - 1}_graphs_{epoch}.pt'))

        fig = plt.figure(figsize=(22, 5))
        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='k')

        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)

        if ('PDE_T' not in model_config.particle_model_name) & ('PDE_K' not in model_config.particle_model_name) & (
                'PDE_MLPs' not in model_config.particle_model_name) & (
                'PDE_F' not in model_config.particle_model_name) & ('PDE_M' not in model_config.particle_model_name) & (
                has_bounding_box == False):

            ax = fig.add_subplot(1, 5, 2)
            embedding = get_embedding(model.a, 1)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0],
                            embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)

            ax = fig.add_subplot(1, 5, 3)
            func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config,
                                                                model_MLP=model.lin_edge, model=model,
                                                                n_nodes=0,
                                                                n_particles=n_particles, ynorm=ynorm,
                                                                type_list=to_numpy(x[:, 1 + 2 * dimension]),
                                                                cmap=cmap, update_type='NA', device=device)

            labels, n_clusters, new_labels = sparsify_cluster(train_config.cluster_method, proj_interaction, embedding,
                                                              train_config.cluster_distance_threshold, type_list,
                                                              n_particle_types, embedding_cluster)

            accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
            print(f'accuracy: {np.round(accuracy, 3)}   n_clusters: {n_clusters}')
            logger.info(f'accuracy: {np.round(accuracy, 3)}    n_clusters: {n_clusters}')

            ax = fig.add_subplot(1, 5, 4)
            for n in np.unique(new_labels):
                pos = np.array(np.argwhere(new_labels == n).squeeze().astype(int))
                if pos.size > 0:
                    plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=5)
            plt.xlabel('proj 0', fontsize=12)
            plt.ylabel('proj 1', fontsize=12)
            plt.text(0, 1.1, f'accuracy: {np.round(accuracy, 3)},  {n_clusters} clusters', ha='left', va='top',
                     transform=ax.transAxes, fontsize=10)

            ax = fig.add_subplot(1, 5, 5)
            model_a_ = model.a[1].clone().detach()
            for n in range(n_clusters):
                pos = np.argwhere(labels == n).squeeze().astype(int)
                pos = np.array(pos)
                if pos.size > 0:
                    median_center = model_a_[pos, :]
                    median_center = torch.median(median_center, dim=0).values
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=1, c='r', alpha=0.25)
                    model_a_[pos, :] = median_center
                    plt.scatter(to_numpy(model_a_[pos, 0]), to_numpy(model_a_[pos, 1]), s=10, c='k')

            plt.xlabel('ai0', fontsize=12)
            plt.ylabel('ai1', fontsize=12)
            plt.xticks(fontsize=10.0)
            plt.yticks(fontsize=10.0)

            if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq - 1) & (
                    epoch < n_epochs - sparsity_freq):
                # Constrain embedding domain
                with torch.no_grad():
                    model.a[1] = model_a_.clone().detach()
                print(f'regul_embedding: replaced')
                logger.info(f'regul_embedding: replaced')

                # Constrain function domain
                if train_config.sparsity == 'replace_embedding_function':

                    logger.info(f'replace_embedding_function')
                    y_func_list = func_list * 0

                    ax, fig = fig_init()
                    for n in np.unique(new_labels):
                        pos = np.argwhere(new_labels == n)
                        pos = pos.squeeze()
                        if pos.size > 0:
                            target_func = torch.median(func_list[pos, :], dim=0).values.squeeze()
                            y_func_list[pos] = target_func
                        plt.plot(to_numpy(target_func) * to_numpy(ynorm), linewidth=2, alpha=1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}_before training function.tif")
                    plt.close()

                    lr_embedding = 1E-12
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    for sub_epochs in range(20):
                        loss = 0
                        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                        pred = []
                        optimizer.zero_grad()
                        for n in range(n_particles):
                            embedding_ = model.a[1, n, :].clone().detach() * torch.ones(
                                (1000, model_config.embedding_dim), device=device)
                            match model_config.particle_model_name:
                                case 'PDE_ParticleField_A' | 'PDE_A':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                                case 'PDE_ParticleField_B' | 'PDE_B':
                                    in_features = torch.cat(
                                        (rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                case 'PDE_G':
                                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                             rr[:, None] / max_radius, 0 * rr[:, None],
                                                             0 * rr[:, None],
                                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                                case 'PDE_E':
                                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                             rr[:, None] / max_radius, embedding_, embedding_), dim=1)
                                case 'PDE_K':
                                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                             rr[:, None] / max_radius), dim=1)
                            pred.append(model.lin_edge(in_features.float()))
                        pred = torch.stack(pred)
                        loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                        logger.info(f'    loss: {np.round(loss.item() / n_particles, 3)}')
                        loss.backward()
                        optimizer.step()

                if train_config.fix_cluster_embedding:
                    lr_embedding = 1E-12
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

            else:
                if epoch > n_epochs - sparsity_freq:
                    lr_embedding = train_config.learning_rate_embedding_end
                    lr = train_config.learning_rate_end
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')
                else:
                    lr_embedding = train_config.learning_rate_embedding_start
                    lr = train_config.learning_rate_start
                    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                    logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20,
              step=15, ratio=1, run=1, test_mode='', sample_embedding=False, particle_of_interest=1, device=[]):
    """Route testing to the particle testing function.

    This simplified version only supports particle-based testing.
    """

    data_test_particle(config, config_file, visualize, style, True, best_model, step, ratio, run, test_mode,
                       sample_embedding, particle_of_interest, device)


def data_test_particle(config=None, config_file=None, visualize=False, style='color frame', verbose=True,
                       best_model=20, step=15, ratio=1, run=1, test_mode='', sample_embedding=False,
                       particle_of_interest=1, device=[]):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_ghost_particles = config.training.n_ghosts > 0
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    time_window = training_config.time_window
    time_step = training_config.time_step
    sub_sampling = simulation_config.sub_sampling
    cmap = CustomColorMap(config=config)
    dimension = simulation_config.dimension
    field_type = model_config.field_type
    has_field = (field_type != '')
    omega = model_config.omega

    do_tracking = training_config.do_tracking
    has_state = (config.simulation.state_type != 'discrete')
    has_bounding_box = 'PDE_F' in model_config.particle_model_name

    if has_field:
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))

    log_dir = 'log/' + config.config_file
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')
    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"

    n_sub_population = n_particles // n_particle_types
    first_cell_id_particles = []
    for n in range(n_particle_types):
        index = np.arange(n_particles * n // n_particle_types, n_particles * (n + 1) // n_particle_types)
        first_cell_id_particles.append(index)

    print(f'load data run {run} ...')

    x_list = []
    y_list = []

    if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
        x = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
        ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
        vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)
        if vnorm == 0:
            vnorm = ynorm

    else:

        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        y = torch.tensor(y, dtype=torch.float32, device=device)
        x_list.append(x)
        y_list.append(y)
        x = x_list[0][0].clone().detach()
        if ('PDE_MLPs' not in model_config.particle_model_name) & ('PDE_F' not in model_config.particle_model_name) & ('PDE_M' not in model_config.particle_model_name):
            n_particles = int(x.shape[0] / ratio)
            config.simulation.n_particles = n_particles
        n_frames = len(x_list[0])
        index_particles = get_index_particles(x, n_particle_types, dimension)
        if n_particle_types > 1000:
            index_particles = []
            for n in range(3):
                index = np.arange(n_particles * n // 3, n_particles * (n + 1) // 3)
                index_particles.append(index)
                n_particle_types = 3
        ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
        vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)
        if vnorm == 0:
            vnorm = ynorm

    if do_tracking | has_state:
        for k in range(len(x_list[0])):
            type = x_list[0][k][:, 2 * dimension + 1]
            if k == 0:
                type_list = type
            else:
                type_list = torch.concatenate((type_list, type))
        n_particles_max = len(type_list) + 1
        config.simulation.n_particles_max = n_particles_max

    if ratio > 1:
        new_nparticles = int(n_particles * ratio)
        model.a = nn.Parameter(
            torch.tensor(np.ones((n_runs, int(new_nparticles), 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        n_particles = new_nparticles
        index_particles = get_index_particles(x, n_particle_types, dimension)
    if sample_embedding:
        model_a_ = nn.Parameter(
            torch.tensor(np.ones((int(n_particles), model.embedding_dim)), device=device, requires_grad=False,
                         dtype=torch.float32))
        for n in range(n_particles):
            t = to_numpy(x[n, 5]).astype(int)
            index = first_cell_id_particles[t][np.random.randint(n_sub_population)]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((model.n_dataset, int(n_particles), model.embedding_dim)), device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_
    if has_ghost_particles:
        model_ghost = Ghost_Particles(config, n_particles, vnorm, device)
        net_ghost = f"{log_dir}/models/best_ghost_particles_with_{n_runs - 1}_graphs_20.pt"
        state_dict = torch.load(net_ghost, map_location=device)
        model_ghost.load_state_dict(state_dict['model_state_dict'])
        model_ghost.eval()
        x_removed_list = torch.load(f'graphs_data/{dataset_name}/x_removed_list_0.pt', map_location=device)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)

    # Create model and load weights
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    model.particle_of_interest = particle_of_interest

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    state_dict = torch.load(net, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    if 'PDE_K' in model_config.particle_model_name:
        model.connection_matrix = torch.load(f'graphs_data/{dataset_name}/connection_matrix_list.pt', map_location=device)
        timeit = np.load(f'graphs_data/{dataset_name}/times_train_springs_example.npy',
                         allow_pickle=True)
        timeit = timeit[run][0]
        time_id = 0

    if has_field:
        n_nodes_per_axis = int(np.sqrt(simulation_config.n_nodes))
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=model_config.input_size_nnr,
                                out_features=model_config.output_size_nnr,
                                hidden_features=model_config.hidden_dim_nnr,
                                hidden_layers=model_config.n_layers_nnr, outermost_linear=True, device=device,
                                first_omega_0=model_config.omega, hidden_omega_0=model_config.omega)
        net_f = f'{log_dir}/models/best_model_f_with_1_graphs_{best_model}.pt'
        state_dict = torch.load(net_f, map_location=device)
        model_f.load_state_dict(state_dict['model_state_dict'])
        model_f.to(device=device)
        model_f.eval()
        table_f = PrettyTable(["Modules", "Parameters"])
        total_params_f = 0
        for name, parameter in model_f.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table_f.add_row([name, param])
            total_params_f += param
        if verbose:
            print(table_f)
            print(f"Total Trainable Params: {total_params_f}")

    if verbose:
        print(f'test data ... {model_config.particle_model_name} {model_config.mesh_model_name}')
        print('log_dir: {}'.format(log_dir))
        print(f'network: {net}')
        print(table)
        print(f"total trainable Params: {total_params}")

    if 'test_simulation' in 'test_mode':
        model, bc_pos, bc_dpos = choose_model(config, device=device)

    rmserr_list = []
    pred_err_list = []
    gloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    geomloss_list = []
    angle_list = []
    time.sleep(1)

    if time_window > 0:
        start_it = time_window
        stop_it = n_frames - 1
    else:
        start_it = 0
        stop_it = n_frames - 1

    start_it = 0

    x = x_list[0][start_it].clone().detach()
    n_particles = x.shape[0]
    x_inference_list = []

    for it in trange(start_it, start_it + 800):

        check_and_clear_memory(device=device, iteration_number=it, every_n_iterations=25,
                               memory_percentage_threshold=0.6)

        if it < n_frames - 4:
            x0 = x_list[0][it].clone().detach()
            x0_next = x_list[0][(it + time_step)].clone().detach()
            if not (model_config.particle_model_name == 'PDE_R'):
                y0 = y_list[0][it].clone().detach()

        if do_tracking:
            x = x0.clone().detach()

        # error calculations
        if has_bounding_box:
            rmserr = torch.sqrt(
                torch.mean(torch.sum(bc_dpos(x[:, 1:dimension + 1] - x0[:, 1:dimension + 1]) ** 2, axis=1)))
        else:
            if (do_tracking) | (x.shape[0] != x0.shape[0]):
                rmserr = torch.zeros(1, device=device)
            else:
                rmserr = torch.sqrt(
                    torch.mean(torch.sum(bc_dpos(x[:, 1:dimension + 1] - x0[:, 1:dimension + 1]) ** 2, axis=1)))
            if x.shape[0] > 5000:
                geomloss = gloss(x[0:5000, 1:3], x0[0:5000, 1:3])
            else:
                geomloss = gloss(x[:, 1:3], x0[:, 1:3])
            geomloss_list.append(geomloss.item())
        rmserr_list.append(rmserr.item())

        if config.training.shared_embedding:
            data_id = torch.ones((n_particles, 1), dtype=torch.int, device=device)
        else:
            data_id = torch.ones((n_particles, 1), dtype=torch.int, device=device) * run

        # update calculations
        with torch.no_grad():
            if has_ghost_particles:
                x_ = x
                x_ghost = model_ghost.get_pos(dataset_id=run, frame=it, bc_pos=bc_pos)
                x_ = torch.cat((x_, x_ghost), 0)
            # compute connectivity and prediction

            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            if has_field:
                field = model_f(time=it / n_frames) ** 2
                x[:, 6:7] = field

            if time_window > 0:
                xt = []
                for t in range(time_window):
                    x_ = x_list[0][it - t].clone().detach()
                    xt.append(x_[:, :])
                dataset = data.Data(x=xt, edge_index=edge_index)
            else:
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

            if 'test_simulation' in test_mode:
                y = y0 / ynorm
                pred = y
            else:
                pred = model(dataset, data_id=data_id, training=False, has_field=has_field, k=it)
                y = pred

            if has_ghost_particles:
                y = y[mask_ghost]

            if sub_sampling > 1:
                # predict position, does not work with rotation_augmentation
                if time_step == 1:
                    x_next = bc_pos(y[:, 0:dimension])
                elif time_step == 2:
                    x_next = bc_pos(y[:, dimension:2 * dimension])
                x[:, dimension + 1:2 * dimension + 1] = (x_next - x[:, 1:dimension + 1]) / delta_t
                x[:, 1:dimension + 1] = x_next
                loss = (x[:, 1:dimension + 1] - x0_next[:, 1:dimension + 1]).norm(2)
                pred_err_list.append(to_numpy(torch.sqrt(loss)))
            elif do_tracking:
                x_pos_next = x0_next[:, 1:dimension + 1].clone().detach()
                if pred.shape[1] != dimension:
                    pred = torch.cat((pred, torch.zeros(pred.shape[0], 1, device=pred.device)), dim=1)
                if model_config.prediction == '2nd_derivative':
                    x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * (
                                x[:, dimension + 1:2 * dimension + 1] + delta_t * time_step * pred * ynorm))
                else:
                    x_pos_pred = (x[:, 1:dimension + 1] + delta_t * time_step * pred * ynorm)
                distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_value = result.values
                indices = result.indices
                loss = torch.std(torch.sqrt(min_value))
                pred_err_list.append(to_numpy(torch.sqrt(loss)))
                if 'inference' in test_mode:
                    x[:, dimension + 1:2 * dimension + 1] = pred.clone().detach() / (delta_t * time_step)

            else:
                if model_config.prediction == '2nd_derivative':
                    y = y * ynorm * delta_t
                    x[:n_particles, dimension + 1:2 * dimension + 1] = x[:n_particles, dimension + 1:2 * dimension + 1] + y[:n_particles]  # speed update
                else:
                    y = y * vnorm
                    x[:n_particles, dimension + 1:2 * dimension + 1] = y[:n_particles]
                x[:, 1:dimension + 1] = bc_pos(
                    x[:, 1:dimension + 1] + x[:, dimension + 1:2 * dimension + 1] * delta_t)  # position update

            if 'inference' in test_mode:
                x_inference_list.append(x)

            if (time_window > 1) & ('plot_data' not in test_mode):
                moving_pos = torch.argwhere(x[:, 5] != 0)
                x_list[0][it + 1, moving_pos.squeeze(), 1:2 * dimension + 1] = x[moving_pos.squeeze(),
                                                                               1:2 * dimension + 1].clone().detach()

        # vizualization
        if 'plot_data' in test_mode:
            x = x_list[0][it].clone().detach()

        if (it % step == 0) & (it >= 0) & visualize:

            num = f"{it:06}"

            if 'latex' in style:
                plt.rcParams['text.usetex'] = True
                rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            if 'black' in style:
                plt.style.use('dark_background')
                mc = 'w'
            else:
                plt.style.use('default')
                mc = 'k'

            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            ax.tick_params(axis='both', which='major', pad=15)

            if 'PDE_K' in model_config.particle_model_name:

                plt.close()
                fig = plt.figure(figsize=(12, 12))
                plt.scatter(x[:, 2].detach().cpu().numpy(),
                            x[:, 1].detach().cpu().numpy(), s=20, color='r')
                if it < n_frames - 1:
                    x0_ = x_list[0][it + 1].clone().detach()
                    plt.scatter(x0_[:, 2].detach().cpu().numpy(),
                                x0_[:, 1].detach().cpu().numpy(), s=40, color='w', alpha=1, edgecolors='None')

                plt.xlim([-3, 3])
                plt.ylim([-3, 3])

            elif do_tracking:

                plt.scatter(to_numpy(x0[:, 2]), to_numpy(x0[:, 1]), s=10, c='w', alpha=0.5)
                plt.scatter(to_numpy(x_pos_pred[:, 1]), to_numpy(x_pos_pred[:, 0]), s=10, c='r')
                x1 = x_list[0][it + time_step].clone().detach()
                plt.scatter(to_numpy(x1[:, 2]), to_numpy(x1[:, 1]), s=10, c='g')

                plt.xticks([])
                plt.yticks([])

                if 'zoom' in style:
                    for m in range(x.shape[0]):
                        plt.arrow(x=to_numpy(x0[m, 2]), y=to_numpy(x0[m, 1]),
                                  dx=to_numpy(x[m, dimension + 2]) * delta_t,
                                  dy=to_numpy(x[m, dimension + 1]) * delta_t, head_width=0.004,
                                  length_includes_head=True, color='g')
                    plt.xlim([300, 400])
                    plt.ylim([300, 400])
                else:
                    plt.xlim([0, 700])
                    plt.ylim([0, 700])
                plt.tight_layout()

            else:
                s_p = 10
                index_particles = get_index_particles(x, n_particle_types, dimension)
                for n in range(n_particle_types):
                    if 'bw' in style:
                        plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                                    x[index_particles[n], 1].detach().cpu().numpy(), s=s_p, color='w')
                    else:
                        plt.scatter(x[index_particles[n], 2].detach().cpu().numpy(),
                                    x[index_particles[n], 1].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
                plt.xlim([0, 1])
                plt.ylim([0, 1])

                if ('field' in style) & has_field:
                    if 'zoom' in style:
                        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=s_p * 50, c=to_numpy(x[:, 6]) * 20,
                                    alpha=0.5, cmap='viridis', vmin=0, vmax=1.0)
                    else:
                        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=s_p * 2, c=to_numpy(x[:, 6]) * 20,
                                    alpha=0.5, cmap='viridis', vmin=0, vmax=1.0)

                if particle_of_interest > 1:

                    xc = to_numpy(x[particle_of_interest, 2])
                    yc = to_numpy(x[particle_of_interest, 1])
                    pos = torch.argwhere(edge_index[1, :] == particle_of_interest)
                    pos = pos[:, 0]
                    if 'zoom' in style:
                        plt.scatter(to_numpy(x[edge_index[0, pos], 2]), to_numpy(x[edge_index[0, pos], 1]), s=s_p * 10,
                                    color=mc, alpha=1.0)
                    else:
                        plt.scatter(to_numpy(x[edge_index[0, pos], 2]), to_numpy(x[edge_index[0, pos], 1]), s=s_p * 1,
                                    color=mc, alpha=1.0)

                    plt.arrow(x=to_numpy(x[particle_of_interest, 2]), y=to_numpy(x[particle_of_interest, 1]),
                              dx=to_numpy(x[particle_of_interest, 4]) * delta_t * 100,
                              dy=to_numpy(x[particle_of_interest, 3]) * delta_t * 100, head_width=0.004,
                              length_includes_head=True, color='b')
                    if model_config.prediction == '2nd_derivative':
                        plt.arrow(x=to_numpy(x[particle_of_interest, 2]), y=to_numpy(x[particle_of_interest, 1]),
                                  dx=to_numpy(y0[particle_of_interest, 1]) * delta_t ** 2 * 100,
                                  dy=to_numpy(y0[particle_of_interest, 0]) * delta_t ** 2 * 100, head_width=0.004,
                                  length_includes_head=True, color='g')
                        plt.arrow(x=to_numpy(x[particle_of_interest, 2]), y=to_numpy(x[particle_of_interest, 1]),
                                  dx=to_numpy(y[particle_of_interest, 1]) * delta_t * 100,
                                  dy=to_numpy(y[particle_of_interest, 0]) * delta_t * 100, head_width=0.004,
                                  length_includes_head=True, color='r')

                if 'zoom' in style:
                    plt.xlim([xc - 0.1, xc + 0.1])
                    plt.ylim([yc - 0.1, yc + 0.1])
                    plt.xticks([])
                    plt.yticks([])

            if 'latex' in style:
                plt.xlabel(r'$x$', fontsize=78)
                plt.ylabel(r'$y$', fontsize=78)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
            if 'frame' in style:
                plt.xlabel('x', fontsize=48)
                plt.ylabel('y', fontsize=48)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
                plt.text(0, 1.1, f'   ', ha='left', va='top', transform=ax.transAxes, fontsize=48)
                ax.tick_params(axis='both', which='major', pad=15)
            if 'arrow' in style:
                for m in range(x.shape[0]):
                    if x[m, 4] != 0:
                        if 'speed' in style:
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(x[m, 4]) * delta_t * 2,
                                      dy=to_numpy(x[m, 3]) * delta_t * 2, head_width=0.004, length_includes_head=True,
                                      color='g')
                        if 'acc_true' in style:
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]), dx=to_numpy(y0[m, 1]) / 5E3,
                                      dy=to_numpy(y0[m, 0]) / 5E3, head_width=0.004, length_includes_head=True,
                                      color='r')
                        if 'acc_learned' in style:
                            plt.arrow(x=to_numpy(x[m, 2]), y=to_numpy(x[m, 1]),
                                      dx=to_numpy(pred[m, 1] * ynorm.squeeze()) / 5E3,
                                      dy=to_numpy(pred[m, 0] * ynorm.squeeze()) / 5E3, head_width=0.004,
                                      length_includes_head=True, color='r')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
            if 'name' in style:
                plt.title(f"{os.path.basename(log_dir)}", fontsize=24)
            if 'no_ticks' in style:
                plt.xticks([])
                plt.yticks([])
            if 'PDE_G' in model_config.particle_model_name:
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
            if 'PDE_GS' in model_config.particle_model_name:

                object_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune',
                               'pluto', 'io',
                               'europa', 'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea',
                               'titan', 'hyperion', 'moon',
                               'phobos', 'deimos', 'charon']

                masses = torch.tensor(
                    [1.989e30, 3.30e23, 4.87e24, 5.97e24, 6.42e23, 1.90e27, 5.68e26, 8.68e25, 1.02e26, 1.31e22,
                     8.93e22, 4.80e22, 1.48e23, 1.08e23, 3.75e19, 1.08e20,
                     6.18e20, 1.10e21, 2.31e21, 1.35e23, 5.62e18, 7.35e22, 1.07e16, 1.48e15, 1.52e21],
                    device=device)

                pos = x[:, 1:dimension + 1]
                distance = torch.sqrt(torch.sum(bc_dpos(pos[:, None, :] - pos[None, 0, :]) ** 2, dim=2))
                unit_vector = pos / distance

                if it == 0:
                    log_coeff = torch.log(distance[1:])
                    log_coeff_min = torch.min(log_coeff)
                    log_coeff_max = torch.max(log_coeff)
                    log_coeff_edge_diff = log_coeff_max - log_coeff_min
                    d_log = [log_coeff_min, log_coeff_max, log_coeff_edge_diff]

                    log_coeff = torch.log(masses)
                    log_coeff_min = torch.min(log_coeff)
                    log_coeff_max = torch.max(log_coeff)
                    log_coeff_edge_diff = log_coeff_max - log_coeff_min
                    m_log = [log_coeff_min, log_coeff_max, log_coeff_edge_diff]

                    m_ = torch.log(masses) / m_log[2]

                distance_ = (torch.log(distance) - d_log[0]) / d_log[2]
                pos = distance_ * unit_vector
                pos = to_numpy(pos)
                pos[0] = 0

                for n in range(25):
                    plt.scatter(pos[n, 1], pos[n, 0], s=200 * to_numpy(m_[n] ** 3), color=cmap.color(n))
                plt.xlim([-1.2, 1.2])
                plt.ylim([-1.2, 1.2])
                plt.xticks([])
                plt.yticks([])
            if 'PDE_K' in model_config.particle_model_name:
                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)

            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{config_file}_{run}_{num}.tif", dpi=100)
            plt.close()

            if ('feature' in style) & ('PDE_MLPs_A' in config.graph_model.particle_model_name):
                if 'PDE_MLPs_A_bis' in model.model:
                    fig = plt.figure(figsize=(22, 5))
                else:
                    fig = plt.figure(figsize=(22, 6))
                for k in range(model.new_features.shape[1]):
                    ax = fig.add_subplot(1, model.new_features.shape[1], k + 1)
                    plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c=to_numpy(model.new_features[:, k]), s=5,
                                cmap='viridis')
                    ax.set_title(f'new_features {k}')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Features_{config_file}_{run}_{num}.tif", dpi=100)
                plt.close()

            if 'boundary' in style:
                fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                t = torch.min(x[:, 7:], -1).values
                plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=25, c=to_numpy(t), vmin=-1, vmax=1)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/Boundary_{config_file}_{num}.tif", dpi=80)
                plt.close()
