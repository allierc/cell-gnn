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

from cell_gnn.models.utils import *
from cell_gnn.models.utils import LossRegularizer
from cell_gnn.utils import *
from cell_gnn.models.Siren_Network import *
from cell_gnn.plot import (
    plot_training, plot_training_cell_field,
    get_embedding, build_edge_features, batched_sparsity_mlp_eval,
    plot_training_summary_panels, plot_loss_components,
)
from cell_gnn.sparsify import EmbeddingCluster
from cell_gnn.generators.utils import choose_model
from cell_gnn.fitting_models import linear_model
from cell_gnn.cell_state import CellState, CellTimeSeries, FieldState, FieldTimeSeries
from cell_gnn.zarr_io import load_simulation_data, load_field_data, load_raw_array

from geomloss import SamplesLoss
from scipy.optimize import curve_fit
from cell_gnn.graph_utils import GraphData, collate_graph_batch
from tqdm import trange
from prettytable import PrettyTable


def data_train(config=None, erase=False, best_model=None, device=None):
    """Route training to the appropriate training function."""

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    has_cell_field = 'field_ode' in config.graph_model.cell_model_name

    if has_cell_field:
        data_train_cell_field(config, erase, best_model, device)
    else:
        data_train_cell(config, erase, best_model, device)


def data_train_cell(config, erase, best_model, device):
    sim = config.simulation
    tc = config.training
    mc = config.graph_model
    pc = config.plotting

    print(f'training data ... {mc.cell_model_name}')

    dimension = sim.dimension
    n_cells = sim.n_cells
    n_cell_types = sim.n_cell_types
    n_frames = sim.n_frames
    max_radius = sim.max_radius
    min_radius = sim.min_radius
    delta_t = sim.delta_t
    n_epochs = tc.n_epochs
    time_window = tc.time_window
    time_step = tc.time_step
    data_augmentation_loop = tc.data_augmentation_loop
    recursive_loop = tc.recursive_loop
    batch_ratio = tc.batch_ratio
    sparsity_freq = tc.sparsity_freq
    dataset_name = config.dataset
    field_type = mc.field_type
    omega = mc.omega

    target_batch_size = tc.batch_size
    replace_with_cluster = 'replace' in tc.sparsity
    if tc.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)
    embedding_cluster = EmbeddingCluster(config)
    n_runs = tc.n_runs

    log_dir, logger = create_log_dir(config, erase)
    time.sleep(0.5)
    print('load data ...')

    x_ts = load_simulation_data(f'graphs_data/{dataset_name}/x_list_0', dimension)
    y_raw = load_raw_array(f'graphs_data/{dataset_name}/y_list_0')
    n_cells_max = x_ts.n_cells
    n_ts_frames = x_ts.n_frames

    # compute normalization from sampled frames
    x = x_ts.frame(0).to_packed().to(device)
    y = torch.tensor(y_raw[0], dtype=torch.float32, device=device)
    time.sleep(0.5)
    for k in trange(n_ts_frames - 5, ncols=100):
        if (k % 10 == 0) | (n_frames < 1000):
            try:
                x = torch.cat((x, x_ts.frame(k).to_packed().to(device)), 0)
            except:
                print(f'error in frame {k}')
            y = torch.cat((y, torch.tensor(y_raw[k], dtype=torch.float32, device=device)), 0)
    time.sleep(0.5)
    if torch.isnan(x).any() | torch.isnan(y).any():
        print('Pb isnan')
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)

    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'N cells: {n_cells}')
    logger.info(f'N cells: {n_cells}')
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
    lr = tc.learning_rate_start
    lr_embedding = tc.learning_rate_embedding_start
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

    x_plot = x_ts.frame(0).to_packed().to(device)
    index_cells = get_index_cells(x_plot, n_cell_types, dimension)
    type_list = get_type_list(x_plot, dimension)
    print(f'N cells: {n_cells} {len(torch.unique(type_list))} types')
    logger.info(f'N cells:  {n_cells} {len(torch.unique(type_list))} types')

    if sim.state_type == 'sequence':
        ind_a = torch.tensor(np.arange(1, n_cells * 100), device=device)
        pos = torch.argwhere(ind_a % 100 != 99).squeeze()
        ind_a = ind_a[pos]

    if field_type != '':
        print('create Siren network')
        has_field = True
        n_nodes_per_axis = int(np.sqrt(n_cells))
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=mc.input_size_nnr,
                                out_features=mc.output_size_nnr, hidden_features=mc.hidden_dim_nnr,
                                hidden_layers=mc.n_layers_nnr, outermost_linear=True, device=device,
                                first_omega_0=omega, hidden_omega_0=omega)
        model_f.to(device=device)
        optimizer_f = torch.optim.Adam(lr=tc.learning_rate_nnr, params=model_f.parameters())
        model_f.train()
    else:
        has_field = False

    print("start training cells ...")
    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    loss_dict = {'loss': []}
    regularizer = LossRegularizer(tc, mc, sim, n_cells, plot_frequency=1)

    time.sleep(1)
    for epoch in range(start_epoch, n_epochs):

        logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        batch_size = int(get_batch_size(epoch))
        logger.info(f'batch_size: {batch_size}')

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio)
        else:
            Niter = n_frames * data_augmentation_loop // batch_size
        plot_frequency = int(Niter // 20)

        if epoch == 0:
            print(f'{Niter} iterations per epoch')
            logger.info(f'{Niter} iterations per epoch')
            print(f'plot every {plot_frequency} iterations')

        regularizer.set_epoch(epoch)
        regularizer.plot_frequency = max(1, plot_frequency)

        time.sleep(1)
        total_loss = 0
        total_loss_regul = 0

        pbar = trange(Niter, ncols=100)
        for N in pbar:

            if has_field:
                optimizer_f.zero_grad()

            regularizer.reset_iteration()
            recurrent_active = recursive_loop > 0 and (not tc.recursive_training or epoch >= tc.recursive_training_start_epoch)
            dataset_batch = []
            ids_batch = []
            ids_index = 0
            loss = 0
            for batch in range(batch_size):

                run = 0
                k = time_window + np.random.randint(n_ts_frames - 1 - time_window - time_step - recursive_loop)
                x = x_ts.frame(k).to_packed().to(device).clone().detach()
                field_col = 2 + 2 * dimension
                vel_start = 1 + dimension
                vel_end = 1 + 2 * dimension
                if has_field:
                    field = model_f(time=k / n_frames) ** 2
                    x[:, field_col:field_col + 1] = field

                edges = edges_radius_blockwise(x, dimension, bc_dpos, min_radius, max_radius, block=4096)

                if batch_ratio < 1:
                    ids = np.random.permutation(x.shape[0])[:int(x.shape[0] * batch_ratio)]
                    ids = np.sort(ids)
                    mask = torch.isin(edges[1, :], torch.tensor(ids, device=device))
                    edges = edges[:, mask]

                if time_window == 0:
                    dataset = GraphData(x=x[:, :], edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)
                else:
                    xt = []
                    for t in range(time_window):
                        x_ = x_ts.frame(k - t).to_packed().to(device)
                        xt.append(x_[:, :])
                    dataset = GraphData(x=xt, edge_index=edges, num_nodes=x.shape[0])
                    dataset_batch.append(dataset)

                if recurrent_active:
                    y = x_ts.frame(k + recursive_loop).pos.to(device).clone().detach()
                elif time_step == 1:
                    y = torch.tensor(y_raw[k], dtype=torch.float32, device=device).clone().detach() / ynorm
                elif time_step > 1:
                    y = x_ts.frame(k + time_step).pos.to(device).clone().detach()

                if tc.shared_embedding:
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

            batch = collate_graph_batch(dataset_batch)
            optimizer.zero_grad()

            batch_state = CellState.from_packed(batch.x, dimension)
            pred = model(batch_state, batch.edge_index, data_id=data_id, training=True, k=k_batch, has_field=has_field)

            if recurrent_active:
                for loop in range(recursive_loop):
                    ids_index = 0
                    for b in range(batch_size):
                        x = dataset_batch[b].x.clone().detach()

                        pos_start = 1
                        pos_end = 1 + dimension
                        vel_start = 1 + dimension
                        vel_end = 1 + 2 * dimension
                        X1 = x[:, pos_start:pos_end]
                        V1 = x[:, vel_start:vel_end]
                        if mc.prediction == '2nd_derivative':
                            V1 += pred[ids_index:ids_index + x.shape[0]] * ynorm * delta_t
                        else:
                            V1 = pred[ids_index:ids_index + x.shape[0]] * ynorm
                        x[:, pos_start:pos_end] = bc_pos(X1 + V1 * delta_t)
                        x[:, vel_start:vel_end] = V1
                        if tc.noise_level > 0:
                            x[:, pos_start:pos_end] += tc.noise_level * torch.randn_like(x[:, pos_start:pos_end])
                        dataset_batch[b].x = x

                        ids_index += x.shape[0]

                    batch = collate_graph_batch(dataset_batch)
                    batch_state = CellState.from_packed(batch.x, dimension)
                    pred = model(batch_state, batch.edge_index, data_id=data_id, training=True, k=k_batch)

            if sim.state_type == 'sequence':
                loss = (pred - y_batch).norm(2)
                loss = loss + tc.coeff_model_a * (model.a[run, ind_a + 1] - model.a[run, ind_a]).norm(2)

            regul_loss = regularizer.compute(model, device)
            loss = loss + regul_loss

            if recurrent_active and recursive_loop > 1:
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
                pos_batch = x_batch[:, 1:dimension + 1]
                vel_batch = x_batch[:, 1 + dimension:1 + 2 * dimension]
                if mc.prediction == '2nd_derivative':
                    x_pos_pred = pos_batch + delta_t * time_step * (
                                vel_batch + delta_t * time_step * pred * ynorm)
                else:
                    x_pos_pred = pos_batch + delta_t * time_step * pred * ynorm

                if batch_ratio < 1:
                    loss = loss + (x_pos_pred[ids_batch] - y_batch[ids_batch]).norm(2)
                else:
                    loss = loss + (x_pos_pred - y_batch).norm(2)


            loss.backward()
            optimizer.step()

            if has_field:
                optimizer_f.step()

            total_loss += loss.item()
            total_loss_regul += regularizer.get_iteration_total()

            if (N + 1) % 1000 == 0:
                avg_loss = total_loss / (N + 1) / n_cells
                pbar.set_postfix(loss=f'{avg_loss:.6f}')
                logger.info(f'Epoch {epoch}  iter {N + 1}  avg loss: {avg_loss:.6f}')

            if (N % plot_frequency == 0) or (N == 0):
                loss_dict['loss'].append(loss.item() / n_cells)

            regularizer.finalize_iteration()

            if ((epoch < 30) & (N % plot_frequency == 0)) | (N == 0):
                plot_loss_components(loss_dict, regularizer.get_history(), log_dir, epoch=epoch, Niter=Niter)
                plot_training(config=config, pred=pred, gt=y_batch, log_dir=log_dir,
                              epoch=epoch, N=N, x=x_plot, model=model, n_nodes=0, n_node_types=0, index_nodes=0,
                              dataset_num=1,
                              index_cells=index_cells, n_cells=n_cells,
                              n_cell_types=n_cell_types, ynorm=ynorm, cmap=cmap, axis=True, device=device)
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

        print("Epoch {}. Loss: {:.6f}  Regul: {:.6f}".format(epoch, total_loss / n_cells, total_loss_regul / n_cells))
        logger.info("Epoch {}. Loss: {:.6f}  Regul: {:.6f}".format(epoch, total_loss / n_cells, total_loss_regul / n_cells))
        list_loss.append(total_loss / n_cells)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        scheduler.step()
        logger.info(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}')

        from cell_gnn.figure_style import default_style as fig_style
        fig = plt.figure(figsize=(12, 10))

        labels, n_clusters, new_labels, func_list, model_a_, accuracy = \
            plot_training_summary_panels(fig, log_dir, model, config, n_cells, n_cell_types,
                                         index_cells, type_list, ynorm, cmap,
                                         embedding_cluster, epoch, logger, device,
                                         loss_dict=loss_dict, regul_history=regularizer.get_history())

        if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq - 1) & (
                epoch < n_epochs - sparsity_freq):
            # Constrain embedding domain
            with torch.no_grad():
                model.a[0] = model_a_.clone().detach()
            print(f'regul_embedding: replaced')
            logger.info(f'regul_embedding: replaced')

            # Constrain function domain
            if tc.sparsity == 'replace_embedding_function':

                logger.info(f'replace_embedding_function')
                y_func_list = func_list * 0

                fig_tmp, ax_tmp = fig_init()
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
                fig_style.savefig(fig_tmp, f"./{log_dir}/tmp_training/Fig_{epoch}_before training function.tif")

                lr_embedding = 1E-12
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                all_embeddings = model.a[0, :n_cells, :].clone().detach()
                features = build_edge_features(rr, all_embeddings, mc.cell_model_name, max_radius,
                                                dimension=sim.dimension)
                N_feat, n_pts, input_dim = features.shape
                for sub_epochs in range(20):
                    optimizer.zero_grad()
                    pred_flat = model.lin_edge(features.reshape(N_feat * n_pts, input_dim).float())
                    pred = pred_flat.reshape(N_feat, n_pts, -1)
                    loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                    logger.info(f'    loss: {np.round(loss.item() / n_cells, 3)}')
                    loss.backward()
                    optimizer.step()

            if tc.fix_cluster_embedding:
                lr_embedding = 1E-12
            else:
                lr_embedding = tc.learning_rate_embedding_start
            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
            logger.info(f'Learning rates: {lr}, {lr_embedding}')

        else:
            if epoch > n_epochs - sparsity_freq:
                lr_embedding = tc.learning_rate_embedding_end
                lr = tc.learning_rate_end
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')
            else:
                lr_embedding = tc.learning_rate_embedding_start
                lr = tc.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        fig_style.savefig(fig, f"./{log_dir}/tmp_training/Fig_{epoch}.tif")


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20,
              step=15, ratio=1, run=0, test_mode='', sample_embedding=False, cell_of_interest=1, device=[]):
    """Route testing to the cell testing function.

    This simplified version only supports cell-based testing.
    """

    data_test_cell(config, config_file, visualize, style, True, best_model, step, ratio, run, test_mode,
                       sample_embedding, cell_of_interest, device)


def data_test_cell(config=None, config_file=None, visualize=False, style='color frame', verbose=True,
                       best_model=20, step=15, ratio=1, run=0, test_mode='', sample_embedding=False,
                       cell_of_interest=1, device=[]):

    dataset_name = config.dataset
    sim = config.simulation
    mc = config.graph_model
    tc = config.training

    n_cells = sim.n_cells
    n_frames = sim.n_frames
    n_runs = tc.n_runs
    max_radius = sim.max_radius
    min_radius = sim.min_radius
    n_cell_types = sim.n_cell_types
    delta_t = sim.delta_t
    time_window = tc.time_window
    time_step = tc.time_step
    sub_sampling = sim.sub_sampling
    dimension = sim.dimension
    omega = mc.omega

    # packed tensor column indices
    pos_start = 1
    pos_end = 1 + dimension
    vel_start = 1 + dimension
    vel_end = 1 + 2 * dimension
    type_col = 1 + 2 * dimension
    field_col = 2 + 2 * dimension
    do_tracking = tc.do_tracking
    cmap = CustomColorMap(config=config)

    has_field = (mc.field_type != '')
    has_state = (sim.state_type != 'discrete')
    has_bounding_box = 'PDE_F' in mc.cell_model_name

    if has_field:
        n_nodes = sim.n_nodes
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

    n_sub_population = n_cells // n_cell_types
    first_cell_id_cells = []
    for n in range(n_cell_types):
        index = np.arange(n_cells * n // n_cell_types, n_cells * (n + 1) // n_cell_types)
        first_cell_id_cells.append(index)

    print(f'load data ...')

    if os.path.exists(f'graphs_data/{dataset_name}/x_list_{run}.pt'):
        x_raw = torch.load(f'graphs_data/{dataset_name}/x_list_{run}.pt', map_location=device)
        y_raw = torch.load(f'graphs_data/{dataset_name}/y_list_{run}.pt', map_location=device)
        x_ts = CellTimeSeries.from_packed(x_raw, dimension)
        # mutable packed copy for rollout write-back (time_window)
        x_packed = x_raw.clone().to(device)
        ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
        vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)
        if vnorm == 0:
            vnorm = ynorm
    else:
        x_ts = load_simulation_data(f'graphs_data/{dataset_name}/x_list_{run}', dimension).to(device)
        y_raw_np = load_raw_array(f'graphs_data/{dataset_name}/y_list_{run}')
        y_raw = torch.tensor(y_raw_np, dtype=torch.float32, device=device)
        # mutable packed copy for rollout write-back (time_window)
        # reconstruct (T, N, C) packed tensor from timeseries fields
        x_packed = torch.stack([x_ts.frame(t).to_packed() for t in range(x_ts.n_frames)])
        x = x_ts.frame(0).to_packed()
        if ('PDE_MLPs' not in mc.cell_model_name) & ('PDE_F' not in mc.cell_model_name) & ('PDE_M' not in mc.cell_model_name):
            n_cells = int(x.shape[0] / ratio)
            config.simulation.n_cells = n_cells
        n_frames = x_ts.n_frames
        index_cells = get_index_cells(x, n_cell_types, dimension)
        if n_cell_types > 1000:
            index_cells = []
            for n in range(3):
                index = np.arange(n_cells * n // 3, n_cells * (n + 1) // 3)
                index_cells.append(index)
                n_cell_types = 3
        ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
        vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)
        if vnorm == 0:
            vnorm = ynorm

    if do_tracking | has_state:
        for k in range(x_ts.n_frames):
            ptype = x_ts.frame(k).cell_type.to(device)
            if k == 0:
                type_list = ptype
            else:
                type_list = torch.concatenate((type_list, ptype))
        n_cells_max = len(type_list) + 1
        config.simulation.n_cells_max = n_cells_max

    if ratio > 1:
        new_ncells = int(n_cells * ratio)
        model.a = nn.Parameter(
            torch.tensor(np.ones((n_runs, int(new_ncells), 2)), device=device, dtype=torch.float32,
                         requires_grad=False))
        n_cells = new_ncells
        index_cells = get_index_cells(x, n_cell_types, dimension)
    if sample_embedding:
        model_a_ = nn.Parameter(
            torch.tensor(np.ones((int(n_cells), model.embedding_dim)), device=device, requires_grad=False,
                         dtype=torch.float32))
        for n in range(n_cells):
            t = to_numpy(x[n, type_col]).astype(int)
            index = first_cell_id_cells[t][np.random.randint(n_sub_population)]
            with torch.no_grad():
                model_a_[n] = first_embedding[index].clone().detach()
        model.a = nn.Parameter(
            torch.tensor(np.ones((model.n_dataset, int(n_cells), model.embedding_dim)), device=device,
                         requires_grad=False, dtype=torch.float32))
        with torch.no_grad():
            for n in range(model.a.shape[0]):
                model.a[n] = model_a_
    # create model and load weights
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    model.cell_of_interest = cell_of_interest

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

    if has_field:
        n_nodes_per_axis = int(np.sqrt(sim.n_nodes))
        model_f = Siren_Network(image_width=n_nodes_per_axis, in_features=mc.input_size_nnr,
                                out_features=mc.output_size_nnr,
                                hidden_features=mc.hidden_dim_nnr,
                                hidden_layers=mc.n_layers_nnr, outermost_linear=True, device=device,
                                first_omega_0=mc.omega, hidden_omega_0=mc.omega)
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
        print(f'test data ... {mc.cell_model_name}')
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

    x = x_ts.frame(start_it).to_packed()
    n_cells = x.shape[0]
    x_inference_list = []

    for it in trange(start_it, stop_it, ncols=100):

        check_and_clear_memory(device=device, iteration_number=it, every_n_iterations=25,
                               memory_percentage_threshold=0.6)

        if it < n_frames - 4:
            x0 = x_ts.frame(it).to_packed()
            x0_next = x_ts.frame(it + time_step).to_packed()
            if not (mc.cell_model_name == 'PDE_R'):
                y0 = y_raw[it].clone().detach()

        if do_tracking:
            x = x0.clone().detach()

        # error calculations
        if has_bounding_box:
            rmserr = torch.sqrt(
                torch.mean(torch.sum(bc_dpos(x[:, pos_start:pos_end] - x0[:, pos_start:pos_end]) ** 2, axis=1)))
        else:
            if (do_tracking) | (x.shape[0] != x0.shape[0]):
                rmserr = torch.zeros(1, device=device)
            else:
                rmserr = torch.sqrt(
                    torch.mean(torch.sum(bc_dpos(x[:, pos_start:pos_end] - x0[:, pos_start:pos_end]) ** 2, axis=1)))
            if x.shape[0] > 5000:
                geomloss = gloss(x[0:5000, pos_start:pos_end], x0[0:5000, pos_start:pos_end])
            else:
                geomloss = gloss(x[:, pos_start:pos_end], x0[:, pos_start:pos_end])
            geomloss_list.append(geomloss.item())
        rmserr_list.append(rmserr.item())

        if config.training.shared_embedding:
            data_id = torch.ones((n_cells, 1), dtype=torch.int, device=device)
        else:
            data_id = torch.ones((n_cells, 1), dtype=torch.int, device=device) * run

        # update calculations
        with torch.no_grad():

            distance = torch.sum(bc_dpos(x[:, None, pos_start:pos_end] - x[None, :, pos_start:pos_end]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()

            if has_field:
                field = model_f(time=it / n_frames) ** 2
                x[:, field_col:field_col + 1] = field

            if time_window > 0:
                xt = []
                for t in range(time_window):
                    x_ = x_packed[it - t].clone().detach()
                    xt.append(x_[:, :])
                dataset = GraphData(x=xt, edge_index=edge_index)
            else:
                dataset = GraphData(x=x, edge_index=edge_index)

            if 'test_simulation' in test_mode:
                y = y0 / ynorm
                pred = y
            else:
                test_state = CellState.from_packed(dataset.x, dimension)
                pred = model(test_state, dataset.edge_index, data_id=data_id, training=False, has_field=has_field, k=it)
                y = pred

            if sub_sampling > 1:
                # predict position, does not work with rotation_augmentation
                if time_step == 1:
                    x_next = bc_pos(y[:, 0:dimension])
                elif time_step == 2:
                    x_next = bc_pos(y[:, dimension:2 * dimension])
                x[:, vel_start:vel_end] = (x_next - x[:, pos_start:pos_end]) / delta_t
                x[:, pos_start:pos_end] = x_next
                loss = (x[:, pos_start:pos_end] - x0_next[:, pos_start:pos_end]).norm(2)
                pred_err_list.append(to_numpy(torch.sqrt(loss)))
            elif do_tracking:
                x_pos_next = x0_next[:, pos_start:pos_end].clone().detach()
                if pred.shape[1] != dimension:
                    pred = torch.cat((pred, torch.zeros(pred.shape[0], 1, device=pred.device)), dim=1)
                if mc.prediction == '2nd_derivative':
                    x_pos_pred = (x[:, pos_start:pos_end] + delta_t * time_step * (
                                x[:, vel_start:vel_end] + delta_t * time_step * pred * ynorm))
                else:
                    x_pos_pred = (x[:, pos_start:pos_end] + delta_t * time_step * pred * ynorm)
                distance = torch.sum(bc_dpos(x_pos_pred[:, None, :] - x_pos_next[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                min_value = result.values
                indices = result.indices
                loss = torch.std(torch.sqrt(min_value))
                pred_err_list.append(to_numpy(torch.sqrt(loss)))
                if 'inference' in test_mode:
                    x[:, vel_start:vel_end] = pred.clone().detach() / (delta_t * time_step)

            else:
                if mc.prediction == '2nd_derivative':
                    y = y * ynorm * delta_t
                    x[:n_cells, vel_start:vel_end] = x[:n_cells, vel_start:vel_end] + y[:n_cells]  # speed update
                else:
                    y = y * vnorm
                    x[:n_cells, vel_start:vel_end] = y[:n_cells]
                x[:, pos_start:pos_end] = bc_pos(
                    x[:, pos_start:pos_end] + x[:, vel_start:vel_end] * delta_t)  # position update

            if 'inference' in test_mode:
                x_inference_list.append(x)

            if (time_window > 1) & ('plot_data' not in test_mode):
                moving_pos = torch.argwhere(x[:, type_col] != 0)
                x_packed[it + 1, moving_pos.squeeze(), pos_start:vel_end] = x[moving_pos.squeeze(),
                                                                               pos_start:vel_end].clone().detach()

        # vizualization
        if 'plot_data' in test_mode:
            x = x_ts.frame(it).to_packed()

        if (it % step == 0) & (it >= 0) & visualize:

            num = f"{it:06}"

            from cell_gnn.figure_style import default_style as fig_style, dark_style
            if 'latex' in style:
                plt.rcParams['text.usetex'] = True
                rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            if 'black' in style:
                active_style = dark_style
            else:
                active_style = fig_style
            active_style.apply_globally()

            fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
            ax.tick_params(axis='both', which='major', pad=15)

            if do_tracking:

                plt.scatter(to_numpy(x0[:, pos_start + 1]), to_numpy(x0[:, pos_start]), s=10, c=active_style.foreground, alpha=0.5)
                plt.scatter(to_numpy(x_pos_pred[:, 1]), to_numpy(x_pos_pred[:, 0]), s=10, c='r')
                x1 = x_ts.frame(it + time_step).to_packed()
                plt.scatter(to_numpy(x1[:, pos_start + 1]), to_numpy(x1[:, pos_start]), s=10, c='g')

                plt.xticks([])
                plt.yticks([])

                if 'zoom' in style:
                    ax.quiver(to_numpy(x0[:, pos_start + 1]), to_numpy(x0[:, pos_start]),
                              to_numpy(x[:, vel_start + 1]) * delta_t, to_numpy(x[:, vel_start]) * delta_t,
                              color='g', angles='xy', scale_units='xy', scale=1, width=0.002)
                    plt.xlim([300, 400])
                    plt.ylim([300, 400])
                else:
                    plt.xlim([0, 700])
                    plt.ylim([0, 700])
                plt.tight_layout()

            else:
                s_p = 10
                index_cells = get_index_cells(x, n_cell_types, dimension)
                for n in range(n_cell_types):
                    if 'bw' in style:
                        plt.scatter(x[index_cells[n], pos_start + 1].detach().cpu().numpy(),
                                    x[index_cells[n], pos_start].detach().cpu().numpy(), s=s_p, color=active_style.foreground)
                    else:
                        plt.scatter(x[index_cells[n], pos_start + 1].detach().cpu().numpy(),
                                    x[index_cells[n], pos_start].detach().cpu().numpy(), s=s_p, color=cmap.color(n))
                plt.xlim([0, 1])
                plt.ylim([0, 1])

                if ('field' in style) & has_field:
                    if 'zoom' in style:
                        plt.scatter(to_numpy(x[:, pos_start + 1]), to_numpy(x[:, pos_start]), s=s_p * 50, c=to_numpy(x[:, field_col]) * 20,
                                    alpha=0.5, cmap='viridis', vmin=0, vmax=1.0)
                    else:
                        plt.scatter(to_numpy(x[:, pos_start + 1]), to_numpy(x[:, pos_start]), s=s_p * 2, c=to_numpy(x[:, field_col]) * 20,
                                    alpha=0.5, cmap='viridis', vmin=0, vmax=1.0)

                if cell_of_interest > 1:

                    xc = to_numpy(x[cell_of_interest, pos_start + 1])
                    yc = to_numpy(x[cell_of_interest, pos_start])
                    pos = torch.argwhere(edge_index[1, :] == cell_of_interest)
                    pos = pos[:, 0]
                    if 'zoom' in style:
                        plt.scatter(to_numpy(x[edge_index[0, pos], pos_start + 1]), to_numpy(x[edge_index[0, pos], pos_start]), s=s_p * 10,
                                    color=active_style.foreground, alpha=1.0)
                    else:
                        plt.scatter(to_numpy(x[edge_index[0, pos], pos_start + 1]), to_numpy(x[edge_index[0, pos], pos_start]), s=s_p * 1,
                                    color=active_style.foreground, alpha=1.0)

                    plt.arrow(x=to_numpy(x[cell_of_interest, pos_start + 1]), y=to_numpy(x[cell_of_interest, pos_start]),
                              dx=to_numpy(x[cell_of_interest, vel_start + 1]) * delta_t * 100,
                              dy=to_numpy(x[cell_of_interest, vel_start]) * delta_t * 100, head_width=0.004,
                              length_includes_head=True, color='b')
                    if mc.prediction == '2nd_derivative':
                        plt.arrow(x=to_numpy(x[cell_of_interest, pos_start + 1]), y=to_numpy(x[cell_of_interest, pos_start]),
                                  dx=to_numpy(y0[cell_of_interest, 1]) * delta_t ** 2 * 100,
                                  dy=to_numpy(y0[cell_of_interest, 0]) * delta_t ** 2 * 100, head_width=0.004,
                                  length_includes_head=True, color='g')
                        plt.arrow(x=to_numpy(x[cell_of_interest, pos_start + 1]), y=to_numpy(x[cell_of_interest, pos_start]),
                                  dx=to_numpy(y[cell_of_interest, 1]) * delta_t * 100,
                                  dy=to_numpy(y[cell_of_interest, 0]) * delta_t * 100, head_width=0.004,
                                  length_includes_head=True, color='r')

                if 'zoom' in style:
                    plt.xlim([xc - 0.1, xc + 0.1])
                    plt.ylim([yc - 0.1, yc + 0.1])
                    plt.xticks([])
                    plt.yticks([])

            if 'latex' in style:
                plt.xlabel(r'$x$', fontsize=active_style.frame_title_font_size * 1.6)
                plt.ylabel(r'$y$', fontsize=active_style.frame_title_font_size * 1.6)
                plt.xticks(fontsize=active_style.frame_title_font_size)
                plt.yticks(fontsize=active_style.frame_title_font_size)
            if 'frame' in style:
                plt.xlabel('x', fontsize=active_style.frame_title_font_size)
                plt.ylabel('y', fontsize=active_style.frame_title_font_size)
                plt.xticks(fontsize=active_style.frame_title_font_size)
                plt.yticks(fontsize=active_style.frame_title_font_size)
                plt.text(0, 1.1, f'   ', ha='left', va='top', transform=ax.transAxes, fontsize=active_style.frame_title_font_size)
                ax.tick_params(axis='both', which='major', pad=15)
            if 'arrow' in style:
                mask = to_numpy(x[:, vel_start + 1]) != 0
                px = to_numpy(x[:, pos_start + 1])
                py = to_numpy(x[:, pos_start])
                if 'speed' in style:
                    ax.quiver(px[mask], py[mask],
                              to_numpy(x[:, vel_start + 1])[mask] * delta_t * 2,
                              to_numpy(x[:, vel_start])[mask] * delta_t * 2,
                              color='g', angles='xy', scale_units='xy', scale=1, width=0.002)
                if 'acc_true' in style:
                    ax.quiver(px[mask], py[mask],
                              to_numpy(y0[:, 1])[mask] / 5E3,
                              to_numpy(y0[:, 0])[mask] / 5E3,
                              color='r', angles='xy', scale_units='xy', scale=1, width=0.002)
                if 'acc_learned' in style:
                    ynorm_s = to_numpy(ynorm.squeeze())
                    ax.quiver(px[mask], py[mask],
                              to_numpy(pred[:, 1])[mask] * ynorm_s / 5E3,
                              to_numpy(pred[:, 0])[mask] * ynorm_s / 5E3,
                              color='r', angles='xy', scale_units='xy', scale=1, width=0.002)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
            if 'name' in style:
                plt.title(f"{os.path.basename(log_dir)}", fontsize=active_style.font_size * 1.7)
            if 'no_ticks' in style:
                plt.xticks([])
                plt.yticks([])
            if 'PDE_G' in mc.cell_model_name:
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
            plt.tight_layout()
            plt.tight_layout()
            active_style.savefig(fig, f"./{log_dir}/tmp_recons/Fig_{config_file}_{run}_{num}.tif")

            if ('feature' in style) & ('PDE_MLPs_A' in config.graph_model.cell_model_name):
                n_feat_cols = model.new_features.shape[1]
                fig_f, axes_f = fig_style.figure(ncols=n_feat_cols, width=22, height=6)
                if not isinstance(axes_f, np.ndarray):
                    axes_f = np.array([axes_f])
                for k in range(n_feat_cols):
                    ax_f = axes_f[k]
                    ax_f.scatter(to_numpy(x[:, pos_start + 1]), to_numpy(x[:, pos_start]), c=to_numpy(model.new_features[:, k]), s=5,
                                 cmap='viridis')
                    ax_f.set_title(f'new_features {k}')
                    ax_f.set_xlim([0, 1])
                    ax_f.set_ylim([0, 1])
                plt.tight_layout()
                fig_style.savefig(fig_f, f"./{log_dir}/tmp_recons/Features_{config_file}_{run}_{num}.tif")

            if 'boundary' in style:
                fig_b, ax_b = fig_init(formatx='%.1f', formaty='%.1f')
                t = torch.min(x[:, field_col:], -1).values
                ax_b.scatter(to_numpy(x[:, pos_start + 1]), to_numpy(x[:, pos_start]), s=25, c=to_numpy(t), vmin=-1, vmax=1)
                ax_b.set_xlim([0, 1])
                ax_b.set_ylim([0, 1])
                plt.tight_layout()
                fig_style.savefig(fig_b, f"./{log_dir}/tmp_recons/Boundary_{config_file}_{num}.tif")

    # --- One-step residual field ---
    print('computing one-step residual field ...')
    from cell_gnn.plot import plot_residual_field_3d
    from cell_gnn.zarr_io import ZarrArrayWriter

    residual_list = []
    os.makedirs(f'./{log_dir}/results/residual', exist_ok=True)
    n_test_frames = min(n_frames - 4, x_ts.n_frames - 4)

    with torch.no_grad():
        for it in trange(n_test_frames, ncols=100, desc='one-step residual'):
            x0 = x_ts.frame(it).to_packed().to(device)
            y_gt = y_raw[it].clone().detach()

            distance = torch.sum(bc_dpos(x0[:, None, pos_start:pos_end] - x0[None, :, pos_start:pos_end]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float()
            edge_index = adj_t.nonzero().t().contiguous()

            dataset = GraphData(x=x0, edge_index=edge_index)
            test_state = CellState.from_packed(dataset.x, dimension)

            if config.training.shared_embedding:
                data_id = torch.ones((n_cells, 1), dtype=torch.int, device=device)
            else:
                data_id = torch.ones((n_cells, 1), dtype=torch.int, device=device) * run

            pred = model(test_state, dataset.edge_index, data_id=data_id, training=False, has_field=has_field, k=it)

            residual = y_gt[:n_cells] - pred[:n_cells] * ynorm
            residual_list.append(to_numpy(residual))

            if (it % step == 0) and visualize:
                pos_np = to_numpy(x0[:n_cells, pos_start:pos_end])
                res_np = to_numpy(residual)
                plot_residual_field_3d(pos_np, res_np, it, dimension, log_dir, cmap, sim)

    residual_arr = np.stack(residual_list, axis=0)  # (T, N, dim)
    np.save(f'./{log_dir}/results/residual_field.npy', residual_arr)

    residual_writer = ZarrArrayWriter(
        path=f'graphs_data/{dataset_name}/residual_list_{run}',
        n_cells=n_cells, n_features=dimension)
    for frame in residual_arr:
        residual_writer.append(frame)
    residual_writer.finalize()

    residual_mag = np.sqrt((residual_arr ** 2).sum(axis=-1))
    logger.info(f'Residual field: mean magnitude = {residual_mag.mean():.6f}, max = {residual_mag.max():.6f}')
    print(f'Residual field: mean magnitude = {residual_mag.mean():.6f}, max = {residual_mag.max():.6f}')
    print(f'Saved to graphs_data/{dataset_name}/residual_list_{run}.zarr')

    # Write structured results log
    results = {
        'rollout_RMSE_mean': float(np.mean(rmserr_list)) if rmserr_list else 0.0,
        'rollout_RMSE_final': float(rmserr_list[-1]) if rmserr_list else 0.0,
        'rollout_geomloss_mean': float(np.mean(geomloss_list)) if geomloss_list else 0.0,
        'rollout_geomloss_final': float(geomloss_list[-1]) if geomloss_list else 0.0,
        'residual_mean_magnitude': float(residual_mag.mean()),
        'residual_max_magnitude': float(residual_mag.max()),
    }
    results_log_path = os.path.join(log_dir, 'results.log')
    with open(results_log_path, 'w') as f:
        for key, value in results.items():
            f.write(f'{key}: {value}\n')
    print(f'results written to {results_log_path}')


def data_train_cell_field(config, erase, best_model, device):
    sim = config.simulation
    tc = config.training
    mc = config.graph_model

    print(f'training cell field data ... {mc.cell_model_name}')

    dimension = sim.dimension
    n_epochs = tc.n_epochs
    max_radius = sim.max_radius
    min_radius = sim.min_radius
    n_cells = sim.n_cells
    n_cell_types = sim.n_cell_types
    n_nodes = sim.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    delta_t = sim.delta_t
    noise_level = tc.noise_level
    dataset_name = config.dataset
    n_frames = sim.n_frames
    has_siren = 'siren' in mc.field_type
    has_siren_time = 'siren_with_time' in mc.field_type
    rotation_augmentation = tc.rotation_augmentation
    data_augmentation_loop = tc.data_augmentation_loop
    target_batch_size = tc.batch_size
    replace_with_cluster = 'replace' in tc.sparsity

    if tc.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)
    embedding_cluster = EmbeddingCluster(config)
    n_runs = tc.n_runs
    sparsity_freq = tc.sparsity_freq

    log_dir, logger = create_log_dir(config, erase)
    print(f'Graph files N: {n_runs}')
    logger.info(f'Graph files N: {n_runs}')
    time.sleep(0.5)

    x_ts = load_simulation_data(f'graphs_data/{dataset_name}/x_list_0', dimension)
    y_raw_np = load_raw_array(f'graphs_data/{dataset_name}/y_list_0')
    y_raw = torch.tensor(y_raw_np, dtype=torch.float32, device=device)
    n_cells_max = x_ts.n_cells

    edge_p_p_list = torch.load(f'graphs_data/{dataset_name}/edge_p_p_list0.pt', map_location=device,
                               weights_only=False)
    edge_f_p_list = torch.load(f'graphs_data/{dataset_name}/edge_f_p_list0.pt', map_location=device,
                               weights_only=False)

    x = x_ts.frame(0).to_packed().to(device)
    y = y_raw[0].clone().detach()
    time.sleep(0.5)
    for k in trange(n_frames - 5, ncols=100):
        if (k % 10 == 0) | (n_frames < 1000):
            try:
                x = torch.cat((x, x_ts.frame(k).to_packed().to(device)), 0)
            except:
                print(f'error in frame {k}')
            y = torch.cat((y, y_raw[k].clone().detach()), 0)
    time.sleep(0.5)
    if torch.isnan(x).any() | torch.isnan(y).any():
        print('Pb isnan')
    vnorm = norm_velocity(x, dimension, device)
    ynorm = norm_acceleration(y, device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'N cells: {n_cells}')
    logger.info(f'N cells: {n_cells}')
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')
    logger.info(f'vnorm ynorm: {to_numpy(vnorm)} {to_numpy(ynorm)}')

    time.sleep(0.5)
    mesh_ts = load_field_data(f'graphs_data/{dataset_name}/x_mesh_list_0', dimension)
    y_mesh_raw_np = load_raw_array(f'graphs_data/{dataset_name}/y_mesh_list_0')
    y_mesh_raw = torch.tensor(y_mesh_raw_np, dtype=torch.float32, device=device)
    h = y_mesh_raw[0].clone().detach()
    for k in range(n_frames - 5):
        h = torch.cat((h, y_mesh_raw[k].clone().detach()), 0)
    hnorm = torch.std(h)
    torch.save(hnorm, os.path.join(log_dir, 'hnorm.pt'))
    print(f'hnorm: {to_numpy(hnorm)}')
    logger.info(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/{dataset_name}/mesh_data_0.pt', map_location=device, weights_only=False)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    x = []
    y = []
    h = []

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    if (best_model != None) & (best_model != ''):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
    model.ynorm = ynorm
    model.vnorm = vnorm

    lr = tc.learning_rate_start
    lr_embedding = tc.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
    logger.info(f"total trainable Params: {n_total_params}")
    logger.info(f'learning rates: {lr}, {lr_embedding}')
    model.train()

    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"
    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    print('')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    # update variable if cell_dropout, cell_division, etc ...
    x = x_ts.frame(n_frames - 5).to_packed().to(device)
    n_cells = x.shape[0]
    index_cells = get_index_cells(x, n_cell_types, dimension)
    type_list = get_type_list(x, dimension)
    print(f'N cells: {n_cells} {len(torch.unique(type_list))} types')
    logger.info(f'N cells:  {n_cells} {len(torch.unique(type_list))} types')
    config.simulation.n_cells = n_cells

    if has_siren:
        image_width = int(np.sqrt(n_nodes))
        model_f = Siren_Network(image_width=image_width, in_features=mc.input_size_nnr,
                                out_features=mc.output_size_nnr,
                                hidden_features=mc.hidden_dim_nnr,
                                hidden_layers=mc.n_layers_nnr, outermost_linear=True, device=device,
                                first_omega_0=80, hidden_omega_0=80.)
        model_f.to(device=device)
        model_f.train()
        optimizer_f = torch.optim.Adam(lr=1e-5, params=model_f.parameters())

    print("start training ...")
    print(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')
    logger.info(f'{n_frames * data_augmentation_loop // batch_size} iterations per epoch')

    list_loss = []
    loss_dict = {'loss': []}
    regularizer = LossRegularizer(tc, mc, sim, n_cells, plot_frequency=1)

    time.sleep(1)

    for epoch in range(n_epochs + 1):

        batch_size = get_batch_size(epoch)
        regularizer.set_epoch(epoch)

        f_p_mask = []
        for k in range(batch_size):
            if k == 0:
                f_p_mask = np.zeros((n_nodes, 1))
                f_p_mask = np.concatenate((f_p_mask, np.ones((n_cells, 1))), axis=0)
            else:
                f_p_mask = np.concatenate((f_p_mask, np.zeros((n_nodes, 1))), axis=0)
                f_p_mask = np.concatenate((f_p_mask, np.ones((n_cells, 1))), axis=0)
        f_p_mask = np.argwhere(f_p_mask == 1)
        f_p_mask = f_p_mask[:, 0]

        logger.info(f'batch_size: {batch_size}')

        total_loss = 0
        total_loss_regul = 0
        Niter = n_frames * data_augmentation_loop // batch_size
        plot_frequency = int(Niter // 10)
        regularizer.plot_frequency = max(1, plot_frequency)

        if epoch == 0:
            print(f'{Niter} iterations per epoch')
            logger.info(f'{Niter} iterations per epoch')
            print(f'plot every {plot_frequency} iterations')

        pbar = trange(Niter)
        for N in pbar:

            phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=device) * np.pi * 2
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            dataset_batch_p_p = []
            dataset_batch_f_p = []
            time_batch = []

            for batch in range(batch_size):

                k = np.random.randint(n_frames - 3)
                x = x_ts.frame(k).to_packed().to(device)
                mesh_frame = mesh_ts.frame(k).to(device)

                match mc.field_type:
                    case 'tensor':
                        mesh_frame.field[:, 0:1] = model.field[0]
                    case 'siren':
                        mesh_frame.field[:, 0:1] = model_f() ** 2
                    case 'siren_with_time':
                        mesh_frame.field[:, 0:1] = model_f(time=k / n_frames) ** 2
                x_mesh = mesh_frame.to_packed()
                x_cell_field = torch.concatenate((x_mesh, x), dim=0)

                edges = edge_p_p_list[k]
                dataset_p_p = GraphData(x=x[:, :], edge_index=edges)
                dataset_batch_p_p.append(dataset_p_p)

                edges = edge_f_p_list[k]
                dataset_f_p = GraphData(x=x_cell_field[:, :], edge_index=edges)
                dataset_batch_f_p.append(dataset_f_p)

                y = y_raw[k].clone().detach()
                if noise_level > 0:
                    y = y * (1 + torch.randn_like(y) * noise_level)
                y = y / ynorm

                if rotation_augmentation:
                    new_x = cos_phi * y[:, 0] + sin_phi * y[:, 1]
                    new_y = -sin_phi * y[:, 0] + cos_phi * y[:, 1]
                    y[:, 0] = new_x
                    y[:, 1] = new_y
                if batch == 0:
                    y_batch = y[:, 0:2]
                else:
                    y_batch = torch.cat((y_batch, y[:, 0:2]), dim=0)

            batch_p_p = collate_graph_batch(dataset_batch_p_p)
            batch_f_p = collate_graph_batch(dataset_batch_f_p)

            regularizer.reset_iteration()
            optimizer.zero_grad()

            if has_siren:
                optimizer_f.zero_grad()
            batch_state = CellState.from_packed(batch_f_p.x, dimension)
            pred_f_p = model(batch_state, batch_f_p.edge_index, data_id=0, training=True, phi=phi, has_field=True)
            batch_state = CellState.from_packed(batch_p_p.x, dimension)
            pred_p_p = model(batch_state, batch_p_p.edge_index, data_id=0, training=True, phi=phi, has_field=False)

            pred_f_p = pred_f_p[f_p_mask]

            loss = (pred_p_p + pred_f_p - y_batch).norm(2)
            regul_loss = regularizer.compute(model, device)
            loss = loss + regul_loss

            loss.backward()
            optimizer.step()
            if has_siren:
                optimizer_f.step()
            total_loss += loss.item()
            total_loss_regul += regularizer.get_iteration_total()

            if (N + 1) % 1000 == 0:
                avg_loss = total_loss / (N + 1) / n_cells
                pbar.set_postfix(loss=f'{avg_loss:.6f}')
                logger.info(f'Epoch {epoch}  iter {N + 1}  avg loss: {avg_loss:.6f}')

            if (N % plot_frequency == 0) or (N == 0):
                loss_dict['loss'].append(loss.item() / n_cells)

            regularizer.finalize_iteration()

            visualize_embedding = True
            if visualize_embedding & (((epoch < 30) & (N % plot_frequency == 0)) | (N == 0)):
                plot_loss_components(loss_dict, regularizer.get_history(), log_dir, epoch=epoch, Niter=Niter)
                plot_training_cell_field(config=config, has_siren=has_siren, has_siren_time=has_siren_time,
                                             model_f=model_f, n_frames=n_frames,
                                             model_name=mc.cell_model_name, log_dir=log_dir,
                                             epoch=epoch, N=N, x=x, x_mesh=x_mesh, model=model, n_nodes=0,
                                             n_node_types=0, index_nodes=0, dataset_num=1,
                                             index_cells=index_cells, n_cells=n_cells,
                                             n_cell_types=n_cell_types, ynorm=ynorm, cmap=cmap, axis=True,
                                             device=device)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                if has_siren:
                    torch.save({'model_state_dict': model_f.state_dict(),
                                'optimizer_state_dict': optimizer_f.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
            if ((epoch == 0) & (N % (Niter // 200) == 0)):
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                if has_siren:
                    torch.save({'model_state_dict': model_f.state_dict(),
                                'optimizer_state_dict': optimizer_f.state_dict()},
                               os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

        print("Epoch {}. Loss: {:.6f}  Regul: {:.6f}".format(epoch, total_loss / n_cells, total_loss_regul / n_cells))
        logger.info("Epoch {}. Loss: {:.6f}  Regul: {:.6f}".format(epoch, total_loss / n_cells, total_loss_regul / n_cells))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))
        if has_siren:
            torch.save({'model_state_dict': model_f.state_dict(),
                        'optimizer_state_dict': optimizer_f.state_dict()},
                       os.path.join(log_dir, 'models', f'best_model_f_with_{n_runs - 1}_graphs_{epoch}.pt'))
        list_loss.append(total_loss / n_cells)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        from cell_gnn.figure_style import default_style as fig_style
        fig = plt.figure(figsize=(12, 10))

        labels, n_clusters, new_labels, func_list, model_a_, accuracy = \
            plot_training_summary_panels(fig, log_dir, model, config, n_cells, n_cell_types,
                                         index_cells, type_list, ynorm, cmap,
                                         embedding_cluster, epoch, logger, device,
                                         loss_dict=loss_dict, regul_history=regularizer.get_history())

        if (replace_with_cluster) & (epoch % sparsity_freq == sparsity_freq - 1) & (epoch < n_epochs - sparsity_freq):

            with torch.no_grad():
                model.a[0] = model_a_.clone().detach()
            print(f'regul_embedding: replaced')
            logger.info(f'regul_embedding: replaced')

            if tc.sparsity == 'replace_embedding':

                logger.info(f'replace_embedding_function')
                y_func_list = func_list * 0

                fig_tmp, ax_tmp = fig_init()
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
                fig_style.savefig(fig_tmp, f"./{log_dir}/tmp_training/Fig_{epoch}_before training function.tif")

                lr_embedding = 1E-12
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                all_embeddings = model.a[0, :n_cells, :].clone().detach()
                features = build_edge_features(rr, all_embeddings, mc.cell_model_name, max_radius,
                                                dimension=sim.dimension)
                N_feat, n_pts, input_dim = features.shape
                for sub_epochs in range(20):
                    optimizer.zero_grad()
                    pred_flat = model.lin_edge(features.reshape(N_feat * n_pts, input_dim).float())
                    pred = pred_flat.reshape(N_feat, n_pts, -1)
                    loss = (pred[:, :, 0] - y_func_list.clone().detach()).norm(2)
                    logger.info(f'    loss: {np.round(loss.item() / n_cells, 3)}')
                    loss.backward()
                    optimizer.step()

            if tc.fix_cluster_embedding:
                lr_embedding = 1E-12
            else:
                lr_embedding = tc.learning_rate_embedding_start
            optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
            logger.info(f'Learning rates: {lr}, {lr_embedding}')

        else:
            if epoch > n_epochs - sparsity_freq:
                lr_embedding = tc.learning_rate_embedding_end
                lr = tc.learning_rate_end
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')
            else:
                lr_embedding = tc.learning_rate_embedding_start
                lr = tc.learning_rate_start
                optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)
                logger.info(f'Learning rates: {lr}, {lr_embedding}')

        plt.tight_layout()
        fig_style.savefig(fig, f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
