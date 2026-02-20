import glob
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.data as data
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from tqdm import trange

from particle_gnn.generators.utils import choose_model, init_particles
from particle_gnn.utils import (
    to_numpy,
    CustomColorMap,
    check_and_clear_memory,
    get_index_particles,
    fig_init,
    get_equidistant_points,
    get_edges_with_cache,
    NeighborCache,
    choose_boundary_values,
)


def data_generate(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    device=None,
    bSave=True,
    timer=False,
):
    dataset_name = config.dataset

    print("")
    print(f"dataset_name: {dataset_name}")

    if (os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.npy")) | (
        os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.pt")
    ):
        print("watch out: data already generated")
        # return

    data_generate_particle(
        config,
        visualize=visualize,
        run_vizualized=run_vizualized,
        style=style,
        erase=erase,
        step=step,
        alpha=alpha,
        ratio=ratio,
        scenario=scenario,
        device=device,
        bSave=bSave,
        timer=timer,
    )

    plt.style.use("default")


def data_generate_particle(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    device=None,
    bSave=True,
    timer=False,
):
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    connection_matrix_list = []

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (f[-3:] != "Fig")
                & (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)

    # create GNN
    model, bc_pos, bc_dpos = choose_model(config=config, device=device)

    particle_dropout_mask = np.arange(n_particles)
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []

    if simulation_config.angular_Bernouilli != [-1]:
        b = simulation_config.angular_Bernouilli
        generative_m = np.array([stats.norm(b[0], b[2]), stats.norm(b[1], b[2])])

    for run in range(config.training.n_runs):
        check_and_clear_memory(
            device=device,
            iteration_number=0,
            every_n_iterations=250,
            memory_percentage_threshold=0.6,
        )

        if "PDE_K" in model_config.particle_model_name:
            p = config.simulation.params
            edges = np.random.choice(p[0], size=(n_particles, n_particles), p=p[1])
            edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)
            connection_matrix = torch.tensor(edges, dtype=torch.float32, device=device)
            model.connection_matrix = connection_matrix.detach().clone()
            connection_matrix_list.append(connection_matrix)

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        edge_p_p_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_particles(
            config=config, scenario=scenario, ratio=ratio, device=device
        )
        edge_cache = NeighborCache()

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1, ncols=150):
            # calculate type change
            if simulation_config.state_type == "sequence":
                sample = torch.rand((len(T1), 1), device=device)
                sample = (
                    sample < (1 / config.simulation.state_params[0])
                ) * torch.randint(0, n_particle_types, (len(T1), 1), device=device)
                T1 = (T1 + sample) % n_particle_types

            x = torch.concatenate(
                (
                    N1.clone().detach(),
                    X1.clone().detach(),
                    V1.clone().detach(),
                    T1.clone().detach(),
                    H1.clone().detach(),
                    A1.clone().detach(),
                ),
                1,
            )

            index_particles = get_index_particles(
                x, n_particle_types, dimension
            )  # can be different from frame to frame

            # compute connectivity rule
            with torch.no_grad():

                if timer:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                pos = x[:, 1:1+dimension]
                edge_index = get_edges_with_cache(
                    pos=pos,
                    bc_dpos=bc_dpos,
                    cache=edge_cache,
                    r_cut=max_radius,
                    r_skin=0.05,
                    min_radius=min_radius,
                    block=2048,
                )

                if timer:
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    print(
                        f"[edge build] "
                        f"N={x.shape[0]:6d}, "
                        f"E={edge_index.shape[1]:8d}, "
                        f"time={(t1 - t0)*1000:.2f} ms"
                    )

                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

                # model prediction
                y = model(dataset)

            if simulation_config.angular_sigma > 0:
                phi = (
                    torch.randn(n_particles, device=device)
                    * simulation_config.angular_sigma
                    / 360
                    * np.pi
                    * 2
                )
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)
                new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
                new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
                y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()
            if simulation_config.angular_Bernouilli != [-1]:
                z_i = stats.bernoulli(b[3]).rvs(n_particles)
                phi = np.array([g.rvs() for g in generative_m[z_i]]) / 360 * np.pi * 2
                phi = torch.tensor(phi, device=device, dtype=torch.float32)
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)
                new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
                new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
                y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()

            # append list
            if (it >= 0) & bSave:
                if has_particle_dropout:
                    x_ = x[particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_list.append(x_)
                    x_ = x[inv_particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                    y_list.append(y[particle_dropout_mask].clone().detach())
                else:
                    x_list.append(x.clone().detach())
                    y_list.append(y.clone().detach())

            # Particle update
            if model_config.prediction == "2nd_derivative":
                V1 += y * delta_t
            else:
                V1 = y
            X1 = bc_pos(X1 + V1 * delta_t)
            A1 = A1 + 1

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                if "black" in style:
                    plt.style.use("dark_background")

                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                if "bw" in style:
                    fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                    s_p = 100
                    for n in range(n_particle_types):
                        plt.scatter(
                            to_numpy(x[index_particles[n], 1]),
                            to_numpy(x[index_particles[n], 2]),
                            s=s_p,
                            color="k",
                        )
                    if training_config.particle_dropout > 0:
                        plt.scatter(
                            x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                            s=25,
                            color="k",
                            alpha=0.75,
                        )
                        plt.plot(
                            x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                            "+",
                            color="w",
                        )
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if "PDE_G" in model_config.particle_model_name:
                        plt.xlim([-2, 2])
                        plt.ylim([-2, 2])
                    if "latex" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif "frame" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7
                    )
                    plt.close()

                if "color" in style:
                    if model_config.particle_model_name == "PDE_O":
                        fig = plt.figure(figsize=(12, 12))
                        plt.scatter(
                            H1[:, 0].detach().cpu().numpy(),
                            H1[:, 1].detach().cpu().numpy(),
                            s=100,
                            c=np.sin(to_numpy(H1[:, 2])),
                            vmin=-1,
                            vmax=1,
                            cmap="viridis",
                        )
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Lut_Fig_{run}_{it}.jpg",
                            dpi=170.7,
                        )
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=10,
                            c="lawngreen",
                            alpha=0.75,
                        )
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Rot_{run}_Fig{it}.jpg",
                            dpi=170.7,
                        )
                        plt.close()

                    elif "PDE_N" in model_config.signal_model_name:
                        matplotlib.rcParams["savefig.pad_inches"] = 0
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                        plt.scatter(
                            to_numpy(X1[:, 1]),
                            to_numpy(X1[:, 0]),
                            s=200,
                            c=to_numpy(H1[:, 0]) * 3,
                            cmap="viridis",
                        )
                        plt.colorbar()
                        plt.xlim([-1.2, 1.2])
                        plt.ylim([-1.2, 1.2])
                        if "latex" in style:
                            plt.xlabel(r"$x$", fontsize=78)
                            plt.ylabel(r"$y$", fontsize=78)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                        elif "frame" in style:
                            plt.xlabel("x", fontsize=48)
                            plt.ylabel("y", fontsize=48)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                            ax.tick_params(axis="both", which="major", pad=15)
                            plt.text(
                                0,
                                1.1,
                                f"frame {it}",
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                                fontsize=48,
                            )
                        else:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()
                        num = f"{it:06}"
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=70,
                        )
                        plt.close()

                    elif (model_config.particle_model_name == "PDE_A") & (dimension == 3):
                        fig = plt.figure(figsize=(20, 10))

                        # Left panel: 3D view
                        ax1 = fig.add_subplot(121, projection="3d")
                        for n in range(n_particle_types):
                            ax1.scatter(
                                to_numpy(x[index_particles[n], 1]),
                                to_numpy(x[index_particles[n], 2]),
                                to_numpy(x[index_particles[n], 3]),
                                s=10,
                                color=cmap.color(n),
                                alpha=0.5,
                                edgecolors="none",
                            )
                        ax1.set_xlim([0, 1])
                        ax1.set_ylim([0, 1])
                        ax1.set_zlim([0, 1])
                        ax1.set_xlabel("X")
                        ax1.set_ylabel("Y")
                        ax1.set_zlabel("Z")

                        # Right panel: 2D cross-section (z slice at middle)
                        ax2 = fig.add_subplot(122)
                        z_center = 0.5
                        z_thickness = 0.1
                        for n in range(n_particle_types):
                            z_vals = to_numpy(x[index_particles[n], 3])
                            mask = np.abs(z_vals - z_center) < z_thickness
                            ax2.scatter(
                                to_numpy(x[index_particles[n], 1])[mask],
                                to_numpy(x[index_particles[n], 2])[mask],
                                s=15,
                                color=cmap.color(n),
                                alpha=0.7,
                                edgecolors="none",
                            )
                        ax2.set_xlim([0, 1])
                        ax2.set_ylim([0, 1])
                        ax2.set_xlabel("X")
                        ax2.set_ylabel("Y")
                        ax2.set_title(
                            f"Z cross-section ({z_center-z_thickness:.1f} < z < {z_center+z_thickness:.1f})"
                        )
                        ax2.set_aspect("equal")

                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg",
                            dpi=170.7,
                        )
                        plt.close()

                    else:
                        fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                        s_p = 25

                        for n in range(n_particle_types):
                            plt.scatter(
                                to_numpy(x[index_particles[n], 2]),
                                to_numpy(x[index_particles[n], 1]),
                                s=s_p,
                                color=cmap.color(n),
                            )
                        if training_config.particle_dropout > 0:
                            plt.scatter(
                                x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                                x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                s=25,
                                color="k",
                                alpha=0.75,
                            )
                            plt.plot(
                                x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                                x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                "+",
                                color="w",
                            )

                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        if "PDE_G" in model_config.particle_model_name:
                            plt.xlim([-2, 2])
                            plt.ylim([-2, 2])
                        if "latex" in style:
                            plt.xlabel(r"$x$", fontsize=78)
                            plt.ylabel(r"$y$", fontsize=78)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                        if "frame" in style:
                            plt.xlabel("x", fontsize=48)
                            plt.ylabel("y", fontsize=48)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                            ax.tick_params(axis="both", which="major", pad=15)
                            plt.text(
                                0,
                                1.1,
                                f"frame {it}",
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                                fontsize=48,
                            )
                        if "no_ticks" in style:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()

                        num = f"{it:06}"
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=80,
                        )
                        plt.close()

        if bSave:
            x_list = np.array(to_numpy(torch.stack(x_list)))
            y_list = np.array(to_numpy(torch.stack(y_list)))
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            if has_particle_dropout:
                torch.save(
                    x_removed_list,
                    f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
                )
                np.save(
                    f"graphs_data/{dataset_name}/particle_dropout_mask.npy",
                    particle_dropout_mask,
                )
                np.save(
                    f"graphs_data/{dataset_name}/inv_particle_dropout_mask.npy",
                    inv_particle_dropout_mask,
                )
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)

            torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")

    if "PDE_K" in model_config.particle_model_name:
        torch.save(
            connection_matrix_list,
            f"graphs_data/{dataset_name}/connection_matrix_list.pt",
        )
