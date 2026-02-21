import glob
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from tqdm import trange

import logging
from shutil import copyfile
from tifffile import imread

from particle_gnn.generators.utils import choose_model, init_particlestate, init_mesh
from particle_gnn.particle_state import ParticleState
from particle_gnn.zarr_io import ZarrSimulationWriterV3, ZarrArrayWriter
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
    save=True,
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

    has_particle_field = "PDE_ParticleField" in config.graph_model.particle_model_name

    if has_particle_field:
        data_generate_particle_field(
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
            save=save,
            timer=timer,
        )
    else:
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
            save=save,
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
    save=True,
    timer=False,
):
    sim = config.simulation
    tc = config.training
    mc = config.graph_model

    print(f"generating data ... {mc.particle_model_name}")

    dimension = sim.dimension
    max_radius = sim.max_radius
    min_radius = sim.min_radius
    n_particle_types = sim.n_particle_types
    n_particles = sim.n_particles
    delta_t = sim.delta_t
    n_frames = sim.n_frames
    has_particle_dropout = tc.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        import shutil
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
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
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
        cut = int(n_particles * (1 - tc.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []

    if sim.angular_Bernouilli != [-1]:
        b = sim.angular_Bernouilli
        generative_m = np.array([stats.norm(b[0], b[2]), stats.norm(b[1], b[2])])

    run = 0

    check_and_clear_memory(
        device=device,
        iteration_number=0,
        every_n_iterations=250,
        memory_percentage_threshold=0.6,
    )

    n_particles = sim.n_particles

    # zarr V3 writers for incremental saving (memory efficient)
    x_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{dataset_name}/x_list_{run}",
        n_particles=n_particles,
        dimension=dimension,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_list_{run}",
        n_particles=n_particles,
        n_features=dimension,
        time_chunks=2000,
    )

    # initialize particle state
    x = init_particlestate(config=config, scenario=scenario, ratio=ratio, device=device)
    edge_cache = NeighborCache()

    time.sleep(0.5)
    for it in trange(sim.start_frame, n_frames + 1, ncols=100):
        # calculate type change
        if sim.state_type == "sequence":
            sample = torch.rand((n_particles, 1), device=device)
            sample = (
                sample < (1 / config.simulation.state_params[0])
            ) * torch.randint(0, n_particle_types, (n_particles, 1), device=device)
            x.particle_type = (x.particle_type + sample.squeeze(-1).long()) % n_particle_types

        x_packed = x.to_packed()
        index_particles = get_index_particles(
            x_packed, n_particle_types, dimension
        )  # can be different from frame to frame

        # compute connectivity rule
        with torch.no_grad():

            if timer:
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            edge_index = get_edges_with_cache(
                pos=x.pos,
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
                    f"N={x.n_particles:6d}, "
                    f"E={edge_index.shape[1]:8d}, "
                    f"time={(t1 - t0)*1000:.2f} ms"
                )

            # model prediction
            y = model(x, edge_index)

        if sim.angular_sigma > 0:
            phi = (
                torch.randn(n_particles, device=device)
                * sim.angular_sigma
                / 360
                * np.pi
                * 2
            )
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
            new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
            y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()
        if sim.angular_Bernouilli != [-1]:
            z_i = stats.bernoulli(b[3]).rvs(n_particles)
            phi = np.array([g.rvs() for g in generative_m[z_i]]) / 360 * np.pi * 2
            phi = torch.tensor(phi, device=device, dtype=torch.float32)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
            new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
            y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()

        # save frame to zarr
        if (it >= 0) & save:
            if has_particle_dropout:
                x_sub = x.subset(particle_dropout_mask).detach()
                x_sub.index = torch.arange(x_sub.n_particles, device=device)
                x_writer.append_state(x_sub)
                x_removed = x.subset(inv_particle_dropout_mask).detach()
                x_removed_list.append(x_removed.to_packed())
                y_writer.append(to_numpy(y[particle_dropout_mask].clone().detach()))
            else:
                x_writer.append_state(x.detach())
                y_writer.append(to_numpy(y.clone().detach()))

        # particle update
        if mc.prediction == "2nd_derivative":
            x.vel = x.vel + y * delta_t
        else:
            x.vel = y
        x.pos = bc_pos(x.pos + x.vel * delta_t)

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
                        to_numpy(x.pos[index_particles[n], 0]),
                        to_numpy(x.pos[index_particles[n], 1]),
                        s=s_p,
                        color="k",
                    )
                if tc.particle_dropout > 0:
                    plt.scatter(
                        x.pos[inv_particle_dropout_mask, 0].detach().cpu().numpy(),
                        x.pos[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                        s=25,
                        color="k",
                        alpha=0.75,
                    )
                    plt.plot(
                        x.pos[inv_particle_dropout_mask, 0].detach().cpu().numpy(),
                        x.pos[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                        "+",
                        color="w",
                    )
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                if "PDE_G" in mc.particle_model_name:
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
                if mc.particle_model_name == "PDE_O":
                    fig = plt.figure(figsize=(12, 12))
                    plt.scatter(
                        x.field[:, 0].detach().cpu().numpy(),
                        x.field[:, 1].detach().cpu().numpy(),
                        s=100,
                        c=np.sin(to_numpy(x.field[:, 2])),
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
                        to_numpy(x.pos[:, 0]),
                        to_numpy(x.pos[:, 1]),
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

                elif (mc.particle_model_name == "PDE_A") & (dimension == 3):
                    fig = plt.figure(figsize=(20, 10))

                    # Left panel: 3D view
                    ax1 = fig.add_subplot(121, projection="3d")
                    for n in range(n_particle_types):
                        ax1.scatter(
                            to_numpy(x.pos[index_particles[n], 0]),
                            to_numpy(x.pos[index_particles[n], 1]),
                            to_numpy(x.pos[index_particles[n], 2]),
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
                        z_vals = to_numpy(x.pos[index_particles[n], 2])
                        mask = np.abs(z_vals - z_center) < z_thickness
                        ax2.scatter(
                            to_numpy(x.pos[index_particles[n], 0])[mask],
                            to_numpy(x.pos[index_particles[n], 1])[mask],
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
                            to_numpy(x.pos[index_particles[n], 1]),
                            to_numpy(x.pos[index_particles[n], 0]),
                            s=s_p,
                            color=cmap.color(n),
                        )
                    if tc.particle_dropout > 0:
                        plt.scatter(
                            x.pos[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x.pos[inv_particle_dropout_mask, 0].detach().cpu().numpy(),
                            s=25,
                            color="k",
                            alpha=0.75,
                        )
                        plt.plot(
                            x.pos[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x.pos[inv_particle_dropout_mask, 0].detach().cpu().numpy(),
                            "+",
                            color="w",
                        )

                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if "PDE_G" in mc.particle_model_name:
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

    if save:
        # finalize zarr writers
        n_frames_written = x_writer.finalize()
        y_writer.finalize()
        print(f"generated {n_frames_written} frames total (saved as .zarr)")

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

        torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")


def data_generate_particle_field(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=1,
    alpha=0.2,
    ratio=1,
    scenario="none",
    device=None,
    save=True,
    timer=False
):

    sim = config.simulation
    tc = config.training
    mc = config.graph_model

    print(
        f"generating data ... {mc.particle_model_name}"
    )

    dimension = sim.dimension
    max_radius = sim.max_radius
    min_radius = sim.min_radius
    n_particle_types = sim.n_particle_types
    n_particles = sim.n_particles
    n_nodes = sim.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    delta_t = sim.delta_t
    n_frames = sim.n_frames
    has_particle_dropout = tc.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    bounce = sim.bounce
    bounce_coeff = sim.bounce_coeff

    # Create log directory
    log_dir = f"./graphs_data/{dataset_name}/"
    log_file = f"{log_dir}/generator.log"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=log_file, format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        import shutil as _shutil
        files = glob.glob(f"{folder}/Fig/*")
        for f in files:
            if (f[-14:] != "generated_data") & (f != "p.pt") & (f != "model_config.json") & (f != "generation_code.py"):
                if os.path.isdir(f):
                    _shutil.rmtree(f)
                else:
                    os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, "generation_code.py"))

    mesh_state, mesh_data = init_mesh(config, device=device)
    mask_mesh = mesh_data["mask"].squeeze()

    if hasattr(sim, 'boundary') and sim.boundary == 'periodic':
        mask_mesh = torch.ones_like(mask_mesh, dtype=torch.bool)

    model, bc_pos, bc_dpos = choose_model(config=config, device=device)

    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - tc.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        particle_dropout_mask = np.arange(n_particles)

    run = 0
    n_particles = sim.n_particles

    # zarr V3 writers for incremental saving (memory efficient)
    x_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{dataset_name}/x_list_{run}",
        n_particles=n_particles,
        dimension=dimension,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_list_{run}",
        n_particles=n_particles,
        n_features=dimension,
        time_chunks=2000,
    )
    x_mesh_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{dataset_name}/x_mesh_list_{run}",
        n_particles=n_nodes,
        dimension=dimension,
        time_chunks=2000,
    )
    y_mesh_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_mesh_list_{run}",
        n_particles=n_nodes,
        n_features=2,
        time_chunks=2000,
    )
    edge_p_p_list = []
    edge_f_p_list = []
    id_fig = 0

    x = init_particlestate(config=config, scenario=scenario, ratio=ratio, device=device)

    if sim.shuffle_particle_types:
        shuffle_index = torch.randperm(n_particles, device=device)
        x.particle_type = x.particle_type[shuffle_index]

    mesh_state.field[mask_mesh == 0.0] = 0.0
    edge_cache = NeighborCache()

    torch.save(mesh_data, f"graphs_data/{dataset_name}/mesh_data_{run}.pt")

    check_and_clear_memory(
        device=device,
        iteration_number=0,
        every_n_iterations=250,
        memory_percentage_threshold=0.6,
    )
    time.sleep(1)

    for it in trange(sim.start_frame, n_frames + 1, ncols=100):
        if ("siren" in mc.field_type) & (it >= 0):
            im = imread(
                f"graphs_data/{sim.node_value_map}"
            )
            im = im[it].squeeze()
            im = np.rot90(im, 3)
            im = np.reshape(im, (n_nodes_per_axis * n_nodes_per_axis))
            mesh_state.field[:, 0:1] = torch.tensor(
                im[:, None], dtype=torch.float32, device=device
            )

        x_packed = x.to_packed()
        index_particles = get_index_particles(x_packed, n_particle_types, dimension)

        x_mesh_packed = mesh_state.clone().detach().to_packed()
        x_pf_packed = torch.concatenate((x_mesh_packed, x_packed), dim=0)
        x_pf_state = ParticleState.from_packed(x_pf_packed, dimension)

        with torch.no_grad():

            edge_index = get_edges_with_cache(
                pos=x.pos,
                bc_dpos=bc_dpos,
                cache=edge_cache,
                r_cut=max_radius,
                r_skin=0.05,
                min_radius=min_radius,
                block=2048
            )

            if not has_particle_dropout:
                edge_p_p_list.append(edge_index)

            pos_pf = x_pf_state.pos
            distance = torch.sum(
                bc_dpos(
                    pos_pf[:, None, :]
                    - pos_pf[None, :, :]
                )
                ** 2,
                dim=2,
            )
            adj_t = (
                (distance < (max_radius / 2) ** 2) & (distance > min_radius**2)
            ).float() * 1
            edge_index_fp = adj_t.nonzero().t().contiguous()
            pos_fp = torch.argwhere(
                (edge_index_fp[1, :] >= n_nodes) & (edge_index_fp[0, :] < n_nodes)
            )
            pos_fp = to_numpy(pos_fp[:, 0])
            edge_index_fp = edge_index_fp[:, pos_fp]
            if not has_particle_dropout:
                edge_f_p_list.append(edge_index_fp)

            y0 = model(x, edge_index, has_field=False)
            y1 = model(x_pf_state, edge_index_fp, has_field=True)[n_nodes:]
            y = y0 + y1

        # save frame to zarr
        if (it >= 0) & save:
            if has_particle_dropout:
                x_removed = x.subset(inv_particle_dropout_mask).detach()
                x_removed_list.append(x_removed.to_packed())
                x_sub = x.subset(particle_dropout_mask).detach()
                x_sub.index = torch.arange(x_sub.n_particles, device=device)
                x_writer.append_state(x_sub)
                y_writer.append(to_numpy(y[particle_dropout_mask].clone().detach()))

                x_sub_packed = x_sub.to_packed()
                distance = torch.sum(
                    bc_dpos(
                        x_sub.pos[:, None, :]
                        - x_sub.pos[None, :, :]
                    )
                    ** 2,
                    dim=2,
                )
                adj_t = (
                    (distance < max_radius**2) & (distance > min_radius**2)
                ).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                edge_p_p_list.append(edge_index)

                x_pf_dropout = torch.concatenate((x_mesh_packed, x_sub_packed), dim=0)
                x_pf_dropout_state = ParticleState.from_packed(x_pf_dropout, dimension)
                pos_pf = x_pf_dropout_state.pos

                distance = torch.sum(
                    bc_dpos(
                        pos_pf[:, None, :]
                        - pos_pf[None, :, :]
                    )
                    ** 2,
                    dim=2,
                )
                adj_t = (
                    (distance < (max_radius / 2) ** 2) & (distance > min_radius**2)
                ).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                pos_idx = torch.argwhere(
                    (edge_index[1, :] >= n_nodes) & (edge_index[0, :] < n_nodes)
                )
                pos_idx = to_numpy(pos_idx[:, 0])
                edge_index = edge_index[:, pos_idx]
                edge_f_p_list.append(edge_index)
            else:
                x_writer.append_state(x.detach())
                y_writer.append(to_numpy(y.clone().detach()))

            x_mesh_writer.append_state(mesh_state.detach())
            y_mesh_writer.append(np.zeros((n_nodes, 2), dtype=np.float32))

        # particle update
        with torch.no_grad():
            if mc.prediction == "2nd_derivative":
                x.vel = x.vel + y * delta_t
            else:
                x.vel = y

            if bounce:
                x.pos = x.pos + x.vel * delta_t
                gap = 0.005
                bouncing_pos = torch.argwhere(
                    (x.pos[:, 0] <= 0.1 + gap) | (x.pos[:, 0] >= 0.9 - gap)
                ).squeeze()
                if bouncing_pos.numel() > 0:
                    x.vel[bouncing_pos, 0] = -0.7 * bounce_coeff * x.vel[bouncing_pos, 0]
                    x.pos[bouncing_pos, 0] = x.pos[bouncing_pos, 0] + x.vel[bouncing_pos, 0] * delta_t * 10
                bouncing_pos = torch.argwhere(
                    (x.pos[:, 1] <= 0.1 + gap) | (x.pos[:, 1] >= 0.9 - gap)
                ).squeeze()
                if bouncing_pos.numel() > 0:
                    x.vel[bouncing_pos, 1] = -0.7 * bounce_coeff * x.vel[bouncing_pos, 1]
                    x.pos[bouncing_pos, 1] = x.pos[bouncing_pos, 1] + x.vel[bouncing_pos, 1] * delta_t * 10
            else:
                x.pos = bc_pos(x.pos + x.vel * delta_t)

        if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
            num = f"{id_fig:06}"
            id_fig += 1

            matplotlib.rcParams["savefig.pad_inches"] = 0
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", which="major", pad=15)
            s_p = 20
            for n in range(n_particle_types):
                plt.scatter(
                    to_numpy(x.pos[index_particles[n], 1]),
                    to_numpy(x.pos[index_particles[n], 0]),
                    s=s_p,
                    color=cmap.color(n),
                )
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(
                f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7
            )
            plt.close()

    if save:
        # finalize zarr writers
        n_frames_written = x_writer.finalize()
        y_writer.finalize()
        x_mesh_writer.finalize()
        y_mesh_writer.finalize()
        print(f"generated {n_frames_written} frames total (saved as .zarr)")

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

        torch.save(
            edge_p_p_list, f"graphs_data/{dataset_name}/edge_p_p_list{run}.pt"
        )
        torch.save(
            edge_f_p_list, f"graphs_data/{dataset_name}/edge_f_p_list{run}.pt"
        )

    print('data generated...')
