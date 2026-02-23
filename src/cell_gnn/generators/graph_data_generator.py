import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import trange

import logging
from shutil import copyfile
from tifffile import imread

from cell_gnn.generators.utils import choose_model, init_cellstate, init_mesh
from cell_gnn.cell_state import CellState
from cell_gnn.zarr_io import ZarrSimulationWriterV3, ZarrArrayWriter
from cell_gnn.figure_style import default_style, dark_style
from cell_gnn.utils import (
    to_numpy,
    CustomColorMap,
    check_and_clear_memory,
    get_index_cells,
    fig_init,
    get_equidistant_points,
    get_edges_with_cache,
    edges_radius_blockwise,
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

    has_embryo_data = 'embryo' in config.data_folder_name
    has_external_data = config.data_folder_name != "none"
    has_cell_field = "field_ode" in config.graph_model.cell_model_name

    if has_embryo_data:
        from cell_gnn.generators.embryo_loader import load_from_embryo
        load_from_embryo(
            config,
            visualize=visualize,
            step=step,
            device=device,
            save=save,
        )
        return

    has_gland_data = 'gland' in config.data_folder_name
    if has_gland_data:
        from cell_gnn.generators.gland_loader import load_from_gland
        load_from_gland(
            config,
            visualize=visualize,
            step=step,
            device=device,
            save=save,
        )
        return

    if has_external_data:
        load_from_data(
            config,
            visualize=visualize,
            step=step,
            device=device,
            save=save,
            erase=erase,
        )
        return

    if has_cell_field:
        data_generate_cell_field(
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
        data_generate_cell(
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

    default_style.apply_globally()


def load_from_data(
    config,
    visualize=True,
    step=100,
    device=None,
    save=True,
    erase=False,
):
    """Load external simulation data from NPZ and write to zarr V3 format.

    Expects an NPZ file at graphs_data/{config.data_folder_name} with keys:
        X:     (T, N, dim) positions
        PDIR:  (T, N, dim) polarity direction
        FPAIR: (T, N, dim) pair forces
        params: dict with at least 'L' (box size)

    Positions are normalized to [0, 1] using the box size L.
    Velocity is computed from finite differences of normalized positions.

    Target selection depends on config.graph_model.prediction:
        "first_derivative"  → target = velocity  (dx/dt)
        "2nd_derivative"    → target = acceleration (d²x/dt²)
    """

    sim = config.simulation
    dataset_name = config.dataset
    data_file = config.data_folder_name
    dimension = sim.dimension
    n_cells = sim.n_cells
    delta_t = sim.delta_t
    cmap = CustomColorMap(config=config)

    print(f"loading data from graphs_data/{data_file} ...")

    # load NPZ
    data = np.load(f"graphs_data/{data_file}", allow_pickle=True)
    X = data['X']           # (T, N, dim) positions
    FPAIR = data['FPAIR']   # (T, N, dim) pair forces
    params = data['params'].item() if hasattr(data['params'], 'item') else data['params']
    L = float(params['L'])

    n_frames_total = X.shape[0]
    print(f"  source: {n_frames_total} frames, {X.shape[1]} cells, {X.shape[2]}D, L={L}")

    # normalize positions to [0, 1]
    pos = X / L

    # compute velocity from finite differences with periodic minimum image
    dpos = np.diff(pos, axis=0)                  # (T-1, N, dim)
    dpos = dpos - np.round(dpos)                 # minimum image in [0, 1]
    vel = dpos / delta_t
    vel = np.concatenate([vel, vel[-1:]], axis=0)  # repeat last frame velocity

    # determine target based on prediction type
    prediction = config.graph_model.prediction
    if prediction == "2nd_derivative":
        # acceleration = d(vel)/dt via finite differences
        dvel = np.diff(vel, axis=0)                    # (T-1, N, dim)
        acc = dvel / delta_t
        acc = np.concatenate([acc, acc[-1:]], axis=0)  # repeat last frame
        target = acc
        print(f"  prediction: {prediction} → target = acceleration (2nd derivative)")
    else:
        target = vel
        print(f"  prediction: {prediction} → target = velocity (1st derivative)")

    # prepare output directory
    folder = f"./graphs_data/{dataset_name}/"
    os.makedirs(folder, exist_ok=True)
    fig_folder = f"{folder}/Fig/"
    os.makedirs(fig_folder, exist_ok=True)
    if erase:
        for f in glob.glob(f"{fig_folder}*"):
            os.remove(f)

    if not save:
        print("save=False, skipping zarr write")
        return

    run = 0

    x_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{dataset_name}/x_list_{run}",
        n_cells=n_cells,
        dimension=dimension,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_list_{run}",
        n_cells=n_cells,
        n_features=dimension,
        time_chunks=2000,
    )

    cell_type = torch.zeros(n_cells, dtype=torch.long)
    _, bc_dpos = choose_boundary_values(sim.boundary)
    edge_counts_per_frame = []

    for t in trange(n_frames_total, ncols=100):
        state = CellState(
            index=torch.arange(n_cells, dtype=torch.long),
            pos=torch.tensor(pos[t], dtype=torch.float32),
            vel=torch.tensor(vel[t], dtype=torch.float32),
            cell_type=cell_type,
        )
        x_writer.append_state(state)
        y_writer.append(target[t].astype(np.float32))

        # collect edge stats every frame
        edge_index = edges_radius_blockwise(
            state.pos, bc_dpos, sim.min_radius, sim.max_radius, block=4096)
        n_edges = edge_index.shape[1] // 2
        degree = torch.zeros(n_cells, dtype=torch.long)
        degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
        edge_counts_per_frame.append({
            'n_edges': n_edges,
            'degree_min': int(degree.min()),
            'degree_max': int(degree.max()),
            'degree_mean': float(degree.float().mean()),
        })

        # 3D plot with edges
        if visualize and (t % step == 0):
            if dimension == 3:
                from mpl_toolkits.mplot3d.art3d import Line3DCollection
                from matplotlib.collections import LineCollection

                ei_np = to_numpy(edge_index)
                pos_all = pos[t]
                fwd = ei_np[0] < ei_np[1]
                ei_fwd = ei_np[:, fwd]
                dx = pos_all[ei_fwd[1]] - pos_all[ei_fwd[0]]
                no_wrap = np.sqrt((dx ** 2).sum(axis=1)) < sim.max_radius * 1.1
                ei_fwd = ei_fwd[:, no_wrap]

                fig = plt.figure(figsize=(12, 6), facecolor=default_style.background)

                ax1 = fig.add_subplot(121, projection="3d")
                ax1.scatter(
                    pos_all[:, 0], pos_all[:, 1], pos_all[:, 2],
                    s=10, color=cmap.color(0), alpha=0.5, edgecolors="none", zorder=2,
                )
                segments_3d = np.stack([pos_all[ei_fwd[0]], pos_all[ei_fwd[1]]], axis=1)
                lc3d = Line3DCollection(segments_3d, colors='#888888', linewidths=0.5, alpha=0.2)
                ax1.add_collection3d(lc3d)
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])
                ax1.set_zlim([0, 1])
                default_style.xlabel(ax1, "X")
                default_style.ylabel(ax1, "Y")
                ax1.set_zlabel("Z", fontsize=default_style.label_font_size, color=default_style.foreground)
                ax1.set_title(f"frame {t}", fontsize=default_style.font_size, color=default_style.foreground)

                ax2 = fig.add_subplot(122)
                z_center, z_thickness = 0.5, sim.max_radius / 2
                z_vals = pos_all[:, 2]
                z_mask = np.abs(z_vals - z_center) < z_thickness
                pos_slice = pos_all[z_mask, :2]
                ax2.scatter(
                    pos_slice[:, 0], pos_slice[:, 1],
                    s=15, color=cmap.color(0), alpha=0.7, edgecolors="none", zorder=2,
                )
                slice_set = set(np.where(z_mask)[0].tolist())
                slice_mask = np.array([ei_fwd[0, k] in slice_set and ei_fwd[1, k] in slice_set
                                       for k in range(ei_fwd.shape[1])])
                if slice_mask.any():
                    global_to_local = np.full(len(pos_all), -1, dtype=int)
                    global_to_local[z_mask] = np.arange(z_mask.sum())
                    ei_slice = ei_fwd[:, slice_mask]
                    src_local = global_to_local[ei_slice[0]]
                    dst_local = global_to_local[ei_slice[1]]
                    segments_2d = np.stack([pos_slice[src_local], pos_slice[dst_local]], axis=1)
                    lc2d = LineCollection(segments_2d, colors='#888888', linewidths=0.5, alpha=0.2)
                    ax2.add_collection(lc2d)
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                default_style.xlabel(ax2, "X")
                default_style.ylabel(ax2, "Y")
                ax2.set_title(
                    f"Z cross-section ({z_center - z_thickness:.1f} < z < {z_center + z_thickness:.1f})",
                    fontsize=default_style.font_size, color=default_style.foreground,
                )
                ax2.set_aspect("equal")

                plt.tight_layout()
                num = f"{t:06}"
                default_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png")

    n_written = x_writer.finalize()
    y_writer.finalize()
    print(f"loaded {n_written} frames from {data_file} (saved as .zarr)")

    if edge_counts_per_frame:
        edges_arr = np.array([s['n_edges'] for s in edge_counts_per_frame])
        deg_min_arr = np.array([s['degree_min'] for s in edge_counts_per_frame])
        deg_max_arr = np.array([s['degree_max'] for s in edge_counts_per_frame])
        deg_mean_arr = np.array([s['degree_mean'] for s in edge_counts_per_frame])
        print(f"edges per frame: min={edges_arr.min()} max={edges_arr.max()} mean={edges_arr.mean():.0f}")
        print(f"degree per cell: min={deg_min_arr.min()} max={deg_max_arr.max()} mean={deg_mean_arr.mean():.1f}")


def data_generate_cell(
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

    print(f"generating data ... {mc.cell_model_name}")

    dimension = sim.dimension
    max_radius = sim.max_radius
    min_radius = sim.min_radius
    n_cell_types = sim.n_cell_types
    n_cells = sim.n_cells
    delta_t = sim.delta_t
    n_frames = sim.n_frames
    has_cell_dropout = tc.cell_dropout > 0
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

    cell_dropout_mask = np.arange(n_cells)
    if has_cell_dropout:
        draw = np.random.permutation(np.arange(n_cells))
        cut = int(n_cells * (1 - tc.cell_dropout))
        cell_dropout_mask = draw[0:cut]
        inv_cell_dropout_mask = draw[cut:]
        x_removed_list = []

    if sim.angular_bernoulli != [-1]:
        b = sim.angular_bernoulli
        generative_m = np.array([stats.norm(b[0], b[2]), stats.norm(b[1], b[2])])

    run = 0

    check_and_clear_memory(
        device=device,
        iteration_number=0,
        every_n_iterations=250,
        memory_percentage_threshold=0.6,
    )

    n_cells = sim.n_cells

    # zarr V3 writers for incremental saving (memory efficient)
    x_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{dataset_name}/x_list_{run}",
        n_cells=n_cells,
        dimension=dimension,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_list_{run}",
        n_cells=n_cells,
        n_features=dimension,
        time_chunks=2000,
    )

    # initialize cell state
    x = init_cellstate(config=config, scenario=scenario, ratio=ratio, device=device)
    edge_cache = NeighborCache()

    edge_counts_per_frame = []

    time.sleep(0.5)
    for it in trange(sim.start_frame, n_frames + 1, ncols=100):
        # calculate type change
        if sim.state_type == "sequence":
            sample = torch.rand((n_cells, 1), device=device)
            sample = (
                sample < (1 / config.simulation.state_params[0])
            ) * torch.randint(0, n_cell_types, (n_cells, 1), device=device)
            x.cell_type = (x.cell_type + sample.squeeze(-1).long()) % n_cell_types

        x_packed = x.to_packed()
        index_cells = get_index_cells(
            x_packed, n_cell_types, dimension
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
                    f"N={x.n_cells:6d}, "
                    f"E={edge_index.shape[1]:8d}, "
                    f"time={(t1 - t0)*1000:.2f} ms"
                )

            # collect edge stats (bidirectional count / 2 = undirected edges)
            n_edges = edge_index.shape[1] // 2
            degree = torch.zeros(n_cells, dtype=torch.long, device=edge_index.device)
            degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long, device=edge_index.device))
            edge_counts_per_frame.append({
                'n_edges': n_edges,
                'degree_min': int(degree.min()),
                'degree_max': int(degree.max()),
                'degree_mean': float(degree.float().mean()),
            })

            # model prediction
            y = model(x, edge_index)

        if sim.angular_sigma > 0:
            phi = (
                torch.randn(n_cells, device=device)
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
        if sim.angular_bernoulli != [-1]:
            z_i = stats.bernoulli(b[3]).rvs(n_cells)
            phi = np.array([g.rvs() for g in generative_m[z_i]]) / 360 * np.pi * 2
            phi = torch.tensor(phi, device=device, dtype=torch.float32)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
            new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
            y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()

        # save frame to zarr
        if (it >= 0) & save:
            if has_cell_dropout:
                x_sub = x.subset(cell_dropout_mask).detach()
                x_sub.index = torch.arange(x_sub.n_cells, device=device)
                x_writer.append_state(x_sub)
                x_removed = x.subset(inv_cell_dropout_mask).detach()
                x_removed_list.append(x_removed.to_packed())
                y_writer.append(to_numpy(y[cell_dropout_mask].clone().detach()))
            else:
                x_writer.append_state(x.detach())
                y_writer.append(to_numpy(y.clone().detach()))

        # cell update
        if mc.prediction == "2nd_derivative":
            x.vel = x.vel + y * delta_t
        else:
            x.vel = y
        x.pos = bc_pos(x.pos + x.vel * delta_t)

        # output plots
        if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
            # recompute edges at current positions for visualization
            if "edge" in style:
                edge_index = edges_radius_blockwise(
                    x.pos, bc_dpos, min_radius, max_radius, block=4096)


            active_style = dark_style if "black" in style else default_style
            active_style.apply_globally()

            if "latex" in style:
                plt.rcParams["text.usetex"] = True
                plt.rcParams["font.family"] = "serif"
                plt.rcParams["font.serif"] = ["Palatino"]

            if "bw" in style:
                fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                s_p = 100
                for n in range(n_cell_types):
                    plt.scatter(
                        to_numpy(x.pos[index_cells[n], 0]),
                        to_numpy(x.pos[index_cells[n], 1]),
                        s=s_p,
                        color="k",
                    )
                if tc.cell_dropout > 0:
                    plt.scatter(
                        x.pos[inv_cell_dropout_mask, 0].detach().cpu().numpy(),
                        x.pos[inv_cell_dropout_mask, 1].detach().cpu().numpy(),
                        s=25,
                        color="k",
                        alpha=0.75,
                    )
                    plt.plot(
                        x.pos[inv_cell_dropout_mask, 0].detach().cpu().numpy(),
                        x.pos[inv_cell_dropout_mask, 1].detach().cpu().numpy(),
                        "+",
                        color="w",
                    )
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                if "gravity_ode" in mc.cell_model_name:
                    plt.xlim([-2, 2])
                    plt.ylim([-2, 2])
                if "latex" in style or "frame" in style:
                    active_style.xlabel(ax, r"$x$", fontsize=active_style.frame_title_font_size * 1.6)
                    active_style.ylabel(ax, r"$y$", fontsize=active_style.frame_title_font_size * 1.6)
                    ax.tick_params(axis="both", labelsize=active_style.frame_title_font_size)
                else:
                    plt.xticks([])
                    plt.yticks([])
                plt.tight_layout()
                active_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg")


            if "color" in style:
                if mc.cell_model_name == "PDE_O":
                    fig, ax = active_style.figure(height=12)
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
                    active_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Lut_Fig_{run}_{it}.jpg")

                    fig, ax = active_style.figure(height=12)
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
                    active_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Rot_{run}_Fig{it}.jpg")

                elif (mc.cell_model_name == "arbitrary_ode") & (dimension == 3):
                    from mpl_toolkits.mplot3d.art3d import Line3DCollection
                    from matplotlib.collections import LineCollection as LC

                    fig, _ = active_style.figure(width=12, height=6)
                    pos_np = to_numpy(x.pos)

                    # prepare edge segments for drawing
                    ei_fwd = None
                    if "edge" in style:
                        ei_np = to_numpy(edge_index)
                        fwd_m = ei_np[0] < ei_np[1]
                        ei_fwd = ei_np[:, fwd_m]
                        dx = pos_np[ei_fwd[1]] - pos_np[ei_fwd[0]]
                        no_wrap = np.sqrt((dx ** 2).sum(axis=1)) < max_radius * 1.1
                        ei_fwd = ei_fwd[:, no_wrap]

                    # Left panel: 3D view
                    ax1 = fig.add_subplot(121, projection="3d")
                    if ei_fwd is not None:
                        seg3d = np.stack([pos_np[ei_fwd[0]], pos_np[ei_fwd[1]]], axis=1)
                        ax1.add_collection3d(Line3DCollection(seg3d, colors='#888888', linewidths=0.5, alpha=0.2))
                    for n in range(n_cell_types):
                        ax1.scatter(
                            to_numpy(x.pos[index_cells[n], 0]),
                            to_numpy(x.pos[index_cells[n], 1]),
                            to_numpy(x.pos[index_cells[n], 2]),
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
                    z_vals = pos_np[:, 2]
                    z_mask = np.abs(z_vals - z_center) < z_thickness
                    if ei_fwd is not None:
                        slice_set = set(np.where(z_mask)[0].tolist())
                        sl_mask = np.array([ei_fwd[0, k] in slice_set and ei_fwd[1, k] in slice_set
                                            for k in range(ei_fwd.shape[1])])
                        if sl_mask.any():
                            ei_sl = ei_fwd[:, sl_mask]
                            seg2d = np.stack([pos_np[ei_sl[0], :2], pos_np[ei_sl[1], :2]], axis=1)
                            ax2.add_collection(LC(seg2d, colors='#888888', linewidths=0.5, alpha=0.2))
                    for n in range(n_cell_types):
                        z_n = to_numpy(x.pos[index_cells[n], 2])
                        mask = np.abs(z_n - z_center) < z_thickness
                        ax2.scatter(
                            to_numpy(x.pos[index_cells[n], 0])[mask],
                            to_numpy(x.pos[index_cells[n], 1])[mask],
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
                    active_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.png")

                else:
                    from matplotlib.collections import LineCollection as LC2

                    fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                    s_p = 25

                    if "edge" in style:
                        pos_np = to_numpy(x.pos)
                        ei_np = to_numpy(edge_index)
                        fwd_m = ei_np[0] < ei_np[1]
                        ei_fwd = ei_np[:, fwd_m]
                        dx = pos_np[ei_fwd[1]] - pos_np[ei_fwd[0]]
                        no_wrap = np.sqrt((dx ** 2).sum(axis=1)) < max_radius * 1.1
                        ei_fwd = ei_fwd[:, no_wrap]
                        # note: plot uses (y, x) ordering
                        seg = np.stack([pos_np[ei_fwd[0]][:, [1, 0]], pos_np[ei_fwd[1]][:, [1, 0]]], axis=1)
                        ax.add_collection(LC2(seg, colors='#888888', linewidths=0.5, alpha=0.2))

                    for n in range(n_cell_types):
                        plt.scatter(
                            to_numpy(x.pos[index_cells[n], 1]),
                            to_numpy(x.pos[index_cells[n], 0]),
                            s=s_p,
                            color=cmap.color(n),
                        )
                    if tc.cell_dropout > 0:
                        plt.scatter(
                            x.pos[inv_cell_dropout_mask, 1].detach().cpu().numpy(),
                            x.pos[inv_cell_dropout_mask, 0].detach().cpu().numpy(),
                            s=25,
                            color="k",
                            alpha=0.75,
                        )
                        plt.plot(
                            x.pos[inv_cell_dropout_mask, 1].detach().cpu().numpy(),
                            x.pos[inv_cell_dropout_mask, 0].detach().cpu().numpy(),
                            "+",
                            color="w",
                        )

                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if "gravity_ode" in mc.cell_model_name:
                        plt.xlim([-2, 2])
                        plt.ylim([-2, 2])
                    if "latex" in style:
                        active_style.xlabel(ax, r"$x$", fontsize=active_style.frame_title_font_size * 1.6)
                        active_style.ylabel(ax, r"$y$", fontsize=active_style.frame_title_font_size * 1.6)
                        ax.tick_params(axis="both", labelsize=active_style.frame_title_font_size)
                    if "frame" in style:
                        active_style.xlabel(ax, "x", fontsize=active_style.frame_title_font_size)
                        active_style.ylabel(ax, "y", fontsize=active_style.frame_title_font_size)
                        ax.tick_params(axis="both", labelsize=active_style.frame_title_font_size, pad=15)
                        active_style.annotate(ax, f"frame {it}", (0, 1.1),
                            ha="left", va="top",
                            fontsize=active_style.frame_title_font_size)
                    if "no_ticks" in style:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()

                    num = f"{it:06}"
                    active_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png")

    if save:
        # finalize zarr writers
        n_frames_written = x_writer.finalize()
        y_writer.finalize()
        print(f"generated {n_frames_written} frames total (saved as .zarr)")

        if has_cell_dropout:
            torch.save(
                x_removed_list,
                f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
            )
            np.save(
                f"graphs_data/{dataset_name}/cell_dropout_mask.npy",
                cell_dropout_mask,
            )
            np.save(
                f"graphs_data/{dataset_name}/inv_cell_dropout_mask.npy",
                inv_cell_dropout_mask,
            )

        torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")

        # print edge statistics
        if edge_counts_per_frame:
            edges_arr = np.array([s['n_edges'] for s in edge_counts_per_frame])
            deg_min_arr = np.array([s['degree_min'] for s in edge_counts_per_frame])
            deg_max_arr = np.array([s['degree_max'] for s in edge_counts_per_frame])
            deg_mean_arr = np.array([s['degree_mean'] for s in edge_counts_per_frame])
            print(f"edges per frame: min={edges_arr.min()} max={edges_arr.max()} mean={edges_arr.mean():.0f}")
            print(f"degree per cell: min={deg_min_arr.min()} max={deg_max_arr.max()} mean={deg_mean_arr.mean():.1f}")


def data_generate_cell_field(
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
        f"generating data ... {mc.cell_model_name}"
    )

    dimension = sim.dimension
    max_radius = sim.max_radius
    min_radius = sim.min_radius
    n_cell_types = sim.n_cell_types
    n_cells = sim.n_cells
    n_nodes = sim.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    delta_t = sim.delta_t
    n_frames = sim.n_frames
    has_cell_dropout = tc.cell_dropout > 0
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

    if has_cell_dropout:
        draw = np.random.permutation(np.arange(n_cells))
        cut = int(n_cells * (1 - tc.cell_dropout))
        cell_dropout_mask = draw[0:cut]
        inv_cell_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        cell_dropout_mask = np.arange(n_cells)

    run = 0
    n_cells = sim.n_cells

    # zarr V3 writers for incremental saving (memory efficient)
    x_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{dataset_name}/x_list_{run}",
        n_cells=n_cells,
        dimension=dimension,
        time_chunks=2000,
    )
    y_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_list_{run}",
        n_cells=n_cells,
        n_features=dimension,
        time_chunks=2000,
    )
    x_mesh_writer = ZarrSimulationWriterV3(
        path=f"graphs_data/{dataset_name}/x_mesh_list_{run}",
        n_cells=n_nodes,
        dimension=dimension,
        time_chunks=2000,
    )
    y_mesh_writer = ZarrArrayWriter(
        path=f"graphs_data/{dataset_name}/y_mesh_list_{run}",
        n_cells=n_nodes,
        n_features=2,
        time_chunks=2000,
    )
    edge_p_p_list = []
    edge_f_p_list = []
    id_fig = 0

    x = init_cellstate(config=config, scenario=scenario, ratio=ratio, device=device)

    if sim.shuffle_cell_types:
        shuffle_index = torch.randperm(n_cells, device=device)
        x.cell_type = x.cell_type[shuffle_index]

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
        index_cells = get_index_cells(x_packed, n_cell_types, dimension)

        x_mesh_packed = mesh_state.clone().detach().to_packed()
        x_pf_packed = torch.concatenate((x_mesh_packed, x_packed), dim=0)
        x_pf_state = CellState.from_packed(x_pf_packed, dimension)

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

            if not has_cell_dropout:
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
            if not has_cell_dropout:
                edge_f_p_list.append(edge_index_fp)

            y0 = model(x, edge_index, has_field=False)
            y1 = model(x_pf_state, edge_index_fp, has_field=True)[n_nodes:]
            y = y0 + y1

        # save frame to zarr
        if (it >= 0) & save:
            if has_cell_dropout:
                x_removed = x.subset(inv_cell_dropout_mask).detach()
                x_removed_list.append(x_removed.to_packed())
                x_sub = x.subset(cell_dropout_mask).detach()
                x_sub.index = torch.arange(x_sub.n_cells, device=device)
                x_writer.append_state(x_sub)
                y_writer.append(to_numpy(y[cell_dropout_mask].clone().detach()))

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
                x_pf_dropout_state = CellState.from_packed(x_pf_dropout, dimension)
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

        # cell update
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

            fig, ax = default_style.figure(height=12, formatx="%.1f", formaty="%.1f")
            ax.tick_params(axis="both", which="major", pad=15)
            s_p = 20
            for n in range(n_cell_types):
                plt.scatter(
                    to_numpy(x.pos[index_cells[n], 1]),
                    to_numpy(x.pos[index_cells[n], 0]),
                    s=s_p,
                    color=cmap.color(n),
                )
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            default_style.savefig(fig, f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg")

    if save:
        # finalize zarr writers
        n_frames_written = x_writer.finalize()
        y_writer.finalize()
        x_mesh_writer.finalize()
        y_mesh_writer.finalize()
        print(f"generated {n_frames_written} frames total (saved as .zarr)")

        if has_cell_dropout:
            torch.save(
                x_removed_list,
                f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
            )
            np.save(
                f"graphs_data/{dataset_name}/cell_dropout_mask.npy",
                cell_dropout_mask,
            )
            np.save(
                f"graphs_data/{dataset_name}/inv_cell_dropout_mask.npy",
                inv_cell_dropout_mask,
            )

        torch.save(
            edge_p_p_list, f"graphs_data/{dataset_name}/edge_p_p_list{run}.pt"
        )
        torch.save(
            edge_f_p_list, f"graphs_data/{dataset_name}/edge_f_p_list{run}.pt"
        )

    print('data generated...')
