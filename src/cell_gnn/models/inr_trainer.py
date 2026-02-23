"""Train an Implicit Neural Representation (INR) on a cell-gnn field.

Supports SIREN (siren_txy, siren_t) and instantNGP (ngp).
Optional gradient mode: learn scalar potential phi where -grad(phi) = target field.
"""

import os
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import linregress
from tqdm import trange

from cell_gnn.config import CellGNNConfig, INRConfig, INRType
from cell_gnn.zarr_io import load_simulation_data, load_raw_array


def _load_field(data_folder, field_name, run, x_ts):
    """Load target field by name. Returns (T, N, C) numpy array."""
    if field_name == 'residual':
        return load_raw_array(f'{data_folder}/residual_list_{run}')
    elif field_name == 'velocity':
        return x_ts.vel.numpy()
    elif field_name == 'y':
        return load_raw_array(f'{data_folder}/y_list_{run}')
    else:
        return load_raw_array(f'{data_folder}/{field_name}_list_{run}')


def _build_model(inr_cfg, input_dim, output_dim, device):
    """Construct the INR model based on config."""
    inr_type = inr_cfg.inr_type

    if inr_type in (INRType.SIREN_TXY, INRType.SIREN_T):
        from cell_gnn.models.Siren_Network import Siren
        model = Siren(
            in_features=input_dim,
            hidden_features=inr_cfg.hidden_dim_inr,
            hidden_layers=inr_cfg.n_layers_inr,
            out_features=output_dim,
            outermost_linear=True,
            first_omega_0=inr_cfg.omega_inr,
            hidden_omega_0=inr_cfg.omega_inr,
            learnable_omega=inr_cfg.omega_inr_learnable,
        ).to(device)
    elif inr_type == INRType.NGP:
        from cell_gnn.models.HashEncoding_Network import HashEncodingMLP
        model = HashEncodingMLP(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            n_levels=inr_cfg.ngp_n_levels,
            n_features_per_level=inr_cfg.ngp_n_features_per_level,
            log2_hashmap_size=inr_cfg.ngp_log2_hashmap_size,
            base_resolution=inr_cfg.ngp_base_resolution,
            per_level_scale=inr_cfg.ngp_per_level_scale,
            n_neurons=inr_cfg.ngp_n_neurons,
            n_hidden_layers=inr_cfg.ngp_n_hidden_layers,
        ).to(device)
    else:
        raise ValueError(f"Unknown INR type: {inr_type}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  INR model: {inr_type.value}, params: {n_params:,}')
    return model


def _predict_all(model, inr_cfg, n_frames, n_cells, n_components, dimension,
                 pos_data, device):
    """Predict on all frames. Returns (T, N, C) tensor."""
    inr_type = inr_cfg.inr_type
    gradient_mode = inr_cfg.inr_gradient_mode
    t_period = inr_cfg.inr_t_period
    xy_period = inr_cfg.inr_xy_period

    results = []
    with torch.no_grad() if not gradient_mode else torch.enable_grad():
        if inr_type == INRType.SIREN_T:
            for t_idx in range(n_frames):
                t_val = torch.tensor([[t_idx / n_frames / t_period]], device=device)
                pred = model(t_val).reshape(n_cells, n_components)
                results.append(pred.detach())
        else:
            for t_idx in range(n_frames):
                t_val = torch.full((n_cells, 1), t_idx / n_frames / t_period, device=device)
                pos_t = torch.tensor(pos_data[t_idx] / xy_period, dtype=torch.float32, device=device)

                if gradient_mode:
                    spatial = pos_t.clone().requires_grad_(True)
                    inp = torch.cat([t_val, spatial], dim=1)
                    phi = model(inp)
                    from cell_gnn.models.Siren_Network import gradient as siren_gradient
                    grad_phi = siren_gradient(phi, spatial)
                    results.append(-grad_phi.detach())
                else:
                    inp = torch.cat([t_val, pos_t], dim=1)
                    results.append(model(inp).detach())

    return torch.stack(results, dim=0)


def data_train_INR(config, device, field_name=None, run=0, erase=False):
    """Train an INR on a cell-gnn field.

    Args:
        config: CellGNNConfig
        device: torch device
        field_name: override field name (defaults to config.inr.inr_field_name)
        run: dataset run index
        erase: whether to erase existing INR outputs
    """
    sim = config.simulation
    inr_cfg = config.inr if config.inr else INRConfig()
    dimension = sim.dimension
    dataset_name = config.dataset
    data_folder = f'graphs_data/{dataset_name}'
    log_dir = f'log/{config.config_file}'

    if field_name is None:
        field_name = inr_cfg.inr_field_name

    inr_type = inr_cfg.inr_type
    gradient_mode = inr_cfg.inr_gradient_mode
    total_steps = inr_cfg.inr_total_steps
    batch_size = inr_cfg.inr_batch_size
    learning_rate = inr_cfg.inr_learning_rate
    t_period = inr_cfg.inr_t_period
    xy_period = inr_cfg.inr_xy_period

    print(f'training INR on field "{field_name}" ...')
    print(f'  type: {inr_type.value}, gradient_mode: {gradient_mode}')

    # --- output directories ---
    output_folder = f'./{log_dir}/tmp_training/inr'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f'./{log_dir}/models', exist_ok=True)
    if erase:
        for f in glob.glob(f'{output_folder}/*'):
            os.remove(f)

    # --- load data ---
    print(f'  loading data from {data_folder} ...')
    x_ts = load_simulation_data(f'{data_folder}/x_list_{run}', dimension)
    pos_data = x_ts.pos.numpy()  # (T, N, dim)
    field_data = _load_field(data_folder, field_name, run, x_ts)

    # ensure 3D shape
    if field_data.ndim == 2:
        field_data = field_data[:, :, np.newaxis]

    n_frames, n_cells, n_components = field_data.shape
    print(f'  data: {n_frames} frames, {n_cells} cells, {n_components} components, dim={dimension}')

    if batch_size > n_frames:
        batch_size = n_frames

    ground_truth = torch.tensor(field_data, dtype=torch.float32, device=device)

    # --- model construction ---
    if inr_type == INRType.SIREN_TXY:
        input_dim = 1 + dimension
        output_dim = 1 if gradient_mode else n_components
    elif inr_type == INRType.SIREN_T:
        if gradient_mode:
            raise ValueError("gradient_mode requires siren_txy or ngp (need spatial coords)")
        input_dim = 1
        output_dim = n_cells * n_components
    elif inr_type == INRType.NGP:
        input_dim = 1 + dimension
        output_dim = 1 if gradient_mode else n_components
    else:
        raise ValueError(f"Unknown INR type: {inr_type}")

    model = _build_model(inr_cfg, input_dim, output_dim, device)

    # --- optimizer ---
    omega_params = [p for name, p in model.named_parameters() if 'omega' in name]
    other_params = [p for name, p in model.named_parameters() if 'omega' not in name]

    if omega_params and inr_cfg.omega_inr_learnable:
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': omega_params, 'lr': inr_cfg.learning_rate_omega_inr}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.01
    )

    max_grad_norm = 0.5 if gradient_mode else 1.0

    # --- training loop ---
    loss_list = []
    report_interval = max(1, total_steps // 10)
    viz_interval = max(1, total_steps // 5)
    t_start = time.time()

    print(f'  training for {total_steps} steps, batch_size={batch_size} ...')

    for step in trange(total_steps + 1, ncols=100, desc=f'INR {field_name}'):
        optimizer.zero_grad()

        # sample random frames
        sample_ids = np.random.choice(n_frames, batch_size, replace=(batch_size > n_frames))
        gt_batch = ground_truth[sample_ids]  # (B, N, C)

        if inr_type == INRType.SIREN_T:
            # input: (B, 1) time, output: (B, N*C)
            t_batch = torch.tensor(sample_ids, dtype=torch.float32, device=device).unsqueeze(1) / n_frames / t_period
            pred_flat = model(t_batch)  # (B, N*C)
            pred = pred_flat.reshape(batch_size, n_cells, n_components)
            loss = F.mse_loss(pred, gt_batch)

        elif inr_type in (INRType.SIREN_TXY, INRType.NGP):
            # input: (B*N, 1+dim), output: (B*N, C)
            pos_batch = torch.tensor(
                pos_data[sample_ids] / xy_period,
                dtype=torch.float32, device=device
            )  # (B, N, dim)
            t_indices = torch.tensor(sample_ids, dtype=torch.float32, device=device)
            t_expanded = (t_indices / n_frames / t_period).unsqueeze(1).unsqueeze(2).expand(
                batch_size, n_cells, 1
            )  # (B, N, 1)

            pos_flat = pos_batch.reshape(-1, dimension)
            t_flat = t_expanded.reshape(-1, 1)
            gt_flat = gt_batch.reshape(-1, n_components)

            if gradient_mode:
                spatial = pos_flat.clone().requires_grad_(True)
                inp = torch.cat([t_flat, spatial], dim=1)
                phi = model(inp)  # (B*N, 1)
                from cell_gnn.models.Siren_Network import gradient as siren_gradient
                grad_phi = siren_gradient(phi, spatial)  # (B*N, dim)
                pred = -grad_phi
                loss = F.mse_loss(pred, gt_flat)
            else:
                inp = torch.cat([t_flat, pos_flat], dim=1)
                pred = model(inp)  # (B*N, C)
                loss = F.mse_loss(pred, gt_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()

        loss_list.append(loss.item())

        # --- progress report ---
        if step > 0 and step % report_interval == 0:
            pct = step / total_steps * 100
            elapsed = time.time() - t_start
            its = step / elapsed
            with torch.no_grad():
                pred_all = _predict_all(model, inr_cfg, n_frames, n_cells, n_components,
                                        dimension, pos_data, device)
                gt_all = ground_truth.cpu().numpy().reshape(-1)
                pred_all_np = pred_all.cpu().numpy().reshape(-1)
                _, _, r_value, _, _ = linregress(gt_all, pred_all_np)
                r2 = r_value ** 2
            print(f'  {pct:5.1f}% | step {step:6d}/{total_steps} | loss: {loss.item():.6f} | R2: {r2:.4f} | {its:.1f} it/s')

        # --- visualization ---
        if step > 0 and step % viz_interval == 0:
            from cell_gnn.plot_inr import plot_inr_training_summary
            with torch.no_grad():
                pred_all = _predict_all(model, inr_cfg, n_frames, n_cells, n_components,
                                        dimension, pos_data, device)
            plot_inr_training_summary(
                loss_list, ground_truth.cpu().numpy(), pred_all.cpu().numpy(),
                pos_data, field_name, inr_type.value, step, n_frames, n_cells,
                n_components, output_folder, gradient_mode=gradient_mode
            )

    # --- final evaluation ---
    print('  final evaluation ...')
    with torch.no_grad():
        pred_all = _predict_all(model, inr_cfg, n_frames, n_cells, n_components,
                                dimension, pos_data, device)

    gt_np = ground_truth.cpu().numpy()
    pred_np = pred_all.cpu().numpy()

    gt_flat = gt_np.reshape(-1)
    pred_flat = pred_np.reshape(-1)
    slope, intercept, r_value, _, _ = linregress(gt_flat, pred_flat)
    final_r2 = r_value ** 2
    final_mse = np.mean((gt_np - pred_np) ** 2)

    elapsed = time.time() - t_start
    print(f'  training complete: {elapsed / 60:.1f} min')
    print(f'  final MSE: {final_mse:.6e}, R2: {final_r2:.6f}')

    # --- save model ---
    model_path = f'./{log_dir}/models/inr_{field_name}.pt'
    torch.save(model.state_dict(), model_path)
    print(f'  model saved to {model_path}')

    # --- save predictions ---
    np.save(os.path.join(output_folder, f'pred_{field_name}.npy'), pred_np)
    np.save(os.path.join(output_folder, f'gt_{field_name}.npy'), gt_np)

    # --- final visualizations ---
    from cell_gnn.plot_inr import plot_inr_training_summary, plot_inr_kinograph

    plot_inr_training_summary(
        loss_list, gt_np, pred_np, pos_data,
        field_name, inr_type.value, total_steps, n_frames, n_cells,
        n_components, output_folder, gradient_mode=gradient_mode
    )

    plot_inr_kinograph(gt_np, pred_np, field_name, n_components, n_cells, output_folder)

    # gradient mode: save potential visualization for a few frames
    if gradient_mode:
        from cell_gnn.plot_inr import plot_inr_gradient_field
        viz_frames = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
        for t_idx in viz_frames:
            t_val = torch.full((n_cells, 1), t_idx / n_frames / t_period, device=device)
            pos_t = torch.tensor(pos_data[t_idx] / xy_period, dtype=torch.float32, device=device)
            spatial = pos_t.clone().requires_grad_(True)
            inp = torch.cat([t_val, spatial], dim=1)
            phi = model(inp)
            from cell_gnn.models.Siren_Network import gradient as siren_gradient
            grad_phi = siren_gradient(phi, spatial)
            pred_field = -grad_phi.detach().cpu().numpy()
            phi_np = phi.detach().cpu().numpy().squeeze()
            gt_field = gt_np[t_idx]
            pos_np_t = pos_data[t_idx]
            plot_inr_gradient_field(pos_np_t, gt_field, pred_field, phi_np, t_idx,
                                     dimension, output_folder)

    # --- results log ---
    results_path = os.path.join(output_folder, 'results.log')
    with open(results_path, 'w') as f:
        f.write(f'field_name: {field_name}\n')
        f.write(f'inr_type: {inr_type.value}\n')
        f.write(f'gradient_mode: {gradient_mode}\n')
        f.write(f'final_mse: {final_mse:.6e}\n')
        f.write(f'final_r2: {final_r2:.6f}\n')
        f.write(f'slope: {slope:.4f}\n')
        f.write(f'n_cells: {n_cells}\n')
        f.write(f'n_frames: {n_frames}\n')
        f.write(f'n_components: {n_components}\n')
        f.write(f'total_steps: {total_steps}\n')
        f.write(f'training_time_min: {elapsed / 60:.1f}\n')
    print(f'  results written to {results_path}')

    return model, loss_list
