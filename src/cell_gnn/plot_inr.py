"""Visualization functions for INR (Implicit Neural Representation) training."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from cell_gnn.figure_style import default_style, dark_style


def compute_kinograph_metrics(gt, pred):
    """Compare kinograph matrices [n_entities, n_frames].

    Returns dict with 'r2' (mean per-frame R2) and 'ssim' (structural similarity).
    """
    try:
        from skimage.metrics import structural_similarity
        has_ssim = True
    except ImportError:
        has_ssim = False

    n_entities, n_frames = gt.shape
    r2_list = []
    for t in range(n_frames):
        gt_col = gt[:, t]
        pred_col = pred[:, t]
        ss_tot = np.sum((gt_col - np.mean(gt_col)) ** 2)
        if ss_tot > 0:
            ss_res = np.sum((gt_col - pred_col) ** 2)
            r2_list.append(1 - ss_res / ss_tot)
    r2_mean = np.mean(r2_list) if r2_list else 0.0

    ssim_val = 0.0
    if has_ssim and gt.shape[0] > 1 and gt.shape[1] > 1:
        data_range = max(np.abs(gt).max(), np.abs(pred).max()) * 2
        if data_range > 0:
            ssim_val = structural_similarity(gt, pred, data_range=data_range)

    return {'r2': r2_mean, 'ssim': ssim_val}


def _smooth(values, window=50):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_inr_training_summary(loss_list, gt_np, pred_np, pos_np,
                               field_name, inr_type, step, n_frames, n_cells,
                               n_components, output_folder, gradient_mode=False):
    """2x3 training summary panel on dark background.

    Args:
        loss_list: list of loss values
        gt_np: (T, N, C) ground truth
        pred_np: (T, N, C) predictions
        pos_np: (T, N, dim) positions
        field_name: name of the field
        inr_type: 'siren_txy', 'siren_t', or 'ngp'
        step: current training step
        n_frames, n_cells, n_components: data dimensions
        output_folder: directory for output images
        gradient_mode: whether gradient mode was used
    """
    style = dark_style
    style.apply_globally()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=style.background)
    for ax in axes.flat:
        ax.set_facecolor(style.background)
        ax.tick_params(colors=style.foreground, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(style.foreground)

    # top-left: loss curve
    ax = axes[0, 0]
    ax.plot(loss_list, color='white', alpha=0.3, linewidth=0.5)
    if len(loss_list) > 100:
        smoothed = _smooth(loss_list, window=max(1, len(loss_list) // 50))
        ax.plot(np.linspace(0, len(loss_list), len(smoothed)), smoothed, color='r', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_title('loss', fontsize=10, color=style.foreground)
    ax.set_xlabel('step', fontsize=8, color=style.foreground)

    # top-middle: per-frame MSE
    ax = axes[0, 1]
    per_frame_mse = np.mean((gt_np - pred_np) ** 2, axis=(1, 2))
    ax.plot(per_frame_mse, color='cyan', linewidth=0.8)
    ax.set_title('per-frame mse', fontsize=10, color=style.foreground)
    ax.set_xlabel('frame', fontsize=8, color=style.foreground)

    # top-right: info text
    ax = axes[0, 2]
    ax.axis('off')
    n_params = '?'
    gt_flat = gt_np.reshape(-1)
    pred_flat = pred_np.reshape(-1)
    slope, intercept, r_value, _, _ = linregress(gt_flat, pred_flat)
    r2 = r_value ** 2

    info_text = (
        f"field: {field_name}\n"
        f"inr: {inr_type}\n"
        f"gradient mode: {gradient_mode}\n"
        f"step: {step}\n"
        f"frames: {n_frames}\n"
        f"cells: {n_cells}\n"
        f"components: {n_components}\n"
        f"R2: {r2:.6f}\n"
        f"final loss: {loss_list[-1]:.6f}" if loss_list else ""
    )
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            color=style.foreground, verticalalignment='top', fontfamily='monospace')

    # bottom-left: GT spatial field at frame T/2
    mid = n_frames // 2
    ax = axes[1, 0]
    pos_mid = pos_np[mid]
    gt_mid = gt_np[mid]
    mag_gt = np.sqrt((gt_mid ** 2).sum(axis=-1))
    if n_components >= 2:
        q_step = max(1, n_cells // 500)
        ax.quiver(pos_mid[::q_step, 0], pos_mid[::q_step, 1],
                  gt_mid[::q_step, 0], gt_mid[::q_step, 1],
                  mag_gt[::q_step], cmap='coolwarm', alpha=0.7)
    else:
        ax.scatter(pos_mid[:, 0], pos_mid[:, 1], c=gt_mid[:, 0], cmap='coolwarm', s=5, alpha=0.7)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title(f'gt (frame {mid})', fontsize=10, color=style.foreground)
    ax.set_aspect('equal')

    # bottom-middle: predicted field at frame T/2
    ax = axes[1, 1]
    pred_mid = pred_np[mid]
    mag_pred = np.sqrt((pred_mid ** 2).sum(axis=-1))
    if n_components >= 2:
        ax.quiver(pos_mid[::q_step, 0], pos_mid[::q_step, 1],
                  pred_mid[::q_step, 0], pred_mid[::q_step, 1],
                  mag_pred[::q_step], cmap='coolwarm', alpha=0.7)
    else:
        ax.scatter(pos_mid[:, 0], pos_mid[:, 1], c=pred_mid[:, 0], cmap='coolwarm', s=5, alpha=0.7)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title(f'predicted (frame {mid})', fontsize=10, color=style.foreground)
    ax.set_aspect('equal')

    # bottom-right: pred vs gt scatter
    ax = axes[1, 2]
    subsample = max(1, len(gt_flat) // 10000)
    ax.scatter(gt_flat[::subsample], pred_flat[::subsample], s=1, alpha=0.3, color='cyan')
    lims = [min(gt_flat.min(), pred_flat.min()), max(gt_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, 'r--', linewidth=0.5)
    ax.set_title(f'pred vs gt  R2={r2:.4f}', fontsize=10, color=style.foreground)
    ax.set_xlabel('gt', fontsize=8, color=style.foreground)
    ax.set_ylabel('pred', fontsize=8, color=style.foreground)
    ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f'{inr_type}_{step}.png'),
                dpi=100, facecolor=style.background, bbox_inches='tight')
    plt.close(fig)


def plot_inr_kinograph(gt_np, pred_np, field_name, n_components, n_cells,
                        output_folder):
    """Kinograph montage: GT, prediction, residual, scatter.

    Args:
        gt_np: (T, N, C) ground truth
        pred_np: (T, N, C) predictions
        field_name: field name for labeling
        n_components: number of field components
        n_cells: number of cells
        output_folder: directory for output
    """
    style = dark_style
    style.apply_globally()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=style.background)
    for ax in axes.flat:
        ax.set_facecolor(style.background)
        ax.tick_params(colors=style.foreground, labelsize=8)

    # reshape to kinograph format: (n_cells * n_components, n_frames)
    gt_kino = gt_np.reshape(gt_np.shape[0], -1).T  # (N*C, T)
    pred_kino = pred_np.reshape(pred_np.shape[0], -1).T
    residual_kino = gt_kino - pred_kino

    vmax = max(np.abs(gt_kino).max(), 1e-8)

    ax = axes[0, 0]
    ax.imshow(gt_kino, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title('ground truth', fontsize=10, color=style.foreground)
    ax.set_ylabel('cell x component', fontsize=8, color=style.foreground)

    ax = axes[0, 1]
    ax.imshow(pred_kino, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title('prediction', fontsize=10, color=style.foreground)

    ax = axes[1, 0]
    res_vmax = max(np.abs(residual_kino).max(), 1e-8)
    ax.imshow(residual_kino, aspect='auto', cmap='RdBu_r', vmin=-res_vmax, vmax=res_vmax)
    ax.set_title('residual (gt - pred)', fontsize=10, color=style.foreground)
    ax.set_xlabel('frame', fontsize=8, color=style.foreground)
    ax.set_ylabel('cell x component', fontsize=8, color=style.foreground)

    # compute metrics per component
    metrics_text = ''
    for c in range(n_components):
        gt_c = gt_np[:, :, c].T  # (N, T)
        pred_c = pred_np[:, :, c].T
        m = compute_kinograph_metrics(gt_c, pred_c)
        metrics_text += f'comp {c}: R2={m["r2"]:.4f} SSIM={m["ssim"]:.4f}\n'

    ax = axes[1, 1]
    gt_flat = gt_np.reshape(-1)
    pred_flat = pred_np.reshape(-1)
    subsample = max(1, len(gt_flat) // 20000)
    ax.scatter(gt_flat[::subsample], pred_flat[::subsample], s=1, alpha=0.2, color='cyan')
    lims = [min(gt_flat.min(), pred_flat.min()), max(gt_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, 'r--', linewidth=0.5)
    ax.set_title('scatter', fontsize=10, color=style.foreground)
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=9,
            color=style.foreground, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f'kinograph_{field_name}.png'),
                dpi=100, facecolor=style.background, bbox_inches='tight')
    plt.close(fig)


def plot_inr_gradient_field(pos_np, gt_field, pred_field, potential, frame_idx,
                             dimension, output_folder):
    """Gradient mode visualization: potential surface + quiver comparison.

    Args:
        pos_np: (N, dim) positions
        gt_field: (N, dim) ground truth vector field
        pred_field: (N, dim) predicted = -grad(phi)
        potential: (N,) scalar potential values
        frame_idx: frame number
        dimension: spatial dimension
        output_folder: directory for output
    """
    style = dark_style
    style.apply_globally()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=style.background)
    for ax in axes:
        ax.set_facecolor(style.background)
        ax.tick_params(colors=style.foreground, labelsize=8)

    q_step = max(1, len(pos_np) // 500)

    # left: scalar potential
    ax = axes[0]
    sc = ax.scatter(pos_np[:, 0], pos_np[:, 1], c=potential, cmap='viridis', s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='phi')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title(f'potential phi (frame {frame_idx})', fontsize=10, color=style.foreground)
    ax.set_aspect('equal')

    # middle: -grad(phi) quiver
    ax = axes[1]
    mag = np.sqrt((pred_field ** 2).sum(axis=-1))
    ax.quiver(pos_np[::q_step, 0], pos_np[::q_step, 1],
              pred_field[::q_step, 0], pred_field[::q_step, 1],
              mag[::q_step], cmap='coolwarm', alpha=0.7)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title('-grad(phi)', fontsize=10, color=style.foreground)
    ax.set_aspect('equal')

    # right: gt field quiver
    ax = axes[2]
    mag_gt = np.sqrt((gt_field ** 2).sum(axis=-1))
    ax.quiver(pos_np[::q_step, 0], pos_np[::q_step, 1],
              gt_field[::q_step, 0], gt_field[::q_step, 1],
              mag_gt[::q_step], cmap='coolwarm', alpha=0.7)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_title('ground truth', fontsize=10, color=style.foreground)
    ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f'gradient_field_{frame_idx:06d}.png'),
                dpi=100, facecolor=style.background, bbox_inches='tight')
    plt.close(fig)
