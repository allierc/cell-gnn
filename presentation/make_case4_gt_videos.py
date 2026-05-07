"""
Generate 3 GT diffusion field videos (case 4a, 4b, 4c) using saved zarr data.

Output format matches test_diffusion_field_generator.ipynb:
- Left: 3D scatter colored by local field value
- Center: top-down particles colored by field
- Right: top-down field (z=0.5 slice) with particles overlaid

Each video uses:
- pos.zarr, field.zarr from the corresponding dataset (already on disk)
- The diffusion PDE is re-simulated to get the grid field slices (we don't
  save full grid fields in zarr — only per-particle field values).

Since v1 and v5 have identical simulator params, their videos will look
nearly identical (up to random init).
"""
import sys, os
sys.path.insert(0, '/groups/jingyiliu/home/liuj4/cell-gnn')
os.chdir('/groups/jingyiliu/home/liuj4/cell-gnn')

import torch
import numpy as np
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess, shutil
import yaml
from tqdm import trange

from cell_gnn.cell_state import CellState
from cell_gnn.utils import edges_radius_blockwise, choose_boundary_values
from cell_gnn.generators.particle_spring_force_diffusion_field import ParticleSpringForceDiffusionField
from cell_gnn.integrators import rk4_step

PROJ = Path('/groups/jingyiliu/home/liuj4/cell-gnn')
VID_DIR = PROJ / 'presentation' / 'videos'
VID_DIR.mkdir(parents=True, exist_ok=True)


def load_config_dict(config_name):
    with open(PROJ / 'config' / 'misc' / f'{config_name}.yaml') as f:
        return yaml.safe_load(f)


def load_pos(dataset):
    return np.array(zarr.open(str(PROJ/'graphs_data'/'misc'/dataset/'x_list_0'/'pos.zarr'), 'r')[:])


def load_field(dataset):
    return np.array(zarr.open(str(PROJ/'graphs_data'/'misc'/dataset/'x_list_0'/'field.zarr'), 'r')[:])


def frames_to_mp4(frame_dir, output_path, fps=30):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
                    '-i', f'{frame_dir}/frame_%06d.png',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p',
                    str(output_path)], check=True)
    print(f'  -> {output_path.name}')


def clean_3d(ax):
    ax.grid(False)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')


def make_gt_field_video(config_name, output_name, label,
                        save_every=10, max_snapshots=400, fps=30, device='cpu'):
    """Replay the diffusion PDE simulator using the config, saving grid slices.
    Then overlay with saved particle positions from zarr."""
    print(f'\n=== {label} ===')
    print(f'  config: {config_name}')
    cfg = load_config_dict(config_name)
    sim = cfg['simulation']
    fp = sim['field_params']

    n_cells = sim['n_cells']
    dimension = sim['dimension']
    n_frames = sim['n_frames']
    delta_t = sim['delta_t']
    max_radius = sim['max_radius']
    min_radius = sim['min_radius']
    cell_params = torch.tensor(sim['cell_params'][0], device=device)
    grid_resolution = fp['grid_resolution']

    # Load saved positions + field
    pos_all = load_pos(config_name)      # (T, N, 3)
    field_all = load_field(config_name)  # (T, N, 1)
    T_saved = pos_all.shape[0]
    print(f'  Saved data: {T_saved} frames, {n_cells} cells')

    # Build field_params dict for simulator
    sim_fp = {
        'diffusion_coeff': fp['diffusion_coeff'],
        'lambda_decay': fp['lambda_decay'],
        'grid_resolution': grid_resolution,
        'source_strength': fp.get('source_strength', 0.0),
        'source_fraction': fp.get('source_fraction', 1.0),
        'pulse_period': fp.get('pulse_period', 0),
        'pulse_duty': fp.get('pulse_duty', 1.0),
        'center_0': torch.tensor(fp['center_0'], device=device),
        'amplitude': fp['amplitude'],
        'sigma': fp['sigma'],
        'mu_chem': fp['mu_chem'],
        'delta_t': delta_t,
        'periodic': True,
    }
    sat = fp.get('chem_saturation_scale')
    if sat is not None:
        sim_fp['chem_saturation_scale'] = sat

    # Re-simulate PDE *only* to get grid slices.
    # We don't re-integrate particles; we feed the real saved positions
    # at each step via model._advance_field.
    model_gt = ParticleSpringForceDiffusionField(
        aggr_type='add', p=cell_params, bc_dpos=choose_boundary_values('periodic')[1],
        dimension=dimension, noise_model_level=0.0, field_params=sim_fp,
    )
    model_gt._init_grid(device)

    # Replay PDE: advance 1 step at a time using saved positions
    pos_snapshots = []
    field_snapshots = []   # per-particle c values (saved from zarr)
    grid_snapshots = []    # full z=0.5 slices (computed here)
    time_snapshots = []

    n_steps = min(n_frames, T_saved, max_snapshots * save_every)

    for it in trange(n_steps, ncols=100, desc='  PDE replay'):
        pos_t = torch.tensor(pos_all[it], dtype=torch.float32, device=device)
        model_gt._advance_field(pos_t, n_steps=1)
        model_gt._last_step = it

        if it % save_every == 0:
            pos_snapshots.append(pos_all[it])
            field_snapshots.append(field_all[it, :, 0])
            grid = model_gt._field_grid.detach().cpu().numpy()
            grid_snapshots.append(grid[grid_resolution // 2, :, :].copy())
            time_snapshots.append(it * delta_t)

    pos_snapshots = np.array(pos_snapshots)
    field_snapshots = np.array(field_snapshots)
    grid_snapshots = np.array(grid_snapshots)
    time_snapshots = np.array(time_snapshots)
    n_snap = len(time_snapshots)
    print(f'  {n_snap} snapshots, field range [{grid_snapshots.min():.4f}, {grid_snapshots.max():.4f}]')

    # Render 3-panel video
    vmax_field = grid_snapshots.max()
    vmax_part = max(field_snapshots.max(), 1e-6)

    tmp = VID_DIR / f'_tmp_{output_name}'; tmp.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for i in trange(n_snap, ncols=100, desc='  rendering'):
        p = pos_snapshots[i]
        t = time_snapshots[i]
        f = field_snapshots[i]
        g = grid_snapshots[i]

        ax1.cla()
        ax1.scatter(p[:,0], p[:,1], p[:,2], s=8, c=f, cmap='YlOrRd',
                   vmin=0, vmax=vmax_part, alpha=0.7, edgecolors='none', depthshade=True)
        clean_3d(ax1)
        ax1.set_title(f'3D  t={t:.2f}')
        ax1.view_init(elev=25, azim=30 + i * 0.3)

        ax2.cla()
        ax2.scatter(p[:,0], p[:,1], s=4, c=f, cmap='YlOrRd',
                   vmin=0, vmax=vmax_part, alpha=0.7, edgecolors='none')
        ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.set_aspect('equal')
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        ax2.set_title('Top-down (color = local c)')

        ax3.cla()
        ax3.imshow(g, extent=[0,1,0,1], origin='lower',
                  cmap='Blues', vmin=0, vmax=vmax_field, alpha=0.8)
        ax3.scatter(p[:,0], p[:,1], s=2, c='red', alpha=0.4)
        ax3.set_xlim(0,1); ax3.set_ylim(0,1); ax3.set_aspect('equal')
        ax3.set_xlabel('x'); ax3.set_ylabel('y')
        ax3.set_title('Field (z=0.5) + particles')

        fig.suptitle(f'{label}   t={t:.2f}', fontsize=14)
        fig.tight_layout()
        fig.savefig(tmp / f'frame_{i:06d}.png', dpi=100, bbox_inches='tight')

    plt.close(fig)
    frames_to_mp4(tmp, VID_DIR / f'{output_name}.mp4', fps)
    shutil.rmtree(tmp)


if __name__ == '__main__':
    cases = [
        ('dicty_spring_force_rk4_diffusion_field_v1',
         'case4a_gt_field',
         'Case 4a: Diffusion Field (v1, linear chemotaxis)'),
        ('dicty_spring_force_rk4_diffusion_field_siren_grad_v2',
         'case4b_gt_field',
         'Case 4b: Diffusion Field (v2, Weber-Fechner saturation)'),
        ('dicty_spring_force_rk4_diffusion_field_siren_grad_v5',
         'case4c_gt_field',
         'Case 4c: Diffusion Field (v5, linear chemotaxis)'),
    ]

    for config_name, out_name, label in cases:
        make_gt_field_video(config_name, out_name, label)

    print('\nDone. Videos:')
    for _, out_name, _ in cases:
        print(f'  {VID_DIR / f"{out_name}.mp4"}')
