"""Generate paper figures from trained models and ground-truth data.

Usage:
    cd /groups/jingyiliu/home/liuj4/cell-gnn
    python paper/generate_figures.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell_gnn.config import CellGNNConfig
from cell_gnn.utils import to_numpy

def load_config(path):
    return CellGNNConfig.from_yaml(str(path))

PROJ = Path(__file__).resolve().parent.parent
FIG_DIR = PROJ / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ── configs for each ablation ──────────────────────────────────────
CONFIGS = {
    "gnn_only":       "dicty_spring_force_rk4",
    "siren_grad_v1":  "dicty_spring_force_rk4_diffusion_field_siren_grad_v1",
    "siren_grad_pde": "dicty_spring_force_rk4_diffusion_field_siren_grad_v4",
}

LABELS = {
    "gnn_only":       "GNN only",
    "siren_grad_v1":  "GNN + SIREN grad",
    "siren_grad_pde": "GNN + SIREN grad + PDE",
}

COLORS = {
    "gnn_only":       "#d62728",
    "siren_grad_v1":  "#2ca02c",
    "siren_grad_pde": "#1f77b4",
}


# =====================================================================
# Figure 1: Ground-truth force profile F(r)
# =====================================================================
def fig_ground_truth_force():
    """Plot the ground-truth spring force profile used in all experiments."""
    r = torch.linspace(0, 0.15, 500)

    # Parameters from config: [k_rep, r0, kadh, r_on, delta, mu_f]
    k_rep, r0, kadh, r_on, delta, mu_f = 50.0, 0.05, 50.0, 0.07, 0.001, 0.033
    delta_safe = max(delta, 1e-8)

    F_rep = k_rep * torch.relu(r0 - r)
    g_on = torch.sigmoid((r - r0) / delta_safe)
    g_off = torch.sigmoid(-(r - r_on) / delta_safe)
    F_adh = -kadh * g_on * g_off * (r - r0)
    F_total = mu_f * (F_rep + F_adh)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(to_numpy(r), to_numpy(mu_f * F_rep), '--', color='#ff7f0e', alpha=0.7, label='Repulsion')
    ax.plot(to_numpy(r), to_numpy(mu_f * F_adh), '--', color='#9467bd', alpha=0.7, label='Adhesion')
    ax.plot(to_numpy(r), to_numpy(F_total), '-', color='k', linewidth=2, label='Total $F(r)$')
    ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
    ax.axvline(r0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5, label=f'$r_0 = {r0}$')
    ax.set_xlabel('Distance $r$', fontsize=12)
    ax.set_ylabel('Force $F(r)$', fontsize=12)
    ax.set_title('Ground-Truth Pairwise Force Profile', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gt_force_profile.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_gt_force_profile.png", dpi=300)
    plt.close(fig)
    print("  -> fig_gt_force_profile.pdf")


# =====================================================================
# Figure 2: Loss curves comparison across ablations
# =====================================================================
def fig_loss_curves():
    """Compare training loss curves across model ablations."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for key, cfg_name in CONFIGS.items():
        loss_path = PROJ / "log" / "misc" / cfg_name / "loss.pt"
        if not loss_path.exists():
            print(f"  [skip] {loss_path} not found")
            continue
        loss_data = torch.load(loss_path, map_location="cpu", weights_only=False)
        if isinstance(loss_data, dict):
            loss_vals = loss_data.get("loss", loss_data.get("train_loss", None))
            if loss_vals is None:
                loss_vals = list(loss_data.values())[0]
        elif isinstance(loss_data, (list, torch.Tensor)):
            loss_vals = loss_data
        else:
            print(f"  [skip] Unknown loss format for {cfg_name}: {type(loss_data)}")
            continue

        loss_np = to_numpy(torch.tensor(loss_vals)) if not isinstance(loss_vals, np.ndarray) else loss_vals
        # Smooth with running average
        if len(loss_np) > 100:
            kernel = 50
            smoothed = np.convolve(loss_np, np.ones(kernel)/kernel, mode='valid')
            x = np.arange(len(smoothed))
        else:
            smoothed = loss_np
            x = np.arange(len(smoothed))

        ax.semilogy(x, smoothed, color=COLORS[key], label=LABELS[key], linewidth=1.5)

    ax.set_xlabel('Training iteration', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_loss_curves.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_loss_curves.png", dpi=300)
    plt.close(fig)
    print("  -> fig_loss_curves.pdf")


# =====================================================================
# Figure 3: Learned vs GT force profiles (per ablation)
# =====================================================================
def fig_learned_force_profiles():
    """Load trained models and evaluate learned F(r) vs ground truth."""
    from cell_gnn.models.MLP import MLP

    r_gt = torch.linspace(0, 0.15, 500)
    k_rep, r0, kadh, r_on, delta, mu_f = 50.0, 0.05, 50.0, 0.07, 0.001, 0.033
    delta_safe = max(delta, 1e-8)
    F_rep = k_rep * torch.relu(r0 - r_gt)
    g_on = torch.sigmoid((r_gt - r0) / delta_safe)
    g_off = torch.sigmoid(-(r_gt - r_on) / delta_safe)
    F_adh = -kadh * g_on * g_off * (r_gt - r0)
    F_gt = mu_f * (F_rep + F_adh)

    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4.5 * len(CONFIGS), 3.5), sharey=True)
    if len(CONFIGS) == 1:
        axes = [axes]

    for idx, (key, cfg_name) in enumerate(CONFIGS.items()):
        ax = axes[idx]
        ax.plot(to_numpy(r_gt), to_numpy(F_gt), 'k-', linewidth=2, label='Ground truth', zorder=10)

        # Find the latest checkpoint
        model_dir = PROJ / "log" / "misc" / cfg_name / "models"
        if not model_dir.exists():
            ax.set_title(LABELS[key] + '\n(no model)', fontsize=11)
            continue

        ckpts = sorted(model_dir.glob("best_model_with_0_graphs_0_*.pt"),
                       key=lambda p: int(p.stem.split('_')[-1]))
        if not ckpts:
            ckpts = list(model_dir.glob("best_model_with_0_graphs_0.pt"))
        if not ckpts:
            ax.set_title(LABELS[key] + '\n(no checkpoint)', fontsize=11)
            continue

        # Load the final checkpoint
        ckpt_path = ckpts[-1]
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

            # Try to load config to get model dimensions
            cfg = load_config(PROJ / "config" / "misc" / f"{cfg_name}.yaml")
            model_cfg = cfg.graph_model
            dim = cfg.simulation.dimension
            emb_dim = model_cfg.embedding_dim
            input_size = dim + 1 + emb_dim  # delta_pos + r + embedding

            # Build MLP with same architecture
            lin_edge = MLP(input_size=input_size, output_size=model_cfg.output_size,
                          nlayers=model_cfg.n_layers, hidden_size=model_cfg.hidden_dim,
                          device="cpu")

            # Load weights for lin_edge
            edge_state = {k.replace("lin_edge.", ""): v for k, v in state.items()
                         if k.startswith("lin_edge.")}
            if edge_state:
                lin_edge.load_state_dict(edge_state)
            else:
                ax.set_title(LABELS[key] + '\n(no edge weights)', fontsize=11)
                continue

            # Get median embedding
            emb_key = "a"
            if emb_key in state:
                embeddings = state[emb_key]  # (n_dataset, n_cells, emb_dim)
                median_emb = embeddings[0].median(dim=0).values  # (emb_dim,)
            else:
                median_emb = torch.zeros(emb_dim)

            # Evaluate learned F(r): create input features
            r_eval = torch.linspace(0.001, 0.15 / cfg.simulation.max_radius,
                                    500).unsqueeze(1)
            # delta_pos = (r, 0, 0) normalized by max_radius
            delta_pos = torch.zeros(500, dim)
            delta_pos[:, 0] = r_eval[:, 0]
            emb_repeat = median_emb.unsqueeze(0).expand(500, -1)
            inp = torch.cat([delta_pos, r_eval, emb_repeat], dim=-1)

            with torch.no_grad():
                F_learned = lin_edge(inp)
                # Take the x-component as the radial force
                F_r = F_learned[:, 0]

            r_phys = to_numpy(r_eval[:, 0] * cfg.simulation.max_radius)
            ax.plot(r_phys, to_numpy(F_r), color=COLORS[key], linewidth=1.5,
                    label='Learned', alpha=0.8)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='red')

        ax.set_xlabel('Distance $r$', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Force $F(r)$', fontsize=11)
        ax.set_title(LABELS[key], fontsize=11)
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlim(0, 0.15)

    fig.suptitle('Learned vs Ground-Truth Pairwise Force Profile', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_learned_force_profiles.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_learned_force_profiles.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig_learned_force_profiles.pdf")


# =====================================================================
# Figure 4: Learned SIREN field visualization (z-slice)
# =====================================================================
def fig_siren_field():
    """Visualize learned scalar field c(x,y,z=0.5,t) at several time slices."""
    from cell_gnn.models.Siren_Network import Siren

    n_time = 4
    resolution = 80

    fig, axes = plt.subplots(2, n_time, figsize=(3.5 * n_time, 7))

    for row, (key, cfg_name) in enumerate([
        ("siren_grad_v1", CONFIGS["siren_grad_v1"]),
        ("siren_grad_pde", CONFIGS["siren_grad_pde"]),
    ]):
        cfg = load_config(PROJ / "config" / "misc" / f"{cfg_name}.yaml")
        model_cfg = cfg.graph_model
        dim = cfg.simulation.dimension
        n_frames = cfg.simulation.n_frames
        omega_field = getattr(model_cfg, 'omega_field', 30.0)

        # Load weights first to infer architecture
        model_dir = PROJ / "log" / "misc" / cfg_name / "models"
        ckpts = sorted(model_dir.glob("best_model_with_0_graphs_0_*.pt"),
                       key=lambda p: int(p.stem.split('_')[-1]))
        if not ckpts:
            ckpts = list(model_dir.glob("best_model_with_0_graphs_0.pt"))
        if not ckpts:
            continue

        raw = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        siren_state = {k.replace("siren_field.", ""): v for k, v in state.items()
                       if k.startswith("siren_field.")}
        if not siren_state:
            print(f"  [skip] No siren_field weights in {cfg_name}")
            continue

        # Infer SIREN architecture from checkpoint keys
        siren_net_keys = [k for k in state if k.startswith("siren_field.net.")]
        n_modules = len(set(k.split('.')[2] for k in siren_net_keys))
        inferred_hidden = max(n_modules - 2, 0)
        hidden_dim_field = state.get("siren_field.net.0.linear.weight",
                                      state.get("siren_field.net.0.weight",
                                                torch.zeros(128, 4))).shape[0]

        siren = Siren(
            in_features=dim + 1,
            out_features=1,
            hidden_features=hidden_dim_field,
            hidden_layers=inferred_hidden,
            outermost_linear=True,
            first_omega_0=omega_field,
            hidden_omega_0=omega_field,
        )
        siren.load_state_dict(siren_state)
        siren.eval()

        # Evaluate on grid
        coords_1d = torch.linspace(0, 1, resolution)
        gy, gx = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
        grid_2d = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (res^2, 2)
        z_mid = 0.5 * torch.ones(resolution**2, 1)
        grid_3d = torch.cat([grid_2d, z_mid], dim=1)  # (res^2, 3)

        time_fracs = np.linspace(0.05, 0.95, n_time)

        vmin_global, vmax_global = None, None
        fields = []
        with torch.no_grad():
            for ti, tf in enumerate(time_fracs):
                t_col = tf * torch.ones(resolution**2, 1)
                inp = torch.cat([grid_3d, t_col], dim=1)
                c = siren(inp).reshape(resolution, resolution)
                fields.append(to_numpy(c))

        # Get global color range
        all_vals = np.concatenate([f.ravel() for f in fields])
        vmin_global = np.percentile(all_vals, 2)
        vmax_global = np.percentile(all_vals, 98)

        for ti, (tf, field) in enumerate(zip(time_fracs, fields)):
            ax = axes[row, ti]
            im = ax.imshow(field, origin='lower', extent=[0, 1, 0, 1],
                          cmap='viridis', vmin=vmin_global, vmax=vmax_global)
            ax.set_title(f'$t/T = {tf:.2f}$', fontsize=10)
            if ti == 0:
                ax.set_ylabel(LABELS[key], fontsize=11)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

        plt.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, label='$c(\\mathbf{x}, t)$')

    fig.suptitle('Learned Scalar Concentration Field ($z = 0.5$ slice)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_siren_field.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_siren_field.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig_siren_field.pdf")


# =====================================================================
# Figure 5: Architecture schematic
# =====================================================================
def fig_architecture():
    """Draw architecture diagram using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Colors
    c_gnn = '#2196F3'
    c_siren = '#4CAF50'
    c_sum = '#FF9800'
    c_data = '#9E9E9E'
    c_pde = '#E91E63'

    def box(x, y, w, h, text, color, fontsize=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='k',
                             linewidth=1.2, alpha=0.85, zorder=5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', zorder=10)

    def arrow(x1, y1, x2, y2, text='', color='k'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=color),
                    zorder=3)
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, text, ha='center', va='bottom', fontsize=7,
                    color=color, style='italic')

    # Input: Trajectories
    box(0.2, 2, 1.6, 0.8, 'Trajectories\n$\\{\\mathbf{x}_i(t)\\}$', c_data)

    # GNN branch
    box(3, 3.5, 2, 0.8, 'GNN Branch\n$\\mathrm{MLP}_{\\mathrm{edge}}(\\Delta\\mathbf{x}, r, \\mathbf{a}_i)$', c_gnn)
    arrow(1.8, 2.8, 3.0, 3.9, '$\\Delta\\mathbf{x}_{ji}$ (relative)')

    # SIREN branch
    box(3, 0.7, 2, 0.8, 'SIREN Branch\n$c_\\phi(\\mathbf{x}, t) \\to \\nabla c$', c_siren)
    arrow(1.8, 2.0, 3.0, 1.1, '$(\\mathbf{x}_i, t)$ (absolute)')

    # Sum
    box(6.5, 2, 1.5, 0.8, '$\\hat{\\mathbf{v}}_i = $\nGNN + $\\nabla c$', c_sum)
    arrow(5.0, 3.9, 6.5, 2.8, '$\\hat{\\mathbf{v}}^{\\mathrm{pair}}$', c_gnn)
    arrow(5.0, 1.1, 6.5, 2.0, '$\\hat{\\mathbf{v}}^{\\mathrm{field}}$', c_siren)

    # PDE loss
    box(3, -0.3, 2, 0.6, 'PDE residual\n$\\partial_t c - D\\nabla^2 c + \\lambda c - S$', c_pde, fontsize=7)
    arrow(4.0, 0.7, 4.0, 0.3, '', c_pde)

    # Agent->Field feedback
    ax.annotate('', xy=(3.0, -0.0), xytext=(7.25, 1.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=c_pde,
                               connectionstyle='arc3,rad=0.3', linestyle='--'),
                zorder=3)
    ax.text(6.2, 0.15, 'cells source\nthe field', fontsize=7, color=c_pde,
            ha='center', style='italic')

    # Output
    box(8.5, 2, 1.3, 0.8, 'Loss\n$||\\hat{\\mathbf{v}} - \\mathbf{v}||^2$', '#FFCDD2')
    arrow(8.0, 2.4, 8.5, 2.4)

    # Embeddings
    box(3, 4.6, 1.2, 0.5, '$\\mathbf{a}_i$\nembeddings', '#E1BEE7', fontsize=8)
    arrow(3.6, 4.6, 3.6, 4.3, '', '#7B1FA2')

    # Title
    ax.text(5, 5.3, 'CellGNN Architecture: Agent--Field Co-Dynamics',
            ha='center', fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_architecture.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_architecture.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig_architecture.pdf")


# =====================================================================
# Figure 6: Agent-field coupling schematic
# =====================================================================
def fig_agent_field_loop():
    """Schematic showing the bidirectional agent-field feedback loop."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Agents box
    rect1 = plt.Rectangle((0.5, 2.5), 2, 1, facecolor='#BBDEFB', edgecolor='#1565C0',
                           linewidth=2, zorder=5)
    ax.add_patch(rect1)
    ax.text(1.5, 3.0, 'Cells\n$\\{\\mathbf{x}_i(t)\\}$', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Field box
    rect2 = plt.Rectangle((3.5, 2.5), 2, 1, facecolor='#C8E6C9', edgecolor='#2E7D32',
                           linewidth=2, zorder=5)
    ax.add_patch(rect2)
    ax.text(4.5, 3.0, 'Field\n$c(\\mathbf{x}, t)$', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Arrows
    # Agents -> Field (top arc)
    ax.annotate('', xy=(3.5, 3.7), xytext=(2.5, 3.7),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#E65100',
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(3.0, 4.2, 'Cells secrete chemical\n$\\partial_t c = D\\nabla^2 c - \\lambda c + \\alpha\\sum_i \\delta(\\mathbf{x} - \\mathbf{x}_i)$',
            ha='center', fontsize=8, color='#E65100', fontweight='bold')

    # Field -> Agents (bottom arc)
    ax.annotate('', xy=(2.5, 2.3), xytext=(3.5, 2.3),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0',
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(3.0, 1.6, 'Gradient guides motion\n$d\\mathbf{x}_i/dt = F_{\\mathrm{pair}} + \\mu \\nabla c$',
            ha='center', fontsize=8, color='#1565C0', fontweight='bold')

    # Pairwise self-loop on agents
    ax.annotate('', xy=(0.5, 3.3), xytext=(0.5, 2.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='#6A1B9A',
                               connectionstyle='arc3,rad=-1.5'))
    ax.text(0.0, 3.0, 'Pairwise\nforces', ha='center', fontsize=7, color='#6A1B9A')

    # Title
    ax.text(3.0, 0.5, 'Agent--Field Feedback Loop\n(closed-loop: agents generate the field they navigate by)',
            ha='center', fontsize=10, style='italic', color='#424242')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_agent_field_loop.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_agent_field_loop.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig_agent_field_loop.pdf")


# =====================================================================
# Figure 7: Embedding space visualization
# =====================================================================
def fig_embeddings():
    """Visualize learned cell embeddings across ablations."""
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 3.5))
    if len(CONFIGS) == 1:
        axes = [axes]

    for idx, (key, cfg_name) in enumerate(CONFIGS.items()):
        ax = axes[idx]
        model_dir = PROJ / "log" / "misc" / cfg_name / "models"
        ckpts = sorted(model_dir.glob("best_model_with_0_graphs_0_*.pt"),
                       key=lambda p: int(p.stem.split('_')[-1]))
        if not ckpts:
            ckpts = list(model_dir.glob("best_model_with_0_graphs_0.pt"))
        if not ckpts:
            ax.set_title(LABELS[key] + '\n(no model)', fontsize=11)
            continue

        raw = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        if "a" not in state:
            ax.text(0.5, 0.5, 'No embeddings', transform=ax.transAxes,
                    ha='center', fontsize=10)
            ax.set_title(LABELS[key], fontsize=11)
            continue

        emb = to_numpy(state["a"][0])  # (n_cells, emb_dim)
        if emb.shape[1] >= 2:
            ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.5, c=COLORS[key])
        elif emb.shape[1] == 1:
            ax.hist(emb[:, 0], bins=50, color=COLORS[key], alpha=0.7)

        ax.set_title(LABELS[key], fontsize=11)
        ax.set_xlabel('$a_1$', fontsize=10)
        if idx == 0:
            ax.set_ylabel('$a_2$', fontsize=10)

    fig.suptitle('Learned Cell Embeddings', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_embeddings.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_embeddings.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig_embeddings.pdf")


# =====================================================================
# Figure 8: Ground-truth field evolution (from simulator)
# =====================================================================
def fig_gt_field_evolution():
    """Show ground-truth diffusion field at several time snapshots."""
    from cell_gnn.generators.particle_spring_force_diffusion_field import (
        init_gaussian_field, build_spectral_decay, spectral_gradient,
        deposit_to_grid
    )

    dim = 3
    res = 64  # lower res for speed
    D, lam, alpha = 0.01, 0.1, 20.0
    dt = 0.002
    n_cells = 200  # fewer cells for visualization
    n_steps_per_snap = 500

    # Random cell positions
    torch.manual_seed(42)
    pos = torch.rand(n_cells, dim)

    # Initial field: zero (no initial blob, cells generate everything)
    center = torch.tensor([[0.5, 0.5, 0.5]])
    field = init_gaussian_field(res, dim, center, amplitude=0.0, sigma=0.1,
                                periodic=True, device='cpu')
    decay = build_spectral_decay(res, dim, D, lam, dt, device='cpu')

    n_snaps = 4
    fig, axes = plt.subplots(1, n_snaps, figsize=(3.5 * n_snaps, 3))

    for snap in range(n_snaps):
        # Advance field
        for _ in range(n_steps_per_snap):
            source = deposit_to_grid(pos, alpha * dt, res, dim, 'cpu')
            field = field + source
            s = [res] * dim
            C_hat = torch.fft.rfftn(field, s=s)
            C_hat = C_hat * decay
            field = torch.fft.irfftn(C_hat, s=s)
            # Jiggle positions slightly
            pos = (pos + 0.0005 * torch.randn_like(pos)) % 1.0

        # Plot z=res//2 slice
        ax = axes[snap]
        slice_2d = to_numpy(field[res//2, :, :])
        im = ax.imshow(slice_2d, origin='lower', extent=[0, 1, 0, 1],
                       cmap='inferno')
        t_val = (snap + 1) * n_steps_per_snap * dt
        ax.set_title(f'$t = {t_val:.1f}$', fontsize=11)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        # Scatter cells on top
        cell_z = pos[:, 2]
        near_mid = (cell_z > 0.4) & (cell_z < 0.6)
        ax.scatter(to_numpy(pos[near_mid, 0]), to_numpy(pos[near_mid, 1]),
                   s=5, c='white', alpha=0.6, edgecolors='none')

    plt.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Concentration $c$')
    fig.suptitle('Ground-Truth Diffusion Field (cell-sourced, $z=0.5$ slice)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gt_field_evolution.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_gt_field_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> fig_gt_field_evolution.pdf")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    print()

    print("[1/7] Ground-truth force profile")
    fig_ground_truth_force()

    print("[2/7] Agent-field feedback loop schematic")
    fig_agent_field_loop()

    print("[3/7] Architecture schematic")
    fig_architecture()

    print("[4/7] Loss curves")
    fig_loss_curves()

    print("[5/7] Learned force profiles")
    fig_learned_force_profiles()

    print("[6/7] Learned SIREN field")
    fig_siren_field()

    print("[7/7] Embeddings")
    fig_embeddings()

    # This one is slow (simulates PDE) - run last
    print("[bonus] Ground-truth field evolution")
    fig_gt_field_evolution()

    print()
    print(f"All figures saved to: {FIG_DIR}")
