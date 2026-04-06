"""Systematic parameter sweep for learning DYNAMIC 3D vector fields with MLP and SIREN.

The field is now time-dependent: a moving Gaussian blob.
Input: (x, y, z, t) -> gradient or scalar value.

Batch size now works like GNN training: sample time frames first, then spatial
points within each frame.

Tests different:
  - Network architectures: MLP (ReLU/Tanh/GELU) vs SIREN
  - Hyperparameters: learning rate, hidden size, depth
  - Target functions: single moving blob, multi-blob orbiting, pulsating blob
  - Learning targets: gradient (vector field) vs C (scalar field)

Usage:
    python -m cell_gnn.experiments.field_learning_sweep_dynamic --device cuda:0
    python -m cell_gnn.experiments.field_learning_sweep_dynamic --device cuda:0 --quick
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cell_gnn.models.Siren_Network import Siren


# ---------------------------------------------------------------------------
# Dynamic field generators
# ---------------------------------------------------------------------------

def periodic_disp(a, a0, L):
    d = a - a0
    return d - L * torch.round(d / L)


def spectral_gradient(c, Nx, L):
    """Compute grad c via FFT."""
    dx = L / Nx
    k = 2.0 * torch.pi * torch.fft.fftfreq(Nx, d=dx, device=c.device, dtype=c.dtype)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    chat = torch.fft.fftn(c)
    gx = torch.real(torch.fft.ifftn(1j * kx * chat))
    gy = torch.real(torch.fft.ifftn(1j * ky * chat))
    gz = torch.real(torch.fft.ifftn(1j * kz * chat))
    return gx, gy, gz


def make_grid(Nx, L, device):
    dx = L / Nx
    coords = torch.arange(Nx, dtype=torch.float64, device=device) * dx
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    return X, Y, Z


def gaussian_blob(X, Y, Z, L, amp, sigma, cx, cy, cz):
    """Single Gaussian blob at given center (periodic)."""
    dxp = periodic_disp(X, cx, L)
    dyp = periodic_disp(Y, cy, L)
    dzp = periodic_disp(Z, cz, L)
    r2 = dxp**2 + dyp**2 + dzp**2
    return amp * torch.exp(-0.5 * r2 / sigma**2)


# ---------------------------------------------------------------------------
# Dynamic field registry
# ---------------------------------------------------------------------------

DYNAMIC_FIELDS = {}


def register_dynamic_field(name):
    def decorator(fn):
        DYNAMIC_FIELDS[name] = fn
        return fn
    return decorator


@register_dynamic_field("moving_blob")
def _moving_blob(Nx, L, n_frames, device):
    """Single Gaussian blob moving in a circle around center of domain."""
    X, Y, Z = make_grid(Nx, L, device)
    times = torch.linspace(0, 1, n_frames, dtype=torch.float64, device=device)

    radius = 0.2  # orbit radius
    sigma = 0.15
    amp = 1.0

    c_all = []
    gc_all = []
    for t_val in times:
        angle = 2 * torch.pi * t_val
        cx = 0.5 + radius * torch.cos(angle)
        cy = 0.5 + radius * torch.sin(angle)
        cz = 0.5
        c = gaussian_blob(X, Y, Z, L, amp, sigma, cx, cy, cz)
        gx, gy, gz = spectral_gradient(c, Nx, L)
        c_all.append(c)
        gc_all.append(torch.stack([gx, gy, gz], dim=-1))

    return X, Y, Z, times, torch.stack(c_all), torch.stack(gc_all)


@register_dynamic_field("multi_orbit")
def _multi_orbit(Nx, L, n_frames, device):
    """Two blobs orbiting in opposite directions."""
    X, Y, Z = make_grid(Nx, L, device)
    times = torch.linspace(0, 1, n_frames, dtype=torch.float64, device=device)

    c_all = []
    gc_all = []
    for t_val in times:
        angle = 2 * torch.pi * t_val
        # blob 1: clockwise
        c1 = gaussian_blob(X, Y, Z, L, 1.0, 0.12,
                           0.5 + 0.2 * torch.cos(angle),
                           0.5 + 0.2 * torch.sin(angle), 0.5)
        # blob 2: counter-clockwise, offset in z
        c2 = gaussian_blob(X, Y, Z, L, 0.7, 0.10,
                           0.5 + 0.15 * torch.cos(-angle + torch.pi),
                           0.5 + 0.15 * torch.sin(-angle + torch.pi),
                           0.5 + 0.1 * torch.sin(angle))
        c = c1 + c2
        gx, gy, gz = spectral_gradient(c, Nx, L)
        c_all.append(c)
        gc_all.append(torch.stack([gx, gy, gz], dim=-1))

    return X, Y, Z, times, torch.stack(c_all), torch.stack(gc_all)


@register_dynamic_field("pulsating_blob")
def _pulsating_blob(Nx, L, n_frames, device):
    """Blob at center with time-varying amplitude and width."""
    X, Y, Z = make_grid(Nx, L, device)
    times = torch.linspace(0, 1, n_frames, dtype=torch.float64, device=device)

    c_all = []
    gc_all = []
    for t_val in times:
        amp = 0.5 + 0.5 * torch.sin(2 * torch.pi * t_val)
        sigma = 0.10 + 0.08 * torch.cos(2 * torch.pi * t_val)
        c = gaussian_blob(X, Y, Z, L, amp, sigma, 0.5, 0.5, 0.5)
        gx, gy, gz = spectral_gradient(c, Nx, L)
        c_all.append(c)
        gc_all.append(torch.stack([gx, gy, gz], dim=-1))

    return X, Y, Z, times, torch.stack(c_all), torch.stack(gc_all)


@register_dynamic_field("moving_narrow")
def _moving_narrow(Nx, L, n_frames, device):
    """Narrow moving blob — harder due to sharp gradients."""
    X, Y, Z = make_grid(Nx, L, device)
    times = torch.linspace(0, 1, n_frames, dtype=torch.float64, device=device)

    c_all = []
    gc_all = []
    for t_val in times:
        angle = 2 * torch.pi * t_val
        cx = 0.5 + 0.25 * torch.cos(angle)
        cy = 0.5 + 0.25 * torch.sin(angle)
        cz = 0.5
        c = gaussian_blob(X, Y, Z, L, 1.0, 0.05, cx, cy, cz)
        gx, gy, gz = spectral_gradient(c, Nx, L)
        c_all.append(c)
        gc_all.append(torch.stack([gx, gy, gz], dim=-1))

    return X, Y, Z, times, torch.stack(c_all), torch.stack(gc_all)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, n_layers, hidden_size, activation='relu'):
        super().__init__()
        act_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[activation]
        layers = [nn.Linear(input_size, hidden_size), act_fn()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), act_fn()])
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(arch, input_size, output_size, n_layers, hidden_size, device, **kwargs):
    if arch == 'siren':
        omega = kwargs.get('omega', 30.0)
        model = Siren(
            in_features=input_size, out_features=output_size,
            hidden_features=hidden_size, hidden_layers=n_layers - 2,
            outermost_linear=True,
            first_omega_0=omega, hidden_omega_0=omega,
        )
    else:
        model = MLPModel(input_size, output_size, n_layers, hidden_size, activation=arch)
    return model.to(device)


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    field_name: str = "moving_blob"
    target: str = "gradient"       # "gradient" or "scalar"
    arch: str = "relu"             # "relu", "tanh", "gelu", "siren"
    n_layers: int = 5
    hidden_size: int = 128
    lr: float = 1e-3
    n_iters: int = 5000
    batch_frames: int = 8         # number of time frames per batch (like GNN)
    batch_points: int = 512       # spatial points per frame
    omega: float = 30.0           # SIREN only
    Nx: int = 48                  # smaller grid to fit memory with many frames
    L: float = 1.0
    n_frames: int = 100           # number of time snapshots


@dataclass
class ExperimentResult:
    config: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)
    rel_errors: list = field(default_factory=list)
    final_rel_error: float = 0.0
    final_loss: float = 0.0
    n_params: int = 0
    time_seconds: float = 0.0
    converged: bool = False


def run_experiment(cfg: ExperimentConfig, device: str, eval_every: int = 100) -> ExperimentResult:
    """Run a single dynamic field learning experiment."""

    # --- build dynamic field ---
    field_fn = DYNAMIC_FIELDS[cfg.field_name]
    X, Y, Z, times, c_all, gc_all = field_fn(cfg.Nx, cfg.L, cfg.n_frames, device)
    # c_all:  (n_frames, Nx, Nx, Nx)
    # gc_all: (n_frames, Nx, Nx, Nx, 3)

    # flatten spatial grid — shared across all frames
    N_spatial = cfg.Nx ** 3
    pos_flat = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1).float()
    # (N_spatial, 3)

    if cfg.target == "gradient":
        # (n_frames, N_spatial, 3)
        targets_all = gc_all.reshape(cfg.n_frames, N_spatial, 3).float()
        output_size = 3
    else:
        # (n_frames, N_spatial, 1)
        targets_all = c_all.reshape(cfg.n_frames, N_spatial, 1).float()
        output_size = 1

    # normalize
    scale = targets_all.abs().max().clamp(min=1e-12)
    targets_norm = targets_all / scale

    # precompute time values as float
    times_f = times.float()  # (n_frames,)

    # full dataset norm for relative error
    true_norm = torch.norm(targets_all).item()

    # --- build model: input is (x, y, z, t) ---
    input_size = 4  # spatial 3 + time 1
    model = build_model(cfg.arch, input_size, output_size, cfg.n_layers,
                        cfg.hidden_size, device, omega=cfg.omega)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_iters)

    # --- train ---
    losses = []
    rel_errors = []
    t0 = time.time()

    model.train()
    for it in range(cfg.n_iters):
        # Sample batch_frames random time frames
        frame_ids = torch.randint(0, cfg.n_frames, (cfg.batch_frames,), device=device)
        # Sample batch_points random spatial points per frame
        point_ids = torch.randint(0, N_spatial, (cfg.batch_points,), device=device)

        # Build input: (batch_frames * batch_points, 4)
        pos_batch = pos_flat[point_ids]  # (batch_points, 3)
        input_list = []
        target_list = []
        for fi in frame_ids:
            t_val = times_f[fi]
            t_col = t_val.expand(cfg.batch_points, 1)
            input_list.append(torch.cat([pos_batch, t_col], dim=1))
            target_list.append(targets_norm[fi, point_ids])

        x_batch = torch.cat(input_list, dim=0)     # (batch_frames * batch_points, 4)
        y_batch = torch.cat(target_list, dim=0)     # (batch_frames * batch_points, output_size)

        pred = model(x_batch)
        loss = nn.functional.mse_loss(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Periodic full evaluation
        if it % eval_every == 0 or it == cfg.n_iters - 1:
            model.eval()
            with torch.no_grad():
                # Evaluate on all frames, all points
                total_sq_err = 0.0
                for fi in range(cfg.n_frames):
                    t_val = times_f[fi]
                    t_col = t_val.expand(N_spatial, 1)
                    x_full = torch.cat([pos_flat, t_col], dim=1)
                    pred_full = model(x_full) * scale
                    true_full = targets_all[fi]
                    total_sq_err += ((pred_full - true_full) ** 2).sum().item()
                rel_e = (total_sq_err ** 0.5) / true_norm
                rel_errors.append((it, rel_e))
            model.train()

    elapsed = time.time() - t0

    result = ExperimentResult(
        config=asdict(cfg),
        losses=losses,
        rel_errors=rel_errors,
        final_rel_error=rel_errors[-1][1],
        final_loss=losses[-1],
        n_params=n_params,
        time_seconds=elapsed,
        converged=rel_errors[-1][1] < 0.05,
    )
    return result


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

def get_sweep_configs(quick=False):
    """Generate all experiment configs for the sweep."""
    configs = []

    if quick:
        fields = ["moving_blob"]
        archs = ["relu", "siren"]
        lrs = [1e-3]
        hidden_sizes = [128]
        n_layers_list = [5]
        n_iters = 1000
        n_frames = 50
    else:
        fields = ["moving_blob", "multi_orbit", "pulsating_blob", "moving_narrow"]
        archs = ["relu", "tanh", "gelu", "siren"]
        lrs = [1e-4, 5e-4, 1e-3, 3e-3]
        hidden_sizes = [64, 128, 256]
        n_layers_list = [3, 5, 7]
        n_iters = 5000
        n_frames = 100

    # --- Sweep 1: Architecture & LR ---
    for field_name in fields:
        for arch in archs:
            for lr in lrs:
                for target in ["gradient", "scalar"]:
                    cfg = ExperimentConfig(
                        field_name=field_name, arch=arch, lr=lr,
                        target=target, n_iters=n_iters, n_frames=n_frames,
                    )
                    if arch == "siren":
                        for omega in [30.0, 60.0]:
                            cfg_s = ExperimentConfig(**{**asdict(cfg), 'omega': omega})
                            configs.append(cfg_s)
                    else:
                        configs.append(cfg)

    # --- Sweep 2: Depth & Width (fixed LR=1e-3, relu & siren) ---
    for field_name in fields:
        for arch in ["relu", "siren"]:
            for hs in hidden_sizes:
                for nl in n_layers_list:
                    for target in ["gradient", "scalar"]:
                        cfg = ExperimentConfig(
                            field_name=field_name, arch=arch, lr=1e-3,
                            hidden_size=hs, n_layers=nl,
                            target=target, n_iters=n_iters, n_frames=n_frames,
                        )
                        configs.append(cfg)

    # deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = json.dumps(asdict(c), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dynamic field learning parameter sweep")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="Run minimal test sweep")
    parser.add_argument("--output_dir", type=str, default="results/field_sweep_dynamic")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    configs = get_sweep_configs(quick=args.quick)
    print(f"Total experiments: {len(configs)}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    all_results = []
    for i, cfg in enumerate(configs):
        tag = f"[{i+1}/{len(configs)}]"
        print(f"\n{tag} {cfg.arch} | {cfg.field_name} | target={cfg.target} | "
              f"lr={cfg.lr} | layers={cfg.n_layers} | hidden={cfg.hidden_size} | "
              f"frames={cfg.n_frames}"
              + (f" | omega={cfg.omega}" if cfg.arch == "siren" else ""))

        try:
            result = run_experiment(cfg, device)
            all_results.append(asdict(result))
            status = "CONVERGED" if result.converged else "NOT CONVERGED"
            print(f"  -> rel_error={result.final_rel_error:.4f}  "
                  f"loss={result.final_loss:.2e}  "
                  f"params={result.n_params:,}  "
                  f"time={result.time_seconds:.1f}s  {status}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            all_results.append({"config": asdict(cfg), "error": str(e)})

        # save incrementally
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            np.savez_compressed(
                os.path.join(args.output_dir, "sweep_results.npz"),
                results=json.dumps(all_results),
            )

    out_path = os.path.join(args.output_dir, "sweep_results.npz")
    np.savez_compressed(out_path, results=json.dumps(all_results))
    print(f"\nResults saved to {out_path}")

    n_converged = sum(1 for r in all_results if r.get("converged", False))
    print(f"\nSummary: {n_converged}/{len(all_results)} converged (rel_error < 0.05)")


if __name__ == "__main__":
    main()
