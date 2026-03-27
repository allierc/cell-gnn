"""Systematic parameter sweep for learning 3D vector fields with MLP and SIREN.

Tests different:
  - Network architectures: MLP (ReLU/Tanh/GELU) vs SIREN
  - Hyperparameters: learning rate, hidden size, depth
  - Target functions: single Gaussian, multi-Gaussian, sharp gradient
  - Learning targets: gradient of C (vector field) vs C itself (scalar field)

Usage:
    python -m cell_gnn.experiments.field_learning_sweep --device cuda:0
    python -m cell_gnn.experiments.field_learning_sweep --device cuda:0 --quick  # fast test run
"""

import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from cell_gnn.models.Siren_Network import Siren


# ---------------------------------------------------------------------------
# Target field generators
# ---------------------------------------------------------------------------

def periodic_disp(a, a0, L):
    d = a - a0
    return d - L * torch.round(d / L)


def make_gaussian_field(Nx, L, blob_amp, blob_sigma, center, device):
    """Single Gaussian blob scalar field + spectral gradient."""
    dx = L / Nx
    coords = torch.arange(Nx, dtype=torch.float64, device=device) * dx
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')

    dxp = periodic_disp(X, center[0], L)
    dyp = periodic_disp(Y, center[1], L)
    dzp = periodic_disp(Z, center[2], L)

    r2 = dxp**2 + dyp**2 + dzp**2
    c = blob_amp * torch.exp(-0.5 * r2 / blob_sigma**2)
    return X, Y, Z, c


def make_multi_gaussian_field(Nx, L, blobs, device):
    """Multiple Gaussian blobs summed together.

    blobs: list of (amp, sigma, (cx, cy, cz))
    """
    dx = L / Nx
    coords = torch.arange(Nx, dtype=torch.float64, device=device) * dx
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')

    c = torch.zeros_like(X)
    for amp, sigma, center in blobs:
        dxp = periodic_disp(X, center[0], L)
        dyp = periodic_disp(Y, center[1], L)
        dzp = periodic_disp(Z, center[2], L)
        r2 = dxp**2 + dyp**2 + dzp**2
        c = c + amp * torch.exp(-0.5 * r2 / sigma**2)
    return X, Y, Z, c


def make_sharp_gradient_field(Nx, L, device):
    """Sharp step-like field — hard for smooth networks."""
    dx = L / Nx
    coords = torch.arange(Nx, dtype=torch.float64, device=device) * dx
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')

    dxp = periodic_disp(X, 0.5, L)
    dyp = periodic_disp(Y, 0.5, L)
    dzp = periodic_disp(Z, 0.5, L)
    r = torch.sqrt(dxp**2 + dyp**2 + dzp**2)
    c = torch.sigmoid((0.15 - r) / 0.01)  # sharp shell
    return X, Y, Z, c


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


# ---------------------------------------------------------------------------
# Target field registry
# ---------------------------------------------------------------------------

TARGET_FIELDS = {}


def register_field(name):
    def decorator(fn):
        TARGET_FIELDS[name] = fn
        return fn
    return decorator


@register_field("single_gaussian")
def _single_gaussian(Nx, L, device):
    X, Y, Z, c = make_gaussian_field(Nx, L, 1.0, 0.20, (0.5, 0.5, 0.5), device)
    return X, Y, Z, c


@register_field("multi_gaussian")
def _multi_gaussian(Nx, L, device):
    blobs = [
        (1.0, 0.15, (0.3, 0.3, 0.3)),
        (0.8, 0.10, (0.7, 0.7, 0.5)),
        (-0.5, 0.12, (0.5, 0.2, 0.8)),
    ]
    X, Y, Z, c = make_multi_gaussian_field(Nx, L, blobs, device)
    return X, Y, Z, c


@register_field("narrow_gaussian")
def _narrow_gaussian(Nx, L, device):
    X, Y, Z, c = make_gaussian_field(Nx, L, 1.0, 0.05, (0.5, 0.5, 0.5), device)
    return X, Y, Z, c


@register_field("sharp_shell")
def _sharp_shell(Nx, L, device):
    X, Y, Z, c = make_sharp_gradient_field(Nx, L, device)
    return X, Y, Z, c


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
    field_name: str = "single_gaussian"
    target: str = "gradient"       # "gradient" (vector) or "scalar" (learn C)
    arch: str = "relu"             # "relu", "tanh", "gelu", "siren"
    n_layers: int = 5
    hidden_size: int = 128
    lr: float = 1e-3
    n_iters: int = 5000
    batch_size: int = 4096
    omega: float = 30.0           # SIREN only
    Nx: int = 64
    L: float = 1.0


@dataclass
class ExperimentResult:
    config: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)
    rel_errors: list = field(default_factory=list)  # (iter, rel_err) pairs
    final_rel_error: float = 0.0
    final_loss: float = 0.0
    n_params: int = 0
    time_seconds: float = 0.0
    converged: bool = False


def run_experiment(cfg: ExperimentConfig, device: str, eval_every: int = 100) -> ExperimentResult:
    """Run a single field learning experiment."""

    # --- build target field ---
    field_fn = TARGET_FIELDS[cfg.field_name]
    X, Y, Z, c = field_fn(cfg.Nx, cfg.L, device)

    positions = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1).float()

    if cfg.target == "gradient":
        gx, gy, gz = spectral_gradient(c, cfg.Nx, cfg.L)
        targets = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1).float()
        output_size = 3
    else:  # scalar
        targets = c.reshape(-1, 1).float()
        output_size = 1

    # normalize
    scale = targets.abs().max().clamp(min=1e-12)
    targets_norm = targets / scale
    true_norm = torch.norm(targets).item()

    N = positions.shape[0]

    # --- build model ---
    model = build_model(cfg.arch, 3, output_size, cfg.n_layers, cfg.hidden_size, device, omega=cfg.omega)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_iters)

    # --- train ---
    losses = []
    rel_errors = []
    t0 = time.time()

    model.train()
    for it in range(cfg.n_iters):
        idx = torch.randint(0, N, (cfg.batch_size,), device=device)
        x_batch = positions[idx]
        y_batch = targets_norm[idx]

        pred = model(x_batch)
        loss = nn.functional.mse_loss(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if it % eval_every == 0 or it == cfg.n_iters - 1:
            model.eval()
            with torch.no_grad():
                pred_all = model(positions) * scale
                rel_e = (torch.norm(pred_all - targets) / true_norm).item()
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
        # minimal test sweep
        fields = ["single_gaussian"]
        archs = ["relu", "siren"]
        lrs = [1e-3]
        hidden_sizes = [128]
        n_layers_list = [5]
        n_iters = 1000
    else:
        fields = ["single_gaussian", "multi_gaussian", "narrow_gaussian", "sharp_shell"]
        archs = ["relu", "tanh", "gelu", "siren"]
        lrs = [1e-4, 5e-4, 1e-3, 3e-3]
        hidden_sizes = [64, 128, 256]
        n_layers_list = [3, 5, 7]
        n_iters = 5000

    # --- Sweep 1: Architecture & LR (fixed depth/width) ---
    for field_name in fields:
        for arch in archs:
            for lr in lrs:
                for target in ["gradient", "scalar"]:
                    cfg = ExperimentConfig(
                        field_name=field_name, arch=arch, lr=lr,
                        target=target, n_iters=n_iters,
                    )
                    if arch == "siren":
                        for omega in [30.0, 60.0]:
                            cfg_s = ExperimentConfig(**{**asdict(cfg), 'omega': omega})
                            configs.append(cfg_s)
                    else:
                        configs.append(cfg)

    # --- Sweep 2: Depth & Width (fixed LR=1e-3, relu & siren only) ---
    for field_name in fields:
        for arch in ["relu", "siren"]:
            for hs in hidden_sizes:
                for nl in n_layers_list:
                    for target in ["gradient", "scalar"]:
                        cfg = ExperimentConfig(
                            field_name=field_name, arch=arch, lr=1e-3,
                            hidden_size=hs, n_layers=nl,
                            target=target, n_iters=n_iters,
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
    parser = argparse.ArgumentParser(description="Field learning parameter sweep")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="Run minimal test sweep")
    parser.add_argument("--output_dir", type=str, default="results/field_sweep")
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
              f"lr={cfg.lr} | layers={cfg.n_layers} | hidden={cfg.hidden_size}"
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

    # save final results
    out_path = os.path.join(args.output_dir, "sweep_results.npz")
    np.savez_compressed(out_path, results=json.dumps(all_results))
    print(f"\nResults saved to {out_path}")

    # summary
    n_converged = sum(1 for r in all_results if r.get("converged", False))
    print(f"\nSummary: {n_converged}/{len(all_results)} converged (rel_error < 0.05)")


if __name__ == "__main__":
    main()
