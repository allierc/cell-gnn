#!/usr/bin/env python3
"""Optuna hyperparameter optimization benchmark for cell-gnn INR training.

Fair comparison against GNN_LLM_INR_parallel.py (LLM-in-the-loop UCB exploration).
Uses same search space, same objective (maximize final_r2), same budget (144 trials),
and same parallelism (4 workers).

Usage:
    python GNN_Optuna_INR.py dicty
    python GNN_Optuna_INR.py dicty --n_trials 144 --n_parallel 4 --device auto
    python GNN_Optuna_INR.py dicty --resume  # resume interrupted run
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import time
import warnings

import yaml
import optuna
from optuna.samplers import TPESampler

from cell_gnn.config import CellGNNConfig
from cell_gnn.utils import add_pre_folder

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

DEFAULT_N_TRIALS = 144
DEFAULT_N_PARALLEL = 4
MAX_TRAINING_TIME_MIN = 60.0


# ── search space ────────────────────────────────────────────────────────────

def suggest_inr_params(trial):
    """Suggest INR hyperparameters — identical to LLM exploration search space."""
    return {
        'omega_inr': trial.suggest_float('omega_inr', 1.0, 100.0, log=True),
        'inr_learning_rate': trial.suggest_float('inr_learning_rate', 1e-6, 1e-3, log=True),
        'hidden_dim_inr': trial.suggest_int('hidden_dim_inr', 128, 1024, step=64),
        'n_layers_inr': trial.suggest_int('n_layers_inr', 2, 5),
        'inr_total_steps': trial.suggest_int('inr_total_steps', 50000, 2000000, log=True),
        'inr_batch_size': trial.suggest_categorical('inr_batch_size', [1, 2, 4, 8, 16]),
        'inr_xy_period': trial.suggest_float('inr_xy_period', 0.5, 10.0),
        'inr_t_period': trial.suggest_float('inr_t_period', 0.5, 10.0),
    }


# ── config builder ──────────────────────────────────────────────────────────

def build_trial_config(base_config_path, params, trial_number, exploration_dir,
                       pre_folder):
    """Create a per-trial YAML config with overridden INR params."""
    with open(base_config_path, 'r') as f:
        raw = yaml.safe_load(f)

    if 'inr' not in raw:
        raw['inr'] = {}
    for key, value in params.items():
        raw['inr'][key] = value

    # Unique config_file avoids log-directory collisions between parallel trials
    trial_config_file = f"{pre_folder}optuna_trial_{trial_number:03d}"

    config_save_dir = os.path.join(exploration_dir, 'config')
    os.makedirs(config_save_dir, exist_ok=True)
    config_yaml_path = os.path.join(config_save_dir, f'trial_{trial_number:03d}.yaml')

    with open(config_yaml_path, 'w') as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    config = CellGNNConfig.from_yaml(config_yaml_path)
    config.dataset = pre_folder + config.dataset
    config.config_file = trial_config_file

    return config, config_yaml_path


# ── results parser ──────────────────────────────────────────────────────────

def parse_results_log(results_path):
    """Parse key-value results.log written by data_train_INR."""
    if not os.path.exists(results_path):
        return {}
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, _, value = line.partition(':')
                key, value = key.strip(), value.strip()
                try:
                    results[key] = float(value)
                except ValueError:
                    results[key] = value
    return results


# ── objective ───────────────────────────────────────────────────────────────

def _objective_fn(trial, base_config_path, exploration_dir, pre_folder, device_str):
    """Train INR with suggested params and return final_r2."""
    from cell_gnn.models.inr_trainer import data_train_INR

    trial_number = trial.number
    params = suggest_inr_params(trial)

    config, config_yaml_path = build_trial_config(
        base_config_path, params, trial_number, exploration_dir, pre_folder
    )

    t_start = time.time()
    try:
        field_name = config.inr.inr_field_name if config.inr else 'velocity'

        model, loss_list = data_train_INR(
            config=config,
            device=device_str,
            field_name=field_name,
            run=0,
            erase=True,
        )

        elapsed_min = (time.time() - t_start) / 60.0

        log_dir = f'log/{config.config_file}'
        results = parse_results_log(f'./{log_dir}/tmp_training/inr/results.log')

        final_r2 = float(results.get('final_r2', 0.0))
        trial.set_user_attr('final_mse', results.get('final_mse', float('inf')))
        trial.set_user_attr('slope', results.get('slope', 0.0))
        trial.set_user_attr('training_time_min', elapsed_min)

        # Copy video
        video_src = f'./{log_dir}/tmp_training/inr/{field_name}_gt_vs_pred.mp4'
        if os.path.exists(video_src):
            video_dir = os.path.join(exploration_dir, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            shutil.copy2(video_src, f'{video_dir}/trial_{trial_number:03d}.mp4')

        if elapsed_min > MAX_TRAINING_TIME_MIN:
            raise optuna.TrialPruned(f"time limit ({elapsed_min:.1f} min)")

        print(f"\033[92m  Trial {trial_number}: R2={final_r2:.6f}, "
              f"time={elapsed_min:.1f}min\033[0m")
        return final_r2

    except optuna.TrialPruned:
        raise
    except Exception as e:
        elapsed_min = (time.time() - t_start) / 60.0
        trial.set_user_attr('error', str(e))
        trial.set_user_attr('training_time_min', elapsed_min)
        print(f"\033[91m  Trial {trial_number}: FAILED ({elapsed_min:.1f}min): {e}\033[0m")
        return 0.0


# ── worker ──────────────────────────────────────────────────────────────────

def run_trial_worker(study_name, storage_url, base_config_path, exploration_dir,
                     pre_folder, device_str, worker_id, n_trials_per_worker):
    """Worker process: connect to shared study and run trials."""
    import matplotlib
    matplotlib.use('Agg')

    study = optuna.load_study(study_name=study_name, storage=storage_url)

    def objective(trial):
        return _objective_fn(trial, base_config_path, exploration_dir,
                             pre_folder, device_str)

    study.optimize(objective, n_trials=n_trials_per_worker, catch=(Exception,))


# ── device assignment ───────────────────────────────────────────────────────

def assign_devices(n_parallel, device_preference='auto'):
    """Assign GPU devices to workers (round-robin across available GPUs)."""
    import torch

    if device_preference == 'cpu':
        return ['cpu'] * n_parallel

    if not torch.cuda.is_available():
        print("  No CUDA GPUs available, using CPU")
        return ['cpu'] * n_parallel

    n_gpus = torch.cuda.device_count()

    if device_preference != 'auto' and device_preference.startswith('cuda:'):
        return [device_preference] * n_parallel

    devices = [f'cuda:{i % n_gpus}' for i in range(n_parallel)]
    for i, dev in enumerate(devices):
        name = torch.cuda.get_device_name(i % n_gpus)
        print(f"  Worker {i} -> {dev} ({name})")
    return devices


# ── results export ──────────────────────────────────────────────────────────

def export_results(study, exploration_dir, base_config_path, pre_folder):
    """Write CSV summary and best-config YAML."""
    csv_path = os.path.join(exploration_dir, 'optuna_results.csv')
    fieldnames = [
        'trial_number', 'state', 'final_r2',
        'omega_inr', 'inr_learning_rate', 'hidden_dim_inr', 'n_layers_inr',
        'inr_total_steps', 'inr_batch_size', 'inr_xy_period', 'inr_t_period',
        'final_mse', 'slope', 'training_time_min',
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'final_r2': trial.value if trial.value is not None else '',
            }
            for key in ['omega_inr', 'inr_learning_rate', 'hidden_dim_inr',
                        'n_layers_inr', 'inr_total_steps', 'inr_batch_size',
                        'inr_xy_period', 'inr_t_period']:
                row[key] = trial.params.get(key, '')
            for key in ['final_mse', 'slope', 'training_time_min']:
                row[key] = trial.user_attrs.get(key, '')
            writer.writerow(row)
    print(f"  Results CSV: {csv_path}")

    # Best config
    try:
        best = study.best_trial
    except ValueError:
        best = None

    if best is not None:
        best_path = os.path.join(exploration_dir, 'optuna_best.yaml')
        with open(base_config_path, 'r') as f:
            raw = yaml.safe_load(f)
        if 'inr' not in raw:
            raw['inr'] = {}
        for key, value in best.params.items():
            raw['inr'][key] = value
        raw['description'] = f'Optuna best (trial {best.number}, R2={best.value:.6f})'
        with open(best_path, 'w') as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
        print(f"  Best config: {best_path}")
        print(f"  Trial {best.number}: R2={best.value:.6f}")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

    # Summary
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if completed:
        r2s = sorted([t.value for t in completed])
        print(f"\n  Completed: {len(completed)}/{len(study.trials)}")
        print(f"  Best R2:   {r2s[-1]:.6f}")
        print(f"  Median R2: {r2s[len(r2s)//2]:.6f}")
        print(f"  Worst R2:  {r2s[0]:.6f}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Optuna benchmark for cell-gnn INR hyperparameter optimization"
    )
    parser.add_argument('dataset', type=str, help='Dataset name (e.g. dicty)')
    parser.add_argument('--n_trials', type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument('--n_parallel', type=int, default=DEFAULT_N_PARALLEL)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_file, pre_folder = add_pre_folder(args.dataset)
    base_config_path = os.path.join(root_dir, 'config', f'{config_file}.yaml')

    if not os.path.exists(base_config_path):
        print(f"Error: config not found: {base_config_path}", file=sys.stderr)
        sys.exit(1)

    exploration_dir = os.path.join(root_dir, 'log', 'Optuna_exploration',
                                   f'{args.dataset}_INR')
    os.makedirs(exploration_dir, exist_ok=True)
    os.makedirs(os.path.join(exploration_dir, 'config'), exist_ok=True)
    os.makedirs(os.path.join(exploration_dir, 'videos'), exist_ok=True)

    # SQLite storage
    db_path = os.path.join(exploration_dir, 'optuna_study.db')
    storage_url = f'sqlite:///{db_path}'
    study_name = f'{args.dataset}_INR_optuna'

    if not args.resume and os.path.exists(db_path):
        print(f"\033[93mExisting study found at {db_path}\033[0m")
        print("Use --resume to continue, or delete the DB to start fresh.")
        sys.exit(1)

    # Create / load study
    sampler = TPESampler(seed=args.seed, multivariate=True, constant_liar=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='maximize',
        sampler=sampler,
        load_if_exists=args.resume,
    )

    n_done = len([t for t in study.trials
                  if t.state in (optuna.trial.TrialState.COMPLETE,
                                 optuna.trial.TrialState.PRUNED,
                                 optuna.trial.TrialState.FAIL)])
    n_remaining = max(0, args.n_trials - n_done)

    if n_remaining == 0:
        print(f"Study already has {n_done} trials — nothing to do.")
        export_results(study, exploration_dir, base_config_path, pre_folder)
        return

    print(f"\n{'='*60}")
    print(f"Optuna INR Benchmark: {args.dataset}")
    print(f"  Trials: {n_remaining} remaining ({n_done} done)")
    print(f"  Parallel workers: {args.n_parallel}")
    print(f"  Sampler: TPE (seed={args.seed})")
    print(f"  Storage: {db_path}")
    print(f"{'='*60}\n")

    devices = assign_devices(args.n_parallel, args.device)

    # Distribute trials across workers
    base_count = n_remaining // args.n_parallel
    extra = n_remaining % args.n_parallel
    trials_per_worker = [base_count + (1 if i < extra else 0)
                         for i in range(args.n_parallel)]

    t_start = time.time()
    processes = []
    for wid in range(args.n_parallel):
        if trials_per_worker[wid] == 0:
            continue
        p = mp.Process(
            target=run_trial_worker,
            args=(study_name, storage_url, base_config_path, exploration_dir,
                  pre_folder, devices[wid], wid, trials_per_worker[wid]),
        )
        p.start()
        processes.append(p)
        print(f"  Worker {wid} started (device={devices[wid]}, "
              f"n_trials={trials_per_worker[wid]})")

    for p in processes:
        p.join()

    elapsed = (time.time() - t_start) / 60.0
    print(f"\nAll workers finished in {elapsed:.1f} minutes")

    # Export final results
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    export_results(study, exploration_dir, base_config_path, pre_folder)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
