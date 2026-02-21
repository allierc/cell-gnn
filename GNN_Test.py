"""
Regression test for particle-gnn training pipeline.

Runs generate/train/test for one or all configs, compares metrics against
reference values, optionally calls Claude CLI for qualitative assessment,
and appends results with timestamp to a persistent test history log.

Usage:
    # Full test of all 4 configs (1 epoch each)
    python GNN_Test.py --config all

    # Test one config, skip data generation
    python GNN_Test.py --config arbitrary --skip-generate

    # Compare existing results only
    python GNN_Test.py --config all --skip-generate --skip-train --skip-test

    # Save current results as reference baseline
    python GNN_Test.py --config all --skip-generate --skip-train --skip-test --save-reference

    # Skip Claude assessment
    python GNN_Test.py --config arbitrary --skip-generate --no-claude
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

from particle_gnn.config import ParticleGNNConfig
from particle_gnn.utils import set_device, add_pre_folder


ALL_CONFIGS = ['arbitrary', 'boids', 'gravity']


# ------------------------------------------------------------------ #
#  Metric parsing
# ------------------------------------------------------------------ #

def parse_results_log(path):
    """Parse results.log written by data_test_particle."""
    if not os.path.exists(path):
        print(f"  warning: {path} not found")
        return {}

    with open(path, 'r') as f:
        content = f.read()

    metrics = {}
    patterns = {
        'rollout_RMSE_mean': r'rollout_RMSE_mean:\s*([\d.eE+-]+)',
        'rollout_RMSE_final': r'rollout_RMSE_final:\s*([\d.eE+-]+)',
        'rollout_geomloss_mean': r'rollout_geomloss_mean:\s*([\d.eE+-]+)',
        'rollout_geomloss_final': r'rollout_geomloss_final:\s*([\d.eE+-]+)',
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, content)
        if m:
            metrics[key] = float(m.group(1))

    return metrics


def parse_training_log(path):
    """Parse training.log for final loss and clustering accuracy."""
    if not os.path.exists(path):
        print(f"  warning: {path} not found")
        return {}

    with open(path, 'r') as f:
        content = f.read()

    metrics = {}

    # Last epoch loss: "Epoch N. Loss: X.XXXXXX"
    losses = re.findall(r'Epoch \d+\. Loss: ([\d.]+)', content)
    if losses:
        metrics['training_final_loss'] = float(losses[-1])

    # Last clustering accuracy: "accuracy: X.XXX"
    accuracies = re.findall(r'accuracy: ([\d.]+)', content)
    if accuracies:
        metrics['training_accuracy'] = float(accuracies[-1])

    return metrics


# ------------------------------------------------------------------ #
#  Comparison
# ------------------------------------------------------------------ #

def compare_metrics(current, reference, thresholds):
    """Compare current metrics against reference values.

    Returns list of row dicts and overall pass/fail.
    For lower-is-better metrics (RMSE, geomloss, loss): PASS if delta <= threshold.
    For higher-is-better metrics (accuracy): PASS if delta >= -threshold.
    """
    rows = []
    all_pass = True

    for key in sorted(set(list(reference.keys()) + list(current.keys()))):
        ref_val = reference.get(key)
        cur_val = current.get(key)

        if ref_val is None or cur_val is None:
            rows.append({
                'metric': key,
                'reference': ref_val,
                'current': cur_val,
                'delta': None,
                'status': 'N/A',
            })
            continue

        delta = cur_val - ref_val
        threshold = thresholds.get(key)

        if threshold is not None:
            # Lower is better for RMSE, geomloss, loss
            if 'RMSE' in key or 'geomloss' in key or 'loss' in key:
                status = 'PASS' if delta <= threshold else 'FAIL'
            else:
                # Higher is better for accuracy
                status = 'PASS' if delta >= -threshold else 'FAIL'
        else:
            status = 'INFO'

        if status == 'FAIL':
            all_pass = False

        rows.append({
            'metric': key,
            'reference': ref_val,
            'current': cur_val,
            'delta': delta,
            'status': status,
        })

    return rows, all_pass


def format_comparison_table(rows):
    """Format comparison rows as a markdown table."""
    lines = []
    lines.append("| Metric | Reference | Current | Delta | Status |")
    lines.append("|--------|-----------|---------|-------|--------|")

    for r in rows:
        ref = f"{r['reference']:.6f}" if r['reference'] is not None else "-"
        cur = f"{r['current']:.6f}" if r['current'] is not None else "-"
        delta = f"{r['delta']:+.6f}" if r['delta'] is not None else "-"
        lines.append(f"| {r['metric']} | {ref} | {cur} | {delta} | {r['status']} |")

    return '\n'.join(lines)


# ------------------------------------------------------------------ #
#  Execution
# ------------------------------------------------------------------ #

def run_generate(config, device):
    """Run data generation (same as GNN_Main.py -o generate)."""
    from particle_gnn.generators.graph_data_generator import data_generate
    data_generate(
        config, device=device, visualize=True, run_vizualized=0,
        style="color", alpha=1, erase=True, save=True, step=10, timer=False,
    )


def run_training_local(config, device):
    """Run training locally (same as GNN_Main.py -o train)."""
    from particle_gnn.models.graph_trainer import data_train
    data_train(config=config, erase=True, best_model=None, device=device)


def run_test_local(config, device):
    """Run test locally (same as GNN_Main.py -o test)."""
    from particle_gnn.models.graph_trainer import data_test
    data_test(
        config=config, visualize=False, style="color name",
        verbose=False, best_model='best', run=0, test_mode="",
        sample_embedding=False, step=5, device=device,
        particle_of_interest=0,
    )


def run_training_cluster(config_name, root_dir, log_dir):
    """Submit training to cluster via SSH + bsub."""
    cluster_home = "/groups/saalfeld/home/allierc"
    cluster_root_dir = f"{cluster_home}/Graph/particle-gnn"

    cluster_train_cmd = f"python GNN_Main.py -o train {config_name}"

    cluster_script_path = os.path.join(log_dir, 'cluster_test_train.sh')
    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {cluster_root_dir}\n")
        f.write(f"conda run -n particle-graph {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = cluster_script_path.replace(root_dir, cluster_root_dir)

    ssh_cmd = (
        f"ssh allierc@login1 \"cd {cluster_root_dir} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_h100 -W 6000 -K "
        f"'bash {cluster_script}'\""
    )

    print(f"\033[96msubmitting training to cluster: {ssh_cmd}\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\033[91mcluster training failed:\033[0m")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError("cluster training failed")

    print(f"\033[92mcluster training completed\033[0m")
    print(result.stdout)
    return result.stdout


def run_test_cluster(config_name, root_dir, log_dir):
    """Submit test to cluster via SSH + bsub."""
    cluster_home = "/groups/saalfeld/home/allierc"
    cluster_root_dir = f"{cluster_home}/Graph/particle-gnn"

    cluster_cmd = f"python GNN_Main.py -o test {config_name} best"

    cluster_script_path = os.path.join(log_dir, 'cluster_test_test.sh')
    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {cluster_root_dir}\n")
        f.write(f"conda run -n particle-graph {cluster_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = cluster_script_path.replace(root_dir, cluster_root_dir)

    ssh_cmd = (
        f"ssh allierc@login1 \"cd {cluster_root_dir} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_h100 -W 6000 -K "
        f"'bash {cluster_script}'\""
    )

    print(f"\033[96msubmitting test to cluster: {ssh_cmd}\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\033[91mcluster test failed:\033[0m")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError("cluster test failed")

    print(f"\033[92mcluster test completed\033[0m")
    print(result.stdout)
    return result.stdout


# ------------------------------------------------------------------ #
#  Claude assessment
# ------------------------------------------------------------------ #

def get_claude_assessment(combined_table, root_dir):
    """Call Claude CLI to generate a qualitative assessment."""
    prompt = f"""You are reviewing a regression test for the particle-gnn training pipeline.

Compare the current training results against the reference baseline and provide a brief assessment.

## Comparison Table
{combined_table}

Please provide:
1. A 2-3 sentence summary of whether results are consistent with the reference
2. Flag any concerning regressions or notable improvements
3. Overall verdict: PASS, WARNING, or FAIL

Keep your response concise (under 200 words)."""

    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', '1',
    ]

    try:
        process = subprocess.Popen(
            claude_cmd, cwd=root_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)

        process.wait()
        output_text = ''.join(output_lines)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print("\033[91mclaude authentication error - skipping assessment\033[0m")
            return "(claude assessment skipped - authentication error)"

        return output_text.strip()

    except FileNotFoundError:
        print("\033[93mclaude CLI not found - skipping assessment\033[0m")
        return "(claude assessment skipped - CLI not available)"
    except Exception as e:
        print(f"\033[93mclaude assessment failed: {e}\033[0m")
        return f"(claude assessment skipped - {e})"


# ------------------------------------------------------------------ #
#  Archive and history
# ------------------------------------------------------------------ #

def archive_results(log_dir, timestamp_str):
    """Copy current results files to archive/ with timestamp."""
    archive_dir = os.path.join(log_dir, 'archive')
    os.makedirs(archive_dir, exist_ok=True)

    for fname in ['results.log', 'training.log']:
        src = os.path.join(log_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(archive_dir, f"{timestamp_str}_{fname}")
            shutil.copy2(src, dst)
            print(f"  archived: {dst}")


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def get_git_branch():
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def format_timings(timings):
    """Format phase timings as a compact string."""
    if not timings:
        return ""
    parts = []
    for phase in ['generate', 'train', 'test']:
        if phase in timings:
            secs = timings[phase]
            if secs >= 60:
                parts.append(f"{phase}: {secs / 60:.1f}m")
            else:
                parts.append(f"{phase}: {secs:.1f}s")
    total = sum(timings.values())
    if total >= 60:
        parts.append(f"total: {total / 60:.1f}m")
    else:
        parts.append(f"total: {total:.1f}s")
    return " | ".join(parts)


def append_test_history(history_path, timestamp_str, commit, branch,
                        config_tables, overall_pass, claude_assessment):
    """Append test entry to log/test_history.md."""
    if not os.path.exists(history_path):
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            f.write("# Particle-GNN Regression Test History\n\n")

    verdict = "PASS" if overall_pass else "FAIL"
    config_names = ', '.join(name for name, _, _, _ in config_tables)

    with open(history_path, 'a') as f:
        f.write(f"## {timestamp_str} -- {config_names} -- commit {commit} ({branch}) -- {verdict}\n\n")
        for name, table, passed, timings in config_tables:
            status = "PASS" if passed else "FAIL"
            timing_str = format_timings(timings)
            f.write(f"### {name} ({status})\n\n")
            if timing_str:
                f.write(f"**time:** {timing_str}\n\n")
            f.write(table)
            f.write("\n\n")
        if claude_assessment:
            f.write(f"**Claude assessment:**\n{claude_assessment}\n\n")
        f.write("---\n\n")

    print(f"test history appended: {history_path}")


def save_reference(ref_path, config_tables_data):
    """Save current metrics as the new reference baseline."""
    ref_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'commit': get_git_commit(),
        'configs': {},
    }

    default_thresholds = {
        'rollout_RMSE_mean': 0.01,
        'rollout_RMSE_final': 0.05,
        'rollout_geomloss_mean': 0.005,
        'rollout_geomloss_final': 0.01,
        'training_accuracy': 0.05,
    }

    for config_name, metrics in config_tables_data.items():
        thresholds = {}
        for key in metrics:
            if key in default_thresholds:
                thresholds[key] = default_thresholds[key]
        ref_data['configs'][config_name] = {
            'metrics': metrics,
            'thresholds': thresholds,
        }

    with open(ref_path, 'w') as f:
        json.dump(ref_data, f, indent=2)

    print(f"\033[92mreference saved to {ref_path}\033[0m")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Regression test for particle-gnn')
    parser.add_argument('--config', type=str, default='all',
                        help='Config name or "all" (default: all)')
    parser.add_argument('--cluster', action='store_true',
                        help='Submit to cluster via SSH+bsub')
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip data generation')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip testing')
    parser.add_argument('--no-claude', action='store_true',
                        help='Skip Claude CLI assessment')
    parser.add_argument('--save-reference', action='store_true',
                        help='Save current metrics as reference baseline')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference JSON (default: config/test_reference.json)')

    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = os.path.join(root_dir, 'config')
    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S')

    # Resolve config list
    if args.config == 'all':
        config_list = ALL_CONFIGS
    else:
        config_list = [args.config]

    # Load reference
    ref_path = args.reference or os.path.join(config_root, 'test_reference.json')
    ref_data = {}
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            ref_data = json.load(f)
    elif not args.save_reference:
        print(f"\033[93mreference file not found: {ref_path} (will show INFO only)\033[0m")

    print(f"\033[94mregression test: {', '.join(config_list)}\033[0m")
    print(f"\033[94mtimestamp: {timestamp_str}\033[0m")
    print(f"\033[94mcommit: {get_git_commit()} ({get_git_branch()})\033[0m")
    if ref_data:
        print(f"\033[94mreference: {ref_path} (date: {ref_data.get('date', '?')})\033[0m")

    overall_pass = True
    all_config_tables = []
    all_config_metrics = {}
    device = None

    for config_name in config_list:
        print(f"\033[96mconfig: {config_name}\033[0m")
        config_timings = {}

        # Load config (same pattern as GNN_Main.py)
        config_file, pre_folder = add_pre_folder(config_name)
        config = ParticleGNNConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_name

        # Override for fast regression testing
        config.training.n_epochs = 1
        config.training.small_init_batch_size = False

        if device is None:
            device = set_device(config.training.device)

        log_dir = os.path.join(root_dir, 'log', config_file)

        # Get reference for this config
        config_ref = ref_data.get('configs', {}).get(config_name, {})
        reference_metrics = config_ref.get('metrics', {})
        thresholds = config_ref.get('thresholds', {})

        # Archive existing results
        archive_results(log_dir, timestamp_str)

        # Phase 0: Generate
        if not args.skip_generate:
            print(f"\033[93m{config_name}: generate\033[0m")
            t0 = time.time()
            if args.cluster:
                run_generate(config, device)
            else:
                run_generate(config, device)
            config_timings['generate'] = time.time() - t0
            print(f"  generate: {config_timings['generate']:.1f}s")

        # Phase 1: Train
        if not args.skip_train:
            print(f"\033[93m{config_name}: train (n_epochs=1)\033[0m")
            t0 = time.time()
            if args.cluster:
                run_training_cluster(config_name, root_dir, log_dir)
            else:
                run_training_local(config, device)
            config_timings['train'] = time.time() - t0
            print(f"  train: {config_timings['train']:.1f}s")

        # Phase 2: Test
        if not args.skip_test:
            print(f"\033[93m{config_name}: test\033[0m")
            t0 = time.time()
            if args.cluster:
                run_test_cluster(config_name, root_dir, log_dir)
            else:
                run_test_local(config, device)
            config_timings['test'] = time.time() - t0
            print(f"  test: {config_timings['test']:.1f}s")

        # Phase 3: Parse metrics
        print(f"\033[93m{config_name}: parse metrics\033[0m")
        current_metrics = {}
        current_metrics.update(parse_results_log(os.path.join(log_dir, 'results.log')))
        current_metrics.update(parse_training_log(os.path.join(log_dir, 'training.log')))

        if not current_metrics:
            print(f"\033[91mno metrics found for {config_name}\033[0m")
            all_config_tables.append((config_name, "(no metrics found)", False, config_timings))
            overall_pass = False
            continue

        all_config_metrics[config_name] = current_metrics
        print(f"parsed {len(current_metrics)} metrics")

        # Phase 4: Compare
        if reference_metrics:
            print(f"\033[93m{config_name}: compare\033[0m")
            rows, config_pass = compare_metrics(current_metrics, reference_metrics, thresholds)
            table = format_comparison_table(rows)
            print(table)
            if not config_pass:
                overall_pass = False
        else:
            # No reference -- show metrics as INFO
            rows = []
            for key, val in sorted(current_metrics.items()):
                rows.append({
                    'metric': key,
                    'reference': None,
                    'current': val,
                    'delta': None,
                    'status': 'INFO',
                })
            table = format_comparison_table(rows)
            print(table)
            config_pass = True

        all_config_tables.append((config_name, table, config_pass, config_timings))

    # Save reference if requested
    if args.save_reference:
        print(f"\033[93msaving reference\033[0m")
        save_reference(ref_path, all_config_metrics)

    # Phase 5: Claude assessment
    claude_assessment = ""
    if not args.no_claude and not args.save_reference:
        print(f"\033[93mclaude assessment\033[0m")
        combined = "\n\n".join(
            f"### {name}\n{table}" for name, table, _, _ in all_config_tables
        )
        claude_assessment = get_claude_assessment(combined, root_dir)

    # Phase 6: Append to history
    if not args.save_reference:
        print(f"\033[93msave results\033[0m")
        history_path = os.path.join(root_dir, 'log', 'test_history.md')
        append_test_history(
            history_path, timestamp_str,
            get_git_commit(), get_git_branch(),
            all_config_tables, overall_pass, claude_assessment,
        )

    # Summary
    for name, _, passed, timings in all_config_tables:
        v = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        timing_str = format_timings(timings)
        suffix = f"  ({timing_str})" if timing_str else ""
        print(f"  {name}: {v}{suffix}")
    verdict = "\033[92mPASS\033[0m" if overall_pass else "\033[91mFAIL\033[0m"
    print(f"overall: {verdict}")

    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()


# python GNN_Test.py --config arbitrary --no-claude