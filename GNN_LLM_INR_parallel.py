"""Parallel LLM-in-the-loop INR parameter exploration.

Runs N_PARALLEL INR training slots in parallel, uses Claude to analyze results
and propose parameter mutations via UCB-guided search.

Usage:
    python GNN_LLM_INR_parallel.py -o train_inr_Claude dicty iterations=48
    python GNN_LLM_INR_parallel.py -o train_inr_Claude dicty iterations=48 --resume
    python GNN_LLM_INR_parallel.py -o train_inr_Claude_cluster dicty iterations=48
"""

import matplotlib
matplotlib.use('Agg')
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
import yaml
import warnings

from cell_gnn.config import CellGNNConfig
from cell_gnn.utils import set_device, add_pre_folder, config_path, log_path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """Detect the last fully completed batch from saved artifacts."""
    found_iters = set()

    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))

    if os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            match = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if match:
                found_iters.add(int(match.group(1)))

    if not found_iters:
        return 1

    last_iter = max(found_iters)
    batch_start = ((last_iter - 1) // n_parallel) * n_parallel + 1
    batch_iters = set(range(batch_start, batch_start + n_parallel))

    if batch_iters.issubset(found_iters):
        return batch_start + n_parallel
    else:
        return batch_start


# ---------------------------------------------------------------------------
# UCB for R2 metric
# ---------------------------------------------------------------------------

def compute_inr_ucb_scores(analysis_path, ucb_path, c=1.414, block_size=12):
    """Parse analysis file, compute UCB scores based on final_r2.

    Reads ## Iter N: entries, extracts final_r2 and parent, computes UCB.
    """
    nodes = {}

    if not os.path.exists(analysis_path):
        return False

    with open(analysis_path, 'r') as f:
        content = f.read()

    # Parse iteration entries
    sections = re.split(r'(?=## Iter \d+:)', content)
    for section in sections:
        iter_match = re.search(r'## Iter (\d+): (\w+)', section)
        if not iter_match:
            continue

        node_id = int(iter_match.group(1))
        status = iter_match.group(2)

        # Parse parent
        parent_match = re.search(r'Node: id=\d+, parent=(\w+)', section)
        parent = None
        if parent_match:
            p = parent_match.group(1)
            parent = int(p) if p != 'root' else None

        # Parse R2
        r2_match = re.search(r'final_r2=([\d.eE+-]+|nan)', section)
        r2 = 0.0
        if r2_match:
            try:
                r2 = float(r2_match.group(1))
            except ValueError:
                r2 = 0.0

        # Parse other metrics
        mse_match = re.search(r'final_mse=([\d.eE+-]+)', section)
        slope_match = re.search(r'slope=([\d.eE+-]+)', section)
        time_match = re.search(r'training_time_min=([\d.]+)', section)
        mutation_match = re.search(r'Mutation: (.+)', section)

        nodes[node_id] = {
            'id': node_id,
            'parent': parent,
            'status': status,
            'final_r2': r2,
            'final_mse': float(mse_match.group(1)) if mse_match else 0.0,
            'slope': float(slope_match.group(1)) if slope_match else 0.0,
            'training_time_min': float(time_match.group(1)) if time_match else 0.0,
            'mutation': mutation_match.group(1).strip() if mutation_match else '',
        }

    if not nodes:
        return False

    # Compute UCB scores
    total_visits = len(nodes)
    ucb_scores = []

    for node_id, node in nodes.items():
        reward = max(0.0, node['final_r2'])  # R2 is already in [0, 1] range
        visits = 1  # each node visited once
        exploration_term = c * math.sqrt(math.log(total_visits + 1) / visits)
        ucb = reward + exploration_term

        ucb_scores.append({
            'id': node_id,
            'ucb': ucb,
            'parent': node['parent'],
            'visits': visits,
            'final_r2': node['final_r2'],
            'final_mse': node['final_mse'],
            'slope': node['slope'],
            'training_time_min': node['training_time_min'],
            'mutation': node['mutation'],
        })

    ucb_scores.sort(key=lambda x: x['ucb'], reverse=True)

    # Write UCB scores
    with open(ucb_path, 'w') as f:
        for score in ucb_scores:
            parent_str = score['parent'] if score['parent'] is not None else 'root'
            f.write(
                f"Node {score['id']}: UCB={score['ucb']:.3f}, "
                f"parent={parent_str}, visits={score['visits']}, "
                f"R2={score['final_r2']:.6f}, "
                f"MSE={score['final_mse']:.6e}, "
                f"slope={score['slope']:.4f}, "
                f"time={score['training_time_min']:.1f}min"
            )
            if score['mutation']:
                f.write(f", mutation={score['mutation']}")
            f.write("\n")

    return True


# ---------------------------------------------------------------------------
# Cluster helpers (reused from GNN_LLM_parallel.py)
# ---------------------------------------------------------------------------

CLUSTER_HOME = os.environ.get('CLUSTER_HOME', '/groups/saalfeld/home/allierc')
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/Graph/cell-gnn"
CLUSTER_USER = os.environ.get('CLUSTER_USER', 'allierc')
CLUSTER_LOGIN = os.environ.get('CLUSTER_LOGIN', 'login1')
CONDA_ENV = os.environ.get('CONDA_ENV', 'neural-graph')


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       log_dir, root_dir, erase=True, node_name='a100'):
    """Submit a single INR training job to the cluster."""
    cluster_script_path = f"{log_dir}/cluster_inr_{slot:02d}.sh"
    error_details_path = f"{log_dir}/inr_error_{slot:02d}.log"

    cluster_config_path = config_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_analysis_log = analysis_log_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_error_log = error_details_path.replace(root_dir, CLUSTER_ROOT_DIR)

    cluster_train_cmd = f"python train_INR_subprocess.py --config '{cluster_config_path}' --device cuda"
    cluster_train_cmd += f" --log_file '{cluster_analysis_log}'"
    cluster_train_cmd += f" --config_file '{config_file_field}'"
    cluster_train_cmd += f" --error_log '{cluster_error_log}'"
    if erase:
        cluster_train_cmd += " --erase"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n {CONDA_ENV} {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = cluster_script_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_log_dir = log_dir.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_stdout = f"{cluster_log_dir}/cluster_inr_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_inr_{slot:02d}.err"

    ssh_cmd = (
        f"ssh {CLUSTER_USER}@{CLUSTER_LOGIN} \"cd {CLUSTER_ROOT_DIR} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_{node_name} -W 6000 "
        f"-o '{cluster_stdout}' -e '{cluster_stderr}' "
        f"'bash {cluster_script}'\""
    )
    print(f"\033[96m  slot {slot}: submitting via SSH\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: job {job_id} submitted\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60):
    """Poll bjobs via SSH until all jobs finish."""
    pending = dict(job_ids)
    results = {}

    while pending:
        ids_str = ' '.join(pending.values())
        ssh_cmd = f'ssh {CLUSTER_USER}@{CLUSTER_LOGIN} "bjobs {ids_str} 2>/dev/null"'
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        print(f"\033[92m  slot {slot} (job {jid}): DONE\033[0m")
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED\033[0m")

            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            print(f"\033[90m  ... waiting ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


def run_claude_cli(prompt, root_dir, max_turns=500):
    """Run Claude CLI with real-time output streaming."""
    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', str(max_turns),
        '--allowedTools',
        'Read', 'Edit', 'Write'
    ]

    output_lines = []
    process = subprocess.Popen(
        claude_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()
    return ''.join(output_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="Cell-GNN — Parallel LLM INR Loop")
    parser.add_argument("-o", "--option", nargs="+", help="option that takes multiple values")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="start from iteration 1 (ignore auto-resume)")
    parser.add_argument("--resume", action="store_true",
                        help="auto-resume from last completed batch")

    print()
    device = []
    args = parser.parse_args()

    N_PARALLEL = 4

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        task = 'train_inr_Claude'
        config_list = ['dicty']
        task_params = {'iterations': 48}

    n_iterations = task_params.get('iterations', 48)
    base_config_name = config_list[0] if config_list else 'dicty'
    instruction_name = f'instruction_{base_config_name}_INR'
    llm_task_name = f'{base_config_name}_INR_Claude'

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = config_path()
    llm_dir = f"{root_dir}/LLM"
    exploration_dir = log_path('Claude_exploration', f'{instruction_name}_parallel')

    # fresh start or resume
    if args.resume:
        analysis_path_probe = f"{exploration_dir}/{llm_task_name}_analysis.md"
        config_save_dir_probe = f"{exploration_dir}/config"
        start_iteration = detect_last_iteration(analysis_path_probe, config_save_dir_probe, N_PARALLEL)
        if start_iteration > 1:
            print(f"\033[93mAuto-resume: resuming from batch starting at {start_iteration}\033[0m")
        else:
            print("\033[93mfresh start (no previous iterations found)\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{exploration_dir}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print("\033[91mWARNING: fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print("\033[93mfresh start\033[0m")

    # --- Initialize slot configs from source ---
    for cfg in config_list:
        cfg_file, pre = add_pre_folder(cfg)
        source_config = f"{config_root}/{pre}{cfg}.yaml"

    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})
    claude_n_iter_block = claude_cfg.get('n_iter_block', 24)
    claude_ucb_c = claude_cfg.get('ucb_c', 1.414)
    claude_node_name = claude_cfg.get('node_name', 'a100')
    n_iter_block = claude_n_iter_block

    print(f"\033[94mCluster node: gpu_{claude_node_name}\033[0m")

    # Slot config paths and analysis log paths
    config_paths = {}
    analysis_log_paths = {}
    slot_names = {}

    for slot in range(N_PARALLEL):
        slot_name = f"{llm_task_name}_{slot:02d}"
        slot_names[slot] = slot_name
        target = f"{config_root}/{pre}{slot_name}.yaml"
        config_paths[slot] = target
        analysis_log_paths[slot] = f"{exploration_dir}/{slot_name}_analysis.log"

        if start_iteration == 1 and not args.resume:
            shutil.copy2(source_config, target)
            with open(target, 'r') as f:
                config_data = yaml.safe_load(f)
            config_data['description'] = 'INR exploration by Claude (parallel)'
            config_data['claude'] = {
                'n_iter_block': claude_n_iter_block,
                'ucb_c': claude_ucb_c,
                'node_name': claude_node_name,
            }
            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target}\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")

    # Shared files
    config_file, pre_folder = add_pre_folder(llm_task_name + '_00')
    analysis_path = f"{exploration_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{exploration_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{exploration_dir}/{llm_task_name}_ucb_scores.txt"
    instruction_path = f"{llm_dir}/{instruction_name}.md"
    reasoning_log_path = f"{exploration_dir}/{llm_task_name}_reasoning.log"

    log_dir = exploration_dir
    os.makedirs(exploration_dir, exist_ok=True)

    cluster_enabled = 'cluster' in task

    if not os.path.exists(instruction_path):
        print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
        sys.exit(1)

    # Initialize shared files on fresh start
    if start_iteration == 1 and not args.resume:
        with open(analysis_path, 'w') as f:
            f.write(f"# INR Experiment Log: {base_config_name} (parallel)\n\n")
        print(f"\033[93mcleared {analysis_path}\033[0m")
        open(reasoning_log_path, 'w').close()

        with open(memory_path, 'w') as f:
            f.write(f"# Working Memory: {base_config_name} INR (velocity field)\n\n")
            f.write("## Knowledge Base (accumulated across all blocks)\n\n")
            f.write("### Regime Comparison Table\n")
            f.write("| Block | omega_inr | lr | hidden_dim | n_layers | steps | batch | Best R2 | slope | time_min | Key finding |\n")
            f.write("| ----- | --------- | -- | ---------- | -------- | ----- | ----- | ------- | ----- | -------- | ----------- |\n\n")
            f.write("### Established Principles\n\n")
            f.write("### Open Questions\n\n")
            f.write("---\n\n")
            f.write("## Previous Block Summary\n\n")
            f.write("---\n\n")
            f.write("## Current Block (Block 1)\n\n")
            f.write("### Block Info\n")
            f.write(f"Field: velocity, inr_type: siren_txyz, n_frames: ~10000, n_cells: 1000, dim: 3\n\n")
            f.write("### Hypothesis\n\n")
            f.write("### Iterations This Block\n\n")
            f.write("### Emerging Observations\n\n")
        print(f"\033[93mcleared {memory_path}\033[0m")

        if os.path.exists(ucb_path):
            os.remove(ucb_path)

    print(f"\033[93m{instruction_name} PARALLEL (N={N_PARALLEL}, {n_iterations} iterations, starting at {start_iteration})\033[0m")

    # -----------------------------------------------------------------------
    # BATCH 0: Claude initializes N_PARALLEL config variations
    # -----------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH 0: Claude initializing {N_PARALLEL} config variations\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        slot_list = "\n".join(
            f"  Slot {s}: {config_paths[s]}"
            for s in range(N_PARALLEL)
        )

        start_prompt = f"""PARALLEL START: Initialize {N_PARALLEL} config variations for INR training.

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}

Config files to edit (all {N_PARALLEL}):
{slot_list}

Read the instructions and the base config. All configs start identical.
Create {N_PARALLEL} diverse initial INR parameter variations by editing ONLY the `inr:` section.
Vary parameters like: omega_inr, inr_learning_rate, hidden_dim_inr, n_layers_inr, inr_total_steps.
Do NOT change: inr_field_name, inr_type, inr_gradient_mode, or anything outside the `inr:` section.

Write the planned mutations to the working memory file."""

        print("\033[93mClaude start call...\033[0m")
        output_text = run_claude_cli(start_prompt, root_dir, max_turns=100)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print("\n\033[91mOAuth token expired during start call\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n=== BATCH 0 (start call) ===\n{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

    # -----------------------------------------------------------------------
    # Main batch loop
    # -----------------------------------------------------------------------
    for batch_start in range(start_iteration, n_iterations + 1, N_PARALLEL):
        iterations = [batch_start + s for s in range(N_PARALLEL)
                      if batch_start + s <= n_iterations]

        batch_first = iterations[0]
        batch_last = iterations[-1]
        n_slots = len(iterations)

        block_number = (batch_first - 1) // n_iter_block + 1
        iter_in_block_first = (batch_first - 1) % n_iter_block + 1
        iter_in_block_last = (batch_last - 1) % n_iter_block + 1
        is_block_end = any((it - 1) % n_iter_block + 1 == n_iter_block for it in iterations)

        # Block boundary: erase UCB at start of new block
        if batch_first > 1 and (batch_first - 1) % n_iter_block == 0:
            if os.path.exists(ucb_path):
                os.remove(ucb_path)
                print(f"\033[93mblock boundary: deleted {ucb_path}\033[0m")

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iterations {batch_first}-{batch_last} / {n_iterations}  (block {block_number})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 1: Run INR training (local or cluster)
        # -------------------------------------------------------------------
        configs = {}
        job_results = {}

        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            config = CellGNNConfig.from_yaml(config_paths[slot])
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + slot_names[slot]
            configs[slot] = config

            if device == []:
                device = set_device(config.training.device)

        if cluster_enabled:
            print(f"\n\033[93mPHASE 1: Submitting {n_slots} INR training jobs to cluster\033[0m")

            job_ids = {}
            for slot_idx, iteration in enumerate(iterations):
                slot = slot_idx
                config = configs[slot]
                jid = submit_cluster_job(
                    slot=slot,
                    config_path=config_paths[slot],
                    analysis_log_path=analysis_log_paths[slot],
                    config_file_field=config.config_file,
                    log_dir=log_dir,
                    root_dir=root_dir,
                    erase=True,
                    node_name=claude_node_name
                )
                if jid:
                    job_ids[slot] = jid
                else:
                    job_results[slot] = False

            if job_ids:
                print(f"\n\033[93mWaiting for {len(job_ids)} cluster jobs\033[0m")
                cluster_results = wait_for_cluster_jobs(job_ids, log_dir=log_dir, poll_interval=60)
                job_results.update(cluster_results)

        else:
            # Local execution — run sequentially
            print(f"\n\033[93mPHASE 1: Training {n_slots} INR models locally\033[0m")

            from cell_gnn.models.inr_trainer import data_train_INR

            for slot_idx, iteration in enumerate(iterations):
                slot = slot_idx
                config = configs[slot]
                field_name = config.inr.inr_field_name if config.inr else 'velocity'
                print(f"\033[90m  slot {slot} (iter {iteration}): training INR on '{field_name}'...\033[0m")

                try:
                    model, loss_list = data_train_INR(
                        config=config,
                        device=device,
                        field_name=field_name,
                        run=0,
                        erase=True,
                    )

                    # Copy results.log to analysis_log_path
                    inr_log_dir = log_path(config.config_file)
                    results_path = f'./{inr_log_dir}/tmp_training/inr/results.log'
                    if os.path.exists(results_path):
                        shutil.copy2(results_path, analysis_log_paths[slot])

                    # Copy video to exploration videos folder
                    video_src = f'./{inr_log_dir}/tmp_training/inr/{field_name}_gt_vs_pred.mp4'
                    if os.path.exists(video_src):
                        video_dir = f"{exploration_dir}/videos"
                        os.makedirs(video_dir, exist_ok=True)
                        iteration = iterations[slot_idx]
                        video_dst = f"{video_dir}/iter_{iteration:03d}_slot_{slot:02d}.mp4"
                        shutil.copy2(video_src, video_dst)

                    job_results[slot] = True
                except Exception as e:
                    print(f"\033[91m  slot {slot}: INR training failed: {e}\033[0m")
                    job_results[slot] = False

        # -------------------------------------------------------------------
        # PHASE 2: Save config snapshots
        # -------------------------------------------------------------------
        config_save_dir = f"{exploration_dir}/config"
        os.makedirs(config_save_dir, exist_ok=True)

        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
            shutil.copy2(config_paths[slot], dst_config)

        # -------------------------------------------------------------------
        # PHASE 3: Compute UCB scores
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 3: Computing UCB scores\033[0m")

        # Build stub entries for current batch
        existing_content = ""
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                existing_content = f.read()

        stub_entries = ""
        for slot_idx, iteration in enumerate(iterations):
            if not job_results.get(slot_idx, False):
                continue
            log_path = analysis_log_paths[slot_idx]
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r') as f:
                log_content = f.read()

            r2_match = re.search(r'final_r2[=:]\s*([\d.eE+-]+|nan)', log_content)
            mse_match = re.search(r'final_mse[=:]\s*([\d.eE+-]+|nan)', log_content)
            slope_match = re.search(r'slope[=:]\s*([\d.eE+-]+|nan)', log_content)
            time_match = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)

            if r2_match and f'## Iter {iteration}:' not in existing_content:
                r2_val = r2_match.group(1)
                mse_val = mse_match.group(1) if mse_match else '0.0'
                slope_val = slope_match.group(1) if slope_match else '0.0'
                time_val = time_match.group(1) if time_match else '0.0'
                stub_entries += (
                    f"\n## Iter {iteration}: pending\n"
                    f"Node: id={iteration}, parent=root\n"
                    f"Metrics: final_r2={r2_val}, final_mse={mse_val}, "
                    f"slope={slope_val}, training_time_min={time_val}\n"
                )

        tmp_analysis = analysis_path + '.tmp_ucb'
        with open(tmp_analysis, 'w') as f:
            f.write(existing_content + stub_entries)

        ucb_c = claude_ucb_c
        with open(config_paths[0], 'r') as f:
            raw_config = yaml.safe_load(f)
        ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

        compute_inr_ucb_scores(tmp_analysis, ucb_path, c=ucb_c, block_size=n_iter_block)
        os.remove(tmp_analysis)
        print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 4: Claude analyzes results + proposes next mutations
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 4: Claude analysis + next mutations\033[0m")

        slot_info_lines = []
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            status = "COMPLETED" if job_results.get(slot, False) else "FAILED"
            slot_info_lines.append(
                f"Slot {slot} (iteration {iteration}) [{status}]:\n"
                f"  Results: {analysis_log_paths[slot]}\n"
                f"  Config: {config_paths[slot]}"
            )
        slot_info = "\n\n".join(slot_info_lines)

        block_end_marker = "\n>>> BLOCK END <<<" if is_block_end else ""

        claude_prompt = f"""Batch iterations {batch_first}-{batch_last} / {n_iterations}
Block info: block {block_number}, iterations {iter_in_block_first}-{iter_in_block_last}/{n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {n_slots} INR training results, then propose next {N_PARALLEL} mutations.

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}

{slot_info}

Analyze all {n_slots} results. For each successful slot, read its results file and write a separate
iteration entry (## Iter N: ...) to the full log and memory file. Then edit all {N_PARALLEL} config
files to set up the next batch of {N_PARALLEL} experiments.

IMPORTANT: Only edit the `inr:` section in each config. Do NOT change inr_field_name, inr_type,
inr_gradient_mode, or anything outside the `inr:` section."""

        print("\033[93mClaude analysis...\033[0m")
        output_text = run_claude_cli(claude_prompt, root_dir)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91mOAuth token expired at batch {batch_first}-{batch_last}\033[0m")
            print("\033[93mRe-run with --resume\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n=== Batch {batch_first}-{batch_last} ===\n{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

        # Recompute UCB after Claude writes entries
        compute_inr_ucb_scores(analysis_path, ucb_path, c=ucb_c, block_size=n_iter_block)

        # Save memory at block end
        if is_block_end:
            memory_save_dir = f"{exploration_dir}/memory"
            os.makedirs(memory_save_dir, exist_ok=True)
            dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
            if os.path.exists(memory_path):
                shutil.copy2(memory_path, dst_memory)
                print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

        n_success = sum(1 for v in job_results.values() if v)
        n_failed = sum(1 for v in job_results.values() if not v)
        print(f"\n\033[92mBatch {batch_first}-{batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")


# Usage examples:
# python GNN_LLM_INR_parallel.py -o train_inr_Claude dicty iterations=48
# python GNN_LLM_INR_parallel.py -o train_inr_Claude dicty iterations=48 --resume
# python GNN_LLM_INR_parallel.py -o train_inr_Claude_cluster dicty iterations=48

