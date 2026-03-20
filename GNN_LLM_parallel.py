import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import re
import shutil
import subprocess
import time
import yaml

import sys

from cell_gnn.config import CellGNNConfig
from cell_gnn.generators.graph_data_generator import data_generate
from cell_gnn.models.graph_trainer import data_train, data_test
from cell_gnn.models.exploration_tree import compute_ucb_scores
from cell_gnn.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from cell_gnn.models.utils import save_exploration_artifacts
from cell_gnn.utils import set_device, add_pre_folder, config_path, log_path

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """Detect the last fully completed batch from saved artifacts.

    Scans two sources:
      1. analysis.md for ``## Iter N:`` entries (written by Claude after training)
      2. config save dir for ``iter_NNN_slot_SS.yaml`` files (saved after test)

    Returns the start_iteration for the next batch (1-indexed), or 1 if nothing found.
    """
    found_iters = set()

    # Source 1: analysis.md — most reliable, written by Claude
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))

    # Source 2: saved config snapshots
    if os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            match = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if match:
                found_iters.add(int(match.group(1)))

    if not found_iters:
        return 1

    last_iter = max(found_iters)

    # Find the batch that contains last_iter
    batch_start = ((last_iter - 1) // n_parallel) * n_parallel + 1
    batch_iters = set(range(batch_start, batch_start + n_parallel))

    # Check if the full batch completed
    if batch_iters.issubset(found_iters):
        # Full batch done → resume from next batch
        resume_at = batch_start + n_parallel
    else:
        # Partial batch → redo this batch
        resume_at = batch_start

    return resume_at


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

CLUSTER_HOME = os.environ.get('CLUSTER_HOME', '/groups/saalfeld/home/allierc')
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/Graph/cell-gnn"
CLUSTER_USER = os.environ.get('CLUSTER_USER', 'allierc')
CLUSTER_LOGIN = os.environ.get('CLUSTER_LOGIN', 'login1')
CONDA_ENV = os.environ.get('CONDA_ENV', 'neural-graph')


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       log_dir, root_dir, erase=True, node_name='a100'):
    """Submit a single training job to the cluster WITHOUT -K (non-blocking).

    Returns the LSF job ID string, or None if submission failed.
    """
    cluster_script_path = f"{log_dir}/cluster_train_{slot:02d}.sh"
    error_details_path = f"{log_dir}/training_error_{slot:02d}.log"

    # Build cluster-side paths
    cluster_config_path = config_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_analysis_log = analysis_log_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_error_log = error_details_path.replace(root_dir, CLUSTER_ROOT_DIR)

    cluster_train_cmd = f"python train_cell_subprocess.py --config '{cluster_config_path}' --device cuda"
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

    # Cluster-side log paths for capturing stdout/stderr
    cluster_log_dir = log_dir.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_stdout = f"{cluster_log_dir}/cluster_train_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_train_{slot:02d}.err"

    # Submit WITHOUT -K so it returns immediately; capture stdout/stderr to files
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
    """Poll bjobs via SSH until all jobs finish.

    Args:
        job_ids: dict {slot: job_id_string}
        log_dir: local directory where cluster_train_XX.err files are written
        poll_interval: seconds between polls

    Returns:
        dict {slot: bool} — True if DONE, False if EXIT/failed
    """
    pending = dict(job_ids)  # {slot: job_id}
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
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)\033[0m")
                        if log_dir:
                            err_file = f"{log_dir}/cluster_train_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        err_content = ef.read().strip()
                                    if err_content:
                                        print(f"\033[91m  --- slot {slot} error log ---\033[0m")
                                        for eline in err_content.splitlines()[-30:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                        print("\033[91m  --- end error log ---\033[0m")
                                except Exception:
                                    pass

            # If job not found in bjobs output, it may have finished and been cleaned up
            if slot in pending and jid not in out.stdout:
                results[slot] = True  # assume done if disappeared from queue
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


def run_claude_cli(prompt, root_dir, max_turns=500):
    """Run Claude CLI with real-time output streaming. Returns output text."""
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
    parser = argparse.ArgumentParser(description="Cell-GNN — Parallel LLM Loop")
    parser.add_argument(
        "-o", "--option", nargs="+", help="option that takes multiple values"
    )
    parser.add_argument(
        "--fresh", action="store_true", default=True, help="start from iteration 1 (ignore auto-resume)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="auto-resume from last completed batch"
    )

    print()
    device = []
    args = parser.parse_args()

    N_PARALLEL = 4

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        best_model = None
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'train_test_Claude'
        config_list = ['dicty']
        task_params = {'iterations': 48}

    n_iterations = task_params.get('iterations', 5)
    base_config_name = config_list[0] if config_list else 'dicty'
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # -----------------------------------------------------------------------
    # Claude mode setup
    # -----------------------------------------------------------------------
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = config_path()
    llm_dir = f"{root_dir}/LLM"
    exploration_dir = log_path('Claude_exploration', f'{instruction_name}_parallel')

    # fresh start (default) or auto-resume (--resume flag)
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
            print(f"\033[91m  {exploration_dir}/{llm_task_name}_memory.md\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print("\033[93mfresh start\033[0m")

    # --- Initialize 4 slot configs from source ---
    for cfg in config_list:
        cfg_file, pre = add_pre_folder(cfg)
        source_config = f"{config_root}/{pre}{cfg}.yaml"

    # Read source config once to extract claude params
    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})
    claude_n_epochs = claude_cfg.get('n_epochs', 1)
    claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)
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
            # fresh start: copy source config, keep original dataset (shared data)
            shutil.copy2(source_config, target)
            with open(target, 'r') as f:
                config_data = yaml.safe_load(f)
            # Keep dataset unchanged — all slots share the same training data.
            # Each slot is differentiated by config_file (used for log directory).
            config_data['training']['n_epochs'] = claude_n_epochs
            config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
            config_data['description'] = 'designed by Claude (parallel)'
            config_data['claude'] = {
                'n_epochs': claude_n_epochs,
                'data_augmentation_loop': claude_data_augmentation_loop,
                'n_iter_block': claude_n_iter_block,
                'ucb_c': claude_ucb_c,
                'node_name': claude_node_name
            }
            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target} (config_file='{slot_name}')\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")

    # Shared files
    config_file, pre_folder = add_pre_folder(llm_task_name + '_00')
    analysis_path = f"{exploration_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{exploration_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{exploration_dir}/{llm_task_name}_ucb_scores.txt"
    instruction_path = f"{llm_dir}/{instruction_name}.md"
    parallel_instruction_path = f"{llm_dir}/instruction_{base_config_name}_parallel.md"
    reasoning_log_path = f"{exploration_dir}/{llm_task_name}_reasoning.log"

    log_dir = exploration_dir
    os.makedirs(exploration_dir, exist_ok=True)

    cluster_enabled = 'cluster' in task

    # Check instruction files exist
    if not os.path.exists(instruction_path):
        print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
        sys.exit(1)
    if not os.path.exists(parallel_instruction_path):
        print(f"\033[93mwarning: parallel instruction file not found: {parallel_instruction_path}\033[0m")
        print("\033[93m  Claude will use base instructions only\033[0m")
        parallel_instruction_path = None

    # Initialize shared files on fresh start
    if start_iteration == 1 and not args.resume:
        with open(analysis_path, 'w') as f:
            f.write(f"# Experiment Log: {base_config_name} (parallel)\n\n")
        print(f"\033[93mcleared {analysis_path}\033[0m")
        open(reasoning_log_path, 'w').close()
        print(f"\033[93mcleared {reasoning_log_path}\033[0m")
        with open(memory_path, 'w') as f:
            f.write(f"# Working Memory: {base_config_name} (parallel)\n\n")
            f.write("## Knowledge Base (accumulated across all blocks)\n\n")
            f.write("### Parameter Comparison Table\n")
            f.write("| Block | lr | lr_emb | batch | hidden | n_layers | emb_dim | max_r | Best RMSE | psi_R2 | Key finding |\n")
            f.write("| ----- | -- | ------ | ----- | ------ | -------- | ------- | ----- | --------- | ------ | ----------- |\n\n")
            f.write("### Established Principles\n\n")
            f.write("### Open Questions\n\n")
            f.write("---\n\n")
            f.write("## Previous Block Summary\n\n")
            f.write("---\n\n")
            f.write("## Current Block (Block 1)\n\n")
            f.write("### Block Info\n\n")
            f.write("### Hypothesis\n\n")
            f.write("### Iterations This Block\n\n")
            f.write("### Emerging Observations\n\n")
        print(f"\033[93mcleared {memory_path}\033[0m")
        if os.path.exists(ucb_path):
            os.remove(ucb_path)
            print(f"\033[93mdeleted {ucb_path}\033[0m")
    else:
        print(f"\033[93mpreserving shared files (resuming from iter {start_iteration})\033[0m")

    print(f"\033[93m{instruction_name} PARALLEL (N={N_PARALLEL}, {n_iterations} iterations, starting at {start_iteration})\033[0m")

    # -----------------------------------------------------------------------
    # BATCH 0: Claude "start" call — initialize 4 config variations
    # -----------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH 0: Claude initializing {N_PARALLEL} config variations\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        slot_list = "\n".join(
            f"  Slot {s}: {config_paths[s]}"
            for s in range(N_PARALLEL)
        )

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        start_prompt = f"""PARALLEL START: Initialize {N_PARALLEL} config variations for the first batch.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}

Config files to edit (all {N_PARALLEL}):
{slot_list}

Read the instructions and the base config, then create {N_PARALLEL} diverse initial training
parameter variations. Each config already has a unique dataset name — do NOT change the
dataset field. Vary training parameters (e.g. learning_rate_start, learning_rate_embedding_start,
batch_size, hidden_dim, n_layers) across the {N_PARALLEL} slots to explore different starting points.

Write the planned mutations to the working memory file."""

        print("\033[93mClaude start call...\033[0m")
        output_text = run_claude_cli(start_prompt, root_dir, max_turns=100)

        # Check for OAuth expiration
        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print("\n\033[91mOAuth token expired during start call\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print("\033[93m  2. Then re-run this script\033[0m")
            sys.exit(1)

        # Save reasoning
        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write("=== BATCH 0 (start call) ===\n")
                f.write(f"{'='*60}\n")
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
        # PHASE 1: Generate datasets (if task includes 'generate')
        # -------------------------------------------------------------------
        configs = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            config = CellGNNConfig.from_yaml(config_paths[slot])
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + slot_names[slot]
            configs[slot] = config

            if device == []:
                device = set_device(config.training.device)

            log_file = open(analysis_log_paths[slot], 'w')

            if "generate" in task:
                print(f"\033[90m  slot {slot} (iter {iteration}): generating data...\033[0m")
                data_generate(
                    config,
                    device=device,
                    visualize=False,
                    run_vizualized=0,
                    style="color",
                    alpha=1,
                    erase=True,
                    save=True,
                    step=50,
                )

            log_file.close()

        # -------------------------------------------------------------------
        # PHASE 2: Submit training jobs to cluster (or run locally)
        # -------------------------------------------------------------------
        job_results = {}

        if "train" in task:
            if cluster_enabled:
                print(f"\n\033[93mPHASE 2: Submitting {n_slots} training jobs to cluster\033[0m")

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

                # Wait for all submitted jobs
                if job_ids:
                    print(f"\n\033[93mPHASE 3: Waiting for {len(job_ids)} cluster jobs to complete\033[0m")
                    cluster_results = wait_for_cluster_jobs(job_ids, log_dir=log_dir, poll_interval=60)
                    job_results.update(cluster_results)

            else:
                # Local execution (no cluster) — run sequentially
                print(f"\n\033[93mPHASE 2: Training {n_slots} models locally (sequential)\033[0m")

                for slot_idx, iteration in enumerate(iterations):
                    slot = slot_idx
                    config = configs[slot]
                    print(f"\033[90m  slot {slot} (iter {iteration}): training locally...\033[0m")

                    log_file = open(analysis_log_paths[slot], 'a')
                    try:
                        data_train(
                            config=config,
                            erase=True,
                            best_model=best_model,
                            device=device,
                            log_file=log_file
                        )
                        job_results[slot] = True
                    except Exception as e:
                        print(f"\033[91m  slot {slot}: training failed: {e}\033[0m")
                        job_results[slot] = False
                    finally:
                        log_file.close()

        else:
            # No training — mark all as success
            for slot in range(n_slots):
                job_results[slot] = True

        # -------------------------------------------------------------------
        # PHASE 4: Test for successful slots
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 4: Test for successful slots\033[0m")

        activity_paths = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            if not job_results.get(slot, False):
                print(f"\033[90m  slot {slot} (iter {iteration}): skipping (training failed)\033[0m")
                continue

            config = configs[slot]

            log_file = open(analysis_log_paths[slot], 'a')

            if "test" in task:
                data_test(
                    config=config,
                    visualize=True,
                    style="color residual true",
                    verbose=False,
                    best_model='best',
                    run=0,
                    test_mode="",
                    sample_embedding=False,
                    step=250,
                    device=device,
                    cell_of_interest=0,
                    log_file=log_file,
                )

            log_file.close()

            # Save exploration artifacts
            iter_in_block = (iteration - 1) % n_iter_block + 1
            artifact_paths = save_exploration_artifacts(
                root_dir, exploration_dir, config, slot_names[slot],
                pre_folder, iteration,
                iter_in_block=iter_in_block, block_number=block_number
            )
            activity_paths[slot] = artifact_paths['activity_path']

            # Save config file for EVERY iteration (not just block start)
            config_save_dir = f"{exploration_dir}/config"
            os.makedirs(config_save_dir, exist_ok=True)
            dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
            shutil.copy2(config_paths[slot], dst_config)

        # -------------------------------------------------------------------
        # PHASE 5: Batch UCB update
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 5: Computing UCB scores\033[0m")

        # Read ucb_c from first slot config (all should share same claude section)
        with open(config_paths[0], 'r') as f:
            raw_config = yaml.safe_load(f)
        ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

        # Build a temporary analysis file with current batch metrics appended,
        # so compute_ucb_scores sees all new nodes at once.
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
            rmse_m = re.search(r'rollout_RMSE_mean[=:]\s*([\d.eE+-]+|nan)', log_content)
            psi_m = re.search(r'training_psi_R2[=:]\s*([\d.eE+-]+|nan)', log_content)
            acc_m = re.search(r'training_accuracy[=:]\s*([\d.eE+-]+|nan)', log_content)
            time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            if rmse_m:
                rmse_val = rmse_m.group(1)
                psi_val = psi_m.group(1) if psi_m else '0.0'
                acc_val = acc_m.group(1) if acc_m else '0.0'
                time_val = time_m.group(1) if time_m else '0.0'
                # Check if this iteration already exists in analysis.md (resume case)
                if f'## Iter {iteration}:' not in existing_content:
                    stub_entries += (
                        f"\n## Iter {iteration}: pending\n"
                        f"Node: id={iteration}, parent=root\n"
                        f"Metrics: rollout_RMSE_mean={rmse_val}, training_psi_R2={psi_val}, "
                        f"training_accuracy={acc_val}\n"
                    )

        tmp_analysis = analysis_path + '.tmp_ucb'
        with open(tmp_analysis, 'w') as f:
            f.write(existing_content + stub_entries)

        compute_ucb_scores(
            tmp_analysis, ucb_path, c=ucb_c,
            current_log_path=None,
            current_iteration=batch_last,
            block_size=n_iter_block
        )
        os.remove(tmp_analysis)
        print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 6: Claude analyzes results + proposes next mutations
        # -------------------------------------------------------------------
        print("\n\033[93mPHASE 6: Claude analysis + next mutations\033[0m")

        # Build per-slot info
        slot_info_lines = []
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            status = "COMPLETED" if job_results.get(slot, False) else "FAILED"
            act_path = activity_paths.get(slot, "N/A")
            slot_info_lines.append(
                f"Slot {slot} (iteration {iteration}) [{status}]:\n"
                f"  Metrics: {analysis_log_paths[slot]}\n"
                f"  Activity: {act_path}\n"
                f"  Config: {config_paths[slot]}"
            )
        slot_info = "\n\n".join(slot_info_lines)

        block_end_marker = "\n>>> BLOCK END <<<" if is_block_end else ""

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        claude_prompt = f"""Batch iterations {batch_first}-{batch_last} / {n_iterations}
Block info: block {block_number}, iterations {iter_in_block_first}-{iter_in_block_last}/{n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {n_slots} results, then propose next {N_PARALLEL} mutations.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}

{slot_info}

Analyze all {n_slots} results. For each successful slot, write a separate iteration entry
(## Iter N: ...) to the full log and memory file. Then edit all {N_PARALLEL} config files
to set up the next batch of {N_PARALLEL} experiments.

IMPORTANT: Do NOT change the 'dataset' field in any config — it must stay as-is for each slot."""

        print("\033[93mClaude analysis...\033[0m")
        output_text = run_claude_cli(claude_prompt, root_dir)

        # Check for OAuth expiration
        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91m{'='*60}\033[0m")
            print(f"\033[91mOAuth token expired at batch {batch_first}-{batch_last}\033[0m")
            print("\033[93mTo resume:\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print(f"\033[93m  2. Re-run with --resume\033[0m")
            print(f"\033[91m{'='*60}\033[0m")
            sys.exit(1)

        # Save reasoning
        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Batch {batch_first}-{batch_last} ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

        # Recompute UCB after Claude writes iteration entries to analysis.md
        compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                           current_log_path=None,
                           current_iteration=batch_last,
                           block_size=n_iter_block)

        # UCB tree visualization
        should_save_tree = (block_number == 1) or is_block_end
        if should_save_tree:
            tree_save_dir = f"{exploration_dir}/exploration_tree"
            os.makedirs(tree_save_dir, exist_ok=True)
            ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{batch_last:03d}.png"
            nodes = parse_ucb_scores(ucb_path) if os.path.exists(ucb_path) else []
            if nodes:
                config = configs[0]
                sim_info = f"n_cells={config.simulation.n_cells}, n_frames={config.simulation.n_frames}"
                sim_info += f", n_cell_types={config.simulation.n_cell_types}"
                sim_info += f", max_radius={config.simulation.max_radius}"
                sim_info += f", delta_t={config.simulation.delta_t}"
                sim_info += f", boundary={config.simulation.boundary}"
                sim_info += f", dimension={config.simulation.dimension}"
                if hasattr(config.training, 'recursive_training'):
                    sim_info += f", recursive={config.training.recursive_training}"
                plot_ucb_tree(nodes, ucb_tree_path,
                              title=f"UCB Tree - Batch {batch_first}-{batch_last}",
                              simulation_info=sim_info)

        # Save instruction file at first iteration of each block
        protocol_save_dir = f"{exploration_dir}/protocol"
        os.makedirs(protocol_save_dir, exist_ok=True)
        if iter_in_block_first == 1:
            dst_instruction = f"{protocol_save_dir}/block_{block_number:03d}.md"
            if os.path.exists(instruction_path):
                shutil.copy2(instruction_path, dst_instruction)

        # Save memory file at end of block
        if is_block_end:
            memory_save_dir = f"{exploration_dir}/memory"
            os.makedirs(memory_save_dir, exist_ok=True)
            dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
            if os.path.exists(memory_path):
                shutil.copy2(memory_path, dst_memory)
                print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

        # Print batch summary
        n_success = sum(1 for v in job_results.values() if v)
        n_failed = sum(1 for v in job_results.values() if not v)
        print(f"\n\033[92mBatch {batch_first}-{batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")


# Usage examples:
# python GNN_LLM_parallel.py -o train_test_Claude dicty iterations=48
# python GNN_LLM_parallel.py -o train_test_Claude dicty iterations=48 --resume
# python GNN_LLM_parallel.py -o train_test_Claude_cluster dicty iterations=48
# python GNN_LLM_parallel.py -o train_test_Claude_cluster misc/dicty iterations=48