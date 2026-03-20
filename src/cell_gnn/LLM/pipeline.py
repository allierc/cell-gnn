"""Pipeline phase functions for the cell-gnn LLM exploration loop.

Each function corresponds to a phase in the main batch loop of GNN_LLM.py.
"""

import os
import re
import shutil
import sys

import yaml

from cell_gnn.config import CellGNNConfig
from cell_gnn.models.exploration_tree import compute_ucb_scores
from cell_gnn.models.graph_trainer import data_test, data_train
from cell_gnn.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from cell_gnn.models.utils import save_exploration_artifacts
from cell_gnn.utils import add_pre_folder, config_path, log_path, set_device

from .claude_cli import run_claude_cli
from .cluster import (
    check_cluster_repo,
    submit_cluster_job,
    wait_for_cluster_jobs,
)
from .prompts import analysis_prompt, batch_0_prompt
from .resume import detect_last_iteration
from .state import BatchInfo, ExplorationState


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_exploration(args, root_dir: str) -> ExplorationState:
    """Parse CLI args, load config, create ExplorationState."""
    print()

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

    n_iterations = task_params.get('iterations', 48)
    base_config_name = config_list[0] if config_list else 'dicty'
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')
    exploration_name = task_params.get('exploration_name', f'LLM_{base_config_name}')

    config_root = config_path()
    llm_dir = f"{root_dir}/LLM"
    exploration_dir = os.path.abspath(log_path('Claude_exploration', exploration_name))

    # Load source config and claude settings
    for cfg in config_list:
        cfg_file, pre = add_pre_folder(cfg)
        source_config = f"{config_root}/{pre}{cfg}.yaml"

    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})

    generate_data = claude_cfg.get('generate_data', "generate" in task)

    state = ExplorationState(
        root_dir=root_dir,
        config_root=config_root,
        llm_dir=llm_dir,
        exploration_dir=exploration_dir,
        source_config=source_config,
        base_config_name=base_config_name,
        pre_folder=pre,
        n_epochs=claude_cfg.get('n_epochs', 1),
        data_augmentation_loop=claude_cfg.get('data_augmentation_loop', 100),
        n_iter_block=claude_cfg.get('n_iter_block', 24),
        ucb_c=claude_cfg.get('ucb_c', 1.414),
        node_name=claude_cfg.get('node_name', 'a100'),
        n_parallel=claude_cfg.get('n_parallel', 4),
        generate_data=generate_data,
        training_time_target_min=claude_cfg.get('training_time_target_min', 60),
        interaction_code=claude_cfg.get('interaction_code', False),
        cluster_enabled=args.cluster,
        n_iterations=n_iterations,
        task=task,
        llm_task_name=llm_task_name,
        best_model=best_model,
    )

    # Detect resume point
    if args.resume:
        analysis_path_probe = f"{exploration_dir}/{llm_task_name}_analysis.md"
        config_save_dir_probe = f"{exploration_dir}/config"
        state.start_iteration = detect_last_iteration(
            analysis_path_probe, config_save_dir_probe, state.n_parallel
        )
        if state.start_iteration > 1:
            print(f"\033[93mAuto-resume: resuming from batch starting at {state.start_iteration}\033[0m")
        else:
            print("\033[93mfresh start (no previous iterations found)\033[0m")
    else:
        state.start_iteration = 1
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

    mode = "cluster" if state.cluster_enabled else "local (sequential)"
    print(f"\033[94mMode: {mode}, node: gpu_{state.node_name}, n_parallel: {state.n_parallel}, "
          f"generate_data: {state.generate_data}, training_time_target_min: {state.training_time_target_min}\033[0m")

    return state


def init_slot_configs(state: ExplorationState, is_resume: bool):
    """Create or preserve per-slot YAML configs."""
    config_file, pre_folder = add_pre_folder(state.llm_task_name + '_00')
    state.config_file = config_file

    for slot in range(state.n_parallel):
        slot_name = f"{state.llm_task_name}_{slot:02d}"
        state.slot_names[slot] = slot_name
        target = f"{state.config_root}/{state.pre_folder}{slot_name}.yaml"
        state.config_paths[slot] = target
        state.analysis_log_paths[slot] = f"{state.exploration_dir}/{slot_name}_analysis.log"

        if state.start_iteration == 1 and not is_resume:
            shutil.copy2(state.source_config, target)
            with open(target, 'r') as f:
                config_data = yaml.safe_load(f)
            if state.generate_data:
                config_data['dataset'] = f"{state.base_config_name}_{slot:02d}"
            config_data['training']['n_epochs'] = state.n_epochs
            config_data['training']['data_augmentation_loop'] = state.data_augmentation_loop
            config_data['description'] = 'designed by Claude (parallel)'
            config_data['claude'] = {
                'n_epochs': state.n_epochs,
                'data_augmentation_loop': state.data_augmentation_loop,
                'n_iter_block': state.n_iter_block,
                'ucb_c': state.ucb_c,
                'n_parallel': state.n_parallel,
                'node_name': state.node_name,
                'generate_data': state.generate_data,
                'training_time_target_min': state.training_time_target_min,
            }
            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target} (config_file='{slot_name}')\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")


def init_shared_files(state: ExplorationState, is_resume: bool):
    """Create analysis/memory/UCB files on fresh start, or preserve on resume."""
    state.analysis_path = f"{state.exploration_dir}/{state.llm_task_name}_analysis.md"
    state.memory_path = f"{state.exploration_dir}/{state.llm_task_name}_memory.md"
    state.ucb_path = f"{state.exploration_dir}/{state.llm_task_name}_ucb_scores.txt"
    instruction_name = f'instruction_{state.base_config_name}'
    state.instruction_path = f"{state.llm_dir}/{instruction_name}.md"
    state.reasoning_log_path = f"{state.exploration_dir}/{state.llm_task_name}_reasoning.log"
    state.user_input_path = f"{state.exploration_dir}/user_input.md"
    state.log_dir = state.exploration_dir

    os.makedirs(state.exploration_dir, exist_ok=True)

    # Check instruction file exists
    if not os.path.exists(state.instruction_path):
        print(f"\033[91merror: instruction file not found: {state.instruction_path}\033[0m")
        sys.exit(1)

    # Create user input file if missing
    if not os.path.exists(state.user_input_path):
        with open(state.user_input_path, 'w') as f:
            f.write("# User Input\n\n")
            f.write("_Write instructions or advice here. The LLM will read this file at each batch and acknowledge below._\n\n")
            f.write("## Pending Instructions\n\n")
            f.write("_(empty — add instructions here)_\n\n")
            f.write("## Acknowledged\n\n")

    # Initialize shared files on fresh start
    if state.start_iteration == 1 and not is_resume:
        with open(state.analysis_path, 'w') as f:
            f.write(f"# Experiment Log: {state.base_config_name} (parallel)\n\n")
        print(f"\033[93mcleared {state.analysis_path}\033[0m")
        open(state.reasoning_log_path, 'w').close()
        print(f"\033[93mcleared {state.reasoning_log_path}\033[0m")
        with open(state.memory_path, 'w') as f:
            f.write(f"# Working Memory: {state.base_config_name} (parallel)\n\n")
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
        print(f"\033[93mcleared {state.memory_path}\033[0m")
        if os.path.exists(state.ucb_path):
            os.remove(state.ucb_path)
            print(f"\033[93mdeleted {state.ucb_path}\033[0m")
    else:
        print(f"\033[93mpreserving shared files (resuming from iter {state.start_iteration})\033[0m")

    print(f"\033[93m{state.base_config_name} PARALLEL "
          f"(N={state.n_parallel}, {state.n_iterations} iterations, starting at {state.start_iteration})\033[0m")


# ---------------------------------------------------------------------------
# Batch info
# ---------------------------------------------------------------------------

def make_batch_info(state: ExplorationState, batch_start: int) -> BatchInfo:
    """Compute BatchInfo for a batch starting at batch_start."""
    iterations = [batch_start + s for s in range(state.n_parallel)
                  if batch_start + s <= state.n_iterations]

    batch_first = iterations[0]
    batch_last = iterations[-1]
    n_slots = len(iterations)

    block_number = (batch_first - 1) // state.n_iter_block + 1
    iter_in_block_first = (batch_first - 1) % state.n_iter_block + 1
    iter_in_block_last = (batch_last - 1) % state.n_iter_block + 1
    is_block_end = any((it - 1) % state.n_iter_block + 1 == state.n_iter_block for it in iterations)
    is_block_start = (batch_first == 1) or ((batch_first - 1) % state.n_iter_block == 0)

    return BatchInfo(
        iterations=iterations,
        batch_first=batch_first,
        batch_last=batch_last,
        n_slots=n_slots,
        block_number=block_number,
        iter_in_block_first=iter_in_block_first,
        iter_in_block_last=iter_in_block_last,
        is_block_start=is_block_start,
        is_block_end=is_block_end,
    )


# ---------------------------------------------------------------------------
# Batch 0
# ---------------------------------------------------------------------------

def run_batch_0(state: ExplorationState):
    """BATCH 0: Claude start call to initialize N config variations."""
    print(f"\n\033[94m{'='*60}\033[0m")
    print(f"\033[94mBATCH 0: Claude initializing {state.n_parallel} config variations\033[0m")
    print(f"\033[94m{'='*60}\033[0m")

    slot_list = "\n".join(
        f"  Slot {s}: {state.config_paths[s]}"
        for s in range(state.n_parallel)
    )
    seed_info = "\n".join(
        f"  Slot {s}: simulation_seed={(state.start_iteration + s) * 1000 + s}, "
        f"training_seed={(state.start_iteration + s) * 1000 + s + 500}"
        for s in range(state.n_parallel)
    )

    prompt = batch_0_prompt(state, slot_list, seed_info)

    print("\033[93mClaude start call...\033[0m")
    output_text = run_claude_cli(prompt, state.root_dir, max_turns=100)

    if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
        print("\n\033[91mOAuth token expired during start call\033[0m")
        print("\033[93m  1. Run: claude /login\033[0m")
        print("\033[93m  2. Then re-run this script\033[0m")
        sys.exit(1)

    if output_text.strip():
        with open(state.reasoning_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write("=== BATCH 0 (start call) ===\n")
            f.write(f"{'='*60}\n")
            f.write(output_text.strip())
            f.write("\n\n")


# ---------------------------------------------------------------------------
# Phase 1: Load configs + force seeds
# ---------------------------------------------------------------------------

def load_configs_and_seeds(state: ExplorationState, batch: BatchInfo):
    """PHASE 1: Load configs, force seeds, write seeds back to YAML."""
    print(f"\n\033[93mPHASE 1: Loading configs for {batch.n_slots} slots\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = CellGNNConfig.from_yaml(state.config_paths[slot])
        if not config.dataset.startswith(state.pre_folder):
            config.dataset = state.pre_folder + config.dataset
        config.config_file = state.pre_folder + state.slot_names[slot]

        # Force seeds (pipeline-controlled)
        sim_seed = iteration * 1000 + slot
        train_seed = iteration * 1000 + slot + 500
        config.simulation.seed = sim_seed
        config.training.seed = train_seed
        batch.slot_seeds[slot] = {'simulation': sim_seed, 'training': train_seed}

        # Write forced seeds back to YAML
        with open(state.config_paths[slot], 'r') as f:
            yaml_data = yaml.safe_load(f)
        yaml_data['simulation']['seed'] = sim_seed
        yaml_data['training']['seed'] = train_seed
        yaml_data['dataset'] = config.dataset
        with open(state.config_paths[slot], 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        batch.configs[slot] = config

        if state.device is None:
            state.device = set_device(config.training.device)

    seed_info = "\n".join(
        f"  Slot {s}: simulation_seed={batch.slot_seeds[s]['simulation']}, "
        f"training_seed={batch.slot_seeds[s]['training']}"
        for s in range(batch.n_slots)
    )
    print(f"\033[90mSeeds (forced by pipeline):\n{seed_info}\033[0m")

    # Handle UCB reset at block boundary
    if batch.batch_first > 1 and (batch.batch_first - 1) % state.n_iter_block == 0:
        if os.path.exists(state.ucb_path):
            os.remove(state.ucb_path)
            print(f"\033[93mblock boundary: deleted {state.ucb_path}\033[0m")


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def generate_data_locally(state: ExplorationState, batch: BatchInfo):
    """PHASE 1.5: Generate data locally for all slots."""
    from cell_gnn.generators.graph_data_generator import data_generate

    print(f"\n\033[93mPHASE 1.5: Generating data locally for {batch.n_slots} slots\033[0m")
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        print(f"\033[90m  slot {slot} (iter {iteration}): generating data...\033[0m")
        data_generate(
            config=config,
            device=state.device,
            visualize=False,
            run_vizualized=0,
            style="color",
            alpha=1,
            erase=True,
            save=True,
            step=50,
        )


def run_cluster_training(state: ExplorationState, batch: BatchInfo):
    """PHASE 2-3: Submit cluster jobs, wait for completion."""
    print(f"\n\033[93mPHASE 2: Submitting {batch.n_slots} training jobs to cluster\033[0m")

    if not check_cluster_repo():
        print("\033[91mAborting batch — fix cluster repo before resubmitting (use --resume)\033[0m")
        sys.exit(1)

    job_ids = {}
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        jid = submit_cluster_job(
            slot=slot,
            config_path=state.config_paths[slot],
            analysis_log_path=state.analysis_log_paths[slot],
            config_file_field=config.config_file,
            log_dir=state.log_dir,
            root_dir=state.root_dir,
            erase=True,
            node_name=state.node_name,
            exploration_dir=state.exploration_dir,
            iteration=iteration,
        )
        if jid:
            job_ids[slot] = jid
        else:
            batch.job_results[slot] = False

    if job_ids:
        print(f"\n\033[93mPHASE 3: Waiting for {len(job_ids)} cluster jobs to complete\033[0m")
        cluster_results = wait_for_cluster_jobs(job_ids, log_dir=state.log_dir, poll_interval=60)
        batch.job_results.update(cluster_results)

    # Test for successful slots
    print("\n\033[93mPHASE 3.5: Test for successful slots\033[0m")
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        if not batch.job_results.get(slot, False):
            print(f"\033[90m  slot {slot} (iter {iteration}): skipping (training failed)\033[0m")
            continue
        config = batch.configs[slot]
        log_file = open(state.analysis_log_paths[slot], 'a')
        if "test" in state.task:
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
                device=state.device,
                cell_of_interest=0,
                log_file=log_file,
            )
        log_file.close()


def run_local_pipeline(state: ExplorationState, batch: BatchInfo):
    """PHASE 2 local: Generate + train + test sequentially."""
    print(f"\n\033[93mPHASE 2: Training {batch.n_slots} models locally (sequential)\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        print(f"\033[90m  slot {slot} (iter {iteration}): training locally...\033[0m")

        log_file = open(state.analysis_log_paths[slot], 'w')

        # Generate data if requested
        if state.generate_data and "generate" in state.task:
            from cell_gnn.generators.graph_data_generator import data_generate
            print(f"\033[90m  slot {slot}: generating data...\033[0m")
            data_generate(
                config=config,
                device=state.device,
                visualize=False,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=True,
                save=True,
                step=50,
            )

        # Train
        try:
            data_train(
                config=config,
                erase=True,
                best_model=state.best_model,
                device=state.device,
                log_file=log_file
            )
            batch.job_results[slot] = True
        except Exception as e:
            print(f"\033[91m  slot {slot}: training failed: {e}\033[0m")
            batch.job_results[slot] = False
            log_file.close()
            continue

        # Test
        if "test" in state.task:
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
                device=state.device,
                cell_of_interest=0,
                log_file=log_file,
            )

        log_file.close()


# ---------------------------------------------------------------------------
# Phase 4: Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(state: ExplorationState, batch: BatchInfo):
    """PHASE 4: Save exploration artifacts."""
    print("\n\033[93mPHASE 4: Saving exploration artifacts\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        if not batch.job_results.get(slot, False):
            print(f"\033[90m  slot {slot} (iter {iteration}): skipping (training failed)\033[0m")
            continue

        config = batch.configs[slot]

        iter_in_block = (iteration - 1) % state.n_iter_block + 1
        artifact_paths = save_exploration_artifacts(
            state.root_dir, state.exploration_dir, config, state.slot_names[slot],
            state.pre_folder, iteration,
            iter_in_block=iter_in_block, block_number=batch.block_number
        )
        batch.activity_paths[slot] = artifact_paths['activity_path']

        # Save config file for EVERY iteration
        config_save_dir = f"{state.exploration_dir}/config"
        os.makedirs(config_save_dir, exist_ok=True)
        dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
        shutil.copy2(state.config_paths[slot], dst_config)

        # Check training time
        slot_log_path = state.analysis_log_paths[slot]
        if os.path.exists(slot_log_path):
            with open(slot_log_path, 'r') as f:
                log_content = f.read()
            time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            if time_m:
                training_time = float(time_m.group(1))
                if training_time > state.training_time_target_min:
                    print(f"\033[91m  WARNING: slot {slot} training took {training_time:.1f} min (>{state.training_time_target_min} min target)\033[0m")
                else:
                    print(f"\033[92m  slot {slot}: training time {training_time:.1f} min\033[0m")


# ---------------------------------------------------------------------------
# Phase 5: UCB scores
# ---------------------------------------------------------------------------

def update_ucb_scores(state: ExplorationState, batch: BatchInfo):
    """PHASE 5: Compute UCB scores from batch results."""
    print("\n\033[93mPHASE 5: Computing UCB scores\033[0m")

    with open(state.config_paths[0], 'r') as f:
        raw_config = yaml.safe_load(f)
    ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

    existing_content = ""
    if os.path.exists(state.analysis_path):
        with open(state.analysis_path, 'r') as f:
            existing_content = f.read()

    stub_entries = ""
    for slot_idx, iteration in enumerate(batch.iterations):
        if not batch.job_results.get(slot_idx, False):
            continue
        slot_log_path = state.analysis_log_paths[slot_idx]
        if not os.path.exists(slot_log_path):
            continue
        with open(slot_log_path, 'r') as f:
            log_content = f.read()
        rmse_m = re.search(r'rollout_RMSE_mean[=:]\s*([\d.eE+-]+|nan)', log_content)
        psi_m = re.search(r'training_psi_R2[=:]\s*([\d.eE+-]+|nan)', log_content)
        acc_m = re.search(r'training_accuracy[=:]\s*([\d.eE+-]+|nan)', log_content)
        time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
        if rmse_m:
            rmse_val = rmse_m.group(1)
            psi_val = psi_m.group(1) if psi_m else '0.0'
            acc_val = acc_m.group(1) if acc_m else '0.0'
            if f'## Iter {iteration}:' not in existing_content:
                stub_entries += (
                    f"\n## Iter {iteration}: pending\n"
                    f"Node: id={iteration}, parent=root\n"
                    f"Metrics: rollout_RMSE_mean={rmse_val}, training_psi_R2={psi_val}, "
                    f"training_accuracy={acc_val}\n"
                )

    tmp_analysis = state.analysis_path + '.tmp_ucb'
    with open(tmp_analysis, 'w') as f:
        f.write(existing_content + stub_entries)

    compute_ucb_scores(
        tmp_analysis, state.ucb_path, c=ucb_c,
        current_log_path=None,
        current_iteration=batch.batch_last,
        block_size=state.n_iter_block
    )
    os.remove(tmp_analysis)
    print(f"\033[92mUCB scores computed (c={ucb_c}): {state.ucb_path}\033[0m")


# ---------------------------------------------------------------------------
# Phase 6: Claude analysis
# ---------------------------------------------------------------------------

def run_claude_analysis(state: ExplorationState, batch: BatchInfo):
    """PHASE 6: Claude analyzes results + proposes next mutations."""
    print("\n\033[93mPHASE 6: Claude analysis + next mutations\033[0m")

    slot_info_lines = []
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        status = "COMPLETED" if batch.job_results.get(slot, False) else "FAILED"
        act_path = batch.activity_paths.get(slot, "N/A")
        slot_info_lines.append(
            f"Slot {slot} (iteration {iteration}) [{status}]:\n"
            f"  Seeds: simulation={batch.slot_seeds[slot]['simulation']}, "
            f"training={batch.slot_seeds[slot]['training']}\n"
            f"  Metrics: {state.analysis_log_paths[slot]}\n"
            f"  Activity: {act_path}\n"
            f"  Config: {state.config_paths[slot]}"
        )
    slot_info = "\n\n".join(slot_info_lines)

    prompt = analysis_prompt(state, batch, slot_info)

    print("\033[93mClaude analysis...\033[0m")
    output_text = run_claude_cli(prompt, state.root_dir)

    if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
        print(f"\n\033[91m{'='*60}\033[0m")
        print(f"\033[91mOAuth token expired at batch {batch.batch_first}-{batch.batch_last}\033[0m")
        print("\033[93mTo resume:\033[0m")
        print("\033[93m  1. Run: claude /login\033[0m")
        print("\033[93m  2. Then re-run with --resume\033[0m")
        print(f"\033[91m{'='*60}\033[0m")
        sys.exit(1)

    if output_text.strip():
        with open(state.reasoning_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"=== Batch {batch.batch_first}-{batch.batch_last} ===\n")
            f.write(f"{'='*60}\n")
            f.write(output_text.strip())
            f.write("\n\n")


# ---------------------------------------------------------------------------
# Finalize batch
# ---------------------------------------------------------------------------

def finalize_batch(state: ExplorationState, batch: BatchInfo):
    """UCB recompute, tree visualization, protocol/memory snapshots."""
    with open(state.config_paths[0], 'r') as f:
        raw_config = yaml.safe_load(f)
    ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

    compute_ucb_scores(state.analysis_path, state.ucb_path, c=ucb_c,
                       current_log_path=None,
                       current_iteration=batch.batch_last,
                       block_size=state.n_iter_block)

    # UCB tree visualization
    should_save_tree = (batch.block_number == 1) or batch.is_block_end
    if should_save_tree:
        tree_save_dir = f"{state.exploration_dir}/exploration_tree"
        os.makedirs(tree_save_dir, exist_ok=True)
        ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{batch.batch_last:03d}.png"
        nodes = parse_ucb_scores(state.ucb_path) if os.path.exists(state.ucb_path) else []
        if nodes:
            config = batch.configs[0]
            sim_info = f"n_cells={config.simulation.n_cells}, n_frames={config.simulation.n_frames}"
            sim_info += f", n_cell_types={config.simulation.n_cell_types}"
            sim_info += f", max_radius={config.simulation.max_radius}"
            sim_info += f", delta_t={config.simulation.delta_t}"
            sim_info += f", boundary={config.simulation.boundary}"
            sim_info += f", dimension={config.simulation.dimension}"
            if hasattr(config.training, 'recursive_training'):
                sim_info += f", recursive={config.training.recursive_training}"
            plot_ucb_tree(nodes, ucb_tree_path,
                          title=f"UCB Tree - Batch {batch.batch_first}-{batch.batch_last}",
                          simulation_info=sim_info)

    # Save instruction file at first iteration of each block
    protocol_save_dir = f"{state.exploration_dir}/protocol"
    os.makedirs(protocol_save_dir, exist_ok=True)
    if batch.iter_in_block_first == 1:
        dst_instruction = f"{protocol_save_dir}/block_{batch.block_number:03d}.md"
        if os.path.exists(state.instruction_path):
            shutil.copy2(state.instruction_path, dst_instruction)

    # Save memory file at end of block
    if batch.is_block_end:
        memory_save_dir = f"{state.exploration_dir}/memory"
        os.makedirs(memory_save_dir, exist_ok=True)
        dst_memory = f"{memory_save_dir}/block_{batch.block_number:03d}_memory.md"
        if os.path.exists(state.memory_path):
            shutil.copy2(state.memory_path, dst_memory)
            print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

    # Print batch summary
    n_success = sum(1 for v in batch.job_results.values() if v)
    n_failed = sum(1 for v in batch.job_results.values() if not v)
    print(f"\n\033[92mBatch {batch.batch_first}-{batch.batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")
