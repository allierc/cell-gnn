"""Pipeline phase functions for the INR LLM exploration loop.

Similar to pipeline.py but specialized for INR (SIREN) training.
"""

import math
import os
import re
import shutil
import sys

import yaml

from cell_gnn.config import CellGNNConfig
from cell_gnn.utils import add_pre_folder, config_path, log_path, set_device

from .claude_cli import run_claude_cli
from .resume import detect_last_iteration
from .state import BatchInfo, ExplorationState


# ---------------------------------------------------------------------------
# INR-specific UCB (based on R2 instead of RMSE)
# ---------------------------------------------------------------------------

def compute_inr_ucb_scores(analysis_path, ucb_path, c=1.414, block_size=12):
    """Parse analysis file, compute UCB scores based on final_r2."""
    nodes = {}

    if not os.path.exists(analysis_path):
        return False

    with open(analysis_path, 'r') as f:
        content = f.read()

    sections = re.split(r'(?=## Iter \d+:)', content)
    for section in sections:
        iter_match = re.search(r'## Iter (\d+): (\w+)', section)
        if not iter_match:
            continue

        node_id = int(iter_match.group(1))

        parent_match = re.search(r'Node: id=\d+, parent=(\w+)', section)
        parent = None
        if parent_match:
            p = parent_match.group(1)
            parent = int(p) if p != 'root' else None

        r2_match = re.search(r'final_r2=([\d.eE+-]+|nan)', section)
        r2 = 0.0
        if r2_match:
            try:
                r2 = float(r2_match.group(1))
            except ValueError:
                r2 = 0.0

        mse_match = re.search(r'final_mse=([\d.eE+-]+)', section)
        slope_match = re.search(r'slope=([\d.eE+-]+)', section)
        time_match = re.search(r'training_time_min=([\d.]+)', section)
        mutation_match = re.search(r'Mutation: (.+)', section)

        nodes[node_id] = {
            'id': node_id,
            'parent': parent,
            'final_r2': r2,
            'final_mse': float(mse_match.group(1)) if mse_match else 0.0,
            'slope': float(slope_match.group(1)) if slope_match else 0.0,
            'training_time_min': float(time_match.group(1)) if time_match else 0.0,
            'mutation': mutation_match.group(1).strip() if mutation_match else '',
        }

    if not nodes:
        return False

    total_visits = len(nodes)
    ucb_scores = []

    for node_id, node in nodes.items():
        reward = max(0.0, node['final_r2'])
        visits = 1
        exploration_term = c * math.sqrt(math.log(total_visits + 1) / visits)
        ucb = reward + exploration_term

        ucb_scores.append({
            'id': node_id, 'ucb': ucb, 'parent': node['parent'],
            'visits': visits, 'final_r2': node['final_r2'],
            'final_mse': node['final_mse'], 'slope': node['slope'],
            'training_time_min': node['training_time_min'],
            'mutation': node['mutation'],
        })

    ucb_scores.sort(key=lambda x: x['ucb'], reverse=True)

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
# Setup
# ---------------------------------------------------------------------------

def setup_inr_exploration(args, root_dir: str) -> ExplorationState:
    """Parse CLI args, load config, create ExplorationState for INR."""
    print()

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
    exploration_name = task_params.get('exploration_name', f'LLM_{base_config_name}_INR')

    config_root = config_path()
    llm_dir = f"{root_dir}/LLM"
    exploration_dir = os.path.abspath(log_path('Claude_exploration', exploration_name))

    for cfg in config_list:
        cfg_file, pre = add_pre_folder(cfg)
        source_config = f"{config_root}/{pre}{cfg}.yaml"

    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})

    state = ExplorationState(
        root_dir=root_dir,
        config_root=config_root,
        llm_dir=llm_dir,
        exploration_dir=exploration_dir,
        source_config=source_config,
        base_config_name=base_config_name,
        pre_folder=pre,
        n_iter_block=claude_cfg.get('n_iter_block', 24),
        ucb_c=claude_cfg.get('ucb_c', 1.414),
        node_name=claude_cfg.get('node_name', 'a100'),
        n_parallel=claude_cfg.get('n_parallel', 4),
        training_time_target_min=claude_cfg.get('training_time_target_min', 60),
        cluster_enabled=args.cluster,
        n_iterations=n_iterations,
        task=task,
        llm_task_name=llm_task_name,
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
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print("\033[93mfresh start\033[0m")

    mode = "cluster" if state.cluster_enabled else "local (sequential)"
    print(f"\033[94mMode: {mode}, node: gpu_{state.node_name}, n_parallel: {state.n_parallel}\033[0m")

    return state


def init_inr_slot_configs(state: ExplorationState, is_resume: bool):
    """Create or preserve per-slot YAML configs for INR."""
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
            config_data['description'] = 'INR exploration by Claude (parallel)'
            config_data['claude'] = {
                'n_iter_block': state.n_iter_block,
                'ucb_c': state.ucb_c,
                'n_parallel': state.n_parallel,
                'node_name': state.node_name,
                'training_time_target_min': state.training_time_target_min,
            }
            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target}\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")


def init_inr_shared_files(state: ExplorationState, is_resume: bool):
    """Create analysis/memory/UCB files for INR exploration."""
    state.analysis_path = f"{state.exploration_dir}/{state.llm_task_name}_analysis.md"
    state.memory_path = f"{state.exploration_dir}/{state.llm_task_name}_memory.md"
    state.ucb_path = f"{state.exploration_dir}/{state.llm_task_name}_ucb_scores.txt"
    instruction_name = f'instruction_{state.base_config_name}_INR'
    state.instruction_path = f"{state.llm_dir}/{instruction_name}.md"
    state.reasoning_log_path = f"{state.exploration_dir}/{state.llm_task_name}_reasoning.log"
    state.user_input_path = f"{state.exploration_dir}/user_input.md"
    state.log_dir = state.exploration_dir

    os.makedirs(state.exploration_dir, exist_ok=True)

    if not os.path.exists(state.instruction_path):
        print(f"\033[91merror: instruction file not found: {state.instruction_path}\033[0m")
        sys.exit(1)

    if not os.path.exists(state.user_input_path):
        with open(state.user_input_path, 'w') as f:
            f.write("# User Input\n\n")
            f.write("_Write instructions or advice here._\n\n")
            f.write("## Pending Instructions\n\n")
            f.write("_(empty)_\n\n")
            f.write("## Acknowledged\n\n")

    if state.start_iteration == 1 and not is_resume:
        with open(state.analysis_path, 'w') as f:
            f.write(f"# INR Experiment Log: {state.base_config_name} (parallel)\n\n")
        print(f"\033[93mcleared {state.analysis_path}\033[0m")
        open(state.reasoning_log_path, 'w').close()

        with open(state.memory_path, 'w') as f:
            f.write(f"# Working Memory: {state.base_config_name} INR (velocity field)\n\n")
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
        print(f"\033[93mcleared {state.memory_path}\033[0m")

        if os.path.exists(state.ucb_path):
            os.remove(state.ucb_path)
    else:
        print(f"\033[93mpreserving shared files (resuming from iter {state.start_iteration})\033[0m")

    print(f"\033[93m{state.base_config_name} INR PARALLEL "
          f"(N={state.n_parallel}, {state.n_iterations} iterations, starting at {state.start_iteration})\033[0m")


# ---------------------------------------------------------------------------
# Batch info (reuse from pipeline)
# ---------------------------------------------------------------------------

def make_inr_batch_info(state: ExplorationState, batch_start: int) -> BatchInfo:
    """Compute BatchInfo for an INR batch."""
    from .pipeline import make_batch_info
    return make_batch_info(state, batch_start)


# ---------------------------------------------------------------------------
# Batch 0
# ---------------------------------------------------------------------------

def run_inr_batch_0(state: ExplorationState):
    """BATCH 0: Claude initializes N INR config variations."""
    print(f"\n\033[94m{'='*60}\033[0m")
    print(f"\033[94mBATCH 0: Claude initializing {state.n_parallel} INR config variations\033[0m")
    print(f"\033[94m{'='*60}\033[0m")

    slot_list = "\n".join(
        f"  Slot {s}: {state.config_paths[s]}"
        for s in range(state.n_parallel)
    )
    seed_info = "\n".join(
        f"  Slot {s}: inr_seed={(state.start_iteration + s) * 1000 + s}"
        for s in range(state.n_parallel)
    )

    prompt = f"""PARALLEL START: Initialize {state.n_parallel} config variations for INR training.

Instructions (follow all instructions): {state.instruction_path}
Working memory: {state.memory_path}
Full log (append only): {state.analysis_path}
User input (read and acknowledge any pending instructions): {state.user_input_path}

Config files to edit (all {state.n_parallel}):
{slot_list}

Seeds (forced by pipeline — DO NOT modify seeds in configs):
{seed_info}

Read the instructions and the base config. All configs start identical.
Create {state.n_parallel} diverse initial INR parameter variations by editing ONLY the `inr:` section.
Vary parameters like: omega_inr, inr_learning_rate, hidden_dim_inr, n_layers_inr, inr_total_steps.
Do NOT change: inr_field_name, inr_type, inr_gradient_mode, or anything outside the `inr:` section.

Write the planned mutations to the working memory file."""

    print("\033[93mClaude start call...\033[0m")
    output_text = run_claude_cli(prompt, state.root_dir, max_turns=100)

    if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
        print("\n\033[91mOAuth token expired during start call\033[0m")
        sys.exit(1)

    if output_text.strip():
        with open(state.reasoning_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n=== BATCH 0 (start call) ===\n{'='*60}\n")
            f.write(output_text.strip())
            f.write("\n\n")


# ---------------------------------------------------------------------------
# Load configs + force seeds
# ---------------------------------------------------------------------------

def load_inr_configs_and_seeds(state: ExplorationState, batch: BatchInfo):
    """Load configs and force seeds for INR batch."""
    print(f"\n\033[93mPHASE 1: Loading configs for {batch.n_slots} INR slots\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = CellGNNConfig.from_yaml(state.config_paths[slot])
        if not config.dataset.startswith(state.pre_folder):
            config.dataset = state.pre_folder + config.dataset
        config.config_file = state.pre_folder + state.slot_names[slot]

        # Force seed
        inr_seed = iteration * 1000 + slot
        batch.slot_seeds[slot] = {'inr': inr_seed}

        # Write seed to YAML
        with open(state.config_paths[slot], 'r') as f:
            yaml_data = yaml.safe_load(f)
        if 'inr' not in yaml_data:
            yaml_data['inr'] = {}
        yaml_data['inr']['seed'] = inr_seed
        yaml_data['dataset'] = config.dataset
        with open(state.config_paths[slot], 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        batch.configs[slot] = config

        if state.device is None:
            state.device = set_device(config.training.device)

    seed_info = "\n".join(
        f"  Slot {s}: inr_seed={batch.slot_seeds[s]['inr']}"
        for s in range(batch.n_slots)
    )
    print(f"\033[90mSeeds (forced by pipeline):\n{seed_info}\033[0m")

    # UCB reset at block boundary
    if batch.batch_first > 1 and (batch.batch_first - 1) % state.n_iter_block == 0:
        if os.path.exists(state.ucb_path):
            os.remove(state.ucb_path)
            print(f"\033[93mblock boundary: deleted {state.ucb_path}\033[0m")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_inr_local_pipeline(state: ExplorationState, batch: BatchInfo):
    """Train INR models locally (sequential)."""
    from cell_gnn.models.inr_trainer import data_train_INR

    print(f"\n\033[93mPHASE 2: Training {batch.n_slots} INR models locally\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        field_name = config.inr.inr_field_name if config.inr else 'velocity'
        print(f"\033[90m  slot {slot} (iter {iteration}): training INR on '{field_name}'...\033[0m")

        try:
            model, loss_list = data_train_INR(
                config=config,
                device=state.device,
                field_name=field_name,
                run=0,
                erase=True,
            )

            inr_log_dir = log_path(config.config_file)
            results_path = f'{inr_log_dir}/tmp_training/inr/results.log'
            if os.path.exists(results_path):
                shutil.copy2(results_path, state.analysis_log_paths[slot])

            video_src = f'{inr_log_dir}/tmp_training/inr/{field_name}_gt_vs_pred.mp4'
            if os.path.exists(video_src):
                video_dir = f"{state.exploration_dir}/videos"
                os.makedirs(video_dir, exist_ok=True)
                video_dst = f"{video_dir}/iter_{iteration:03d}_slot_{slot:02d}.mp4"
                shutil.copy2(video_src, video_dst)

            batch.job_results[slot] = True
        except Exception as e:
            print(f"\033[91m  slot {slot}: INR training failed: {e}\033[0m")
            batch.job_results[slot] = False


# ---------------------------------------------------------------------------
# Save artifacts + UCB
# ---------------------------------------------------------------------------

def save_inr_artifacts(state: ExplorationState, batch: BatchInfo):
    """Save config snapshots for INR batch."""
    print("\n\033[93mPHASE 3: Saving config snapshots\033[0m")
    config_save_dir = f"{state.exploration_dir}/config"
    os.makedirs(config_save_dir, exist_ok=True)

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
        shutil.copy2(state.config_paths[slot], dst_config)


def update_inr_ucb_scores(state: ExplorationState, batch: BatchInfo):
    """Compute INR UCB scores."""
    print("\n\033[93mPHASE 4: Computing UCB scores\033[0m")

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

    tmp_analysis = state.analysis_path + '.tmp_ucb'
    with open(tmp_analysis, 'w') as f:
        f.write(existing_content + stub_entries)

    compute_inr_ucb_scores(tmp_analysis, state.ucb_path, c=ucb_c, block_size=state.n_iter_block)
    os.remove(tmp_analysis)
    print(f"\033[92mUCB scores computed (c={ucb_c}): {state.ucb_path}\033[0m")


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def run_inr_claude_analysis(state: ExplorationState, batch: BatchInfo):
    """Claude analyzes INR results + proposes next mutations."""
    print("\n\033[93mPHASE 5: Claude analysis + next mutations\033[0m")

    slot_info_lines = []
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        status = "COMPLETED" if batch.job_results.get(slot, False) else "FAILED"
        slot_info_lines.append(
            f"Slot {slot} (iteration {iteration}) [{status}]:\n"
            f"  Seed: inr_seed={batch.slot_seeds[slot]['inr']}\n"
            f"  Results: {state.analysis_log_paths[slot]}\n"
            f"  Config: {state.config_paths[slot]}"
        )
    slot_info = "\n\n".join(slot_info_lines)

    block_end_marker = "\n>>> BLOCK END <<<" if batch.is_block_end else ""

    prompt = f"""Batch iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}
Block info: block {batch.block_number}, iterations {batch.iter_in_block_first}-{batch.iter_in_block_last}/{state.n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {batch.n_slots} INR training results, then propose next {state.n_parallel} mutations.

Instructions (follow all instructions): {state.instruction_path}
Working memory: {state.memory_path}
Full log (append only): {state.analysis_path}
UCB scores: {state.ucb_path}
User input (read and acknowledge any pending instructions): {state.user_input_path}

{slot_info}

Seeds are forced by pipeline (DO NOT modify seeds in configs).

Analyze all {batch.n_slots} results. For each successful slot, read its results file and write a separate
iteration entry (## Iter N: ...) to the full log and memory file. Then edit all {state.n_parallel} config
files to set up the next batch of {state.n_parallel} experiments.

IMPORTANT: Only edit the `inr:` section in each config. Do NOT change inr_field_name, inr_type,
inr_gradient_mode, or anything outside the `inr:` section.
IMPORTANT: Training time target is {state.training_time_target_min} min per iteration.
IMPORTANT: Read user_input.md — if there are pending instructions, acknowledge them."""

    print("\033[93mClaude analysis...\033[0m")
    output_text = run_claude_cli(prompt, state.root_dir)

    if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
        print(f"\n\033[91mOAuth token expired at batch {batch.batch_first}-{batch.batch_last}\033[0m")
        print("\033[93mRe-run with --resume\033[0m")
        sys.exit(1)

    if output_text.strip():
        with open(state.reasoning_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n=== Batch {batch.batch_first}-{batch.batch_last} ===\n{'='*60}\n")
            f.write(output_text.strip())
            f.write("\n\n")


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def finalize_inr_batch(state: ExplorationState, batch: BatchInfo):
    """UCB recompute + memory snapshot."""
    with open(state.config_paths[0], 'r') as f:
        raw_config = yaml.safe_load(f)
    ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

    compute_inr_ucb_scores(state.analysis_path, state.ucb_path, c=ucb_c, block_size=state.n_iter_block)

    if batch.is_block_end:
        memory_save_dir = f"{state.exploration_dir}/memory"
        os.makedirs(memory_save_dir, exist_ok=True)
        dst_memory = f"{memory_save_dir}/block_{batch.block_number:03d}_memory.md"
        if os.path.exists(state.memory_path):
            shutil.copy2(state.memory_path, dst_memory)
            print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

    n_success = sum(1 for v in batch.job_results.values() if v)
    n_failed = sum(1 for v in batch.job_results.values() if not v)
    print(f"\n\033[92mBatch {batch.batch_first}-{batch.batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")
