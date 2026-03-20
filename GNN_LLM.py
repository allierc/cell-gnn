"""Cell-GNN — Parallel LLM Exploration Loop.

Orchestrates Claude-driven hyperparameter exploration with UCB-guided
mutation selection across parallel config slots.

Pipeline structure:
  setup -> batch_0 -> loop { load -> train -> artifacts -> UCB -> analysis -> finalize }
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import warnings

from cell_gnn.LLM import (
    setup_exploration,
    init_slot_configs,
    init_shared_files,
    make_batch_info,
    run_batch_0,
    load_configs_and_seeds,
    generate_data_locally,
    run_cluster_training,
    run_local_test_plot,
    run_local_pipeline,
    save_artifacts,
    update_ucb_scores,
    run_claude_analysis,
    finalize_batch,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def parse_args():
    parser = argparse.ArgumentParser(description="Cell-GNN — Parallel LLM Loop")
    parser.add_argument("-o", "--option", nargs="+", help="option that takes multiple values")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="start from iteration 1 (ignore auto-resume)")
    parser.add_argument("--resume", action="store_true",
                        help="auto-resume from last completed batch")
    parser.add_argument("--cluster", action="store_true",
                        help="submit training to LSF cluster (default: run locally)")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Setup ---
    state = setup_exploration(args, root_dir)
    init_slot_configs(state, is_resume=args.resume)
    init_shared_files(state, is_resume=args.resume)

    # --- Batch 0: initialize config variations (fresh start only) ---
    if state.start_iteration == 1 and not args.resume:
        run_batch_0(state)

    # --- Main batch loop ---
    for batch_start in range(state.start_iteration, state.n_iterations + 1, state.n_parallel):
        batch = make_batch_info(state, batch_start)

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}  (block {batch.block_number})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # Load configs + force seeds
        load_configs_and_seeds(state, batch)

        # Training (cluster or local)
        if "train" in state.task:
            if state.cluster_enabled:
                if state.generate_data:
                    generate_data_locally(state, batch)
                run_cluster_training(state, batch)
                run_local_test_plot(state, batch)
            else:
                run_local_pipeline(state, batch)
        else:
            for slot in range(batch.n_slots):
                batch.job_results[slot] = True

        # Save exploration artifacts
        save_artifacts(state, batch)

        # Compute UCB scores
        update_ucb_scores(state, batch)

        # Claude analysis + next mutations
        run_claude_analysis(state, batch)

        # Finalize: tree viz, protocol/memory snapshots
        finalize_batch(state, batch)


# python GNN_LLM.py -o train_test_Claude dicty iterations=48
# python GNN_LLM.py -o train_test_Claude dicty iterations=48 --resume
# python GNN_LLM.py -o train_test_Claude_cluster dicty iterations=48 --cluster
# python GNN_LLM.py -o generate_train_test_Claude dicty_spring_force iterations=48
