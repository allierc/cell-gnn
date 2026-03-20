"""Cell-GNN — Parallel LLM INR Exploration Loop.

Orchestrates Claude-driven SIREN INR hyperparameter exploration with
UCB-guided mutation selection across parallel config slots.

Pipeline structure:
  setup -> batch_0 -> loop { load -> train -> artifacts -> UCB -> analysis -> finalize }
"""

import matplotlib
matplotlib.use('Agg')
import argparse
import os
import warnings

from cell_gnn.LLM.inr_pipeline import (
    setup_inr_exploration,
    init_inr_slot_configs,
    init_inr_shared_files,
    make_inr_batch_info,
    run_inr_batch_0,
    load_inr_configs_and_seeds,
    run_inr_local_pipeline,
    save_inr_artifacts,
    update_inr_ucb_scores,
    run_inr_claude_analysis,
    finalize_inr_batch,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def parse_args():
    parser = argparse.ArgumentParser(description="Cell-GNN — Parallel LLM INR Loop")
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
    state = setup_inr_exploration(args, root_dir)
    init_inr_slot_configs(state, is_resume=args.resume)
    init_inr_shared_files(state, is_resume=args.resume)

    # --- Batch 0: initialize config variations (fresh start only) ---
    if state.start_iteration == 1 and not args.resume:
        run_inr_batch_0(state)

    # --- Main batch loop ---
    for batch_start in range(state.start_iteration, state.n_iterations + 1, state.n_parallel):
        batch = make_inr_batch_info(state, batch_start)

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}  (block {batch.block_number})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # Load configs + force seeds
        load_inr_configs_and_seeds(state, batch)

        # Training (local only for now)
        run_inr_local_pipeline(state, batch)

        # Save config snapshots
        save_inr_artifacts(state, batch)

        # Compute UCB scores
        update_inr_ucb_scores(state, batch)

        # Claude analysis + next mutations
        run_inr_claude_analysis(state, batch)

        # Finalize: UCB recompute, memory snapshots
        finalize_inr_batch(state, batch)


# python GNN_LLM_INR.py -o train_inr_Claude dicty iterations=48
# python GNN_LLM_INR.py -o train_inr_Claude dicty iterations=48 --resume
