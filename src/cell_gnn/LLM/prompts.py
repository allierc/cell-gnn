"""LLM prompt templates for the cell-gnn exploration pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import BatchInfo, ExplorationState


def batch_0_prompt(state: 'ExplorationState', slot_list: str, seed_info: str) -> str:
    """Build the BATCH 0 start prompt for initializing N config variations."""
    return f"""PARALLEL START: Initialize {state.n_parallel} config variations for the first batch.

Instructions (follow all instructions): {state.instruction_path}
Working memory: {state.memory_path}
Full log (append only): {state.analysis_path}
User input (read and acknowledge any pending instructions): {state.user_input_path}

Config files to edit (all {state.n_parallel}):
{slot_list}

Seeds (forced by pipeline — DO NOT modify simulation.seed or training.seed in configs):
{seed_info}
Log these seed values in your iteration entries.

Read the instructions and the base config, then create {state.n_parallel} diverse initial training
parameter variations. Each config already has a unique dataset name — do NOT change the
dataset field. Vary training parameters (e.g. learning_rate_start, learning_rate_embedding_start,
batch_size, hidden_dim, n_layers) across the {state.n_parallel} slots to explore different starting points.

IMPORTANT: Training time target is {state.training_time_target_min} min per iteration — keep configurations within this budget.
IMPORTANT: Read user_input.md — if there are pending instructions, acknowledge them by appending to the "Acknowledged" section with timestamp and moving them out of "Pending Instructions".

Write the planned mutations to the working memory file."""


def analysis_prompt(state: 'ExplorationState', batch: 'BatchInfo',
                    slot_info: str) -> str:
    """Build the PHASE 6 Claude analysis prompt."""
    block_end_marker = "\n>>> BLOCK END <<<" if batch.is_block_end else ""

    return f"""Batch iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}
Block info: block {batch.block_number}, iterations {batch.iter_in_block_first}-{batch.iter_in_block_last}/{state.n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {batch.n_slots} results, then propose next {state.n_parallel} mutations.

Instructions (follow all instructions): {state.instruction_path}
Working memory: {state.memory_path}
Full log (append only): {state.analysis_path}
UCB scores: {state.ucb_path}
User input (read and acknowledge any pending instructions): {state.user_input_path}

{slot_info}

Seeds are forced by pipeline (DO NOT modify simulation.seed or training.seed in configs).
The seed values for this batch are shown in each slot above. Log them in your iteration entries.

Analyze all {batch.n_slots} results. For each successful slot, write a separate iteration entry
(## Iter N: ...) to the full log and memory file. Then edit all {state.n_parallel} config files
to set up the next batch of {state.n_parallel} experiments.

IMPORTANT: Do NOT change the 'dataset' field in any config — it must stay as-is for each slot.
IMPORTANT: Training time target is {state.training_time_target_min} min per iteration. Check training_time_min in the metrics and flag any slot that exceeds this limit.
IMPORTANT: Read user_input.md — if there are pending instructions, acknowledge them by appending to the "Acknowledged" section with a timestamp and moving them out of "Pending Instructions".
"""
