from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExplorationState:
    """All shared state for the LLM exploration pipeline."""

    # Directories
    root_dir: str = ""
    config_root: str = ""
    llm_dir: str = ""
    exploration_dir: str = ""
    source_config: str = ""
    base_config_name: str = ""
    pre_folder: str = ""

    # Claude settings (from YAML)
    n_epochs: int = 1
    data_augmentation_loop: int = 100
    n_iter_block: int = 24
    ucb_c: float = 1.414
    node_name: str = "a100"
    n_parallel: int = 4
    generate_data: bool = False
    training_time_target_min: int = 60
    interaction_code: bool = False

    # Runtime mode
    cluster_enabled: bool = False
    start_iteration: int = 1
    n_iterations: int = 48

    # Slot paths (keyed by slot index 0..N-1)
    config_paths: dict = field(default_factory=dict)
    analysis_log_paths: dict = field(default_factory=dict)
    slot_names: dict = field(default_factory=dict)

    # Shared file paths
    analysis_path: str = ""
    memory_path: str = ""
    ucb_path: str = ""
    instruction_path: str = ""
    reasoning_log_path: str = ""
    user_input_path: str = ""
    log_dir: str = ""

    # Computed once
    task: str = ""
    llm_task_name: str = ""
    config_file: str = ""

    # Runtime (set during execution)
    device: Any = None
    best_model: Any = None


@dataclass
class BatchInfo:
    """Per-batch computed info."""

    iterations: list = field(default_factory=list)
    batch_first: int = 0
    batch_last: int = 0
    n_slots: int = 0
    block_number: int = 0
    iter_in_block_first: int = 0
    iter_in_block_last: int = 0
    is_block_start: bool = False
    is_block_end: bool = False

    # Populated during execution
    configs: dict = field(default_factory=dict)
    slot_seeds: dict = field(default_factory=dict)
    job_results: dict = field(default_factory=dict)
    activity_paths: dict = field(default_factory=dict)
