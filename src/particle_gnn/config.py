from typing import Optional, Literal, Annotated
import yaml
from pydantic import BaseModel, ConfigDict, Field

# Sub-config schemas for particle-gnn


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    dimension: int = 2
    n_frames: int = 1000
    start_frame: int = 0
    seed: int = 42

    model_id: str = "000"
    ensemble_id: str = "0000"

    sub_sampling: int = 1
    delta_t: float = 1

    boundary: Literal["periodic", "no", "periodic_special", "wall"] = "periodic"
    bounce: bool = False
    bounce_coeff: float = 0.1
    min_radius: float = 0.0
    max_radius: float = 0.1

    n_particles: int = 1000
    n_neurons: int = 1000
    n_input_neurons: int = 0
    n_particles_max: int = 20000
    n_edges: int = 0
    max_edges: float = 1.0e6
    n_extra_null_edges: int = 0
    n_particle_types: int = 5
    n_neuron_types: int = 5
    baseline_value: float = -999.0
    n_particle_type_distribution: list[int] = [0]
    shuffle_particle_types: bool = False
    pos_init: str = "uniform"
    dpos_init: float = 0
    len_directed_edges: int = 1

    diffusion_coefficients: list[list[float]] = None

    angular_sigma: float = 0
    angular_Bernouilli: list[float] = [-1]

    noise_visual_input: float = 0.0
    only_noise_visual_input: float = 0.0
    visual_input_type: str = ""
    blank_freq: int = 2
    simulation_initial_state: bool = False

    n_grid: int = 128

    n_nodes: Optional[int] = None
    n_node_types: Optional[int] = None
    node_coeff_map: Optional[str] = None
    node_value_map: Optional[str] = "input_data/pattern_Null.tif"
    node_proliferation_map: Optional[str] = None

    adjacency_matrix: str = ""

    short_term_plasticity_mode: str = "depression"

    connectivity_file: str = ""
    connectivity_init: list[float] = [-1]
    connectivity_filling_factor: float = 1
    connectivity_type: Literal["none", "distance", "voronoi", "k_nearest"] = "distance"
    connectivity_parameter: float = 1.0
    connectivity_distribution: str = "Gaussian"
    connectivity_distribution_params: float = 1

    excitation_value_map: Optional[str] = None
    excitation: str = "none"

    params: list[list[float]] = None
    params_mesh: list[list[float]] = None
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

    state_type: Literal["discrete", "sequence", "continuous"] = "discrete"
    state_params: list[float] = [-1]


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    particle_model_name: str = ""
    prediction: Literal["first_derivative", "2nd_derivative"] = "2nd_derivative"
    integration: Literal["Euler", "Runge-Kutta"] = "Euler"

    field_type: str = ""
    field_grid: Optional[str] = ""

    input_size: int = 1
    output_size: int = 1
    hidden_dim: int = 1
    n_layers: int = 1

    lin_edge_positive: bool = False

    aggr_type: str

    embedding_dim: int = 2
    embedding_init: str = ""

    update_type: Literal[
        "linear",
        "mlp",
        "pre_mlp",
        "2steps",
        "none",
        "no_pos",
        "generic",
        "excitation",
        "generic_excitation",
        "embedding_MLP",
        "test_field",
    ] = "none"

    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1
    init_update_gradient: bool = False

    kernel_type: str = "mlp"


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    colormap: str = "tab10"
    arrow_length: int = 10
    marker_size: int = 100
    xlim: list[float] = [-0.1, 0.1]
    ylim: list[float] = [-0.1, 0.1]
    embedding_lim: list[float] = [-40, 40]
    speedlim: list[float] = [0, 1]
    pic_folder: str = "none"
    pic_format: str = "jpg"
    pic_size: list[int] = [1000, 1100]
    data_embedding: int = 1


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda:\d+)$")] = "auto"

    n_epochs: int = 20
    n_epochs_init: int = 99999
    epoch_reset: int = -1
    epoch_reset_freq: int = 99999
    batch_size: int = 1
    batch_ratio: float = 1
    small_init_batch_size: bool = True
    embedding_step: int = 1000
    shared_embedding: bool = False
    embedding_trial: bool = False
    remove_self: bool = True

    pretrained_model: str = ""

    time_window: int = 0

    n_runs: int = 2
    seed: int = 42
    clamp: float = 0
    pred_limit: float = 1.0e10

    particle_dropout: float = 0
    n_ghosts: int = 0
    ghost_method: Literal["none", "tensor", "MLP"] = "none"
    ghost_logvar: float = -12

    sparsity_freq: int = 5
    sparsity: Literal[
        "none",
        "replace_embedding",
        "replace_embedding_function",
        "replace_state",
        "replace_track",
    ] = "none"
    fix_cluster_embedding: bool = False
    cluster_method: Literal[
        "kmeans",
        "kmeans_auto_plot",
        "kmeans_auto_embedding",
        "distance_plot",
        "distance_embedding",
        "distance_both",
        "inconsistent_plot",
        "inconsistent_embedding",
        "none",
    ] = "distance_plot"
    cluster_distance_threshold: float = 0.01
    cluster_connectivity: Literal["single", "average"] = "single"

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    learning_rate_update_start: float = 0.0

    learning_rate_end: float = 0.0005
    learning_rate_embedding_end: float = 0.0001

    coeff_loss1: float = 1

    noise_level: float = 0
    measurement_noise_level: float = 0
    noise_model_level: float = 0
    loss_noise_level: float = 0.0

    rotation_augmentation: bool = False
    translation_augmentation: bool = False
    reflection_augmentation: bool = False
    velocity_augmentation: bool = False
    data_augmentation_loop: int = 40

    recursive_training: bool = False
    recursive_training_start_epoch: int = 0
    recursive_loop: int = 0
    coeff_loop: list[float] = [2, 4, 8, 16, 32, 64]
    time_step: int = 1
    recursive_sequence: str = ""
    recursive_parameters: list[float] = [0, 0]


# Main config schema for particle-gnn


class ParticleGNNConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    description: Optional[str] = "ParticleGNN"
    dataset: str
    data_folder_name: str = "none"
    connectome_folder_name: str = "none"
    data_folder_mesh_name: str = "none"
    config_file: str = "none"
    simulation: SimulationConfig
    graph_model: GraphModelConfig
    plotting: PlottingConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return ParticleGNNConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == "__main__":
    config_file = "../../config/arbitrary_3.yaml"
    config = ParticleGNNConfig.from_yaml(config_file)
    print(config.pretty())

    print("Successfully loaded config file. Model description:", config.description)
