from enum import Enum
from typing import Optional, Annotated
import yaml
from pydantic import BaseModel, ConfigDict, Field


# Python 3.10 compatibility (StrEnum added in 3.11)
class StrEnum(str, Enum):
    pass


# StrEnum types for config fields

class Boundary(StrEnum):
    PERIODIC = "periodic"
    NO = "no"
    PERIODIC_SPECIAL = "periodic_special"
    WALL = "wall"

class StateType(StrEnum):
    DISCRETE = "discrete"
    SEQUENCE = "sequence"
    CONTINUOUS = "continuous"

class ConnectivityType(StrEnum):
    NONE = "none"
    DISTANCE = "distance"
    VORONOI = "voronoi"
    K_NEAREST = "k_nearest"

class Prediction(StrEnum):
    FIRST_DERIVATIVE = "first_derivative"
    SECOND_DERIVATIVE = "2nd_derivative"

class Integration(StrEnum):
    EULER = "Euler"
    RUNGE_KUTTA = "Runge-Kutta"

class UpdateType(StrEnum):
    LINEAR = "linear"
    MLP = "mlp"
    PRE_MLP = "pre_mlp"
    TWO_STEPS = "2steps"
    NONE = "none"
    NO_POS = "no_pos"
    GENERIC = "generic"
    EXCITATION = "excitation"
    GENERIC_EXCITATION = "generic_excitation"
    EMBEDDING_MLP = "embedding_MLP"
    TEST_FIELD = "test_field"

class GhostMethod(StrEnum):
    NONE = "none"
    TENSOR = "tensor"
    MLP = "MLP"

class Sparsity(StrEnum):
    NONE = "none"
    REPLACE_EMBEDDING = "replace_embedding"
    REPLACE_EMBEDDING_FUNCTION = "replace_embedding_function"
    REPLACE_STATE = "replace_state"
    REPLACE_TRACK = "replace_track"

class ClusterMethod(StrEnum):
    KMEANS = "kmeans"
    KMEANS_AUTO_PLOT = "kmeans_auto_plot"
    KMEANS_AUTO_EMBEDDING = "kmeans_auto_embedding"
    DISTANCE_PLOT = "distance_plot"
    DISTANCE_EMBEDDING = "distance_embedding"
    DISTANCE_BOTH = "distance_both"
    INCONSISTENT_PLOT = "inconsistent_plot"
    INCONSISTENT_EMBEDDING = "inconsistent_embedding"
    NONE = "none"

class ClusterConnectivity(StrEnum):
    SINGLE = "single"
    AVERAGE = "average"


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

    boundary: Boundary = Boundary.PERIODIC
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
    angular_bernoulli: list[float] = [-1]

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
    connectivity_type: ConnectivityType = ConnectivityType.DISTANCE
    connectivity_parameter: float = 1.0
    connectivity_distribution: str = "Gaussian"
    connectivity_distribution_params: float = 1

    excitation_value_map: Optional[str] = None
    excitation: str = "none"

    particle_params: list[list[float]] = None
    params_mesh: list[list[float]] = None
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

    non_discrete_level: float = 0

    state_type: StateType = StateType.DISCRETE
    state_params: list[float] = [-1]


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    particle_model_name: str = ""
    prediction: Prediction = Prediction.SECOND_DERIVATIVE
    integration: Integration = Integration.EULER

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

    update_type: UpdateType = UpdateType.NONE

    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1
    init_update_gradient: bool = False

    input_size_nnr: int = 3
    n_layers_nnr: int = 5
    hidden_dim_nnr: int = 128
    output_size_nnr: int = 1
    omega: float = 80.0

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
    data_embedding: int = 0


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
    ghost_method: GhostMethod = GhostMethod.NONE
    ghost_logvar: float = -12

    sparsity_freq: int = 5
    sparsity: Sparsity = Sparsity.NONE
    fix_cluster_embedding: bool = False
    cluster_method: ClusterMethod = ClusterMethod.DISTANCE_PLOT
    cluster_distance_threshold: float = 0.01
    cluster_connectivity: ClusterConnectivity = ClusterConnectivity.SINGLE

    learning_rate_start: float = 0.001
    learning_rate_embedding_start: float = 0.001
    learning_rate_update_start: float = 0.0

    learning_rate_end: float = 0.0005
    learning_rate_embedding_end: float = 0.0001
    learning_rate_nnr: float = 0.0001

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
    do_tracking: bool = False
    coeff_model_a: float = 0
    coeff_continuous: float = 0

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
