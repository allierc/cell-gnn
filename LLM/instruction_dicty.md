# Cell-GNN Optimization: Dicty 3D Pair Forces

## Goal

Find a GNN training configuration for 3D cell dynamics (dicty — 1000 cells, 1 type, pair forces) that **minimizes rollout RMSE** over the first 2000 steps.

Primary metric: `rollout_RMSE_mean` (lower is better)

**Training time constraint**: Training should complete in **less than 60 minutes** if possible. Prefer configurations that achieve good RMSE within this budget.

**Note on psi_R2**: The `training_psi_R2` metric is **not applicable** for dicty because there are no ground truth interaction functions to compare against. Do not use psi_R2 for evaluating or classifying runs.

## Cell-GNN Model

The Cell-GNN learns pairwise interaction functions from cell trajectories:
```
acceleration_i = aggr_j(lin_edge(delta_pos_ij / max_r, r / max_r, a_i)) * ynorm
```

- **Simulation**: 1000 cells, 1 cell type, 10000 frames, delta_t=0.002
- **Spatial**: 3D, periodic boundary, max_radius=0.2
- **Physics**: Arbitrary pair force function (Gaussian repulsion/attraction)
- **GNN**: Embedding per cell (dim=2), MLP0 (lin_edge) learns interaction, mean aggregation

### PARAMS_DOC Reference
See `CellGNN.PARAMS_DOC` in `src/cell_gnn/models/cell_gnn.py` for detailed model documentation.

## Metrics from analysis.log

| Metric | Source | Better | Description |
|--------|--------|--------|-------------|
| `rollout_RMSE_mean` | test results | Lower | Mean position RMSE across rollout steps (PRIMARY) |
| `rollout_RMSE_final` | test results | Lower | RMSE at final rollout step |
| `training_psi_R2` | training log | N/A | **Not applicable** — no ground truth interaction function for dicty |
| `training_accuracy` | training log | Higher | Clustering accuracy of learned embeddings |
| `training_final_loss` | training log | Lower | Final epoch training loss |
| `training_time_min` | training log | Lower | Training duration in minutes (target: < 60 min) |

## Classification

- **Converged**: rollout_RMSE_mean < 0.05
- **Partial**: rollout_RMSE_mean in [0.05, 0.2]
- **Failed**: rollout_RMSE_mean > 0.2 OR training diverged OR training_time_min > 60

## Explorable Training Parameters

| Parameter | YAML path | Default | Description | Typical range |
|-----------|-----------|---------|-------------|---------------|
| `learning_rate_start` | training.learning_rate_start | 1E-4 | LR for MLP parameters | [1E-5, 1E-3] |
| `learning_rate_embedding_start` | training.learning_rate_embedding_start | 1E-5 | LR for cell embeddings | [1E-6, 1E-4] |
| `batch_size` | training.batch_size | 8 | Frames per gradient step | [1, 16] |
| `data_augmentation_loop` | training.data_augmentation_loop | 100 | Data augmentation multiplier | [10, 200] |
| `hidden_dim` | graph_model.hidden_dim | 128 | MLP0 hidden layer width | [64, 256] |
| `n_layers` | graph_model.n_layers | 5 | MLP0 depth | [3, 7] |
| `embedding_dim` | graph_model.embedding_dim | 2 | Cell embedding dimension | [1, 8] |
| `aggr_type` | graph_model.aggr_type | mean | Message aggregation: mean, add, max | - |
| `coeff_edge_diff` | training.coeff_edge_diff | 0 | Edge similarity regularizer | [0, 100] |
| `coeff_edge_norm` | training.coeff_edge_norm | 0 | Monotonicity regularizer | [0, 10] |
| `max_radius` | simulation.max_radius | 0.2 | Interaction radius | [0.1, 0.5] |
| `rotation_augmentation` | training.rotation_augmentation | True | SO(3) rotation augmentation | - |

## Recurrent Training (Key for Rollout RMSE)

When `recursive_training: True` and `recursive_loop > 0`, the model unrolls multiple steps during training, directly optimizing multi-step prediction stability. This is the **key lever** for reducing rollout RMSE.

| Parameter | YAML path | Default | Description |
|-----------|-----------|---------|-------------|
| `recursive_training` | training.recursive_training | False | Enable recurrent training |
| `recursive_training_start_epoch` | training.recursive_training_start_epoch | 0 | Epoch to start recurrent training |
| `recursive_loop` | training.recursive_loop | 0 | Number of unroll steps (2-8 recommended) |

**Strategy**:
1. First train with standard loss (recursive_training=False) to learn pairwise forces
2. Then enable recurrent training (recursive_loop=2-8) to improve rollout stability
3. Can combine: start standard, switch to recurrent mid-training via recursive_training_start_epoch

## Code Modifications

You may modify code within `LLM-MODIFIABLE` markers in `src/cell_gnn/models/graph_trainer.py`:

- **OPTIMIZER SETUP** (~4 lines): Change optimizer type, learning rate schedule, parameter groups
- **TRAINING LOOP** (~200 lines): Change loss function, gradient clipping, data sampling, LR scheduling, batch size schedule, early stopping

**Rules**:
- Do NOT change: function signature, model construction, data loading, return values
- Do NOT modify code outside the LLM-MODIFIABLE markers
- If a code change causes a crash, it will be auto-repaired (up to 3 attempts)

## Iteration Workflow

For each iteration, follow these 5 steps IN ORDER:

### Step 1: Read Working Memory
Read the memory file to recall established principles, previous block summaries, and current hypotheses.

### Step 2: Analyze Results
Read the analysis.log for the current iteration's metrics. Compare with UCB scores and previous results.

### Step 3: Write to Experiment Log
Append an entry to the analysis file in this format:
```
## Iter N: [converged|partial|failed]
Node: id=N, parent=P
Metrics: rollout_RMSE_mean=X, training_accuracy=X, training_final_loss=X, training_time_min=X
Mutation: [description of what was changed]
Observation: [brief analysis of results]
Next: parent=P
```

### Step 4: Select Parent via UCB
Read UCB scores. Choose the node with highest UCB score as parent for next iteration.
- High UCB = good performance OR under-explored
- The `Next: parent=P` line in Step 3 determines the parent for the NEXT iteration

### Step 5: Propose Mutation
Edit the config YAML file to set up the next experiment:
- Change 1-2 parameters at a time (small, controlled mutations)
- Build on successful configurations (exploit) or try new regions (explore)
- Record what you changed in the mutation description

## Update Working Memory

At block boundaries (every n_iter_block iterations):
- Summarize findings from the current block
- Update the "Established Principles" section
- Update the "Regime Comparison Table"
- Formulate hypotheses for the next block

## Memory File Structure

```markdown
# Working Memory

## Knowledge Base (accumulated across all blocks)
### Regime Comparison Table
| Block | Best RMSE | Best time_min | Key finding |
### Established Principles
### Open Questions

## Previous Block Summary

## Current Block
### Block Info
### Hypothesis
### Iterations This Block
### Emerging Observations
```
