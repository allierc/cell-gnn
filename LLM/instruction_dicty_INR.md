# Cell-GNN INR Training Landscape Study

## Goal

Find optimal SIREN INR configuration for learning the **velocity field** of 3D dicty cell dynamics (1000 cells, 9997 frames, 3 components, dimension=3).

Primary metric: `final_r2` (higher is better, target R2 > 0.99)
Secondary metrics: `final_mse`, `slope` (target slope ~1.0)

**Training time constraint**: Each run should complete in **less than 60 minutes**. Prefer configurations that achieve good R2 within this budget. Runs on A100 GPU.

## INR Model

SIREN (Sinusoidal Implicit Neural Representation) maps `(t, x, y, z) -> field_value(3)`:
```
f(t/n_frames/t_period, pos/xy_period) -> velocity(3 components)
```

- **Data**: 9997 frames, 1000 cells, 3 velocity components, 3D positions
- **Input**: 4D (t_normalized, x, y, z) — time pre-normalized to [0,1] by n_frames
- **Output**: 3D velocity vector
- **Architecture**: SIREN with sin(omega * Wx + b) activations

### Key from MPM-pytorch Knowledge

Prior LLM-in-the-loop exploration on MPM data (400-1000 frames, 2D) established:
- **omega_f**: Field-dependent. Low-complexity fields: 2.5-10. Multi-component: 10-15. Decreases with more frames.
- **n_layers**: 3 is optimal for almost all fields (4 only for F deformation gradient). 5 degrades.
- **hidden_dim**: 256-1280 depending on field complexity and frame count.
- **lr**: Scales UP with more data (more frames = higher lr safe). Typical: 1E-5 to 3E-4.
- **batch_size**: Always 1 for MPM (but cell-gnn uses 8 by default — test this).
- **Steps/frame**: Decreases with more data. At 1000 frames: 500-1000 steps/frame.
- **Period parameters**: Keep at 1.0 (changing causes catastrophic degradation in MPM).
- **LayerNorm/BatchNorm**: INCOMPATIBLE with SIREN. Never use.

This dataset has 10,000 frames (10x more than MPM explored), so:
- omega_f may need to be lower (expect 5-15 range)
- lr can be higher (expect 1E-4+ range)
- Steps/frame can be lower (expect 50-500 steps/frame)
- 4D input (vs 3D in MPM) may need more capacity

## Metrics from results.log

| Metric | Better | Description |
|--------|--------|-------------|
| `final_r2` | Higher | R2 score on full dataset (PRIMARY) |
| `final_mse` | Lower | Mean squared error on full dataset |
| `slope` | ~1.0 | Regression slope pred vs gt (1.0 = unbiased) |
| `training_time_min` | Lower | Training duration in minutes |

## Classification

- **Excellent**: R2 > 0.995
- **Good**: R2 0.990 - 0.995
- **Moderate**: R2 0.90 - 0.99
- **Poor**: R2 < 0.90 OR training_time_min > 60

## Explorable INR Parameters

| Parameter | YAML path | Default | Description | Typical range |
|-----------|-----------|---------|-------------|---------------|
| `omega_inr` | inr.omega_inr | 10.0 | SIREN frequency parameter | [1.0, 100.0] |
| `inr_learning_rate` | inr.inr_learning_rate | 1E-4 | Learning rate | [1E-6, 1E-3] |
| `hidden_dim_inr` | inr.hidden_dim_inr | 384 | Hidden layer width | [128, 1024] |
| `n_layers_inr` | inr.n_layers_inr | 3 | Number of hidden layers | [2, 5] |
| `inr_total_steps` | inr.inr_total_steps | 500000 | Total training steps | [50000, 2000000] |
| `inr_batch_size` | inr.inr_batch_size | 8 | Frames per batch | [1, 16] |
| `inr_xy_period` | inr.inr_xy_period | 1.0 | Spatial coordinate scaling | [0.5, 10.0] |
| `inr_t_period` | inr.inr_t_period | 1.0 | Time coordinate scaling | [0.5, 10.0] |

**DO NOT change**: `inr_field_name`, `inr_type`, `inr_gradient_mode`, or anything outside the `inr:` section.

**Constraint**: `inr_total_steps` should be chosen so training_time < 60 min. Scale inversely with hidden_dim.

## Seeds

Seeds are **controlled by the pipeline** (GNN_LLM_INR.py). Do NOT modify seeds in configs — the pipeline forces `inr.seed` before each batch. The seed values are shown in the slot info and should be logged in iteration entries.

## Iteration Workflow

For each iteration, follow these 5 steps IN ORDER:

### Step 1: Read Working Memory
Read the memory file to recall established principles, previous block summaries, and current hypotheses.

### Step 2: Analyze Results
Read the analysis.log for the current iteration's metrics. Compare with UCB scores and previous results.

### Step 3: Write to Experiment Log
Append an entry to the analysis file in this format:
```
## Iter N: [excellent|good|moderate|poor]
Node: id=N, parent=P
Metrics: final_r2=X, final_mse=X, slope=X, training_time_min=X
Config: omega_inr=X, inr_learning_rate=X, hidden_dim_inr=X, n_layers_inr=X, inr_total_steps=X, inr_batch_size=X
Mutation: [description of what was changed]
Observation: [brief analysis of results]
Next: parent=P
```

### Step 4: Select Parent via UCB
Read UCB scores. Choose the node with highest UCB score as parent for next iteration.

### Step 5: Propose Mutation
Edit the config YAML file to set up the next experiment:
- Change 1-2 parameters at a time (small, controlled mutations)
- Build on successful configurations (exploit) or try new regions (explore)
- Record what you changed in the mutation description

**Strategy guidelines:**

| Condition | Strategy | Action |
|-----------|----------|--------|
| Default | **exploit** | Highest UCB node, try mutation |
| 3+ consecutive R2 >= 0.95 | **failure-probe** | Extreme parameter to find boundary |
| Same param mutated 4+ times | **switch-param** | Mutate different parameter |

**Mutation magnitude:**
- Learning rates: x2 or /2
- Network size: +128 or -128
- Training steps: x1.5 or x2
- omega_inr: +5 or -5
- batch_size: x2 or /2

## Update Working Memory

At block boundaries (every n_iter_block iterations):
- Summarize findings from the current block
- Update the "Established Principles" section
- Update the "Regime Comparison Table"
- Formulate hypotheses for the next block

## Memory File Structure

```markdown
# Working Memory: dicty INR (velocity field)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | omega_inr | lr | hidden_dim | n_layers | steps | batch | Best R2 | slope | time_min | Key finding |
|-------|-----------|-----|------------|----------|-------|-------|---------|-------|----------|-------------|

### Established Principles
[Confirmed patterns that apply across regimes]

### Open Questions
[Patterns needing more testing, contradictions]

---

## Previous Block Summary

---

## Current Block
### Block Info
### Hypothesis
### Iterations This Block
### Emerging Observations
```
