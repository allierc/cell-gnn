# Cell-GNN Exploration: Robust g_phi Recovery on Dicty Spring Force (3D, 2 Types)

## Goal

Find GNN training configurations that **robustly recover the pairwise interaction function g_phi** for 3D overdamped cells with spring repulsion + sigmoid-gated adhesion, across different random seeds.

**ROBUSTNESS IS THE PRIMARY OBJECTIVE.** A config with mean g_phi_R2=0.92 and std=0.05 is BETTER than a config with mean=0.98 and std=0.4.

Primary metric: **training_g_phi_R2** (R² between learned and true g_phi, higher is better)
**Stability metric: training_g_phi_R2_std — target std < 0.1.**
Tertiary metric: **rollout_RMSE_mean** (lower is better)

## Scientific Method

This exploration follows a strict **hypothesize → test → validate/falsify** cycle:

1. **Hypothesize**: Based on available data (metrics, seed variance, prior results), form a specific testable prediction
2. **Design experiment**: Choose a mutation that tests the hypothesis — change ONE parameter at a time
3. **Run training**: 4 slots with different seeds — you cannot predict the outcome
4. **Analyze results**: Use both metrics AND cross-seed variance to evaluate
5. **Update understanding**: Revise hypotheses based on evidence. A falsified hypothesis is valuable information.

**CRITICAL**: You can only hypothesize. Only training results can validate or falsify. Never assume a hypothesis is correct without experimental evidence.

**Evidence hierarchy:**

| Level            | Criterion                                       | Action                       |
| ---------------- | ----------------------------------------------- | ---------------------------- |
| **Established**  | Consistent across 3+ iterations AND 4/4 seeds   | Add to Principles            |
| **Tentative**    | Observed 1-2 times or inconsistent across seeds | Add to Open Questions        |
| **Contradicted** | Conflicting evidence across iterations/seeds    | Note in Falsified Hypotheses |

## Physics

4800 cells of **2 types** in a 3D periodic box. Each cell type has a distinct spring-based interaction:

```
F_rep = k_rep * relu(r0 - r)                           (linear repulsion for r < r0)
g_on  = sigmoid((r - r0) / delta)
g_off = sigmoid(-(r - r_on) / delta)
F_adh = -kadh * g_on * g_off * (r - r0)                (sigmoid-gated adhesion)
F     = -mu_f * (F_rep + F_adh) * rhat
```

Parameters per type: `(k_rep, r0, kadh, r_on, delta, mu_f)`

- Type 0: `[50.0, 0.1, 50.0, 0.14, 0.001, 0.05]` — strong repulsion + adhesion, larger equilibrium
- Type 1: `[30.0, 0.08, 80.0, 0.12, 0.002, 0.03]` — weaker repulsion, stronger adhesion, smaller range

The `cell_params` are FIXED — do not change them.

## Cell-GNN Model

```
velocity_i = aggr_j(g_phi(delta_pos_ij / max_r, r / max_r, a_i)) * ynorm
```

- **g_phi**: Edge message MLP — learns the pairwise interaction function
- **f_theta**: Node update MLP (if update_type='mlp')
- **a**: Learned per-cell embedding encoding cell type
- **prediction**: `first_derivative` (velocity, not acceleration)

See `CellGNN.PARAMS_DOC` in `src/cell_gnn/models/cell_gnn.py` for details.

## Seeds

Seeds are **controlled by the pipeline**. Do NOT modify `simulation.seed` or `training.seed` in configs.

- `simulation.seed = iteration * 1000 + slot` (controls data generation)
- `training.seed = iteration * 1000 + slot + 500` (controls weight init & training randomness)

Log seed values in iteration entries.

## CRITICAL: Data is RE-GENERATED per slot

Each slot re-generates data with a different seed. You MAY vary these simulation parameters:

| Parameter | YAML path          | Default | Explorable range |
| --------- | ------------------ | ------- | ---------------- |
| `delta_t` | simulation.delta_t | 0.00025 | [0.0001, 0.0025] |

**FIXED — DO NOT change**: `n_cells` (4800), `n_cell_types` (2), `n_frames` (8000), `cell_params`, `func_params`, `sigma`, `max_radius`, `boundary`, `dimension`.

Note: `n_frames` is fixed at 8000. Adjust `data_augmentation_loop` to control effective training data volume.

## Metrics from analysis.log

| Metric                  | Better | Description                                  |
| ----------------------- | ------ | -------------------------------------------- |
| `training_g_phi_R2`     | Higher | R² of learned g_phi vs true (PRIMARY)        |
| `training_g_phi_R2_std` | Lower  | Std of per-type R² — consistency (SECONDARY) |
| `rollout_RMSE_mean`     | Lower  | Mean RMSE across rollout steps (TERTIARY)    |
| `training_accuracy`     | Higher | Clustering accuracy of embeddings            |
| `training_final_loss`   | Lower  | Final epoch training loss                    |
| `training_time_min`     | < 50   | Training duration in minutes                 |

## Classification

Per-slot:

- **Excellent**: g_phi_R2 > 0.95 AND g_phi_R2_std < 0.1
- **Good**: g_phi_R2 > 0.90 AND g_phi_R2_std < 0.2
- **Partial**: g_phi_R2 > 0.80
- **Failed**: g_phi_R2 < 0.80 OR training_time_min > 50

Batch-level robustness (across 4 seeds):

- **Stable-Robust**: all 4 slots g_phi_R2 > 0.90 AND CV < 5% — **TARGET**
- **Robust**: all 4 slots g_phi_R2 > 0.90, CV 5-10%
- **Partially robust**: 2-3 slots g_phi_R2 > 0.90
- **Fragile**: 0-1 slots g_phi_R2 > 0.90
- **DISQUALIFIED**: any slot g_phi_R2 < 0.5 — reject config

## Training Time Constraint

Budget: **< 1 hour on H100**. Factors increasing time:

- Larger `hidden_dim` / `n_layers`
- Larger `data_augmentation_loop`
- 8000 frames is a large dataset — keep `data_augmentation_loop` modest
- Smaller `batch_size`

## CRITICAL: GPU Memory Constraint

This simulation has **very high edge counts** (~700K edges/frame, avg degree ~295) because 4800 cells cluster tightly in 3D. This makes memory the binding constraint:

- **hidden_dim > 128 WILL OOM** on H100 (80GB) — do NOT exceed 128
- **batch_size > 8 WILL OOM** — do NOT exceed 8
- **hidden_dim=128 + batch_size=8** is near the memory limit — if using hidden_dim=128, prefer batch_size ≤ 4
- Combining large hidden_dim with many n_layers multiplies memory further

Safe combinations:

- hidden_dim=64, batch_size=8 — safe
- hidden_dim=128, batch_size=4 — safe
- hidden_dim=128, batch_size=8 — borderline, may OOM on dense frames

## Explorable Training Parameters

| Parameter                       | YAML path                              | Default | Description               | Typical range   |
| ------------------------------- | -------------------------------------- | ------- | ------------------------- | --------------- |
| `learning_rate_start`           | training.learning_rate_start           | 1E-4    | LR for g_phi MLP          | [1E-5, 1E-3]    |
| `learning_rate_embedding_start` | training.learning_rate_embedding_start | 1E-5    | LR for embeddings         | [1E-6, 1E-4]    |
| `batch_size`                    | training.batch_size                    | 8       | Frames per gradient step  | [1, 8]          |
| `data_augmentation_loop`        | training.data_augmentation_loop        | 20      | Iterations multiplier     | [5, 100]        |
| `hidden_dim`                    | graph_model.hidden_dim                 | 128     | g_phi hidden width        | [32, 128]       |
| `n_layers`                      | graph_model.n_layers                   | 5       | g_phi depth               | [3, 7]          |
| `embedding_dim`                 | graph_model.embedding_dim              | 2       | Cell embedding dim        | [1, 8]          |
| `coeff_edge_diff`               | training.coeff_edge_diff               | 0       | Same-type edge similarity | [0, 100]        |
| `coeff_edge_norm`               | training.coeff_edge_norm               | 0       | Monotonicity on g_phi     | [0, 10]         |
| `delta_t`                       | simulation.delta_t                     | 0.0005  | Integration time step     | [0.0001, 0.005] |

**Note on `input_size`**: Auto-computed. Do NOT set manually.

## Recurrent Training

| Parameter                        | YAML path                               | Default | Description          |
| -------------------------------- | --------------------------------------- | ------- | -------------------- |
| `recursive_training`             | training.recursive_training             | False   | Multi-step unrolling |
| `recursive_training_start_epoch` | training.recursive_training_start_epoch | 0       | Epoch to start       |
| `recursive_loop`                 | training.recursive_loop                 | 0       | Unroll steps (2-8)   |

## Parallel Mode — 4 Slots Per Batch

4 results per batch, 4 mutations for next batch. Each slot runs with a different random seed.

### Slot Strategy

All 4 slots run the **SAME config** — different seeds test robustness automatically.

When exploring different configs rather than testing robustness of one config:

- **Slot 0**: Exploit — best UCB parent
- **Slot 1**: Exploit — small mutation of best
- **Slot 2**: Explore — different parameter region
- **Slot 3**: Explore — different data regime (delta_t)

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

Compute across the 4 slots: mean, std, CV, min, max of g_phi_R2 and g_phi_R2_std.

### Step 3: Write Log Entries + Update Memory

Append to full log (`analysis.md`) and current block in `memory.md`:

```
## Iter N: [excellent|good|partial|failed]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: hidden_dim=X, n_layers=X, lr=X, lr_emb=X, batch=X, aug_loop=X, delta_t=X
Slot 0: g_phi_R2=X, g_phi_R2_std=X, rollout_RMSE=X, accuracy=X, loss=X, time=X min, sim_seed=S, train_seed=T
Slot 1: g_phi_R2=X, g_phi_R2_std=X, rollout_RMSE=X, accuracy=X, loss=X, time=X min, sim_seed=S, train_seed=T
Slot 2: g_phi_R2=X, g_phi_R2_std=X, rollout_RMSE=X, accuracy=X, loss=X, time=X min, sim_seed=S, train_seed=T
Slot 3: g_phi_R2=X, g_phi_R2_std=X, rollout_RMSE=X, accuracy=X, loss=X, time=X min, sim_seed=S, train_seed=T
Seed stats: mean_g_phi_R2=X, std=Y, CV=Z%, min=W, max=V
Stability: [Stable-Robust / Robust / Partially robust / Fragile / DISQUALIFIED]
Mutation: [param]: [old] -> [new]
Verdict: [supported/falsified/inconclusive] — [one line explanation]
Next: parent=P
```

**CRITICAL**: The `Mutation:` line is parsed by UCB. Always include exact parameter change.
**CRITICAL**: `Next: parent=P` — P must be from a previous or current batch, NEVER `id+1`.

### Step 4: Acknowledge User Input (if any)

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

1. Formulate hypothesis and write to memory.md
2. Design config mutation that tests it
3. Edit all 4 configs (identical unless explicitly exploring different regimes)

## Block Partition

| Block | Iterations | Focus              | Key Questions                                                         |
| ----- | ---------- | ------------------ | --------------------------------------------------------------------- |
| 1     | 1-16       | Baseline sweep     | Does default work? MLP size effect? LR effect? delta_t effect?        |
| 2     | 17-32      | Architecture       | hidden_dim, n_layers, embedding_dim, aggregation type                 |
| 3     | 33-48      | Training scheme    | LR, batch_size, data_augmentation, regularization                     |
| 4     | 49-64      | Regularization     | coeff_edge_diff, coeff_edge_norm — does it reduce R² std?             |
| 5     | 65-80      | Recurrent training | Does multi-step unrolling improve rollout RMSE without hurting g_phi? |
| 6+    | 81+        | Combined best      | Best from blocks 1-5, fine-tune, validation across seeds              |

## Block Boundaries

At end of each block:

1. Summarize findings in memory.md "Previous Block Summary"
2. Update "Established Principles" (require 3+ iterations AND cross-seed consistency)
3. Move falsified hypotheses to "Falsified Hypotheses"
4. Update "Regime Comparison Table"
5. Clear "Current Block" for next block
6. Carry forward best robust config

## Start Call

When prompt says `PARALLEL START`:

- Read base config
- Set all 4 configs identically to baseline
- Write initial hypothesis: "The default config achieves g_phi_R2 > 0.90 robustly across seeds"
- First iteration = baseline — do not change hyperparameters

---

# Working Memory Structure

```markdown
# Working Memory: dicty_spring_force g_phi exploration

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

| Iter | Config summary | g_phi_R2 (mean±std) | g_phi_R2_std (mean) | CV% | min | max | RMSE | time_min | Stability | Hypothesis tested |
| ---- | -------------- | ------------------- | ------------------- | --- | --- | --- | ---- | -------- | --------- | ----------------- |

### Established Principles

[Confirmed patterns — require 3+ iterations AND cross-seed consistency]

### Falsified Hypotheses

[Hypotheses contradicted by evidence]

### Open Questions

---

## Previous Block Summary

---

## Current Block (Block N)

### Block Info

### Current Hypothesis

**Hypothesis**: [specific, testable]
**Rationale**: [why]
**Test**: [what change]
**Expected outcome**: [support vs falsify]
**Status**: untested / supported / falsified

### Iterations This Block

### Emerging Observations
```
