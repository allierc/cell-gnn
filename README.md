# cell-gnn
Graph neural networks for cell simulations

### Setup
Run the following line from the terminal to create a new environment cell-gnn:
```
conda env create -f environment.yaml
```

Activate the environment:
```
conda activate cell-gnn
```

Install the package by executing the following command from the root of this directory:
```
pip install -e .
```

Then, you should be able to import all the modules from the package in python:
```python
from cell_gnn import *
```

### Model Architectures

The `cell_model_name` field in the YAML config selects both the data generator and the ML model:

| `cell_model_name` | Generator | ML Model | Loss |
|---|---|---|---|
| **Pairwise only** | | | |
| `particle_spring_force_ode` | `ParticleSpringForceODE` | **CellGNN** — GNN only | ‖v − v_gt‖² + reg |
| **Prescribed field** | | | |
| `particle_spring_force_prescribed_field` | `ParticleSpringForcePrescribedField` | **CellGNN** — GNN only | ‖v − v_gt‖² + reg |
| `..._prescribed_field_siren` | same ↑ | **CellGNNSirenField** — GNN + SIREN→**vector** | ‖v − v_gt‖² + reg |
| `..._prescribed_field_siren_grad` | same ↑ | **CellGNNSirenGradField** — GNN + ∇(SIREN→scalar) | ‖v − v_gt‖² + reg |
| **Diffusion field** | | | |
| `..._diffusion_field` | `ParticleSpringForceDiffusionField` | **CellGNN** — GNN only | ‖v − v_gt‖² + reg |
| `..._diffusion_field_siren` | same ↑ | **CellGNNSirenField** — GNN + SIREN→**vector** | ‖v − v_gt‖² + reg |
| `..._diffusion_field_siren_pde` | same ↑ | **CellGNNSirenFieldPDE** — GNN + SIREN→(c,**v**) | ‖v − v_gt‖² + w_pde·‖PDE‖² + reg |
| `..._diffusion_field_siren_grad` | same ↑ | **CellGNNSirenGradField** — GNN + ∇(SIREN→scalar) | ‖v − v_gt‖² + reg |
| `..._diffusion_field_siren_grad_pde` | same ↑ | **CellGNNSirenGradFieldPDE** — GNN + ∇(SIREN→scalar) | ‖v − v_gt‖² + w_pde·‖PDE‖² + reg |

All names above are prefixed with `particle_spring_force_` (abbreviated as `...` for readability).

- **SIREN→vector**: SIREN outputs velocity (vx, vy, vz) directly
- **∇(SIREN→scalar)**: SIREN outputs scalar c(x,t), velocity = μ·∇c via autograd
- **SIREN→(c,v)**: SIREN outputs (c, vx, vy, vz); c used for PDE loss, v used for velocity
- **PDE** = ∂c/∂t − D∇²c + λc − S
- **reg** = edge_weight (L1) + edge_diff (monotonicity) + edge_norm (zero at cutoff) + continuous (smoothness)

### Training Loop

for epoch in range(n_epochs=4):                       # n_epochs from config
    for N in range(Niter=20000):                      # iterations per epoch
        # ─── one iteration ─────────────────────────────────
        pick batch_size=8 random frames k_1,...,k_8  # uniform random
        for each frame k_i:
            load 1000 cells (positions, velocities)
            load ground-truth velocity y_i = y_raw[k_i]
            build neighbor edges
        collate into one big batched graph
        pred = model(batched_graph, k_batch)
            └→ pred = GNN(positions, edges) + SIREN(x, y, z, t)
        loss = ‖pred − y_true‖₂                       # ONE loss per iteration
        loss.backward()
        optimizer.step()                              # ONE weight update per iteration

bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o generate_train_test dicty_spring_force_rk4_n3000"