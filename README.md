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
