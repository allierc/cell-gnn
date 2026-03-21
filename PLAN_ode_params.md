# Plan: ODE Parameter Classes for cell-gnn

## Goal

Introduce structured ODE parameter dataclasses for `arbitrary_ode` and `dicty_spring_force_ode`,
following the pattern established in flyvis-gnn (`flyvis_gnn/generators/ode_params.py`).

## Current State

Parameters are raw lists in YAML configs (`cell_params`, `func_params`) converted to
an unstructured tensor `self.p` with magic indices:

- **arbitrary_ode**: `self.p[cell_type, :]` → `[p0, p1, p2, p3]` (unnamed coefficients)
- **dicty_spring_force_ode**: `self.p[cell_type, :]` → `[k_rep, r0, kadh, r_on, delta, mu_f]`

No save/load, no named access, no validation.

## Reference: flyvis-gnn Implementation

File: `flyvis_gnn/generators/ode_params.py`

Key components:
1. **`ODEParamsBase`** — dataclass with `to()`, `clone()`, `save()`, `load()`, dict-style access
2. **`@register_ode_params("name")`** — decorator to register params class under config names
3. **`get_ode_params_class(name)`** — lookup by `cell_model_name`
4. **Concrete classes**: `FlyVisODEParams`, `FlyVisAdExODEParams`, `FlyVisHodgkinHuxleyODEParams`
   each with named tensor fields and `from_*()` constructors

## Plan

### Step 1: Create `cell_gnn/generators/ode_params.py`

Copy the registry + base class from flyvis-gnn (same pattern):

```python
_ODE_PARAMS_REGISTRY: dict[str, type] = {}
register_ode_params(...)
get_ode_params_class(...)
ODEParamsBase  # dataclass with to(), clone(), save(), load()
```

### Step 2: Define `ArbitraryODEParams`

```python
@register_ode_params("arbitrary_ode", "arbitrary_field_ode")
@dataclass
class ArbitraryODEParams(ODEParamsBase):
    p: torch.Tensor = None          # (n_cell_types, 4) interaction coefficients
    sigma: float = 0.005            # interaction scale
    func_p: list = None             # [['arbitrary', i, j], ...] per cell type
    dimension: int = 2

    @classmethod
    def from_config(cls, config, device="cpu"):
        """Construct from CellGNNConfig."""
        sim = config.simulation
        p = torch.tensor(sim.cell_params, dtype=torch.float32, device=device)
        return cls(p=p, sigma=sim.sigma, func_p=sim.func_params,
                   dimension=sim.dimension)
```

### Step 3: Define `DictySpringForceODEParams`

```python
@register_ode_params("dicty_spring_force_ode")
@dataclass
class DictySpringForceODEParams(ODEParamsBase):
    # (n_cell_types, 6) with named columns:
    # k_rep, r0, kadh, r_on, delta, mu_f
    p: torch.Tensor = None
    dimension: int = 3

    @classmethod
    def from_config(cls, config, device="cpu"):
        sim = config.simulation
        p = torch.tensor(sim.cell_params, dtype=torch.float32, device=device)
        return cls(p=p, dimension=sim.dimension)
```

### Step 4: Update simulator classes

Update `ArbitraryODE.__init__` and `DictySpringForceODE.__init__` to accept
an `ODEParamsBase` instance instead of loose `p`, `sigma`, `func_p` args.
Keep backward compat with dict-style access during transition.

```python
# Before (generators/arbitrary_ode.py)
def __init__(self, aggr_type=[], p=[], func_p=None, sigma=[], ...):
    self.p = p
    self.sigma = sigma
    self.func_p = func_p

# After
def __init__(self, aggr_type=[], params=None, bc_dpos=[], ...):
    self.params = params
    self.p = params.p
    self.sigma = params.sigma
    self.func_p = params.func_p
```

### Step 5: Update `choose_model()` in `generators/utils.py`

Replace the manual tensor construction with:

```python
# Before
p = torch.tensor(params, dtype=torch.float32, device=device)
model = sim_cls(aggr_type=aggr_type, p=p, func_p=func_p, sigma=sigma, ...)

# After
ODE_params_class = get_ode_params_class(mc.cell_model_name)
ode_params = ODE_params_class.from_config(config, device=device)
model = sim_cls(aggr_type=aggr_type, params=ode_params, ...)
```

### Step 6: Save ODE params alongside generated data

In `data_generate_cell()`, after data generation:

```python
ode_params.save(graphs_data_path(dataset_name))
```

This creates `ode_params.pt` in the data folder, matching flyvis-gnn's convention.

### Step 7: Load ODE params at test time

In `data_test_cell()`, load params from the data folder instead of re-deriving from config:

```python
ODE_params_class = get_ode_params_class(mc.cell_model_name)
ode_params = ODE_params_class.load(graphs_data_path(dataset_name), device=device)
```

## Files to Modify

| File | Change |
|------|--------|
| `cell_gnn/generators/ode_params.py` | **New** — registry, base class, concrete param classes |
| `cell_gnn/generators/arbitrary_ode.py` | Accept `params` object instead of loose args |
| `cell_gnn/generators/dicty_spring_force_ode.py` | Accept `params` object instead of loose args |
| `cell_gnn/generators/utils.py` | Use `from_config()` + `get_ode_params_class()` |
| `cell_gnn/generators/graph_data_generator.py` | Save `ode_params.pt` after generation |
| `cell_gnn/models/graph_trainer.py` | Load `ode_params.pt` at test time for ground truth |

## Not Changed

- YAML config format (stays as `cell_params` / `func_params` lists)
- CellGNN / CellFieldGNN learnable models
- Training loop
- LLM exploration pipeline
