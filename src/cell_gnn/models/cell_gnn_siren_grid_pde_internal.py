"""GNN + SIREN-on-grid + learned PDE MLP + learned internal-state MLP.

Extends :class:`CellGNNSirenGridPDE`:

  * Per-cell internal state ``s_i`` is *observed* (read from
    ``state.field[:, 1]`` — the fluorescent reporter saved by the matching
    generator).  Source emission to the chemical-field grid is gated by the
    KNOWN sigmoid: ``alpha_i = source_strength * sigmoid(s_i)``.

  * A small MLP ``lin_internal`` learns the unknown ODE law:

        ds_i/dt  ≈  lin_internal([ s_i,  c_local_i,  embedding_i ])

    The per-cell embedding is included so the ODE law can depend on
    cell-intrinsic state (cell type / individual identity), not just on
    the (s, c) pair.  Supervision target is ``state.field[:, 2]`` (the
    analytic ds/dt saved by the generator).  The residual

        R_internal = lin_internal([s, c_local, embed]) - ds/dt_target

    is stored in ``self.last_internal_residual`` for the trainer to add
    to the loss (weight ``training.internal_weight``).
"""

import torch
import torch.nn.functional as F

from cell_gnn.cell_state import CellState
from cell_gnn.models.MLP import MLP
from cell_gnn.models.cell_gnn_siren_grid_pde import CellGNNSirenGridPDE
from cell_gnn.models.registry import register_model


@register_model("particle_spring_force_diffusion_field_siren_grid_pde_internal")
class CellGNNSirenGridPDEInternal(CellGNNSirenGridPDE):
    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):
        super().__init__(config, device, aggr_type=aggr_type, bc_dpos=bc_dpos, dimension=dimension)

        model_config = config.graph_model
        # Internal-state MLP: (s, c_local, embedding) -> ds/dt
        # The embedding lets the ODE law differ between cells of different
        # type/identity even at the same (s, c).
        internal_hidden = int(getattr(model_config, 'internal_hidden_dim', 16))
        internal_layers = int(getattr(model_config, 'internal_n_layers', 2))
        self._internal_input_size = 2 + self.embedding_dim
        self.lin_internal = MLP(
            input_size=self._internal_input_size,
            output_size=1,
            nlayers=internal_layers,
            hidden_size=internal_hidden,
            device=self.device,
        )

        # Cache filled by forward() and consumed by _compute_source().
        self._s_obs_cache = None
        self.last_internal_residual = None
        self.last_internal_pred = None
        self.last_internal_target = None

    # ------------------------------------------------------------------
    def _compute_source(self, pos, k_time):
        """Source emission gated by the KNOWN sigmoid of observed s_i.

        ``self._s_obs_cache`` is set by ``forward()`` immediately before the
        parent computes the source field on the grid.
        """
        if self.pde_alpha <= 0.0 or self._s_obs_cache is None:
            return torch.zeros(self.grid_shape, device=pos.device, dtype=pos.dtype)
        emit = self.pde_alpha * torch.sigmoid(self._s_obs_cache)
        return self._particles_to_grid(pos, emit)

    # ------------------------------------------------------------------
    def forward(self, state: CellState, edge_index: torch.Tensor,
                data_id=[], training=[], has_field=False, k=[]):
        # Stash observed s for _compute_source. state.field is (N, 3): [c, s, ds/dt].
        if state.field is None or state.field.shape[1] < 2:
            raise RuntimeError(
                "CellGNNSirenGridPDEInternal expects state.field = [c, s, ds/dt]")
        self._s_obs_cache = state.field[:, 1].detach()
        ds_dt_target = state.field[:, 2].detach()

        # Run the parent forward — it will call our overridden _compute_source.
        out = super().forward(state, edge_index, data_id=data_id, training=training,
                              has_field=has_field, k=k)

        # Sample c at particle positions from the learned grid (use the
        # SIREN-evaluated grid stashed by the parent).  ``last_scalar_field``
        # is detached in the parent — re-evaluate with grad enabled so the
        # internal-MLP loss can flow through.
        c_grid = self._reeval_c_grid_for_internal(k=k)
        c_local = self._trilinear_interp(c_grid.unsqueeze(0), state.pos).squeeze(-1)  # (N,)

        # Look up the per-cell embedding (same way the parent does for lin_edge).
        cell_id = state.index.unsqueeze(-1)
        embedding = self.a[self.data_id.clone().detach(), cell_id, :].squeeze(1)  # (N, E)

        s_obs = state.field[:, 1]
        internal_in = torch.cat(
            [s_obs.unsqueeze(-1), c_local.unsqueeze(-1), embedding], dim=-1
        )                                                                          # (N, 2+E)
        ds_dt_pred = self.lin_internal(internal_in).squeeze(-1)                    # (N,)

        residual = ds_dt_pred - ds_dt_target
        self.last_internal_residual = residual.unsqueeze(-1)
        self.last_internal_pred = ds_dt_pred.detach()
        self.last_internal_target = ds_dt_target.detach()

        return out

    # ------------------------------------------------------------------
    def _reeval_c_grid_for_internal(self, k):
        """Re-evaluate SIREN c on the fixed grid (with grad) for c_local sampling."""
        if not torch.is_tensor(k):
            k_scalar = float(k)
            t_scalar = torch.tensor(
                k_scalar * self.delta_t,
                dtype=self.grid_coords.dtype, device=self.grid_coords.device,
            )
        else:
            t_scalar = k.reshape(-1)[0].to(self.grid_coords.dtype) * self.delta_t

        t_vec = t_scalar.detach().clone().expand(self.grid_coords.shape[0]).clone().unsqueeze(-1)
        siren_in = torch.cat([self.grid_coords, t_vec], dim=1)
        c_flat = self.siren_field(siren_in)  # (G, 1)
        return c_flat.view(*self.grid_shape)
