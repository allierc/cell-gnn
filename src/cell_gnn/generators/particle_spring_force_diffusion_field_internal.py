"""Diffusion-field cells with a per-cell internal state s_i (fluorescent reporter).

Ground-truth dynamics:

    ds_i/dt = coeff_s * (c_i - s_i)                                    # linear relaxation toward c
    alpha_i = coeff_emit * (1 - s_i) * sigmoid(beta * (c_i - c_thr))   # smoothly gated emission

The sigmoid replaces a Heaviside step so the gate is differentiable —
useful when a learner is asked to recover the gating from data. Emission
shuts off as s saturates (1 - s) and ramps up smoothly across c_threshold,
with sharpness controlled by ``internal_c_sharpness`` (β).

Saved per-cell observables (``state.field`` columns):

    [c_i, s_i, ds_i/dt_target]

where ``ds_i/dt_target`` is the analytic rate evaluated at the current
(c_i, s_i) state — the supervision target for a learner trying to recover
the unknown ds/dt law.
"""

import torch

from cell_gnn.generators.particle_spring_force_diffusion_field import (
    ParticleSpringForceDiffusionField,
    deposit_to_grid_weighted,
    interp_periodic,
    spectral_gradient,
)
from cell_gnn.utils import to_numpy
from cell_gnn.cell_state import CellState
from cell_gnn.graph_utils import remove_self_loops, scatter_aggregate
from cell_gnn.models.registry import register_simulator


@register_simulator("particle_spring_force_diffusion_field_internal",
                    "particle_spring_force_diffusion_field_siren_grid_pde_internal")
class ParticleSpringForceDiffusionFieldInternal(ParticleSpringForceDiffusionField):
    r"""Adds a per-cell internal state ``s_i`` driven by local concentration.

    Extra ``field_params`` keys
    ---------------------------
    internal_coeff_s : float
        Rate for ``ds/dt = coeff_s * (c - s)``.  Default 2.0
        (time constant 1/coeff_s = 0.5).
    internal_coeff_emit : float
        Emission strength: ``alpha = coeff_emit * (1 - s) * sigmoid(beta * (c - c_threshold))``.
        Default 20.0.
    internal_c_threshold : float
        Sigmoid mid-point: emission ramps from 0 to coeff_emit*(1-s) as c
        crosses this value.  Default 0.01.
    internal_c_sharpness : float
        Sigmoid sharpness β.  Large β → sigmoid approaches a Heaviside step;
        small β → very gradual ramp.  Default 200.0 (effective full-on/off
        transition over a c-window of ~4/β = 0.02).
    internal_init_range : (lo, hi)
        Uniform-random init range for s_i.  Default (0.0, 1.0).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Lazy per-cell internal state: (n_cells,) float
        self._s = None
        # ds/dt target stored on the last forward — written into state.field
        self._ds_dt_target = None

    # ------------------------------------------------------------------
    def _init_internal(self, n_cells, device, dtype=torch.float32):
        fp = self.field_params
        lo, hi = fp.get('internal_init_range', (0.0, 1.0))
        self._s = lo + (hi - lo) * torch.rand(n_cells, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def _advance_field(self, pos, n_steps=1):
        """Advance both the diffusion field *and* the per-cell internal state.

        Per step:
          1. emission ``alpha_i = coeff_emit * (1 - s_i) * sigmoid(beta*(c_i - c_thr))``
             deposited via trilinear weights.
          2. spectral diffusion + decay.
          3. forward-Euler update ``s_i += dt * coeff_s * (c_i - s_i)``
             using c sampled before the spectral step.
        """
        dim = self.dimension
        fp = self.field_params
        res = fp['grid_resolution']
        dt = fp['delta_t']
        # Internal-state knobs
        coeff_s = float(fp.get('internal_coeff_s', 2.0))
        coeff_emit = float(fp.get('internal_coeff_emit', 20.0))
        c_threshold = float(fp.get('internal_c_threshold', 0.01))
        c_sharpness = float(fp.get('internal_c_sharpness', 200.0))
        s_shape = [res] * dim

        # Initialise internal state on first call.
        if self._s is None:
            self._init_internal(pos.shape[0], pos.device, dtype=pos.dtype)

        for _ in range(n_steps):
            # 1) Sample c at particle positions BEFORE the spectral step.
            c_at_pos_old = interp_periodic(self._field_grid, pos, res)

            # 2) Sigmoid-gated emission, scaled by (1 - s_i).
            if coeff_emit > 0:
                gate = torch.sigmoid(c_sharpness * (c_at_pos_old - c_threshold))
                emit = coeff_emit * (1.0 - self._s) * gate
                source = deposit_to_grid_weighted(
                    pos, emit * dt, res, dim, pos.device)
                self._field_grid = self._field_grid + source

            # 3) Spectral diffusion + decay step.
            C_hat = torch.fft.rfftn(self._field_grid, s=s_shape)
            C_hat = C_hat * self._decay
            self._field_grid = torch.fft.irfftn(C_hat, s=s_shape)

            # 4) Forward-Euler update of s_i using c_at_pos_old.
            ds_dt_step = coeff_s * (c_at_pos_old - self._s)
            self._s = self._s + dt * ds_dt_step

        C_hat = torch.fft.rfftn(self._field_grid, s=s_shape)
        return C_hat

    # ------------------------------------------------------------------
    def forward(self, state: CellState, edge_index: torch.Tensor,
                has_field=False, k=0):
        edge_index = remove_self_loops(edge_index)
        p = self.p.unsqueeze(0) if self.p.dim() == 1 else self.p
        parameters = p[to_numpy(state.cell_type), :]

        src, dst = edge_index[1], edge_index[0]
        pos_i, pos_j = state.pos[dst], state.pos[src]
        parameters_i = parameters[dst]

        # --- pairwise spring forces ---
        messages = self.message(pos_i, pos_j, parameters_i)
        d_pos = scatter_aggregate(messages, dst, state.n_cells, self.aggr_type)

        # --- diffusion field with internal s_i ---
        device = state.pos.device
        for key in ('center_0',):
            v = self.field_params.get(key)
            if torch.is_tensor(v) and v.device != device:
                self.field_params[key] = v.to(device)

        if self._field_grid is None:
            self._init_grid(device)

        steps = k - self._last_step
        if steps > 0:
            C_hat = self._advance_field(state.pos, n_steps=steps)
        else:
            s_shape = [self.field_params['grid_resolution']] * self.dimension
            C_hat = torch.fft.rfftn(self._field_grid, s=s_shape)
            if self._s is None:
                self._init_internal(state.pos.shape[0], device, dtype=state.pos.dtype)
        self._last_step = k

        res = self.field_params['grid_resolution']
        grad_grid = spectral_gradient(C_hat, res, self.dimension, device)

        field_value = interp_periodic(self._field_grid, state.pos, res)  # (N,)
        field_gradient = interp_periodic(grad_grid, state.pos, res)      # (N, dim)

        # Chemotactic force (same as parent — supports linear & saturating forms).
        mu_chem = self.field_params['mu_chem']
        g0 = self.field_params.get('chem_saturation_scale', None)
        sat_type = self.field_params.get('chem_saturation_type', 'log')
        if g0 is not None and g0 > 0:
            g_norm = field_gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g_hat = field_gradient / g_norm
            if sat_type == 'mm':
                chem_force = mu_chem * (g_norm / (g_norm + g0)) * g_hat
            elif sat_type == 'sigmoid':
                chem_force = mu_chem * (2 * torch.sigmoid(g_norm) - 1) * g_hat
            else:
                chem_force = mu_chem * torch.log1p(g_norm / g0) * g_hat
        else:
            chem_force = mu_chem * field_gradient
        d_pos = d_pos + chem_force

        # --- Compute analytic ds/dt target at the *current* (c, s) state ---
        # This is what a learner should recover when given (s_i, c_i) as input.
        coeff_s = float(self.field_params.get('internal_coeff_s', 2.0))
        ds_dt_target = coeff_s * (field_value - self._s)
        self._ds_dt_target = ds_dt_target

        # state.field channels: [c, s, ds/dt_target]
        state.field = torch.stack([field_value, self._s, ds_dt_target], dim=-1)

        self.last_clean = d_pos.clone()

        if self.noise_model_level > 0:
            noise = self.noise_model_level * torch.randn_like(d_pos)
            self.last_noise = noise
            d_pos = d_pos + noise
        else:
            self.last_noise = torch.zeros_like(d_pos)

        return d_pos
