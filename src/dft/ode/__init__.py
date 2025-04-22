"""
Core ODE solver logic.
Performance-critical.
"""

import jax
import jax.numpy as jnp
from diffrax import (
    Euler,
    ODETerm,
    SaveAt,
    diffeqsolve,
)


# Making a new simulation function will be required if changing the entire model,
# the shape of the grid, the shape/extent of time...
def make_base_solver(model, input_fn, sim_params):
    """Create pure single-item solver without any batching."""
    term = ODETerm(model.population_dynamics(input_fn))

    # Precompute static time array
    save_ts = jnp.arange(sim_params.t_start, sim_params.t_end, sim_params.sampling_dt)
    saveat = SaveAt(ts=save_ts)

    init_state_template = jnp.ones(model.n_xs)

    def solve(model_params, exp_params):
        """Single-item solver - no batching. Batching is handled by vmapping."""
        y0 = model_params.initial_u * init_state_template
        return diffeqsolve(
            term,
            solver=Euler(),
            t0=sim_params.t_start,
            t1=sim_params.t_end,
            dt0=sim_params.dt0,
            y0=y0,
            args={
                "exp_params": exp_params,
                "model_params": model_params,
            },
            max_steps=None,
            saveat=saveat,
        )

    # Not JIT-ing at this level would incur a high performance cost!
    return jax.jit(solve)
