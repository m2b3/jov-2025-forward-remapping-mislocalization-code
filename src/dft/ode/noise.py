"""
Utilities for adding noise to ODE systems.
"""

import jax
import jax.numpy as jnp
from diffrax import (
    ControlTerm,
    MultiTerm,
    UnsafeBrownianPath,
)


def get_shape(item):
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), item
    )


def make_stochastic_term(term, shape, random_key):
    def diffusion(_t, y, args):
        return jax.tree_map(
            lambda x: args["noise_strength"] * jnp.ones_like(x),
            jax.eval_shape(lambda: y),
        )

    brownian = UnsafeBrownianPath(
        # XXX: noise shape
        shape=shape,
        key=random_key,
    )
    # Alternative:
    # brownian = VirtualBrownianTree(
    #     t_start,
    #     t_end,
    #     tol=1e-3,
    #     shape=jax.eval_shape(lambda: y0),
    #     key=jrandom.PRNGKey(0),
    # )

    return MultiTerm(term, ControlTerm(diffusion, brownian))
