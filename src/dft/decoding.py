"""
Decoding = getting a position from population activity.
"""

from typing import cast
import jax
import jax.numpy as jnp


@jax.jit
def compute_thresholded_barycenter(ws, xs):
    """
    In order to avoid skew effects due to the extent of the considered space,
    only consider weights that are more than a % of the max weight.
    We also do not compute a barycenter, and return nan instead, if ALL weights are below an absolute threshold.
    """
    # TODO: make this configurable (and play nicely with vmap below)
    thresh_ratio = 0.3
    thresh_abs = 1e-1

    threshold = thresh_ratio * jnp.max(ws)
    all_below_threshold = jnp.all(ws < thresh_abs)
    ws = cast(jax.Array, jnp.where(ws < threshold, 0, ws))
    sum_ws = jnp.sum(ws)

    return jnp.where(all_below_threshold, jnp.nan, jnp.dot(ws, xs) / sum_ws)


# This expects the weights to have shape (positions, times)
thresholded_barycenters_across_time = jax.vmap(
    compute_thresholded_barycenter, in_axes=(1, None)
)
