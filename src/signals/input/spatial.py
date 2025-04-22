import jax
import jax.numpy as jnp


@jax.jit
def spatial_tuning_curve(xs, center_loc, std):
    """
    Gaussian spatial tuning curve, centered at `center_loc`,
    normalized such that it is 1 at its peak.

    Args:
        xs: Array of shape (n_xs,) - neuron positions
        center_loc: Scalar or array - stimulus center location(s)

    Returns:
        Array of same batch shape as center_loc, broadcast with xs
    """
    # Just let JAX broadcasting handle everything naturally
    spatial_resp = jax.scipy.stats.norm.pdf(xs, scale=std, loc=center_loc)
    return spatial_resp / jnp.max(spatial_resp)
