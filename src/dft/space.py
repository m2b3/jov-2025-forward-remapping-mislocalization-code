import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def make_xs(n_xs, amplitude_coded_for):
    # The center should be 0, not a slight offset from it!
    # So we need an odd number of points.
    # Perfect symmetry. Otherwise, could directionally-bias results.
    assert n_xs % 2 == 1

    # Discretize around the origin of the retinotopic coordinate system.
    # This is "less efficient" in terms of simulation, but more realistic.
    center = 0

    return jnp.linspace(
        center - (amplitude_coded_for / 2),
        center + (amplitude_coded_for / 2),
        n_xs,
    )
