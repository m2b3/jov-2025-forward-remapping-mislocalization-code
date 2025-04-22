import jax
import jax.numpy as jnp


@jax.jit
def eye_position(t, saccade_duration, saccade_amplitude):
    """
    Physical eye position at time t in screen coordinates, with t = 0 being saccade onset. Assumes linear movement (constant speed).
    Here we choose to make the saccade amplitude configurable.
    """
    movement_ratio = jnp.clip(t / saccade_duration, 0.0, 1.0)
    return movement_ratio * saccade_amplitude
