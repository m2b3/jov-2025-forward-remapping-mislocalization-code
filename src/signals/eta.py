"""
Corollary Discharge - temporal
"""

import jax
import jax.lax
import jax.numpy as jnp


@jax.jit
def double_sigmoid_window(t, tau_rise, tau_fall, start, end):
    rise = jax.lax.logistic(+(t - start) / tau_rise)
    fall = jax.lax.logistic(-(t - end) / tau_fall)
    return rise * fall


@jax.jit
def compute_normalization_factor(
    tau_rise, tau_fall, duration, center, t_min=-100, t_max=100, n_points=1000
):
    """Compute the normalization factor for a given parameter set"""
    t = jnp.linspace(t_min, t_max, n_points)
    window = remapping_window_unnormalized(t, tau_rise, tau_fall, duration, center)
    return jnp.trapezoid(window, t)


@jax.jit
def remapping_window_unnormalized(t, tau_rise, tau_fall, duration, center):
    """Original window function without normalization"""
    remapping_start = -duration / 2 + center
    remapping_end = duration / 2 + center
    return double_sigmoid_window(
        t,
        tau_rise=tau_rise,
        tau_fall=tau_fall,
        start=remapping_start,
        end=remapping_end,
    )


@jax.jit
def eta(t, tau_rise, tau_fall, duration, center, t_min=-300, t_max=300):
    """Normalized remapping window with constant integral"""
    window = remapping_window_unnormalized(t, tau_rise, tau_fall, duration, center)
    norm_factor = compute_normalization_factor(
        tau_rise, tau_fall, duration, center, t_min, t_max
    )
    return window / norm_factor
