import jax
import jax.numpy as jnp
from .temporal import alpha_gaussian_exp
from .spatial import spatial_tuning_curve

__all__ = ["driving_fn", "alpha_gaussian_exp", "spatial_tuning_curve"]


@jax.jit
def driving_fn(
    t,
    scale=1.0,
    onset_to_peak=15,
    temporal_sigma=10.0,
    offset_tau=15.0,
    flash_duration=30.0,
    stable_baseline=0.0,
):
    """
    Vectorized driving function with alpha-gaussian-exponential temporal envelope.
    Immediately starts at t = 0 - this function uses its own timebase and has no awareness of flash onset on screen.

    Args:
        t: Time points
        scale: Spatial scale factor (usually from spatial tuning curve)
        onset: Time of stimulus onset
        onset_to_peak: Time to reach peak response
        temporal_sigma: Width of Gaussian phase
        temporal_tau_end: Time constant of final exponential decay
        temporal_width: Width of Gaussian plateau
        temporal_baseline: Baseline activation level
    """
    scale = jnp.asarray(scale)

    f_t = alpha_gaussian_exp(
        t,
        onset_to_peak=onset_to_peak,
        sigma=temporal_sigma,
        tau_end=offset_tau,
        width=flash_duration,
        baseline=stable_baseline,
        amplitude=1.0,  # We use scale for amplitude control
    )
    expanded = jnp.squeeze(jnp.outer(scale, f_t))
    return expanded
