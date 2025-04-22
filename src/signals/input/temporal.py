import jax
import jax.numpy as jnp


@jax.jit
def alpha_gaussian_exp(
    t, onset_to_peak, sigma, tau_end, width, baseline=0.0, amplitude=1.0
):
    """Three-phase temporal response with controlled baseline and continuous transitions"""
    onset_to_peak, sigma, tau_end = [
        jnp.maximum(x, 1e-6) for x in [onset_to_peak, sigma, tau_end]
    ]
    t_peak = onset_to_peak
    t_end = t_peak + width

    # Inline gaussian calculation function
    def gauss(t_rel):
        return (1.0 - baseline) * jnp.exp(-(t_rel**2) / (2 * sigma**2)) + baseline

    # Alpha rise to 1.0
    alpha = (t / onset_to_peak) * jnp.exp(-t / onset_to_peak) / jnp.exp(-1)

    # Gaussian decay to baseline and endpoint value for exp decay
    gaussian = gauss(t - t_peak)
    exp_decay = gauss(width) * jnp.exp(-(t - t_end) / tau_end)

    return amplitude * jnp.where(
        t < 0,
        0.0,
        jnp.where(t <= t_peak, alpha, jnp.where(t <= t_end, gaussian, exp_decay)),  # type: ignore
    )
