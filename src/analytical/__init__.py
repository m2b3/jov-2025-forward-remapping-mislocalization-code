# Mislocalization curve derivation from first principles,
# without actually simulating a population of neurons.

import jax
import jax.numpy as jnp
from src.signals import eta, eye_position

# might be able to speed up with cumsum but not necessary?

@jax.jit
def integrate_eta_future(t_start, tau1, tau2, duration, center, t_dec):
    # Changing -> shape changes -> JIT unhappy - could be filter_jit'd but not important for now.
    t_max = 500.0
    dt = 0.01

    # Create a fixed time grid from -200 ms up to t_max (e.g. 500 ms)
    t = jnp.arange(-200.0, t_max, dt)
    cd = eta(t, tau1, tau2, duration, center)
    # For normalization, integrate only up to t_dec (i.e. using t_dec as the upper bound)
    mask_total = t <= t_dec
    total = jnp.sum(cd * mask_total) * dt
    mask_past = t <= t_start
    past_integral = jnp.sum(cd * mask_past) * dt
    return (total - past_integral) / total

# Vectorized version accepting t_dec as a constant for all onsets.
integrate_etas_future = jax.vmap(
    integrate_eta_future, in_axes=(0, None, None, None, None, None)
)

@jax.jit
def compute_analytical_mislocalization_single(
    t,
    saccade_duration,
    saccade_amplitude,
    retina_delay,
    onset_to_peak,
    eta_tau1,
    eta_tau2,
    eta_duration,
    eta_center,
    t_dec
):
    """
    Compute mislocalization for a flash at time t.
    Returns error in degrees (positive = forward mislocalization)
    """
    # Compute the effective time for neural integration.
    t_neural = t + retina_delay + onset_to_peak
    # Use the provided t_dec as the integration upper bound.
    remapping_left = integrate_eta_future(
        t_neural, eta_tau1, eta_tau2, eta_duration, eta_center, t_dec
    )
    # Compute eye position and the remapped flash location.
    eye_pos = eye_position(t, saccade_duration, saccade_amplitude)
    retinal_flash_loc = -eye_pos
    actual_remapping = -saccade_amplitude * remapping_left
    perceived_pos = retinal_flash_loc + actual_remapping + saccade_amplitude
    return perceived_pos - 0.0

# Vectorized computation over time points.
# Note: we now have 9 arguments, with t_dec passed as a constant for all onsets.
compute_analytical_mislocalization_vect = jax.vmap(
    compute_analytical_mislocalization_single,
    in_axes=(0, None, None, None, None, None, None, None, None, None)
)

@jax.jit
def compute_analytical_mislocalization(onsets, mp, ep):
    """
    Given a set of flash onset times, and model (mp) and experimental (ep) parameters,
    compute the analytical mislocalization.

    The model parameters mp are assumed to include:
      - saccade_duration
      - retina_to_lip_delay
      - eta_tau_rise, eta_tau_fall, eta_duration, eta_center
      - decoding_time (to be used as t_dec)

    The experimental parameters ep should include:
      - saccade_amplitude
    """
    return compute_analytical_mislocalization_vect(
        onsets,
        mp.saccade_duration,
        ep.saccade_amplitude,
        mp.retina_to_lip_delay,
        mp.input_onset_to_peak,
        mp.eta_tau_rise,
        mp.eta_tau_fall,
        mp.eta_duration,
        mp.eta_center,
        mp.decoding_time  # newly added decoding time parameter
    )
