import equinox as eqx
import jax.numpy as jnp
import jax
from .params import DftDriftParams, ExpParams
from src.signals import eye_position, driving_fn, spatial_tuning_curve


@eqx.filter_jit
def ff_input(t, mp: DftDriftParams, ep: ExpParams, xs: jax.Array):  # type: ignore
    # t = 0: saccade onset
    # ep.onset: flash onset *on screen*
    t_since_flash_onset = t - ep.onset

    # This is the time used to derive the feedforward input to the population.
    # This takes into account propagation delays.
    t_since_pop_input_onset = t_since_flash_onset - mp.retina_to_lip_delay

    # With smearing on, we consider where the eye was at each timestep of the flash.
    # With smearing off, we consider where the eye was at the onset of the flash.
    eye_pos = eye_position(
        jnp.where(mp.smearing, t - mp.retina_to_lip_delay, ep.onset),
        saccade_duration=mp.saccade_duration,
        saccade_amplitude=ep.saccade_amplitude,
    )

    # Convert from screen to retinal coords
    # If flash_x=0, this reduces to current -eye_pos
    stim_pres_x = ep.flash_x - eye_pos

    return driving_fn(
        t_since_pop_input_onset,
        scale=spatial_tuning_curve(
            xs=xs, center_loc=stim_pres_x, std=mp.input_spatial_std
        ),
        # Duration of the flash on screen
        flash_duration=ep.flash_duration,
        # Parameters of the LIP input derived from the "real" flash
        onset_to_peak=mp.input_onset_to_peak,
        temporal_sigma=mp.input_temporal_sigma,
        offset_tau=mp.input_offset_tau,
        stable_baseline=mp.input_stable_baseline,
    )
