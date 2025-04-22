import jax.numpy as jnp
from typing import Optional, TypeVar, Generic
from .base import BaseSimulator
from ..decoding import compute_thresholded_barycenter
from ..count_bumps import count_bumps
from ..params import DftDriftParams, ExpParams

EARLY_ONSET_TIME = -200

TSim = TypeVar("TSim", bound=BaseSimulator)

class ApplicationsMixin(Generic[TSim]):
    def decoded_retinal_locations_for_flash_onsets(
        self: TSim,
        onsets: jnp.ndarray,
        model_params: Optional[DftDriftParams] = None,
        exp_params: Optional[ExpParams] = None,
    ) -> jnp.ndarray:
        if model_params is None:
            model_params = self.model_params
        if exp_params is None:
            exp_params = self.exp_params

        ts, ys = self.simulate_flash_response(onsets, mp=model_params, base_ep=exp_params)
        acts = self.model.population_activity(ys.T, model_params)
        assert acts.shape[2] == len(onsets)
        assert acts.shape[0] == self.model.n_xs
        # Instead of taking the final timepoint, choose the one closest to decoding_time.
        # ts is assumed common to all trials.
        # Find the index in ts closest to decoding_time:
        idx = int(jnp.argmin(jnp.abs(ts - model_params.decoding_time)))
        decoded_acts = acts[:, idx, :]

        decoded_locations = jnp.array(
            [
                # Use the thresholded barycenter function as before
                compute_thresholded_barycenter(decoded_acts[:, i], self.model.xs)
                for i in range(len(onsets))
            ]
        )
        return decoded_locations

    def count_bumps(self: BaseSimulator, sol: jnp.ndarray) -> int:
        final_activity = self.model.population_activity(sol, self.model_params)[:, -1]

        # Alpha scaling
        alpha = self.model_params.alpha0 / self.model.n_xs

        n_bumps = count_bumps(
            final_activity, self.model.xs, alpha=alpha
        )
        return n_bumps
