from jax.typing import ArrayLike
from functools import partial
import jax.numpy as jnp
import jax
import equinox as eqx
from typing import TypeVar, Optional, Union, Generic, Callable
from ..model import activation
from ..decoding import compute_thresholded_barycenter
from .base import BaseSimulator
from ..input import ff_input
from ..ode import make_base_solver
from src.signals.eta import eta
from ..params import DftDriftParams, ExpParams

TSim = TypeVar("TSim", bound=BaseSimulator)


# Now make the mixin generic over types implementing SimulatorProtocol
class CoreSimulationMixin(Generic[TSim]):
    vect_input_: Callable
    ff_input: Callable
    base_solve: Callable

    # Parameter sets
    vsim_exp: Callable
    vsim_model: Callable

    # Specialized
    vsim_onsets: Callable
    vsim_rf_mapping: Callable

    sim_beta: Callable
    vsim_betas: Callable

    final_pos_for_beta: Callable
    final_pos_for_betas: Callable

    def recompile(self) -> None:
        if self._recompile_needed:
            print("Recompiling")
            # Must be set BEFORE recompiling solvers!
            self.ff_input = partial(ff_input, xs=self.model.xs)  # type: ignore
            # Create vectorized input
            self.vect_input_ = jax.vmap(  # type: ignore
                self.ff_input,
                in_axes=(0, None, None),
            )
            self._create_solvers()
            self._recompile_needed = False
        else:
            print("Skipping recompilation")

    def _create_solvers(self) -> None:
        xs = self.model.xs  # type: ignore

        base_solve = make_base_solver(
            self.model,  # type: ignore
            self.ff_input,
            self.sim_params,  # type: ignore
        )
        self.base_solve = base_solve

        # Parameter-set-level
        self.vsim_exp = jax.vmap(base_solve, in_axes=(None, 0))
        self.vsim_model = jax.vmap(base_solve, in_axes=(0, None))

        # Specialized versions
        def sim_onset(onset, mp, base_ep):
            ep = eqx.tree_at(lambda p: p.onset, base_ep, onset)
            return base_solve(mp, ep)

        self.vsim_onsets = jax.vmap(sim_onset, in_axes=(0, None, None))  # type: ignore

        def sim_flash_pos(position, mp, base_ep):
            ep = eqx.tree_at(lambda p: p.flash_x, base_ep, position)
            return base_solve(mp, ep)

        def sim_flash_time_pos(time, positions, mp, base_ep):
            ep = eqx.tree_at(lambda p: p.onset, base_ep, time)
            sim_positions = jax.vmap(sim_flash_pos, in_axes=(0, None, None))
            return sim_positions(positions, mp, ep)

        self.vsim_rf_mapping = jax.vmap(
            sim_flash_time_pos, in_axes=(0, None, None, None)
        )

        @jax.jit
        def sim_beta(beta, base_mp, base_ep):
            # Set beta0 to beta and amplitude to 1.
            mp = eqx.tree_at(lambda p: p.beta0, base_mp, beta)
            ep = eqx.tree_at(lambda p: p.saccade_amplitude, base_ep, 1.0)
            return self.base_solve(mp, ep)

        self.sim_beta = sim_beta
        self.vsim_betas = jax.vmap(sim_beta, in_axes=(0, None, None))

        # Beware - what is the onset here?
        def final_pos_for_beta(beta, mp, ep):
            sol = sim_beta(beta, mp, ep)
            _ts, ys = sol.ts, sol.ys
            acts = activation(ys.T, mp)
            final_r = acts[:, -1]
            return compute_thresholded_barycenter(final_r, xs)

        self.final_pos_for_beta = jax.jit(final_pos_for_beta)
        self.final_pos_for_betas = jax.vmap(final_pos_for_beta, in_axes=(0, None, None))

    def simulate_flash_response(
        self: TSim,
        onsets: Union[float, jnp.ndarray],
        mp: Optional[DftDriftParams] = None,
        base_ep: Optional[ExpParams] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if mp is None:
            mp = self.model_params
        if base_ep is None:
            base_ep = self.exp_params
        if jnp.isscalar(onsets):
            ep = eqx.tree_at(lambda p: p.onset, base_ep, onsets)
            sol = self.base_solve(mp, ep)  # type: ignore
        else:
            onsets = jnp.array(onsets)
            sol = self.vsim_onsets(onsets, mp, base_ep)  # type: ignore
        return sol.ts, sol.ys

    def remapping_window(self: TSim, t: ArrayLike, model_params=None) -> float:
        if model_params is None:
            model_params = self.model_params

        return eta(
            t,
            tau_rise=model_params.eta_tau_rise,
            tau_fall=model_params.eta_tau_fall,
            duration=model_params.eta_duration,
            center=model_params.eta_center,
        )

    def vect_input(self: TSim, ts: jnp.ndarray) -> jnp.ndarray:
        return self.vect_input_(ts, self.model_params, self.exp_params).T  # type: ignore

    def population_activity(self: TSim, u: jnp.ndarray) -> jnp.ndarray:
        return self.model.population_activity(u, self.model_params)
