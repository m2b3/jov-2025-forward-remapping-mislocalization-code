"""
DFT model definition. ODE system definition.
"""

import math

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid

from src.signals import eta

from .kernels import MexicanHatParams, w_1, w_2
from .params import DftDriftParams
from .space import make_xs


@jit
def activation(u, p: DftDriftParams):
    """
    Firing rate from membrane potential.
    """
    return sigmoid((u - p.act_offset) / p.act_slope)


# TODO: consider making this into an equinox Module
class DftDriftModel:
    params_class = DftDriftParams

    def __init__(
        self,
        n_xs,
        base_kernel_sigma,
        rel_inh_sigma,
        w_exc,
        w_inh,
        amplitude_coded_for,
        *args,
        **kwargs,
    ):
        self.n_xs = n_xs
        self.xs = make_xs(self.n_xs, amplitude_coded_for=amplitude_coded_for)

        sigma_exc = base_kernel_sigma * 1
        sigma_inh = base_kernel_sigma * rel_inh_sigma
        csp = MexicanHatParams(
            w_exc=w_exc,
            sigma_exc=sigma_exc,
            w_inh=w_inh,
            sigma_inh=sigma_inh,
        )

        delta_x = amplitude_coded_for / n_xs

        # There is no need to compute convolutions
        # with a kernel wider than its non-negligible support.
        n_ker_pts = 6 * (int(math.ceil(csp.sigma_inh) / delta_x)) + 1

        print(f"n_ker_pts: {n_ker_pts}")

        # To ensure symmetry, this must be odd.
        # Without this precaution, there will be a slight bias in one direction.
        assert n_ker_pts % 2 == 1, f"Kernel width must be odd, got {n_ker_pts}"
        x = jnp.linspace(-n_ker_pts * delta_x / 2, n_ker_pts * delta_x / 2, n_ker_pts)

        X, Y = jnp.meshgrid(x, jnp.array([0]))

        self.W1 = w_1(X, Y, csp).squeeze()
        self.W2 = w_2(X, Y, csp).squeeze()

        self.kernel_xs = x

    def population_dynamics(self, input_fn):
        @jit
        def dyn(t, state, args):
            # All user-defined model parameters.
            mp = args["model_params"]
            ep = args["exp_params"]  # TODO fix naming mismatch

            # Single population
            u = state

            # Following the DFT book's naming convention
            s = ep.input_strength * input_fn(t, mp, ep)

            # Alpha scaling with inverse of N neurons.
            alpha = mp.alpha0 / self.n_xs

            # Get CD signal for timestep.
            cur_eta = eta(
                t,
                tau_rise=mp.eta_tau_rise,
                tau_fall=mp.eta_tau_fall,
                duration=mp.eta_duration,
                center=mp.eta_center,
            )

            # Beware, slight difference between meaning of W2 in code
            # and in written equations (whether beta is absorbed into it)
            effective_W1 = self.W1
            beta = mp.beta0 * ep.saccade_amplitude
            effective_W2 = beta * cur_eta * self.W2
            W = effective_W1 + effective_W2

            # Nonlinearity.
            r = activation(u, mp)

            # XXX: check correctness of convolutions at the edges
            # fftconvolve is often slower than jnp.convolve.
            interaction = alpha * jnp.convolve(r, W, mode="same")

            # Final derivative computation.
            du = (-u + mp.h + s + interaction) / mp.tau
            return du

        return dyn

    def population_activity(self, u, p: DftDriftParams):  # type: ignore
        act = activation(u, p)
        return act
