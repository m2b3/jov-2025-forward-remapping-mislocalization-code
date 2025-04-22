"""
Convolution kernels and associated parameters.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import jit


@jit
def gaussian(x, y, sigma):
    return jnp.exp(-(x**2 + y**2) / (2 * sigma**2))


class MexicanHatParams(eqx.Module):
    w_exc: float
    sigma_exc: float
    w_inh: float
    sigma_inh: float


@jit
def w_1(x, y, params: MexicanHatParams):
    exc = params.w_exc * gaussian(x, y, params.sigma_exc)
    inh = params.w_inh * gaussian(x, y, params.sigma_inh)

    # Derived symbolically.
    peak = params.w_exc - params.w_inh

    # Normalized kernel.
    return (exc - inh) / peak


def _derivative_wrt_x(x, y, sigma, w):
    return w * x * gaussian(x, y, sigma) / sigma**2


# We take w1 params because this is based on the derivative of w1.
@jit
def w_2(x, y, w1_p: MexicanHatParams):
    exc_derivative = _derivative_wrt_x(x, y, w1_p.sigma_exc, w1_p.w_exc)
    inh_derivative = _derivative_wrt_x(x, y, w1_p.sigma_inh, w1_p.w_inh)
    dw1 = inh_derivative - exc_derivative
    return dw1
