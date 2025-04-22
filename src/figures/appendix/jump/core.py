"""
Core computations for jump model figures.
Shared between mislocalization and profile figures.
"""
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

@dataclass
class JumpParams:
    """Parameters for jump model responses."""
    # Current RF
    t1: float = 25        # Rise time
    s2: float = 40        # Decay SD
    baseline: float = 2   # Baseline
    amp: float = 20       # Peak amplitude

    # Remapped RF
    t1r: float = 25      # Rise time
    s2r: float = 75      # Decay SD
    baseliner: float = 0 # Baseline
    ampr: float = 15     # Peak amplitude

    # Other
    decaytime: float = 150   # Decay time constant
    decodetime: float = 150  # Default decode time

def alpha_gauss(t, t1, s2, baseline, amp, lat):
    """Combined alpha function (rise) and Gaussian decay."""
    r = np.zeros_like(t)

    # Alpha function part (rise)
    lt = (t <= (t1 + lat)) & (t > lat)
    uset = t[lt] - lat
    r[lt] = amp * uset / t1 * np.exp(1 - (uset / t1))

    # Gaussian part (decay)
    gt = t > (t1 + lat)
    uset = t[gt] - (t1 + lat)
    r[gt] = baseline + (amp - baseline) * np.exp(-0.5 * uset**2 / (2 * s2**2))

    return r

def decay(t, r, t1, tau):
    """Exponential decay after time t1."""
    rout = r.copy()
    uu = t >= t1
    vv = np.argmin(np.abs(t - t1))
    rout[uu] = r[vv] * np.exp(-(t[uu] - t1) / tau)
    return rout

def current_rf(t, lat, params, decodetime=None, decayloc=0):
    """Current RF response."""
    if decodetime is None:
        decodetime = params.decodetime

    r = alpha_gauss(t, params.t1, params.s2, params.baseline,
                   params.amp, lat)

    if decayloc >= 0:
        r = decay(t, r, decayloc, params.decaytime)

    if (lat + params.t1) > decodetime:
        decodetime = lat + params.t1 + 10

    k1ps = r[np.argmin(np.abs(t - decodetime))]
    return r, k1ps

def remap_rf(t, s1lat, params, lat, decodetime=None):
    """Remapped RF response."""
    if decodetime is None:
        decodetime = params.decodetime

    # Scale amplitude based on flash timing
    amprn = params.ampr * (1 - norm.cdf(lat, 200, 50))

    r = alpha_gauss(t, params.t1r, params.s2r, params.baseliner,
                   amprn, s1lat)

    if (s1lat + params.t1r) > decodetime:
        decodetime = s1lat + params.t1r + 10

    k2ps = r[np.argmin(np.abs(t - decodetime))]
    return r, k2ps

def two_norm(x, k, m, s):
    """Population response profile and barycenter."""
    y = k[0] * norm.pdf(x, m[0], s) + k[1] * norm.pdf(x, m[1], s)

    # Normalize by maximum
    y = y / np.max(y)

    # Compute barycenter
    ny = y / np.sum(y)
    baryc = np.sum(x * ny)

    return y, baryc

def compute_responses(t, x, flash_time, s1lat, decayloc, flash_pos,
                     params, decodetime=None):
    """Compute all responses for a given flash configuration."""
    current_r, k1ps = current_rf(t, flash_time, params, decodetime, decayloc)
    remap_r, k2ps = remap_rf(t, s1lat, params, flash_time, decodetime)
    pop_y, baryc = two_norm(x, [k1ps, k2ps], flash_pos, 2.5)

    return current_r, remap_r, pop_y, baryc
