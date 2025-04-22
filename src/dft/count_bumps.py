import numpy as np
from scipy.signal import find_peaks


def count_bumps(act, xs, alpha, min_width_degrees=2):
    """
    If you show one flash and get two bumps, the model parameters are not in a reasonable configuration.
    This function can be used to automate crude model validation in certain
    cases - particularly in cases where the extraneous bump(s) would be out of
    the visible viewport in plots.
    """

    pt_per_deg = len(xs) / np.ptp(xs)
    min_width_points = int(min_width_degrees * pt_per_deg)

    act = np.array(act)

    # Sanity check for prominence
    assert np.max(act) <= alpha
    assert np.min(act) >= 0.0

    peaks, extras = find_peaks(
        act,
        # Min peak width
        # TODO should be derived from sigmas...
        width=min_width_points,
        # With a sigmoid WITH NO ALPHA, this is easy since values are in (0,1)...
        # but be when introducing alpha, must scale the interval!
        prominence=0.5 * alpha,
    )
    return len(peaks)
