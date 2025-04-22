import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import (
    fig_width,
    grid_axis_label_kw,
)
from .labels import time_from_sac_onset_label
from . import plotting

from .base import BaseFigure

from src.constants import notable_onset_times


class BaseMislocFigure(BaseFigure):
    def __init__(self, sim, *, notable_only=True, **kw):
        self.notable_only = notable_only
        self.onsets = jnp.array(
            notable_onset_times if notable_only else jnp.linspace(-200, 200, 401)
        )
        super().__init__(sim, **kw)

    def create_figure(self) -> Figure:
        fig = plt.figure(figsize=(fig_width, 6))
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel(time_from_sac_onset_label, **grid_axis_label_kw)
        self.ax.set_ylabel("Reported horizontal screen position (°)")

        plotting.hmark(self.ax, 0)
        plotting.sac_start(self.ax)

        # FIXME: move away from base class, AND into update method to deal
        # with dynamic saccade duration.
        plotting.sac_end(self.ax, self.sim.model_params.saccade_duration)
        return fig
