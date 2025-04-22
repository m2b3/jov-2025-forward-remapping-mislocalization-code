"""
Mislocalization curve figure for jump model.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.figures.base import BaseFigure
from src.figures.style import fig_width, grid_axis_label_kw
from src.figures.plotting import standard_plot
from .core import JumpParams, compute_responses

class JumpMislocFig(BaseFigure):
    def __init__(self, sim, **kwargs):
        # Create latvals similar to MATLAB version
        self.latvals = np.concatenate([
            np.arange(-280, 0, 20, dtype=np.float32),
            np.arange(40, 281, 20, dtype=np.float32)
        ])

        # Computational axes
        self.t = np.linspace(-500, 500, 1000)
        self.x = np.linspace(-40, 40, 1000)

        # Model parameters
        self.params = JumpParams()

        super().__init__(sim, **kwargs)

    def create_figure(self) -> Figure:
        """Create figure with single axis."""
        fig = plt.figure(figsize=(fig_width, 0.7 * fig_width))
        self.ax = fig.add_subplot(111)

        # Configure axis
        self.ax.set_xlabel('Flash time from saccade onset (ms)',
                          **grid_axis_label_kw)
        self.ax.set_ylabel('Mislocalization (°)', **grid_axis_label_kw)

        # Add reference lines
        self.ax.axhline(y=0, color='k', linestyle=':')
        self.ax.axvline(x=0, color='k', linestyle=':')

        fig.tight_layout()
        return fig

    def create_elements(self):
        """Create plot elements."""
        return {
            'misloc_line': standard_plot(self.ax, color='k', linewidth=1.5),
            'misloc_points': standard_plot(self.ax, 'ko', markersize=3)
        }

    def _update_data(self):
        """Update data for mislocalization curve."""
        baryc = np.zeros_like(self.latvals)

        for i, lat in enumerate(self.latvals):
            # Determine parameters based on flash timing
            if lat > 60:
                s1lat = lat
            else:
                s1lat = 60

            if lat > 0:
                decayloc = lat + 200
                flash_pos = (0, -20)  # flash_x, other_x
            else:
                decayloc = 0
                flash_pos = (20, 0)

            # Compute responses and get barycenter
            _, _, pop_y, baryc[i] = compute_responses(
                self.t, self.x, lat, s1lat, decayloc, flash_pos, self.params
            )

        # Update plot elements
        self.elements['misloc_line'].set_data(self.latvals, baryc)
        self.elements['misloc_points'].set_data(self.latvals, baryc)

    def _update_view(self):
        super()._update_view()
        self.ax.set_xlim(-300, 300)
        self.ax.set_ylim(-15, 15)
