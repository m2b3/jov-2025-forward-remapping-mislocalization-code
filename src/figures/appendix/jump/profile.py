"""
Profile figure showing jump model responses across time.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from typing import List, Dict
from src.figures.base import BaseFigure
from src.figures.style import (
    fig_width, common_gs_kw, grid_axis_label_kw
)
from src.figures.plotting import add_subplot_letter
from .core import JumpParams, compute_responses

class JumpProfileFig(BaseFigure):
    def __init__(self, sim, **kwargs):
        # Configuration for each row
        self.flash_times = [-260, -60, 100, 240]
        self.s1lats = [60, 60, 100, 240]
        self.decaylocs = [0, 0, 300, 440]
        self.flash_positions = [(20, 0), (20, 0), (0, -20), (0, -20)]

        # Row titles
        self.row_titles = [
            [r'$x=0\degree$',
             r'$x=-20\degree$'],
            [r'$x=0\degree$',
             r'$x=-20\degree$'],
            [r'$x=-20\degree$',
             r'$x=-40\degree$'],
            [r'$x=-20\degree$',
             r'$x=-40\degree$'],
        ]

        # Sampling
        self.t = np.linspace(-500, 500, 1000)
        self.x = np.linspace(-40, 40, 1000)

        # Parameters
        self.params = JumpParams()

        super().__init__(sim, **kwargs)

    def create_figure(self) -> Figure:
        """Create figure with 4x3 grid."""
        fig = plt.figure(figsize=(fig_width, 0.9 * fig_width * 4/3))

        # Make header row very small but not zero
        gs = GridSpec(
            4, 3,
            figure=fig,
            height_ratios=[1]*4,  # Super tiny header row
            **(common_gs_kw | {"wspace": 0.5, "hspace": 0.5})
        )

        # Header row with titles
        titles = ['Current RF', 'Remapped RF', 'Post-sac. pop. activity']

        # Create main axes
        self.axes: List[List[Axes]] = []
        for row in range(4):
            row_axes = []
            for col in range(3):
                ax = fig.add_subplot(gs[row, col])

                if row == 0:
                    ax.text(0.5, 1.3, titles[col],ha='center', va='center',  fontsize=12, transform=ax.transAxes)

                if col < 2:
                    ax.set_xlabel('Time from saccade onset (ms)',
                                **grid_axis_label_kw)
                    if col < len(self.row_titles[row]):
                        ax.set_title(self.row_titles[row][col], fontsize=10)
                else:
                    ax.set_xlabel('Pref. horiz. ret. pos. (°)',
                                **grid_axis_label_kw)
                    ax.set_ylabel('Norm. activity' if col == 2 else '')

                # Left-side onset label
                if col == 0:
                    ax.text(
                        -0.3, 0.5,
                        f"Onset = {self.flash_times[row]} ms",
                        rotation=90,
                        transform=ax.transAxes,
                        va="center",
                        ha="center",
                        fontsize=12
                    )

                add_subplot_letter(fig, ax, row * 3 + col)
                row_axes.append(ax)
            self.axes.append(row_axes)

        fig.tight_layout()
        return fig

    def create_elements(self) -> Dict:
        """Create plot elements."""
        elements = {
            'current_lines': [],
            'remap_lines': [],
            'pop_lines': [],
            'decode_lines': [],
            'zero_lines': [],
            'barycenter_lines': [],
        }

        for row in range(4):
            # Main response lines
            elements['current_lines'].append(
                self.axes[row][0].plot([], [], 'k-', lw=1.5)[0]
            )
            elements['remap_lines'].append(
                self.axes[row][1].plot([], [], 'k-', lw=1.5)[0]
            )
            elements['pop_lines'].append(
                self.axes[row][2].plot([], [], 'k-', lw=1.5)[0]
            )

            # Reference lines
            for col in range(2):
                decode_line = self.axes[row][col].axvline(
                    x=0, color='b', ls=':', visible=False
                )
                elements['decode_lines'].append(decode_line)

                zero_line = self.axes[row][col].axvline(
                    x=0, color='k', ls=':', visible=False
                )
                elements['zero_lines'].append(zero_line)

            barycenter_line = self.axes[row][2].axvline(
                x=0, color='k', ls=':', visible=False
            )
            elements['barycenter_lines'].append(barycenter_line)

        return elements

    def _update_data(self):
        """Update data for all rows."""
        for row in range(4):
            # Get configuration for this row
            flash_time = self.flash_times[row]
            s1lat = self.s1lats[row]
            decayloc = self.decaylocs[row]
            flash_pos = self.flash_positions[row]

            # Use later decode time for last row
            decodetime = 275 if row == 3 else 150

            # Compute responses
            current_r, remap_r, pop_y, baryc = compute_responses(
                self.t, self.x, flash_time, s1lat, decayloc,
                flash_pos, self.params, decodetime
            )

            # Update main response lines
            self.elements['current_lines'][row].set_data(self.t, current_r)
            self.elements['remap_lines'][row].set_data(self.t, remap_r)
            self.elements['pop_lines'][row].set_data(self.x, pop_y)

            # Update vertical lines for temporal plots
            for col in range(2):
                idx = row * 2 + col
                # Update decode time line
                self.elements['decode_lines'][idx].set_xdata([decodetime, decodetime])
                self.elements['decode_lines'][idx].set_visible(True)

                # Update zero time line
                self.elements['zero_lines'][idx].set_visible(True)

            # Update barycenter line
            self.elements['barycenter_lines'][row].set_xdata([baryc, baryc])
            self.elements['barycenter_lines'][row].set_visible(True)

    def _update_view(self):
        """Update view limits and formatting."""
        super()._update_view()

        # Set common y-limits for response columns
        ylims_current = [float('inf'), float('-inf')]
        ylims_remap = [float('inf'), float('-inf')]
        ylims_pop = [float('inf'), float('-inf')]

        for row in range(4):
            # Current RF
            ydata = self.elements['current_lines'][row].get_ydata()
            ylims_current[0] = min(ylims_current[0], ydata.min())
            ylims_current[1] = max(ylims_current[1], ydata.max())

            # Remapped RF
            ydata = self.elements['remap_lines'][row].get_ydata()
            ylims_remap[0] = min(ylims_remap[0], ydata.min())
            ylims_remap[1] = max(ylims_remap[1], ydata.max())

            # Population activity
            ydata = self.elements['pop_lines'][row].get_ydata()
            ylims_pop[0] = min(ylims_pop[0], ydata.min())
            ylims_pop[1] = max(ylims_pop[1], ydata.max())

        # Apply limits with margins
        margin = 0.1
        for row in range(4):
            self.axes[row][0].set_ylim(ylims_current[0] - margin,
                                        ylims_current[1] + margin)
            self.axes[row][1].set_ylim(ylims_remap[0] - margin,
                                        ylims_remap[1] + margin)
            self.axes[row][2].set_ylim(ylims_pop[0] - margin,
                                        ylims_pop[1] + margin)

            # Set common x-limits
            for col in range(2):
                self.axes[row][col].set_xlim(-500, 500)
            self.axes[row][2].set_xlim(-40, 40)
