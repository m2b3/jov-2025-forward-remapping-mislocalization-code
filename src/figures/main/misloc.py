from typing import cast
import itertools
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.constants import flash_onset_times
from src.digitized.honda import load_honda_data
from ..plotting import sac_start, sac_end, standard_plot
from ..style import fig_width, grid_axis_label_kw, saccade_start_vline_kw
from ..base import BaseFigure
from ..labels import report_horiz_pos_label, flash_onset_from_sac_onset_label


marker_width = 2 * 0.8
fat_line_width = 2 * 3

def compute_bootstrap_ci(x, y, n_bootstraps=1000, frac=0.15, it=3):
    """Compute bootstrap confidence intervals for LOWESS smoothed data.

    Args:
        x, y: Input data points
        n_bootstraps: Number of bootstrap samples (default 1000 for 95% CI)
        frac: LOWESS smoothing fraction (default 0.15)
        it: Number of LOWESS iterations (default 3)

    Returns:
        tuple: (lower, upper) 95% confidence interval bounds
    """
    bootstrap_curves = np.zeros((n_bootstraps, len(x)))
    indices = np.arange(len(x))

    for i in range(n_bootstraps):
        # Resample with replacement
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        boot_x, boot_y = x[boot_idx], y[boot_idx]

        # Compute LOWESS on bootstrap sample
        smooth = lowess(boot_y, boot_x, frac=frac, it=it, return_sorted=True)
        bootstrap_curves[i] = smooth[:, 1]

    # Compute confidence intervals
    ci_lower = np.percentile(bootstrap_curves, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_curves, 97.5, axis=0)

    return ci_lower, ci_upper

class MislocFig(BaseFigure):
    def __init__(self, sim, *, notable_only=True, show_legend=True, show_individual=True, show_confidence=True, **kw):
        self.show_legend = show_legend
        self.show_individual = show_individual
        self.show_confidence = show_confidence
        self.notable_only = notable_only
        self.onsets = flash_onset_times(notable_only=notable_only)
        self.series_data, (self.x_agg, self.y_agg) = load_honda_data()
        super().__init__(sim, **kw)

    def create_figure(self) -> Figure:
        fig = plt.figure(figsize=(6, 6))
        self.ax = fig.add_subplot(111)

        self.ax.set_xlabel(flash_onset_from_sac_onset_label, **grid_axis_label_kw)
        self.ax.set_ylabel(report_horiz_pos_label, **grid_axis_label_kw)
        self.ax.axhline(y=0, lw=marker_width, **saccade_start_vline_kw)
        sac_start(self.ax, lw=marker_width)

        return fig

    def create_elements(self) -> Dict:
        paper_color: Dict[str, str] = {}  # Map paper names to colors.

        # Get Matplotlib's default color cycle.
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle = itertools.cycle(default_colors)

        raw_lines = []

        if self.show_individual:
            for series in self.series_data:
                paper = cast(str, series.get('paper', '?'))
                extra_kw = dict()
                # Assign a new color, and label (ONLY ONCE) if we haven't seen this paper before.
                if paper not in paper_color:
                    paper_color[paper] = next(color_cycle)
                    extra_kw: dict = dict(label=paper)
                line = standard_plot(
                    self.ax,
                    color=paper_color[paper],
                    alpha=0.3,
                    marker='o',
                    markersize=3,
                    **extra_kw
                )
                line.set_data(series['x'], series['y'])
                raw_lines.append(line)

        if self.show_legend:
            self.ax.legend()


        # Overall smooth trend
        smooth = lowess(self.y_agg, self.x_agg, frac=0.15, it=3, return_sorted=True)
        smooth_line = standard_plot(self.ax, color='red', linewidth=fat_line_width)
        smooth_line.set_data(smooth[:,0], smooth[:,1])

        # Model prediction line and end marker
        model_line = standard_plot(self.ax, linewidth=fat_line_width)
        sac_end_line = sac_end(self.ax, lw=marker_width)

        # Compute confidence intervals
        ci_lower, ci_upper = compute_bootstrap_ci(self.x_agg, self.y_agg)

        # Plot confidence interval
        confidence_fill = self.ax.fill_between(
            smooth[:,0], ci_lower, ci_upper,
            color='red', alpha=0.2
        )

        if not self.show_confidence:
            confidence_fill.set_visible(False)


        # Make all spines thicker
        for spine in self.ax.spines.values():
            spine.set_linewidth(marker_width)


        return {
            "model_line": model_line,
            "raw_lines": raw_lines,
            "smooth_line": smooth_line,
            "sac_end": sac_end_line,
            "confidence_fill": confidence_fill
        }

    def _update_data(self):
        # Update model prediction
        barycenters = self.sim.decoded_retinal_locations_for_flash_onsets(self.onsets)
        reported_locations = barycenters + self.sim.exp_params.saccade_amplitude
        self.elements["model_line"].set_data(self.onsets, reported_locations)

        # Update saccade end marker
        saccade_duration = self.sim.model_params.saccade_duration
        self.elements["sac_end"].set_xdata([saccade_duration, saccade_duration])

    def _update_view(self):
        super()._update_view()
        self.ax.set_xlim(-250, 250)
        self.ax.set_ylim(-6, 6)

    def get_parameters(self):
        return {
            "notable_only": self.notable_only
        }
