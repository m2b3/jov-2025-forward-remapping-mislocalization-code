from src.constants import flash_onset_times
from src.figures.plotting import add_subplot_label
from src.utils.profile import profile
from src.dft.params import DftDriftParams
from src.analytical import (
    compute_analytical_mislocalization,
    integrate_etas_future,
)

from typing import Dict, Optional, Union
from matplotlib.figure import Figure

from .. import plotting
from ..style import (
    grid_axis_label_kw,
)

from src.figures.labels import flash_onset_from_sac_onset_label
from src.dft.sim import Simulator
from src.utils.eqx import update_with_overrides
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from src.figures.labels import report_horiz_pos_label_short, time_from_sac_onset_label


# Make sure to keep math notation in sync with paper!
screen_onset_time_math = r't_\text{on}^\text{scr}'
int_subscript_math = screen_onset_time_math + r' + t_\text{crit}'
effective_integral_math = r'\int_{' + int_subscript_math + r'}^{t_\text{dec}} \eta(t) \text{d} t'
report_horiz_pos_math = r'x_\text{perc}^\text{scr}'

# Column 1
eta_XLABEL = time_from_sac_onset_label
eta_YLABEL = r"$\eta(t)$"

# Column 2
integral_XLABEL = f"${screen_onset_time_math}$ (ms)"
integral_YLABEL = f"${effective_integral_math}$"
integral_twin_YLABEL = 'Analytical $' + report_horiz_pos_math + '$ (°)'

# Column 3
misloc_XLABEL = "Flash onset\nrelative to saccade onset (ms)"
misloc_YLABEL = report_horiz_pos_label_short

class EtaMislocRow:
    """A row showing how eta transforms into mislocalization through integration."""

    def __init__(
        self,
        fig: Figure,
        gs: GridSpec,
        row_idx: int,
        n_rows: int,
        analytical_misloc_color: str = "darkviolet",
        label: Optional[str] = None,
        label_start: int = 0,
        notable_only: bool = True,
    ):
        self.analytical_misloc_color = analytical_misloc_color
        self.eta_ts = np.linspace(-300, 400, 1000)
        self.sim_onsets = flash_onset_times(notable_only=notable_only)
        self.fig = fig
        self.label_start = label_start

        # Create axes for this row
        self.ax_eta = fig.add_subplot(gs[row_idx, 0])
        self.ax_eta.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.ax_integral = fig.add_subplot(gs[row_idx, 1])
        self.ax_integral_misloc = self.ax_integral.twinx()  # Create twin axis
        self.ax_misloc = fig.add_subplot(gs[row_idx, 2])

        self.is_bottom = row_idx == n_rows - 1

        # Add row label if provided
        if label is not None:
            self.ax_eta.text(
                -0.575,
                0.5,
                label,
                rotation=90,
                transform=self.ax_eta.transAxes,
                va="center",
                ha="center",
            )

        self._setup_axes()

    def _setup_axes(self):
        """Configure all axes with proper labels and markers."""
        # Setup individual panels
        self._setup_eta_axis(self.ax_eta)
        self._setup_integral_axis(self.ax_integral)
        self._setup_misloc_axis(self.ax_misloc)

        # Add panel labels (A, B, C...)
        for i, ax in enumerate([self.ax_eta, self.ax_integral, self.ax_misloc]):
            label = chr(self.label_start + i + ord("A"))
            add_subplot_label(self.fig, ax, label)

        # Eta
        plotting.sac_start(self.ax_eta)

        # Integral
        plotting.sac_start(self.ax_integral)

        # Twin
        plotting.hmark(self.ax_integral_misloc, 0)

        # Misloc
        plotting.sac_start(self.ax_misloc)
        plotting.hmark(self.ax_misloc, 0)



    def _setup_eta_axis(self, ax: Axes):
        if self.is_bottom:
            ax.set_xlabel(eta_XLABEL, **grid_axis_label_kw)
        ax.set_ylabel(eta_YLABEL)

    def _setup_integral_axis(self, ax: Axes):
        if self.is_bottom:
            ax.set_xlabel(integral_XLABEL, **grid_axis_label_kw)
        ax.set_ylabel(integral_YLABEL)

        # Configure twin axis
        self.ax_integral_misloc.set_ylabel(integral_twin_YLABEL, color=self.analytical_misloc_color)
        self.ax_integral_misloc.tick_params(axis="y", labelcolor=self.analytical_misloc_color)

    def _setup_misloc_axis(self, ax: Axes):
        if self.is_bottom:
            ax.set_xlabel(misloc_XLABEL, **grid_axis_label_kw)
        ax.set_ylabel(misloc_YLABEL)

    def create_elements(self) -> Dict:
        """Create all plot elements for this row."""
        eta_line = plotting.standard_plot(self.ax_eta)
        integral_line = plotting.standard_plot(self.ax_integral)
        analytical_pattern = plotting.standard_plot(
            self.ax_integral_misloc, color=self.analytical_misloc_color
        )
        misloc_sim = plotting.standard_plot(self.ax_misloc, label="Simulated")
        misloc_analytical = plotting.standard_plot(
            self.ax_misloc,
            label="Analytical",
            # ls="--",
            color=self.analytical_misloc_color,
        )

        # Store lines marking saccade end for updating
        end_lines = []
        for ax in [
            # self.ax_eta,
            self.ax_integral,
            # self.ax_integral_twin,
            self.ax_misloc,
        ]:
            line = plotting.sac_end(ax)
            end_lines.append(line)

        # Store propagation delay lines for updating
        delay_lines = []
        for ax in [self.ax_eta]:
            line = plotting.prop_delay(ax)
            delay_lines.append(line)

        # **Add decoding time vertical marker.**
        # Note: do not use model parameters here; simply create a placeholder line.
        decoding_line = plotting.decoding_time(self.ax_eta)

        return {
            "eta": eta_line,
            "integral": integral_line,
            "analytical_pattern": analytical_pattern,
            "misloc_sim": misloc_sim,
            "misloc_analytical": misloc_analytical,
            "end_lines": end_lines,
            "delay_lines": delay_lines,
            "decoding_line": decoding_line,
        }

    def _update_eta(self, sim: Simulator, elements: Dict, mp: DftDriftParams):
        """Update eta panel with fills."""
        eta_sig = sim.remapping_window(self.eta_ts, mp)
        elements["eta"].set_data(self.eta_ts, eta_sig)

        # Update fills
        if hasattr(self, "_eta_fills"):
            for fill in self._eta_fills:
                fill.remove()

        pop_peak = mp.retina_to_lip_delay + mp.input_onset_to_peak
        self._eta_fills = plotting.fill_split_regions(
            self.ax_eta, self.eta_ts, eta_sig, split_x=pop_peak
        )

        # **Update the decoding time marker using the current model parameter.**
        decode_time = mp.decoding_time
        elements["decoding_line"].set_xdata([decode_time, decode_time])

    def _update_integral(self, sim: Simulator, elements: Dict, mp: DftDriftParams, ep):
        """Update remaining integral panel and analytical pattern using vectorized computation."""
        # Compute population times for all onsets at once
        pop_times = self.sim_onsets + mp.retina_to_lip_delay + mp.input_onset_to_peak

        # Compute all integrals in a single call
        remaining_integrals = integrate_etas_future(
            pop_times, mp.eta_tau_rise, mp.eta_tau_fall, mp.eta_duration, mp.eta_center, mp.decoding_time
        )
        elements["integral"].set_data(self.sim_onsets, remaining_integrals)

        # Compute analytical pattern (reusing computation from mislocalization)
        analytical_pattern = compute_analytical_mislocalization(self.sim_onsets, mp, ep)
        elements["analytical_pattern"].set_data(self.sim_onsets, analytical_pattern)

    def _update_misloc(self, sim: Simulator, elements: Dict, mp: DftDriftParams, ep):
        """Update mislocalization panel with both simulated and analytical curves."""
        decoded_ret = sim.decoded_retinal_locations_for_flash_onsets(
            self.sim_onsets, mp
        )
        reported_scr = decoded_ret + ep.saccade_amplitude
        elements["misloc_sim"].set_data(self.sim_onsets, reported_scr)

        # Update analytical line
        analytical_misloc = compute_analytical_mislocalization(self.sim_onsets, mp, ep)
        elements["misloc_analytical"].set_data(self.sim_onsets, analytical_misloc)

    def _update_data(
        self,
        sim: Simulator,
        elements: Dict,
        model_overrides: Optional[Union[Dict, DftDriftParams]] = None,
    ):
        """Update all panels with new data."""
        # Apply model parameter overrides
        mp = update_with_overrides(sim.model_params, model_overrides)
        ep = sim.exp_params

        # Update panels
        with profile("update eta"):
            self._update_eta(sim, elements, mp)

        with profile("update integral"):
            self._update_integral(sim, elements, mp, ep)

        with profile("update misloc"):
            self._update_misloc(sim, elements, mp, ep)

        # Update dynamic markers
        pop_peak = mp.retina_to_lip_delay + mp.input_onset_to_peak
        for lines, val in zip(
            [elements["end_lines"], elements["delay_lines"]],
            [mp.saccade_duration, pop_peak],
        ):
            for line in lines:
                line.set_xdata([val, val])

    def _update_view(self):
        """Update axis limits and view."""
        for ax in [
            self.ax_eta,
            self.ax_integral,
            self.ax_integral_misloc,
            self.ax_misloc,
        ]:
            ax.relim()
            ax.autoscale_view()


    def set_misloc_limits(self, ymin: float, ymax: float):
        self.ax_misloc.set_ylim(ymin, ymax)
        # self.ax_integral_misloc.set_ylim(ymin, ymax)
