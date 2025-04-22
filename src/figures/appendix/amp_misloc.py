import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List, Literal
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from src.figures.plotting import add_subplot_letter
from src.figures.base import BaseFigure
from src.figures.style import fig_width
from src.figures import plotting
from src.utils.eqx import update_with_overrides
from src.analytical import compute_analytical_mislocalization

MARKER_ALPHA = 0.5
LEGEND_FONTSIZE = 8

MISLOC_XLABEL = "Flash onset relative to saccade onset (ms)"
ETA_XLABEL = "Time relative to saccade onset (ms)"
MISLOC_YLABEL = r"Localization error ($\degree$)"
ETA_YLABEL = r"$\eta(t)$"
AMP_XLABEL = "Saccade amplitude (°)"
PEAK_YLABEL = r"Peak forward mislocalization ($\degree$)"

DURATION_KEYPOINTS = {
    "amplitudes": [5, 25],
    "durations": [30, 60]
}

ADD_DURATION_VLINES = False

class SaccadeAmpMislocFig(BaseFigure):
    def __init__(self, sim, param_sets: List[Dict], mode: Literal["analytical", "full"] = "analytical", **kw):
        self.param_sets = param_sets
        self.mode = mode
        self.flash_ts = jnp.linspace(-200, 200, 401)
        self.plot_ts = jnp.linspace(-200, 400, 1001)
        super().__init__(sim, **kw)

    def create_figure(self) -> Figure:
        fig = plt.figure(figsize=(fig_width, fig_width + 1)) # Square + legend
        fig.subplots_adjust(bottom=0.1)
        gs = GridSpec(2, 2, figure=fig,
                      wspace=0.3,
                      hspace=0.5,
                      width_ratios=[1, 1],
                      height_ratios=[1, 1])
        self.ax_misloc_base = fig.add_subplot(gs[0, 0])
        self.ax_misloc_tuned = fig.add_subplot(gs[0, 1])
        self.ax_eta = fig.add_subplot(gs[1, 0])
        self.ax_markers = fig.add_subplot(gs[1, 1])

        self.ax_eta.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        add_subplot_letter(fig, self.ax_misloc_base, 0)
        add_subplot_letter(fig, self.ax_misloc_tuned, 1)
        add_subplot_letter(fig, self.ax_eta, 2)
        add_subplot_letter(fig, self.ax_markers, 3)

        tick_every_n_deg = 5
        ticks = jnp.arange(-15, 15 + tick_every_n_deg, tick_every_n_deg)
        for ax in [self.ax_misloc_base, self.ax_misloc_tuned]:
            ax.set_xlabel(MISLOC_XLABEL, fontsize=LEGEND_FONTSIZE)
            ax.set_ylabel(MISLOC_YLABEL, fontsize=LEGEND_FONTSIZE)
            ax.set_yticks(ticks)
            plotting.sac_start(ax)
            plotting.hmark(ax, 0)

        self.ax_eta.set_xlabel(ETA_XLABEL, fontsize=LEGEND_FONTSIZE)
        self.ax_eta.set_ylabel(ETA_YLABEL, fontsize=LEGEND_FONTSIZE)
        plotting.sac_start(self.ax_eta)
        plotting.hmark(self.ax_eta, 0)

        self.ax_markers.set_xlabel(AMP_XLABEL, fontsize=LEGEND_FONTSIZE)
        self.ax_markers.set_ylabel(PEAK_YLABEL, fontsize=LEGEND_FONTSIZE)

        return fig

    def create_elements(self) -> Dict:
        elements = {
            "misloc_base": [],
            "misloc_tuned": [],
            "eta": [],
            "legend_handles": [],
            "misloc_base_marker": [],
            "misloc_tuned_marker": [],
            "misloc_base_line": None,
            "misloc_tuned_line": None
        }

        line_marker_base, = self.ax_markers.plot([], [], linestyle='--', color='black')
        line_marker_tuned, = self.ax_markers.plot([], [], linestyle='--', color='red')
        elements["misloc_base_line"] = line_marker_base
        elements["misloc_tuned_line"] = line_marker_tuned

        for i, pset in enumerate(self.param_sets):
            ampl = pset["amplitude"]
            color = f"C{i}"
            dur = self._compute_duration(ampl)
            label = f"{ampl}° ({dur:.0f} ms)"

            line_base = plotting.standard_plot(self.ax_misloc_base, color=color)
            line_tuned = plotting.standard_plot(self.ax_misloc_tuned, color=color)
            line_eta = plotting.standard_plot(self.ax_eta, color=color)

            elements["misloc_base"].append(line_base)
            elements["misloc_tuned"].append(line_tuned)
            elements["eta"].append(line_eta)
            elements["legend_handles"].append((line_eta, label))

            if ADD_DURATION_VLINES:
                for ax in [self.ax_misloc_base, self.ax_misloc_tuned]:
                    plotting.vmark(ax, dur, color=color, alpha=MARKER_ALPHA)

            marker_base, = self.ax_markers.plot([], [], marker='o', linestyle='None', color=color)
            marker_tuned, = self.ax_markers.plot([], [], marker='o', linestyle='None', color=color)
            elements["misloc_base_marker"].append(marker_base)
            elements["misloc_tuned_marker"].append(marker_tuned)

        elements["peak_line"] = plotting.prop_delay(self.ax_eta)

        handles, labels = zip(*elements["legend_handles"])
        self.fig.legend(handles, labels,
                        loc='lower center',
                        bbox_to_anchor=(0.5, 0.02),
                        ncol=len(self.param_sets),
                        fontsize=LEGEND_FONTSIZE,
                        borderaxespad=0,
                        bbox_transform=self.fig.transFigure)

        return elements

    def _compute_duration(self, amplitude: float) -> float:
        return float(jnp.interp(
            amplitude,
            xp=jnp.array(DURATION_KEYPOINTS["amplitudes"]),
            fp=jnp.array(DURATION_KEYPOINTS["durations"]),
            left="extrapolate",
            right="extrapolate"
        ))

    def _compute_misloc(self, model_params, exp_params) -> jnp.ndarray:
        if self.mode == "analytical":
            return compute_analytical_mislocalization(self.flash_ts, model_params, exp_params)
        decoded_loc = self.sim.decoded_retinal_locations_for_flash_onsets(
            self.flash_ts, model_params=model_params, exp_params=exp_params
        )
        reported_loc = decoded_loc + exp_params.saccade_amplitude
        return reported_loc - exp_params.flash_x

    def _update_data(self):
        base_marker_data = []
        tuned_marker_data = []

        def update_misloc(model_overrides, exp_overrides, element_key, idx):
            model_params = update_with_overrides(self.sim.model_params, model_overrides)
            exp_params = update_with_overrides(self.sim.exp_params, exp_overrides)
            misloc = self._compute_misloc(model_params, exp_params)
            self.elements[element_key][idx].set_data(self.flash_ts, misloc)
            return misloc, model_params, exp_params

        for i, pset in enumerate(self.param_sets):
            ampl = pset["amplitude"]
            dur = self._compute_duration(ampl)

            model_base = {"saccade_duration": dur}
            model_tuned = {**model_base, **(pset.get("eta", {}))}
            exp_base = {"saccade_amplitude": ampl}

            misloc_base, _, _ = update_misloc(model_base, exp_base, "misloc_base", i)
            misloc_tuned, model_params_b, _ = update_misloc(model_tuned, exp_base, "misloc_tuned", i)

            eta = self.sim.remapping_window(self.plot_ts, model_params_b)
            self.elements["eta"][i].set_data(self.plot_ts, eta)

            # Max FORWARD (positive), not absolute.
            peak_base = float(jnp.max(misloc_base))
            peak_tuned = float(jnp.max(misloc_tuned))
            self.elements["misloc_base_marker"][i].set_data([ampl], [peak_base])
            self.elements["misloc_tuned_marker"][i].set_data([ampl], [peak_tuned])

            base_marker_data.append((ampl, peak_base))
            tuned_marker_data.append((ampl, peak_tuned))

            if i == len(self.param_sets) - 1:
                pop_peak = model_params_b.retina_to_lip_delay + model_params_b.input_onset_to_peak
                self.elements["peak_line"].set_xdata([pop_peak, pop_peak])

        base_marker_data.sort(key=lambda tup: tup[0])
        tuned_marker_data.sort(key=lambda tup: tup[0])
        base_x, base_y = zip(*base_marker_data) if base_marker_data else ([], [])
        tuned_x, tuned_y = zip(*tuned_marker_data) if tuned_marker_data else ([], [])

        self.elements["misloc_base_line"].set_data(base_x, base_y)
        self.elements["misloc_tuned_line"].set_data(tuned_x, tuned_y)

    def get_parameters(self):
        return {
            "param_sets": self.param_sets,
            "duration_interpolation": DURATION_KEYPOINTS
        }

    def _update_view(self):
        super()._update_view()
        DYNAMIC = False

        if DYNAMIC:
            misloc_base_ymin, misloc_base_ymax = self.ax_misloc_base.get_ylim()
            misloc_tuned_ymin, misloc_tuned_ymax = self.ax_misloc_tuned.get_ylim()
            new_ymin = min(misloc_base_ymin, misloc_tuned_ymin)
            new_ymax = max(misloc_base_ymax, misloc_tuned_ymax)
            self.ax_misloc_tuned.set_ylim(new_ymin, new_ymax)
            self.ax_misloc_base.set_ylim(new_ymin, new_ymax)
        else:
            self.ax_misloc_base.set_ylim(-15, 15)
            self.ax_misloc_tuned.set_ylim(-15, 15)
            self.ax_markers.set_ylim(0, 15)
