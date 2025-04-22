
import warnings
from matplotlib.image import imread
from src.utils.typst import compile_task_diagram

from src.figures.plotting import add_subplot_letter
from src.signals import spatial_tuning_curve
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

from src.signals.input.temporal import alpha_gaussian_exp
from src.signals import eye_position
from .. import plotting
from ..base import BaseFigure
from ..style import (
    fig_width,
    common_gs_kw,
    grid_axis_label_kw,
)
from ..labels import time_from_sac_onset_label, time_from_lip_onset_label
from src.dft.sim import Simulator
import numpy as np


class ComponentsFig(BaseFigure):
    def __init__(self, sim: Simulator, active=True, **kw):
        # CD window always skewed to the right.
        t_min = -150
        t_max = 400
        n_eta_pts = 1000

        # Shorter than main t_max to get a clearer view
        # of the shape.
        t_max_input = 150

        self.eye_pos_ts = np.linspace(t_min, t_max, 100)
        self.eta_win_ts = np.linspace(t_min, t_max, n_eta_pts)
        self.input_ts = np.linspace(-10, t_max_input, 501)
        self.input_xs = np.linspace(-15, 15, 100)
        super().__init__(sim, active=active, **kw)

    def create_figure(self) -> Figure:
        fig = plt.figure(figsize=(fig_width, 12))
        self.gs = GridSpec(nrows=3, ncols=2, figure=fig, **common_gs_kw)

        # Create all subplots
        i, j = 0, 0

        def next_subplot():
            nonlocal i, j
            ax = fig.add_subplot(self.gs[i, j])
            subplot_idx = i * 2 + j
            add_subplot_letter(fig, ax, subplot_idx)
            j += 1
            if j == 2:
                j = 0
                i += 1
            return ax

        # Create all subplot axes
        self.ax_typst = next_subplot()  # New first subplot for SVG
        self.ax_eye_pos = next_subplot()
        self.ax_eta_window = next_subplot()
        self.ax_input_temporal = next_subplot()
        self.ax_input_spatial = next_subplot()
        self.ax_kernels = next_subplot()

        # Configure axes
        self.ax_typst.axis('off')

        self.ax_eye_pos.set_xlabel(time_from_sac_onset_label, **grid_axis_label_kw)
        self.ax_eye_pos.set_ylabel("Horiz. eye pos. on screen (°)")

        self.ax_eta_window.set_xlabel(time_from_sac_onset_label, **grid_axis_label_kw)
        self.ax_eta_window.set_ylabel(r"$\eta(t)$")
        self.ax_eta_window.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self._setup_input_temporal_axis()

        self.ax_input_spatial.set_xlabel("Offset (°)", **grid_axis_label_kw)
        self.ax_input_spatial.set_ylabel(r"$I_{\text{space}}(x)$")

        self.ax_kernels.set_xlabel("Offset (°)", **grid_axis_label_kw)
        self.ax_kernels.set_ylabel("Connection strength")

        # Do tight_layout BEFORE adding our image
        fig.tight_layout()

        return fig

    def _setup_input_temporal_axis(self):
        self.ax_input_temporal.set_xlabel(
            "Time relative to flash onset (ms)", **grid_axis_label_kw
        )
        self.ax_input_temporal.set_ylabel(r"$I_{\text{time}}(t)$")
        # Add twin axis for LIP onset time reference
        self.shifted_input_ax = self.ax_input_temporal.twiny()
        self.shifted_input_ax.set_xlabel(
            time_from_lip_onset_label, **grid_axis_label_kw
        )
        self.shifted_input_ax.spines["top"].set_position(("outward", 0))
        self.shifted_input_ax.spines["top"].set_visible(True)


    def create_elements(self) -> Dict:
        # Main lines
        eye_pos_line = plotting.standard_plot(self.ax_eye_pos)
        eta_line = plotting.standard_plot(self.ax_eta_window)
        input_temporal_line = plotting.standard_plot(self.ax_input_temporal)
        input_spatial_line = plotting.standard_plot(self.ax_input_spatial)
        kernel_w1_line = plotting.standard_plot(self.ax_kernels, color='k', label=r'$W_1$')
        kernel_w2_line = plotting.standard_plot(self.ax_kernels, color='r', label=r'$W_2$')
        self.ax_kernels.legend()

        self._plot_task_diagram()

        # Not updated
        plotting.sac_start(self.ax_eye_pos)
        plotting.sac_start(self.ax_eta_window)

        # Updated
        self.sac_end_eye = plotting.sac_end(self.ax_eye_pos)
        self.sac_end_cd = plotting.sac_end(self.ax_eta_window)

        self.onset_plus_delay_cd = plotting.prop_delay(self.ax_eta_window)
        self.onset_plus_delay_inp = plotting.prop_delay(self.ax_input_temporal)

        self.decoding_time_eta = plotting.decoding_time(self.ax_eta_window)

        # Create markers (S is static, E needs updating)
        for ax in [self.ax_eye_pos, self.ax_eta_window]:
            plotting.sac_start_marker(ax)

        # Store E markers for updating
        self.e_markers = {
            "eye_pos": plotting.sac_end_marker(self.ax_eye_pos),
            "eta": plotting.sac_end_marker(self.ax_eta_window),
        }

        # Store P (Peak) marker for updating
        # self.p_marker_cd = plotting.prop_delay_marker(self.ax_eta_window)

        return {
            "eye_pos": eye_pos_line,
            "eta": eta_line,
            "input_temporal": input_temporal_line,
            "input_spatial": input_spatial_line,
            "kernel_w1": kernel_w1_line,
            "kernel_w2": kernel_w2_line,
        }

    def _plot_task_diagram(self):
        # XXX(perf): saccade amplitude is a dynamic param, but we only compile the diagram on creation.
        # This is not terrible though - we might lose too much speed otherwise.
        # Probably not worth the effort.
        if not compile_task_diagram(self.sim.exp_params.saccade_amplitude):
            warnings.warn("required binaries not found in PATH, cannot freshly-compile task diagram")

        img = imread("../img/task_diagram.png", format='png')

        # Get original position - it already has correct margins!
        orig_pos = self.ax_typst.get_position()

        img_height, img_width = img.shape[:2]
        img_aspect = img_height / img_width

        occupied_width_ratio = 1
        new_width = orig_pos.width * occupied_width_ratio
        new_height = new_width * img_aspect

        vertical_overflow = 0.10
        new_bbox = [
            orig_pos.x0,           # Original x (with margins)
            # orig_pos.y1 - 0.25,           # Original top
            orig_pos.y0 - vertical_overflow,
            new_width,             # Same width
            new_height            # Height from aspect ratio
        ]

        # Remove original axes
        self.ax_typst.remove()

        # Create new axes
        overflow_ax = self.fig.add_axes(new_bbox)
        overflow_ax.imshow(img, clip_on=False)
        overflow_ax.axis("off")

    def _update_data(self):
        mp = self.sim.model_params
        ep = self.sim.exp_params

        # Saccade end annotation
        dur = mp.saccade_duration
        self.sac_end_eye.set_xdata([dur, dur])
        self.sac_end_cd.set_xdata([dur, dur])

        # Update E marker positions
        for marker in self.e_markers.values():
            marker.set_x(mp.saccade_duration)

        # LIP peak for a flash occurring exactly at saccade onset, relative to saccade onset.
        lip_peak_delay = mp.retina_to_lip_delay + mp.input_onset_to_peak

        for line in [self.onset_plus_delay_cd, self.onset_plus_delay_inp]:
            line.set_xdata([lip_peak_delay, lip_peak_delay])

        decoding_time = mp.decoding_time
        self.decoding_time_eta.set_xdata([decoding_time, decoding_time])
        # self.p_marker_cd.set_x(lip_peak_delay)

        # Update eye position
        eye_pos = np.array(
            eye_position(
                self.eye_pos_ts,
                saccade_duration=mp.saccade_duration,
                saccade_amplitude=ep.saccade_amplitude,
            )
        )
        self.elements["eye_pos"].set_data(self.eye_pos_ts, eye_pos)

        # Update CD window
        eta_sig = self.sim.remapping_window(self.eta_win_ts)
        self.elements["eta"].set_data(self.eta_win_ts, eta_sig)

        # Update input temporal
        temporal = alpha_gaussian_exp(
            self.input_ts - mp.retina_to_lip_delay,
            width=ep.flash_duration,
            onset_to_peak=mp.input_onset_to_peak,
            sigma=mp.input_temporal_sigma,
            tau_end=mp.input_offset_tau,
            baseline=mp.input_stable_baseline,
            amplitude=1.0,
        )
        self.elements["input_temporal"].set_data(self.input_ts, temporal)

        # Update input spatial
        spatial = spatial_tuning_curve(
            self.input_xs, center_loc=0, std=mp.input_spatial_std
        )
        self.elements["input_spatial"].set_data(self.input_xs, spatial)

        # Update kernels
        self.elements["kernel_w1"].set_data(self.sim.model.kernel_xs, self.sim.model.W1)
        self.elements["kernel_w2"].set_data(self.sim.model.kernel_xs, self.sim.model.W2)

    def _update_view(self):
        # Mostly delegated.
        super()._update_view()

        ep = self.sim.exp_params
        mp = self.sim.model_params

        # Static limits for eye position.
        self.ax_eye_pos.set_ylim(-5, 5 + ep.saccade_amplitude)

        # Special handling for twin axis
        sac_xlim = self.ax_input_temporal.get_xlim()
        shifted_xlim = tuple(x - mp.retina_to_lip_delay for x in sac_xlim)
        self.shifted_input_ax.set_xlim(*shifted_xlim)
