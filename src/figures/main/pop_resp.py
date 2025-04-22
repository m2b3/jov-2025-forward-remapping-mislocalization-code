import equinox as eqx
import matplotlib.colors as mcolors
from typing import Any, Dict
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredOffsetbox, OffsetImage, TextArea

from src.utils.profile import profile
from .. import plotting
from ..plotting import heatmap, standard_plot, add_subplot_letter
from ..base import BaseFigure
from ..style import (
    fig_width,
    common_gs_kw,
    grid_axis_label_kw,
    heatmap_cm,
    barycenter_color,
    retinal_onset_col,
    saccade_start_col,
    saccade_end_col,
    decoding_color,
    decoded_loc_color
)
from ..labels import time_from_sac_onset_label
from src.dft.sim import Simulator
from src.dft.decoding import thresholded_barycenters_across_time, compute_thresholded_barycenter

TEXT_X_POS = 1.225



def up_to_1_dec_place(x: float) -> str:
    return f"{x:.1f}".rstrip('0').rstrip('.')

class PopRespFig(BaseFigure):
    def __init__(self, sim: Simulator, **kw):
        # These are the specific times we want to show.
        # FIXME: one should be right after saccade end -> dynamic.
        sac_dur = sim.model_params.saccade_duration
        self.onset_times = [-250, -1, sac_dur / 2, sac_dur, 250]
        self._init_markers()
        super().__init__(sim, **kw)

    def get_parameters(self):
        return {
            "onset_times": self.onset_times,
            # NOTE: decoded_locations are the results computed in _update_data.
            "decoded_locations": self.decoded_locations,
        }

    def _init_markers(self):
        """Initialize the marker symbols for annotations."""
        symbols = {
            "retinal_onset": ("img", "../img/flash.png", retinal_onset_col),
            "saccade_start": ("text", "S", saccade_start_col),
            "saccade_end": ("text", "E", saccade_end_col),
            "decoding_time": ("text", "D", decoding_color),
        }
        self.markers = {}
        for key, (typ, content, color) in symbols.items():
            if typ == "text":
                self.markers[key] = TextArea(
                    content, textprops=dict(color=color, fontweight="bold")
                )
            elif typ == "img":
                im = plt.imread(content)
                # Color override; leave alpha alone.
                im[:, :, :3] = mcolors.to_rgb(color)
                self.markers[key] = OffsetImage(im, zoom=0.01)
            else:
                raise ValueError(f"Unknown type {typ}")

    def create_figure(self) -> Figure:
        self.fig = plt.figure(figsize=(fig_width, 14))
        gs = GridSpec(
            6, 2, figure=self.fig, height_ratios=[0.1] + [1] * 5, **common_gs_kw
        )

        # Setup colorbars.
        self.cax_input = self.fig.add_subplot(gs[0, 0])
        self.cax_response = self.fig.add_subplot(gs[0, 1])
        self.sm_input = plt.cm.ScalarMappable(cmap=heatmap_cm)
        self.sm_response = plt.cm.ScalarMappable(cmap=heatmap_cm)
        self.cbar_input = self.fig.colorbar(
            self.sm_input, cax=self.cax_input, orientation="horizontal"
        )
        self.cbar_response = self.fig.colorbar(
            self.sm_response, cax=self.cax_response, orientation="horizontal"
        )
        # Adjust colorbar labels.
        self.cbar_input.ax.xaxis.set_ticks_position("bottom")
        self.cbar_response.ax.xaxis.set_ticks_position("bottom")

        # Add column titles.
        self._add_column_titles()

        # Create all row axes.
        self.row_axes = []
        for i, onset in enumerate(self.onset_times):
            ax_input = self.fig.add_subplot(gs[i + 1, 0])
            ax_response = self.fig.add_subplot(gs[i + 1, 1])
            self._setup_row_axes(ax_input, ax_response, onset, i)
            self.row_axes.append((ax_input, ax_response))

        self.fig.tight_layout()
        return self.fig

    def _add_column_titles(self):
        """Add titles above the colorbars."""
        title_style: Any = dict(x=0.5, y=2, ha="center", va="center", fontsize=12)
        self.cax_input.text(
            s="Input strength", transform=self.cax_input.transAxes, **title_style
        )
        self.cax_response.text(
            s="Response strength", transform=self.cax_response.transAxes, **title_style
        )

    def _setup_row_axes(self, ax_input, ax_response, onset, row_idx):
        """Configure a pair of axes for a single row."""
        for ax in [ax_input, ax_response]:
            ampl = self.sim.exp_params.saccade_amplitude
            ax.set_yticks([ampl, 0, -ampl, -2 * ampl])
            margin = 2
            ax.set_ylim(-2.5 * ampl, ampl + margin)  # XXX
            # Spatial markers.
            plotting.hmark(ax, 0)
            plotting.hmark(ax, -ampl)
            # Temporal markers.
            plotting.vmark(ax, 0)
            plotting.sac_start(ax)
            # FIXME: this should be updatable.
            plotting.sac_end(ax, self.sim.model_params.saccade_duration)
            # Only add x-label for bottom row.
            if row_idx == len(self.onset_times) - 1:
                ax.set_xlabel(time_from_sac_onset_label, **grid_axis_label_kw)
        # Add onset time label to left column only.


        ax_input.text(
            -0.175, 0.5,
            f"Onset = {up_to_1_dec_place(onset)} ms",
            rotation=90,
            transform=ax_input.transAxes,
            va="center",
            ha="center",
            fontsize=12,
        )
        # Add subplot labels.
        add_subplot_letter(self.fig, ax_input, 2 * row_idx)
        add_subplot_letter(self.fig, ax_response, 2 * row_idx + 1)

    def create_elements(self) -> Dict:
        """Create all plot elements that will be updated."""
        elements = {
            "heatmaps_input": [],
            "heatmaps_response": [],
            "barycenters": [],
            "retinal_onsets": [],
            "decoding_lines": [],
            # Mark the decoded LOCATION.
            "decoded_lines": [],
            "decoded_texts": [],
        }
        for i, (ax_input, ax_response) in enumerate(self.row_axes):
            # Heatmaps.
            im_input = heatmap(ax_input)
            elements["heatmaps_input"].append(im_input)
            im_response = heatmap(ax_response)
            elements["heatmaps_response"].append(im_response)
            # Barycenter line.
            barycenter_line = standard_plot(ax_response, color=barycenter_color)
            elements["barycenters"].append(barycenter_line)
            # Decoding time vertical line on response plot.
            dec_time_line = plotting.decoding_time(ax_response)
            elements["decoding_lines"].append(dec_time_line)
            # Onset vertical lines on both axes.
            for ax in [ax_input, ax_response]:
                retinal_line = plotting.vmark(ax, self.onset_times[i], color=retinal_onset_col)
                elements["retinal_onsets"].append(retinal_line)
            # --- Create horizontal dashed line for the decoded location ---
            decoded_line = plotting.decoded_location(ax_response)
            elements["decoded_lines"].append(decoded_line)
            # --- Create text label on the right-hand side, outside of the plot ---
            x_text = ax_response.get_xlim()[1] * TEXT_X_POS
            text_obj = ax_response.text(
                x_text, 0, "",
                color=decoded_loc_color,
                va="center", ha="right", fontsize=10, fontweight="bold",
                # transform=ax_response.transAxes,  # Use axes coords (0 to 1)
                clip_on=False
            )
            elements["decoded_texts"].append(text_obj)
        return elements

    def _update_markers(self, ax_idx):
        """Update marker positions for a given axis pair."""
        ax_input, ax_response = self.row_axes[ax_idx]
        onset = self.onset_times[ax_idx]
        for ax in [ax_input, ax_response]:
            for artist in ax.artists[:]:
                if isinstance(artist, AnchoredOffsetbox):
                    artist.remove()
        positions = {
            "retinal_onset": (onset, 1.15),
            "saccade_start": (0, 1.25),
            "saccade_end": (self.sim.model_params.saccade_duration, 1.25),
            "decoding_time": (self.sim.model_params.decoding_time, 1.25),
        }
        for key, marker in self.markers.items():
            x, y = positions[key]
            if key == "decoding_time":
                axes = [ax_response]
            else:
                axes = [ax_input, ax_response]
            for ax in axes:
                x_axes = (x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
                anchored = AnchoredOffsetbox(
                    loc="upper center",
                    child=marker,
                    bbox_to_anchor=(x_axes, y),
                    bbox_transform=ax.transAxes,
                    frameon=False,
                )
                ax.add_artist(anchored)

    def _update_data(self):
        """Update all plot elements with new data."""
        vmin_input, vmax_input = float("inf"), float("-inf")
        vmin_response, vmax_response = float("inf"), float("-inf")
        # Reset decoded_locations to avoid accumulation.
        self.decoded_locations = []
        with profile("simulate"):
            batched_ts, batched_ys = self.sim.simulate_flash_response(
                jnp.array(self.onset_times)
            )
        for i, onset in enumerate(self.onset_times):
            ts = batched_ts[i]
            ys = batched_ys[i]
            # Update input heatmap.
            exp_params = eqx.tree_at(lambda x: x.onset, self.sim.exp_params, onset)
            input_data = self.sim.vect_input_(ts, self.sim.model_params, exp_params).T
            vmin_input = min(vmin_input, input_data.min())
            vmax_input = max(vmax_input, input_data.max())
            self.elements["heatmaps_input"][i].set_data(input_data)
            extent = [ts[0], ts[-1], self.sim.model.xs[0], self.sim.model.xs[-1]]
            self.elements["heatmaps_input"][i].set_extent(extent)
            # Update response heatmap.
            response_data = self.sim.model.population_activity(ys.T, self.sim.model_params)
            vmin_response = min(vmin_response, response_data.min())
            vmax_response = max(vmax_response, response_data.max())
            self.elements["heatmaps_response"][i].set_data(response_data)
            self.elements["heatmaps_response"][i].set_extent(extent)
            # Update barycenter (already computed across time).
            barycenters = thresholded_barycenters_across_time(response_data, self.sim.model.xs)
            self.elements["barycenters"][i].set_data(ts, barycenters)
            # Slice out the decoded location from barycenters.
            decoding_time = self.sim.model_params.decoding_time
            idx = int(jnp.argmin(jnp.abs(ts - decoding_time)))
            decoded_loc = round(float(barycenters[idx]), 1)
            self.decoded_locations.append(decoded_loc)
            # Update decoded location horizontal marker and label.
            ax_response = self.row_axes[i][1]
            self.elements["decoded_lines"][i].set_ydata([decoded_loc, decoded_loc])
            x_text = ax_response.get_xlim()[1] * TEXT_X_POS
            self.elements["decoded_texts"][i].set_text(f"{decoded_loc:.1f}°")
            self.elements["decoded_texts"][i].set_position((x_text, decoded_loc))
            # Update markers.
            self._update_markers(i)
            # Update onset vertical lines.
            for j in (2 * i, (2 * i) + 1):
                line = self.elements["retinal_onsets"][j]
                line.set_xdata([onset])
            # Update decoding time vertical line.
            decode_time = self.sim.model_params.decoding_time
            line = self.elements["decoding_lines"][i]
            line.set_xdata([decode_time, decode_time])
        # Update color scales.
        for i in range(len(self.onset_times)):
            self.elements["heatmaps_input"][i].set_clim(vmin_input, vmax_input)
            self.elements["heatmaps_response"][i].set_clim(vmin_response, vmax_response)
        self.sm_input.set_clim(vmin_input, vmax_input)
        self.sm_response.set_clim(vmin_response, vmax_response)
        self.cbar_input.update_normal(self.sm_input)
        self.cbar_response.update_normal(self.sm_response)
