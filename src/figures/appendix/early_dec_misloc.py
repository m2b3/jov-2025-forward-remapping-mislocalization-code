from src.figures.main.misloc import MislocFig
from src.utils.eqx import update_with_overrides
from src.figures.main.misloc import fat_line_width, standard_plot, sac_end

class EarlyDecMislocFig(MislocFig):
    def __init__(self, sim, **kw):
        # **Define the decoding times we want to show.**
        self.decoding_times = [150, 200, 350]
        super().__init__(sim, **kw)

    def create_elements(self) -> dict:
        """
        Create only the model prediction lines (one per decoding time)
        and a saccade end marker. Experimental data and its CI are omitted.
        """
        # Remove experimental data lines by not calling the parent's create_elements.
        # Instead, create our own collection of model lines.
        model_lines = {}  # maps decoding time -> line handle

        # For each decoding time, create a line with an interpolated color and a LaTeX label.
        linestyles = ['--', '-.', '-']
        for (idx, t) in enumerate(self.decoding_times):
            # 0 for first, 1 for last
            weight = (t - self.decoding_times[0]) / (self.decoding_times[-1] - self.decoding_times[0])
            # **Interpolate color** between gray and black (0, 0, 0)
            color = (
                0.75 * (1 - weight),
                0.75 * (1 - weight),
                0.75 * (1 - weight)
            )
            # **Create a line** using the standard_plot helper.
            # The label uses nice LaTeX formatting.
            line = standard_plot(
                self.ax,
                color=color,
                linewidth=fat_line_width,
                linestyle = linestyles[idx % len(linestyles)],
                label=fr"$t_{{dec}}={t}$ ms"
            )
            model_lines[t] = line

        # Create the saccade end marker (same as in the main figure)
        sac_end_line = sac_end(self.ax)

        # Add legend so the labels appear.
        self.ax.legend()

        return {
            "model_lines": model_lines,
            "sac_end": sac_end_line
        }

    def _update_data(self):
        """
        For each decoding time, update the corresponding model line.
        Uses the simulation’s onset times and the standard retinal decoding method.
        """
        # Loop over each decoding time and update its corresponding line.
        # Could be made faster with vectorization...
        for t, line in self.elements["model_lines"].items():
            # **Override model parameters** with the current decoding time.
            local_model_params = update_with_overrides(self.sim.model_params, {
                "decoding_time": t,
            })

            # Use the standard onset machinery (no filtering on max onset).
            onsets = self.onsets

            # Compute barycenters and then the reported locations.
            barycenters = self.sim.decoded_retinal_locations_for_flash_onsets(
                onsets,
                model_params=local_model_params
            )
            reported_locations = barycenters + self.sim.exp_params.saccade_amplitude

            line.set_data(onsets, reported_locations)

        # Update the saccade end marker as before.
        saccade_duration = self.sim.model_params.saccade_duration
        self.elements["sac_end"].set_xdata([saccade_duration, saccade_duration])

    def get_parameters(self) -> dict:
        """Return current visualization parameters."""
        return {
            'decoding_times': self.decoding_times
        }
