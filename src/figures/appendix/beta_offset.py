import equinox as eqx
import jax.numpy as jnp
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import contextlib

from .. import plotting
from ..style import fig_width
from ..base import BaseFigure
from src.figures.plotting import standard_plot
from src.utils.profile import profile
from src.callback.disabler import CallbackDisabler

@contextlib.contextmanager
def temporary_n_xs(sim, n_xs: int):
    """
    Temporarily modify simulator's n_xs, restoring original value after.
    Temporarily disables callbacks to prevent spurious updates.
    """
    original_params = sim.model_constr_params.copy()
    with CallbackDisabler(sim):
        try:
            sim.update_model(**{**original_params, 'n_xs': n_xs})
            yield
        finally:
            sim.update_model(**original_params)

class BetaOffsetFig(BaseFigure):
    """Figure showing how drift amount varies with beta for different spatial resolutions."""

    def __init__(self, sim, n_xs_values: Optional[List[int]] = None, individual_dots=False, **kw):
        self.n_xs_values = n_xs_values or [101, 201, 401]
        self.individual_dots = individual_dots
        self.early_onset_time = -300
        self.n_betas = 201
        self.beta_extent = 100
        super().__init__(sim, **kw)

    def create_figure(self) -> Figure:
        fig = plt.figure(figsize=(fig_width, 6))
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel(r"$\beta$")
        self.ax.set_ylabel(r"Response peak offset after remapping (°)")
        return fig

    def create_elements(self) -> Dict:
        """Create plot elements for each resolution."""
        elements = {
            'offset_lines': {},
            'markers': []
        }

        # Create a line for each resolution
        for i, n_xs in enumerate(self.n_xs_values):
            kw = {'individual': True, 'markersize': 5} if self.individual_dots else {}
            line = standard_plot(
                self.ax,
                label=f'$N$ = {n_xs}',
                color=f'C{i}',
                **kw
            )
            elements['offset_lines'][n_xs] = line

        elements['markers'].extend([
            plotting.vmark(self.ax, 0),
            plotting.hmark(self.ax, 0)
        ])

        self.ax.legend(framealpha=0.9)
        return elements

    def _compute_single_resolution(self, n_xs: int, base_params) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute offsets for a single resolution value using parameter overrides."""
        with profile(f"compute_resolution_n{n_xs}"):
            betas = jnp.linspace(-self.beta_extent, self.beta_extent, self.n_betas)
            ep = eqx.tree_at(lambda x: x.onset, self.sim.exp_params, self.early_onset_time)

            # Temporarily switch to this resolution
            with temporary_n_xs(self.sim, n_xs):
                with profile("compute_offsets"):
                    offsets = self.sim.final_pos_for_betas(betas, base_params, ep)

            return betas, offsets

    def _update_data(self):
        """Update data for all resolutions."""
        # Get current base parameters from simulator
        base_params = self.sim.model_params

        # Store parameter sets for get_parameters
        self.last_param_sets = []

        with profile("update_data"):
            # Compute results for each resolution
            for n_xs in self.n_xs_values:
                betas, offsets = self._compute_single_resolution(n_xs, base_params)

                # Update line data
                line = self.elements['offset_lines'][n_xs]
                line.set_data(betas, offsets)

                # Store full parameter set
                self.last_param_sets.append({
                    'n_xs': n_xs,
                })

            # Update legend with new labels
            self.ax.legend(framealpha=0.9)

    def get_parameters(self):
        """Return current visualization parameters."""
        return {
            'early_onset_time': self.early_onset_time,
            'n_xs_values': self.n_xs_values,
            'param_sets': self.last_param_sets
        }
