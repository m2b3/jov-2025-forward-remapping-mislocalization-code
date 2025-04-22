from ipywidgets import VBox, Label
import ipywidgets
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec

from src.figures.base import BaseFigure
from .eta_misloc_row import EtaMislocRow
from src.figures.style import fig_width, common_gs_kw

from src.parameterization import ParamSpec
from src.utils.interactivity import widgets_from_spec
from src.dft.params import model_ps
from src.parameterization import spec_with_overrides


def format_params(params):
    """Format parameters for LaTeX display"""
    # XXX: Must keep this notation in sync with the paper's!
    return (
        f"$\\tau_{{\\eta}}^r={params['eta_tau_rise']}$ ms, "
        f"$\\tau_{{\\eta}}^f={params['eta_tau_fall']}$ ms\n"
        f"$\\Delta T_\\eta={params['eta_duration']}$ ms, "
        f"$T_\\eta^c={params['eta_center']}$ ms"
    )


class EtaMislocGridFig(BaseFigure):
    def __init__(self, sim, param_sets, notable_only=True, **kw):
        # For use in code.
        self.analytical_misloc_color = "darkviolet"
        # For use in paper.
        self.analytical_misloc_color_displayname = "purple"

        self.param_sets = param_sets
        self.notable_only = notable_only
        self.n_rows = len(param_sets)
        self.n_cols = 3
        self.labels = [format_params(params) for params in param_sets]

        super().__init__(sim, **kw)

    def get_parameters(self):
        return {"param_sets": self.param_sets, "notable_only": self.notable_only, "analytical_color": self.analytical_misloc_color_displayname}

    def create_figure(self):
        # 3x3, square
        fig = figure(figsize=(
            fig_width,
            # Shrink height a bit to keep subplots square,
            # due to extra spacing / labels
            fig_width * 0.9
        ))
        gs = GridSpec(
            self.n_rows, self.n_cols, figure=fig,
            **(common_gs_kw | {"wspace": 0.7})
        )

        self.rows = []
        for i, (params, label) in enumerate(zip(self.param_sets, self.labels)):
            row = EtaMislocRow(
                fig=fig,
                gs=gs,
                row_idx=i,
                n_rows=self.n_rows,
                analytical_misloc_color=self.analytical_misloc_color,
                label=label,
                label_start=i * self.n_cols,
                notable_only=self.notable_only,
            )
            self.rows.append(row)

        return fig

    def create_elements(self):
        return {f"row_{i}": row.create_elements() for i, row in enumerate(self.rows)}

    def _update_data(self):
        for i, (row, overrides) in enumerate(zip(self.rows, self.param_sets)):
            row._update_data(self.sim, self.elements[f"row_{i}"], overrides)

    def _update_view(self):
        # First collect all mislocalization data
        all_y = []
        for i, row in enumerate(self.rows):
            misloc_sim = self.elements[f"row_{i}"]["misloc_sim"]
            misloc_analytical = self.elements[f"row_{i}"]["misloc_analytical"]
            all_y.extend(misloc_sim.get_ydata())
            all_y.extend(misloc_analytical.get_ydata())

        # Compute limits, handling edge cases
        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            # Add small margin for visual clarity
            margin = 0.1 * (ymax - ymin)
            ymin, ymax = ymin - margin, ymax + margin

            # Apply limits to all rows
            for row in self.rows:
                row.set_misloc_limits(ymin, ymax)

        # Let rows update their other panels as usual
        for row in self.rows:
            row._update_view()

    def create_controls(self):
        row_controls = []

        def make_callback(index):
            def on_change(**values):
                if not self.active:
                    return
                row = self.rows[index]
                row._update_data(self.sim, self.elements[f"row_{index}"], values)
                row._update_view()

            return on_change

        for i, params in enumerate(self.param_sets):
            spec: ParamSpec = spec_with_overrides(model_ps, params)
            widgets = widgets_from_spec(spec)
            ctrl = ipywidgets.interactive(make_callback(i), **widgets)
            row_controls.append(VBox([Label(f"Row {i+1} Parameters:"), ctrl]))
        return VBox(row_controls)
