from typing import Any
import ipywidgets as widgets
from ipywidgets import Button, VBox, HBox, Dropdown, Checkbox, Layout

from ..params import sim_ps
from src.figures.base import BaseFigure
from src.dft.sim import Simulator
from src.parameterization import ParamSpec
from src.utils.interactivity import widgets_from_spec

__all__ = ["make_controls"]


def make_controls(
    sim: Simulator,
    figs: dict[str, BaseFigure],
    model_ps: ParamSpec,
    static_model_ps: ParamSpec,
    exp_ps: ParamSpec,
) -> widgets.HBox:
    # Create all parameter widgets
    sim_widgets = widgets_from_spec(sim_ps)
    static_widgets = widgets_from_spec(static_model_ps)
    model_widgets = widgets_from_spec(model_ps)
    exp_widgets = widgets_from_spec(exp_ps)

    # Create preset buttons
    def on_lightweight_click(_ignore):
        static_widgets["n_xs"].value = 201
        model_widgets["beta0"].value = 20.0811

    def on_full_click(_ignore):
        static_widgets["n_xs"].value = 1001
        model_widgets["beta0"].value = 20.0430

    button_layout = Layout(width="auto")
    lightweight_button = Button(
        description="Apply lightweight-mode override", layout=button_layout
    )
    lightweight_button.on_click(on_lightweight_click)
    full_button = Button(description="Apply full-mode override", layout=button_layout)
    full_button.on_click(on_full_click)
    buttons_box = VBox([lightweight_button, full_button])

    # Create parameter control sections
    control_sections: Any = {
        "Model params": VBox(
            [
                widgets.interactive(
                    lambda **kw: sim.update_model_params(kw), **model_widgets
                ),
            ]
        ),
        "Experiment params": VBox(
            [
                widgets.interactive(sim.update_experiment, **exp_widgets),
            ]
        ),
        "Static model params": VBox(
            [
                widgets.interactive(sim.update_model, **static_widgets),
            ]
        ),
        "ODE Solver": VBox(
            [
                widgets.interactive(sim.update_sim_params, **sim_widgets),
            ]
        ),
    }

    # Create control section dropdown
    control_dropdown = Dropdown(
        options=list(control_sections.keys()),
        value=list(control_sections.keys())[0],
        description="Parameters:",
        style={"description_width": "initial"},
    )

    # Create control containers and set initial visibility
    control_containers = []
    for section in control_sections.values():
        container = VBox([section])
        container.layout.display = "none"
        control_containers.append(container)
    control_containers[0].layout.display = "block"  # Show first section

    def update_control_display(change):
        selected_name = change.new
        selected_idx = list(control_sections.keys()).index(selected_name)
        for i, container in enumerate(control_containers):
            container.layout.display = "block" if i == selected_idx else "none"

    control_dropdown.observe(update_control_display, names="value")

    # Create figure controls
    figure_checkboxes = {}
    for fig_name, fig in figs.items():
        checkbox = Checkbox(
            value=fig.active,
            description=fig_name,
            indent=False,
        )
        checkbox.observe(
            lambda change, f=fig: setattr(f, "active", change.new), names="value"
        )
        figure_checkboxes[fig_name] = checkbox


    # Create figure containers with their custom controls
    figure_containers = []
    for fig in figs.values():
        controls = fig.create_controls()
        container = (
            HBox([fig.fig.canvas, controls]) if controls else HBox([fig.fig.canvas])
        )
        container.layout.display = "none"
        figure_containers.append(container)

    active_figures = [i for i, fig in enumerate(figs.values()) if fig.active]
    print("af", active_figures)
    idx_to_show = 0
    if active_figures:
        idx_to_show = active_figures[0]

    print(f"Enabling figure {idx_to_show}")
    figure_containers[idx_to_show].layout.display = "block"  # Show first figure if no active figure

    # Create figure dropdown
    figure_dropdown = Dropdown(
        options=list(figs.keys()),
        value=list(figs.keys())[idx_to_show],
        description="Select figure:",
        style={"description_width": "initial"},
    )

    def update_figure_display(change):
        selected_name = change.new
        selected_idx = list(figs.keys()).index(selected_name)
        for i, container in enumerate(figure_containers):
            container.layout.display = "block" if i == selected_idx else "none"
        figs[selected_name].active = True
        figure_checkboxes[selected_name].value = True

    figure_dropdown.observe(update_figure_display, names="value")

    # Create the final layouts
    figure_panel = VBox([VBox(figure_containers)])

    controls_panel = HBox(
        [
            VBox(
                [
                    control_dropdown,
                    VBox(control_containers),
                    buttons_box,
                ]
            ),
            VBox(
                [
                    figure_dropdown,
                    VBox(list(figure_checkboxes.values())),
                ]
            ),
        ]
    )

    return HBox([figure_panel, controls_panel])
