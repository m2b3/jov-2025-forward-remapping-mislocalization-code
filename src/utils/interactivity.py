"""
Generic utilities for interactivity.
Keep clean of direct dependencies on model.
"""

import builtins
from typing import Dict, cast

import ipywidgets as widgets
from ipywidgets import (
    Checkbox,
    FloatSlider,
    IntSlider,
)

from src.parameterization import ParamSpec, NumericParamSpec


def widgets_from_spec(param_spec: ParamSpec) -> Dict[str, widgets.ValueWidget]:
    d = {}
    for name, info in param_spec.items():
        common = dict(
            value=info["value"],
            description=name,
            continuous_update=False,
            style={"description_width": "initial"},
        )
        match info["type"]:
            case builtins.float:
                info = cast(NumericParamSpec, info)
                w = FloatSlider(
                    min=info["min"],
                    max=info["max"],
                    step=info["step"],
                    readout_format=".4f",
                    **common,
                )
            case builtins.int:
                info = cast(NumericParamSpec, info)
                w = IntSlider(
                    min=info["min"], max=info["max"], step=info["step"], **common
                )
            case builtins.bool:
                w = Checkbox(**common)
            case _:
                raise ValueError(f"Unsupported type {info['type']}")

        d[name] = w
    return d
