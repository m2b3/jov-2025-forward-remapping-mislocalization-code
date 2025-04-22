# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.insert(0, "..")  # Add the project root to the path

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %pwd

# %%
# %matplotlib widget

# %%
PAPER_READY = True
FORCE_EXPORT = True

# %%
SHOULD_EXPORT = PAPER_READY or FORCE_EXPORT

# %%
import ipywidgets as widgets
from ipywidgets.widgets import HBox, VBox, Dropdown
from IPython.display import display
import jax
import jax.numpy as jnp

from src.dft.sim import Simulator
from src.figures.base import BaseFigure


from src.dft.params import (
    perisaccadic_flash_ps,
    sim_ps,
    model_ps,
    static_model_ps,
    DftDriftParams,
    ChangeType
)
from src.dft.interactivity import (
    make_controls
)
from src.utils.interactivity import widgets_from_spec
from src.utils.export import export_all_parameters, export_all_figures, get_output_dir
from src.utils.profile import print_spans
from typing import Dict, List


sim = Simulator.from_param_specs(
    model_ps=model_ps,
    static_ps=static_model_ps,
    sim_ps=sim_ps,
    exp_ps=perisaccadic_flash_ps
)


# %%
# Main
from src.figures.main.components import ComponentsFig
from src.figures.main.pop_resp import PopRespFig
from src.figures.main.misloc import MislocFig

# Appendix
from src.figures.appendix.eta_misloc_grid import EtaMislocGridFig
from src.figures.appendix.amp_misloc import SaccadeAmpMislocFig
from src.figures.appendix.beta_offset import BetaOffsetFig
from src.figures.appendix.early_dec_misloc import EarlyDecMislocFig
from src.figures.appendix.jump.profile import JumpProfileFig
from src.figures.appendix.jump.misloc import JumpMislocFig

eta_param_sets = [
    # Base
    {
        "eta_tau_rise": 15,
        "eta_tau_fall": 50,
        "eta_duration": 240,
        "eta_center": 75,
    },
    {
        "eta_tau_rise": 50,
        "eta_tau_fall": 40,
        "eta_duration": 100,
        "eta_center": 0,
    },
    {
        "eta_tau_rise": 5,
        "eta_tau_fall": 20,
        "eta_duration": 220,
        "eta_center": 100,
    }
]

ampl_eta_mapping = [
    {
        "amplitude": 9,
        "eta": {
            "eta_tau_rise": 15,
            "eta_tau_fall": 50,
            "eta_duration": 240,
            "eta_center": 75,
        }
    },
    {
        "amplitude": 14,
        "eta": {
            "eta_tau_rise": 10,
            "eta_tau_fall": 40,
            "eta_duration": 200,
            "eta_center": 75,
        }
    },
    {
        "amplitude": 27,
        "eta": {
            "eta_tau_rise": 10,
            "eta_tau_fall": 30,
            "eta_duration": 150,
            "eta_center": 70,
        }
    },
    {
        "amplitude": 35,
        "eta": {
            "eta_tau_rise": 10,
            "eta_tau_fall": 20,
            "eta_duration": 130,
            "eta_center": 70,
        }
    },
]

beta_offset_n_xs = [101, 201] + ([] if not PAPER_READY else [1001])

main_figs: Dict[str, BaseFigure] = {
    "model_components": ComponentsFig(sim),
    "pop_resp": PopRespFig(sim, active=True),
    "misloc": MislocFig(sim, notable_only=(not PAPER_READY)),
}

appendix_figs: Dict[str, BaseFigure] = {
    "eta_misloc": EtaMislocGridFig(sim, eta_param_sets, notable_only=(not PAPER_READY)),
    "ampl_misloc": SaccadeAmpMislocFig(sim, active=False, param_sets=ampl_eta_mapping, mode="analytical"),
    "early_dec_misloc": EarlyDecMislocFig(sim),
    "beta_offset": BetaOffsetFig(sim, active=False, n_xs_values=beta_offset_n_xs),
}

figs = dict(
    **main_figs,
    **appendix_figs,
)

# %%
controls = make_controls(
    sim=sim,
    figs=figs,
    model_ps=model_ps,
    static_model_ps=static_model_ps,
    exp_ps=perisaccadic_flash_ps
)
display(controls)

# %%
if SHOULD_EXPORT:
    for fig in figs.values():
        fig.active = True

# %%
print_spans()

# %%
if not SHOULD_EXPORT:
    raise # halt nb before export when experimenting

# %%
out_dir = get_output_dir()
export_all_parameters(sim, figs, out_dir=out_dir)
export_all_figures(figs, out_dir=out_dir)

# %%
DURATION_KEYPOINTS = {
    "amplitudes": [5, 25],  # degrees
    "durations": [30, 60]   # ms
}

import jax.numpy as jnp
def _compute_duration(amplitude: float) -> float:
    return float(jnp.interp(
        amplitude,
        xp=jnp.array(DURATION_KEYPOINTS["amplitudes"]),
        fp=jnp.array(DURATION_KEYPOINTS["durations"]),
        left="extrapolate",
        right="extrapolate"
    ))


_compute_duration(8)

# %%
