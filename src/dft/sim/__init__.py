from typing import Optional
from .param_manager import ParamManagerMixin
from .core import CoreSimulationMixin
from .applications import ApplicationsMixin
from src.callback import CallbackMixin
from .base import ModelConstructionParams

from ..model import DftDriftModel
from ..params import SimParams, ExpParams, DftDriftParams
from src.parameterization import ParamSpec, static_params_from_spec


class Simulator(
    CallbackMixin,
    ParamManagerMixin,
    CoreSimulationMixin,
    ApplicationsMixin,
):
    """Main simulator class combining all mixins"""

    def __init__(
        self,
        model_params: DftDriftParams,
        model_constr_params: ModelConstructionParams,
        sim_params: Optional[SimParams] = None,
        exp_params: Optional[ExpParams] = None,
    ) -> None:
        super().__init__()
        self._recompile_needed = True

        # parameter initialization
        self.model_params = model_params
        self.sim_params = sim_params if sim_params is not None else SimParams()
        self.exp_params = exp_params if exp_params is not None else ExpParams()
        self.model_constr_params = model_constr_params

        # Model instantiation
        self.model = DftDriftModel(**model_constr_params)

        # Initial compilation
        self.recompile()

    @classmethod
    def from_param_specs(
        cls,
        model_ps: ParamSpec,
        static_ps: ParamSpec,
        sim_ps: ParamSpec,
        exp_ps: ParamSpec,
    ):
        return cls(
            model_params=DftDriftParams.from_spec(model_ps),
            model_constr_params=static_params_from_spec(static_ps),
            sim_params=SimParams.from_spec(sim_ps),
            exp_params=ExpParams.from_spec(exp_ps),
        )  # type: ignore

    def get_parameters(self) -> dict:
        """Get all simulator parameters as a dictionary."""
        extras = {
            "early_onset_time": -200,  # EARLY_ONSET_TIME constant
        }

        exp_params = self.exp_params.get_param_dict()
        m_params = self.model_params.get_param_dict()
        sim_params = self.sim_params.get_param_dict()
        mc_params = self.model_constr_params

        return exp_params | sim_params | mc_params | m_params | extras
