"""
This mixin bridges parameter update calls and the callback system,
while handling recompilation logic.
"""

import equinox as eqx
from typing import TypeVar, Union, Generic
from .base import BaseSimulator, ModelConstructionParams
from ..params import DftDriftParams, SimParams, ExpParams, ChangeType
from ..model import DftDriftModel

TSim = TypeVar("TSim", bound=BaseSimulator)


class ParamManagerMixin(Generic[TSim]):
    def __init__(self) -> None: ...  # Stub impl required by Python

    def update_model(self: TSim, **kwargs: ModelConstructionParams) -> None:
        """Update model construction parameters and recreate model."""
        self.model_constr_params = kwargs  # type: ignore
        self.model = DftDriftModel(**self.model_constr_params)
        self._recompile_needed = True
        self.recompile()
        self._notify(ChangeType.MODEL_PARAMS)

    def update_sim_params(self: TSim, **kwargs) -> None:
        self.sim_params = SimParams(**kwargs)
        self._recompile_needed = True
        self.recompile()
        self._notify(ChangeType.SIM_PARAMS)

    def update_model_params(self: TSim, params: Union[dict, DftDriftParams]) -> None:
        if isinstance(params, dict):
            new_params = DftDriftParams(**params)
        elif isinstance(params, DftDriftParams):
            new_params = params
        else:
            raise ValueError("params must be either a dict or DftDriftParams instance")

        self.model_params = eqx.tree_at(lambda x: x, self.model_params, new_params)
        self._notify(ChangeType.MODEL_PARAMS)

    def update_experiment(self: TSim, **kwargs) -> None:
        self.exp_params = ExpParams(**kwargs)
        self._notify(ChangeType.EXP_PARAMS)
