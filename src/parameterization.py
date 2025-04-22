"""
The "raison-d'être" of this module is to treat parameters as data,
in order to easy generate interactive interfaces, run hyperparameter search, etc,
while staying compatible with Jax's JIT (through proper use of equinox.Module).

Don't touch the parameter_module logic if you don't understand it + JIT implications perfectly.
"""

from typing import Dict, TypedDict, Union
import equinox as eqx


class NumericParamSpec(TypedDict):
    type: type
    min: float
    max: float
    step: float
    value: float


class BoolParamSpec(TypedDict):
    type: type
    value: bool


ParamSpec = dict[str, Union[NumericParamSpec, BoolParamSpec]]

# TODO: assertions / other checks in order to ensure proper structure


class BaseParams(eqx.Module):
    @classmethod
    def from_spec(cls, param_spec: ParamSpec):
        return cls(**{name: info["value"] for name, info in param_spec.items()})

    def get_param_dict(self):
        return {name: getattr(self, name) for name in self.__annotations__}


def parameterized(param_spec: ParamSpec):
    def decorator(cls):
        name = cls.__name__
        attrs: dict = {
            "__annotations__": {name: spec["type"] for name, spec in param_spec.items()}
        }

        # Just set values, let Equinox handle field creation
        for name, spec in param_spec.items():
            attrs[name] = spec["value"]

        return type(name, (BaseParams,), attrs)

    return decorator


def spec_with_overrides(base_spec: ParamSpec, overrides: Dict) -> ParamSpec:
    """Create a new param spec with override values but keeping min/max/step etc."""
    new_spec = {}
    for name, value in overrides.items():
        spec = base_spec[name].copy()
        spec["value"] = value
        new_spec[name] = spec
    return new_spec


def get_default_values(param_spec: ParamSpec) -> dict:
    """Extract default values from a parameter specification"""
    return {name: info["value"] for name, info in param_spec.items()}


def static_params_from_spec(spec: ParamSpec) -> dict:
    return {name: info["value"] for name, info in spec.items()}
