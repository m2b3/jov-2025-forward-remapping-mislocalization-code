from typing import Protocol, Optional, Any
import jax
from ..model import DftDriftModel
from ..params import DftDriftParams, SimParams, ExpParams, ChangeType

ModelConstructionParams = Any


# Core interface that all mixins assume exists
class BaseSimulator(Protocol):
    """Core interface required by all simulator mixins"""

    # Required state
    model_params: DftDriftParams
    sim_params: SimParams
    exp_params: ExpParams
    model_constr_params: ModelConstructionParams
    model: DftDriftModel
    _recompile_needed: bool

    # Core functionality that other mixins depend on
    def population_activity(self, u: jax.Array) -> jax.Array: ...
    def _notify(self, change_type: ChangeType) -> None: ...
    def vect_input(self, ts: jax.Array) -> jax.Array: ...
    def _create_solver(self) -> None: ...
    def _create_vmapped_solvers(self) -> None: ...
    def recompile(self) -> None: ...
    def simulate_flash_response(
        self,
        onsets: jax.Array | float,
        mp: Optional[DftDriftParams] = None,
        base_ep: Optional[ExpParams] = None,
    ) -> tuple[jax.Array, jax.Array]: ...
