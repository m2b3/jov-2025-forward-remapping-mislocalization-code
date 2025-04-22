# New, shiny callback machinery. Very useful for life plots.
# Abstract, with little awareness of the rest of the codebase - except for ChangeType...


from enum import Enum
from typing import Callable, Set, Optional, TypeVar, Generic

T = TypeVar("T", bound=Enum)


class CallbackHandle(Generic[T]):
    def __init__(self, callback: Callable[[], None], dependencies: Optional[Set[T]]):
        self.callback = callback
        self.dependencies = dependencies

    def __call__(self, change_type: T) -> None:
        if self.dependencies is None or change_type in self.dependencies:
            self.callback()

    # No __del__ needed - we're just a data holder
