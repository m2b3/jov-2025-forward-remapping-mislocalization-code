from enum import Enum
from typing import Callable, Set, Optional, TypeVar, Generic

from .handle import CallbackHandle

T = TypeVar("T", bound=Enum)


class CallbackMixin(Generic[T]):
    """
    Mixin class providing callback functionality to simulators.
    Must be used with a class that has appropriate parameter update methods.
    """

    def __init__(self):
        self._callbacks: Set[CallbackHandle[T]] = set()
        self._update_in_progress = False

    def add_callback(
        self, callback: Callable[[], None], dependencies: Optional[Set[T]] = None
    ) -> CallbackHandle[T]:
        handle = CallbackHandle(callback, dependencies)
        self._callbacks.add(handle)
        return handle

    def remove_callback(self, handle: CallbackHandle[T]) -> None:
        self._callbacks.discard(handle)

    def _notify(self, change_type: T) -> None:
        if self._update_in_progress:
            return

        self._update_in_progress = True
        try:
            # Create a frozen copy of the callbacks before iteration
            callbacks = tuple(self._callbacks)
            for handle in callbacks:
                handle(change_type)
        finally:
            self._update_in_progress = False
