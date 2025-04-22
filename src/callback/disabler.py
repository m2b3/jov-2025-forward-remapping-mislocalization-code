from contextlib import ContextDecorator
from typing import TypeVar, Generic, Optional, Set
from enum import Enum

from .handle import CallbackHandle

T = TypeVar('T', bound=Enum)

class CallbackDisabler(Generic[T], ContextDecorator):
    """A context manager that temporarily disables callbacks on a simulator instance.

    Usage:
        with CallbackDisabler(simulator):
            # Do operations that should not trigger callbacks
            simulator.update_model(...)
    """

    def __init__(self, simulator):
        """
        Args:
            simulator: The simulator instance whose callbacks should be temporarily disabled.
                      Must be an instance of a class using CallbackMixin.
        """
        self.simulator = simulator
        self.stored_callbacks: Optional[Set[CallbackHandle[T]]] = None

    def __enter__(self):
        """Temporarily store and remove all callbacks from the simulator."""
        self.stored_callbacks = self.simulator._callbacks.copy()
        self.simulator._callbacks.clear()
        return self.simulator

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the previously stored callbacks."""
        if self.stored_callbacks is not None:
            self.simulator._callbacks = self.stored_callbacks
        # Don't suppress any exceptions
        return False
