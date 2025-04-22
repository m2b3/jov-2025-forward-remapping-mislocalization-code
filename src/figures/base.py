import ipywidgets as widgets
from typing import Optional, Set
import matplotlib.pyplot as plt

from src.dft.params import ChangeType
from src.dft.sim import Simulator
from src.utils.profile import profile


class BaseFigure:
    def __init__(
        self,
        sim: Simulator,
        dependencies: Optional[Set[ChangeType]] = None,
        active: bool = False,
    ):
        self.sim = sim
        self._active = active

        # Prevent automatic display in Jupyter - would get the figure twice otherwise.
        plt.ioff()
        self.fig = self.create_figure()
        plt.ion()

        self.elements = self.create_elements()

        # Store handle for cleanup
        self._callback_handle = sim.add_callback(self._on_sim_update, dependencies)

        # if active:
        #     self.update()

    def __del__(self):
        # Clean up callback
        if hasattr(self, "_callback_handle"):
            self.sim.remove_callback(self._callback_handle)

        if hasattr(self, 'fig'):
            plt.close(self.fig)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        was_active = self._active
        self._active = value
        if value and not was_active:
            self.update()

    def _on_sim_update(self):
        if self._active:
            self.update()

    def create_figure(self):
        raise NotImplementedError

    def create_elements(self):
        raise NotImplementedError

    def _update_data(self):
        raise NotImplementedError

    def _update_view(self):
        """
        Default behavior is to autoscale the axes and redraw the figure.
        """
        axes = self.fig.axes
        for ax in axes:
            ax.relim()
            ax.autoscale_view()

    def get_widget(self):
        """Return the figure as an IPython widget"""
        return self.fig.canvas

    def get_parameters(self):
        """Export parameters relevant to paper reproduction. Default is empty."""
        return {}

    def update(self):
        """Public update method - calls _update which is wrapped"""
        if self._active:
            name = self.__class__.__name__
            with profile(f"{name}._update_data"):
                self._update_data()
            with profile(f"{name}._update_view"):
                self._update_view()

    def create_controls(self) -> Optional[widgets.Widget]:
        """Override to provide figure-specific controls that appear when figure is selected"""
        return None
