"""Abstract simulator interface every backend implements.

The Gymnasium env only ever talks to this interface; adding a new flight
simulator means implementing this class, mapping the canonical
:class:`~aero_gym.variables.FlightVariable` set to the simulator's own
properties and units.
"""

from __future__ import annotations

import abc

from .scenario import InitialConditions
from .variables import FlightVariable


class SimulatorError(RuntimeError):
    """The backend failed to load a model or produced an invalid state."""


class Simulator(abc.ABC):
    """A flight dynamics backend driven at a fixed physics rate."""

    @property
    @abc.abstractmethod
    def dt(self) -> float:
        """Physics timestep in seconds."""

    @abc.abstractmethod
    def reset(
        self,
        aircraft: str,
        initial_conditions: InitialConditions,
        settle_steps: int = 1,
    ) -> None:
        """Load `aircraft`, apply `initial_conditions`, then settle.

        `settle_steps` physics steps are run before returning, because a
        backend may only recompute control-position properties inside run(),
        leaving them stale against a live command until one step has passed.
        The caller passes its own steps-per-agent-step, so the first
        observation of an episode reports settled positions.

        Takes the two values a backend actually needs rather than a Scenario,
        so no backend has to know what a task is.
        """

    @abc.abstractmethod
    def set_control(self, var: FlightVariable, value: float) -> None:
        """Apply a control input, canonical units and sign convention."""

    @abc.abstractmethod
    def get_control(self, var: FlightVariable) -> float:
        """Read back a control *command*, canonical units and sign convention.

        The exact inverse of :meth:`set_control`: ``set_control(v, x)`` then
        ``get_control(v)`` must return ``x``. Deliberately not the control's
        *position*, which on a trimmed aircraft is command plus trim; feeding
        a position back in as a command makes the backend add trim a second
        time.
        """

    @abc.abstractmethod
    def run(self) -> None:
        """Advance the simulation by one physics timestep."""

    @abc.abstractmethod
    def observe(self, var: FlightVariable) -> float:
        """Read one state variable, converted to canonical units."""

    def enable_visualization(self, **kwargs: object) -> None:
        """Start streaming to an external visualizer, if supported."""
        raise NotImplementedError(f"{type(self).__name__} has no visualization support")

    def close(self) -> None:
        """Release simulator resources."""
