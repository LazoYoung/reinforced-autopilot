from __future__ import annotations

import pytest

from aero_gym.scenario import InitialConditions, Scenario, Transition
from aero_gym.simulator import Simulator
from aero_gym.variables import FlightVariable as V


class FakeSimulator(Simulator):
    """Scripted backend for testing the env without any physics."""

    def __init__(self, dt: float = 1.0 / 120.0) -> None:
        self._dt = dt
        self.reset_calls: list[Scenario] = []
        self.controls: dict[V, float] = {}
        self.run_count = 0
        self.closed = False
        self.visualization_enabled = False
        self.state: dict[V, float] = {v: 0.0 for v in V}

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self, scenario: Scenario) -> None:
        self.reset_calls.append(scenario)
        self.state[V.ALTITUDE] = scenario.initial_conditions.altitude_ft
        self.state[V.HEADING] = scenario.initial_conditions.heading_deg

    def set_control(self, var: V, value: float) -> None:
        self.controls[var] = value

    def get_control(self, var: V) -> float:
        # A real backend reads back its own command property; the fake stores
        # commands in `controls`, so reading that back is the same inverse.
        # Defaults to 0.0 for a control never commanded, which is what a
        # freshly reset FDM reports too.
        return self.controls.get(var, 0.0)

    def run(self) -> None:
        self.run_count += 1

    def observe(self, var: V) -> float:
        return self.state[var]

    def enable_visualization(self, **kwargs: object) -> None:
        self.visualization_enabled = True

    def close(self) -> None:
        self.closed = True


def constant_reward(t: Transition) -> tuple[float, bool]:
    return 1.0, False


@pytest.fixture
def initial_conditions() -> InitialConditions:
    return InitialConditions(
        latitude_deg=37.6191,
        longitude_deg=-122.3816,
        altitude_ft=5000.0,
        heading_deg=270.0,
        airspeed_kts=95.0,
    )


@pytest.fixture
def scenario(initial_conditions: InitialConditions) -> Scenario:
    return Scenario(
        name="test",
        initial_conditions=initial_conditions,
        reward_fn=constant_reward,
        agent_hz=10.0,
        max_episode_steps=5,
    )


@pytest.fixture
def fake_simulator() -> FakeSimulator:
    return FakeSimulator()
