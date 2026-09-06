from __future__ import annotations

import pytest

from aero_gym.scenario import Scenario
from aero_gym.variables import (
    ACTION_SPECS,
    ALL_ACTION_VARIABLES,
    ALL_OBSERVATION_VARIABLES,
    OBSERVATION_SPECS,
    FlightVariable as V,
)
from conftest import constant_reward


def test_variable_registry_is_complete():
    assert set(ALL_ACTION_VARIABLES) == {V.ELEVATOR, V.AILERON, V.RUDDER, V.THROTTLE}
    assert len(ALL_OBSERVATION_VARIABLES) == 11
    for spec in list(ACTION_SPECS.values()) + list(OBSERVATION_SPECS.values()):
        assert spec.low < spec.high


def test_scenario_defaults(scenario):
    assert scenario.aircraft == "c172p"
    assert scenario.action_variables == ALL_ACTION_VARIABLES
    assert scenario.observation_variables == ALL_OBSERVATION_VARIABLES


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_variables": ()},
        {"observation_variables": ()},
        {"action_variables": (V.HEADING,)},
        {"observation_variables": (V.ELEVATOR,)},
        {"action_variables": (V.AILERON, V.AILERON)},
        {"agent_hz": 0.0},
    ],
)
def test_scenario_validation(initial_conditions, kwargs):
    with pytest.raises(ValueError):
        Scenario(
            name="bad",
            initial_conditions=initial_conditions,
            reward_fn=constant_reward,
            **kwargs,
        )
