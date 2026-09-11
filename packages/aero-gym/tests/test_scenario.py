from __future__ import annotations

import pytest

from aero_gym.scenario import Scenario, fixed_setup
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
    assert len(ALL_OBSERVATION_VARIABLES) == 15
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
            setup_fn=fixed_setup(initial_conditions),
            reward_fn=constant_reward,
            **kwargs,
        )


def test_features_default_to_a_passthrough_over_the_observed_variables(initial_conditions):
    from aero_gym.scenario import Scenario
    from aero_gym.variables import OBSERVATION_SPECS, FlightVariable as V

    scenario = Scenario(
        name="passthrough",
        setup_fn=fixed_setup(initial_conditions),
        reward_fn=lambda t: (0.0, False),
        observation_variables=(V.ROLL, V.ALTITUDE),
    )
    assert [f.name for f in scenario.observation_features] == ["roll", "altitude"]
    assert scenario.observation_features[1].spec == OBSERVATION_SPECS[V.ALTITUDE]
    assert scenario.observation_features[0].fn({V.ROLL: 12.0}, {}) == 12.0


def test_raw_narrows_bounds_without_touching_the_unit(initial_conditions):
    from aero_gym.scenario import raw
    from aero_gym.variables import FlightVariable as V

    feature = raw(V.ALTITUDE, -500.0, 500.0)
    assert (feature.spec.low, feature.spec.high) == (-500.0, 500.0)
    assert feature.spec.unit == "ft"


def test_duplicate_feature_names_are_rejected(initial_conditions):
    from aero_gym.scenario import Scenario, raw
    from aero_gym.variables import FlightVariable as V

    with pytest.raises(ValueError, match="duplicate observation feature"):
        Scenario(
            name="dup",
            setup_fn=fixed_setup(initial_conditions),
            reward_fn=lambda t: (0.0, False),
            observation_variables=(V.ROLL,),
            observation_features=(raw(V.ROLL), raw(V.ROLL)),
        )


def test_fixed_setup_returns_the_same_setup_for_any_generator():
    import numpy as np

    from aero_gym.scenario import InitialConditions, fixed_setup

    ic = InitialConditions(0.0, 0.0, 5000.0, 270.0)
    setup_fn = fixed_setup(ic, target_altitude_ft=5000.0)
    first = setup_fn(np.random.default_rng(0))
    second = setup_fn(np.random.default_rng(99))
    assert first == second
    assert first.initial_conditions is ic
    assert first.context == {"target_altitude_ft": 5000.0}
