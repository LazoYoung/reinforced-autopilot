"""Golden pin on steady_flight, recorded before the sandbox refactor began.

The numbers in this file are the observation vector and reward the trained
checkpoint was produced against. They must survive every step of the
migration: the feature names and their order fix the Box layout the policy's
first layer expects, and the bounds fix the affine map ScaleObservation
applies to it. A change to any of them silently invalidates runs/*/best.pt.

Only _feature_values() may be edited as feature signatures change. The
constants below must not be touched.
"""

from __future__ import annotations

import pytest

from aero_gym.scenario import Transition
from aero_gym.scenarios import make_steady_flight_scenario
from aero_gym.variables import FlightVariable as V

PROBE_OBS = {
    V.ALTITUDE: 5180.0,
    V.HEADING: 273.5,
    V.VERTICAL_SPEED: -240.0,
    V.ROLL: 6.0,
    V.PITCH: -2.5,
    V.AOA: 3.25,
    V.ROLL_RATE: -9.0,
    V.PITCH_RATE: 4.5,
    V.ELEVATOR_POS: 0.21,
    V.AILERON_POS: -0.06,
    V.RUDDER_POS: 0.02,
    V.THROTTLE_POS: 0.7,
}

PINNED_NAMES = [
    "altitude_error_ft",
    "heading_error_deg",
    "vertical_speed",
    "roll",
    "pitch",
    "aoa",
    "roll_rate",
    "pitch_rate",
    "elevator_pos",
    "aileron_pos",
    "rudder_pos",
    "throttle_pos",
]

PINNED_BOUNDS = [
    (-500.0, 500.0),
    (-10.0, 10.0),
    (-1000.0, 1000.0),
    (-75.0, 75.0),
    (-45.0, 45.0),
    (-20.0, 20.0),
    (-60.0, 60.0),
    (-30.0, 30.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1.0),
]

PINNED_VALUES = [
    180.0, -3.5, -240.0, 6.0, -2.5, 3.25, -9.0, 4.5, 0.21, -0.06, 0.02, 0.7,
]

PINNED_REWARD = 0.6833333333333333


def _scenario():
    return make_steady_flight_scenario()


def _context(scenario):
    """The episode context the features are evaluated against.

    Empty today: the targets are still captured at construction time. Task 5
    replaces this with the sampled context, and nothing else in this file moves.
    """
    return {}


def _feature_values(scenario, obs):
    """The one place a feature-signature change is allowed to land."""
    return [float(f.fn(obs)) for f in scenario.observation_features]


def _reward(scenario, obs):
    return scenario.reward_fn(Transition(prev_obs=obs, obs=obs, action={}))


def test_the_feature_names_and_their_order_are_unchanged():
    """Order is the Box layout: a permutation feeds every input to the wrong
    weight column while every other assertion in this file still passes."""
    assert [f.name for f in _scenario().observation_features] == PINNED_NAMES


def test_the_feature_bounds_are_unchanged():
    """ScaleObservation maps these onto [-1, 1], so a changed bound rescales
    every input the policy was trained on."""
    scenario = _scenario()
    assert [(f.spec.low, f.spec.high) for f in scenario.observation_features] == PINNED_BOUNDS


def test_the_feature_values_are_unchanged():
    scenario = _scenario()
    assert _feature_values(scenario, PROBE_OBS) == pytest.approx(PINNED_VALUES)


def test_the_reward_is_unchanged():
    scenario = _scenario()
    reward, terminated = _reward(scenario, PROBE_OBS)
    assert reward == pytest.approx(PINNED_REWARD)
    assert not terminated


def test_the_action_and_observation_variables_are_unchanged():
    """The action Box layout, which DeltaAction's Discrete(81) is built from."""
    scenario = _scenario()
    assert [v.name for v in scenario.action_variables] == [
        "ELEVATOR", "AILERON", "RUDDER", "THROTTLE",
    ]
    assert [v.name for v in scenario.observation_variables] == [
        "ALTITUDE", "HEADING", "VERTICAL_SPEED", "ROLL", "PITCH", "AOA",
        "ROLL_RATE", "PITCH_RATE", "ELEVATOR_POS", "AILERON_POS",
        "RUDDER_POS", "THROTTLE_POS",
    ]


def test_the_default_flight_condition_is_unchanged():
    scenario = _scenario()
    ic = scenario.initial_conditions
    assert (ic.latitude_deg, ic.longitude_deg) == (37.6191, -122.3816)
    assert (ic.altitude_ft, ic.heading_deg, ic.airspeed_kts) == (5000.0, 270.0, 95.0)
    assert (ic.pitch_deg, ic.roll_deg) == (0.0, 0.0)
    assert scenario.aircraft == "c172p"
    assert scenario.agent_hz == 10.0
    assert scenario.max_episode_steps == 2000
