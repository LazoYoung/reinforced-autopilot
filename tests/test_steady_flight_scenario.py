from __future__ import annotations

import dataclasses
from dataclasses import fields

import numpy as np
import pytest

from aero_gym.scenario import Transition
from aero_gym.variables import FlightVariable as V
from sandbox.scenarios.steady_flight import (
    SteadyFlightParams,
    heading_error_deg,
    make_steady_flight_scenario,
)


def test_the_flight_condition_follows_the_aircraft():
    """5000 ft at 95 kts is a Cessna cruise; a 737 would stall there."""
    ic = _setup(make_steady_flight_scenario(SteadyFlightParams(aircraft="737"))).initial_conditions
    assert ic.altitude_ft == 30000.0
    assert ic.airspeed_kts == 280.0


def test_c172p_keeps_the_flight_condition_it_always_had():
    ic = _setup(make_steady_flight_scenario()).initial_conditions
    assert ic.altitude_ft == 5000.0
    assert ic.airspeed_kts == 95.0


def test_an_explicit_flight_condition_wins_over_the_aircrafts():
    ic = _setup(
        make_steady_flight_scenario(
            SteadyFlightParams(aircraft="737", altitude_ft=12000.0, airspeed_kts=250.0)
        )
    ).initial_conditions
    assert ic.altitude_ft == 12000.0
    assert ic.airspeed_kts == 250.0


def test_one_overridden_value_does_not_drag_the_other_along():
    ic = _setup(
        make_steady_flight_scenario(SteadyFlightParams(aircraft="737", altitude_ft=12000.0))
    ).initial_conditions
    assert ic.altitude_ft == 12000.0
    assert ic.airspeed_kts == 280.0


def test_a_tweaked_band_reaches_the_bounds_the_reward_and_the_terminator():
    """One knob, three places. A params field that only reached the Box would
    leave the reward normalizing against the old number."""
    scenario = make_steady_flight_scenario(SteadyFlightParams(altitude_band_ft=250.0))
    bounds = {f.name: (f.spec.low, f.spec.high) for f in scenario.observation_features}
    assert bounds["altitude_error_ft"] == (-250.0, 250.0)
    # 5250 saturates only the altitude term (250/250 = 1.0); the averaged
    # reward is 1 - 1/3, not 0 (0 needs all three terms at their corner).
    reward, terminated = _evaluate(scenario, {V.ALTITUDE: 5250.0})
    assert reward == pytest.approx(2.0 / 3.0)
    assert not terminated
    _, terminated = _evaluate(scenario, {V.ALTITUDE: 5251.0})
    assert terminated


def test_zero_jitter_gives_the_same_start_every_episode():
    """The default, and the thing that keeps runs/*/best.pt comparable."""
    scenario = make_steady_flight_scenario()
    rng = np.random.default_rng(0)
    setups = [scenario.setup_fn(rng) for _ in range(5)]
    assert {s.initial_conditions.altitude_ft for s in setups} == {5000.0}
    assert {s.initial_conditions.heading_deg for s in setups} == {270.0}


def test_jitter_draws_a_different_start_each_episode():
    scenario = make_steady_flight_scenario(
        SteadyFlightParams(altitude_jitter_ft=300.0, heading_jitter_deg=15.0)
    )
    rng = np.random.default_rng(0)
    setups = [scenario.setup_fn(rng) for _ in range(5)]
    altitudes = [s.initial_conditions.altitude_ft for s in setups]
    assert len(set(altitudes)) == 5
    assert all(4700.0 <= a <= 5300.0 for a in altitudes)


def test_the_same_seed_draws_the_same_sequence_of_starts():
    """Both directions. Without the different-seed half a constant sampler
    passes this and every environment flies the identical episode."""
    scenario = make_steady_flight_scenario(SteadyFlightParams(altitude_jitter_ft=300.0))

    def draw(seed):
        rng = np.random.default_rng(seed)
        return [scenario.setup_fn(rng).initial_conditions.altitude_ft for _ in range(4)]

    assert draw(0) == draw(0)
    assert draw(0) != draw(1)


def test_the_target_follows_the_jittered_start_so_it_stays_a_hold_task():
    """A jittered start against a fixed target would silently turn a hold task
    into a go-to task with a band far too narrow to be reachable."""
    scenario = make_steady_flight_scenario(SteadyFlightParams(altitude_jitter_ft=300.0))
    setup = scenario.setup_fn(np.random.default_rng(0))
    assert setup.context["target_altitude_ft"] == setup.initial_conditions.altitude_ft


def test_a_negative_jitter_is_rejected():
    with pytest.raises(ValueError, match="altitude_jitter_ft"):
        SteadyFlightParams(altitude_jitter_ft=-10.0)


def test_a_non_positive_band_is_rejected():
    """The bands are divisors in reward_fn; zero would produce inf or NaN."""
    with pytest.raises(ValueError, match="altitude_band_ft"):
        SteadyFlightParams(altitude_band_ft=0.0)


def test_max_episode_steps_is_not_a_scenario_parameter():
    """Every scenario has it, so repeating it in every params dataclass is
    boilerplate; the factory applies it with dataclasses.replace instead."""
    assert "max_episode_steps" not in {f.name for f in fields(SteadyFlightParams)}
    scenario = dataclasses.replace(make_steady_flight_scenario(), max_episode_steps=7)
    assert scenario.max_episode_steps == 7
    assert len(scenario.observation_features) == 12


def _setup(scenario):
    return scenario.setup_fn(np.random.default_rng(0))


def _obs(scenario, overrides=None):
    setup = _setup(scenario)
    obs = {v: 0.0 for v in scenario.observation_variables}
    obs[V.ALTITUDE] = setup.initial_conditions.altitude_ft
    obs[V.HEADING] = setup.initial_conditions.heading_deg
    obs.update(overrides or {})
    return obs


def _evaluate(scenario, overrides=None):
    # A plain dict, not **kwargs: the keys are FlightVariable members, and
    # CPython's call-unpacking requires str keys whatever the callee accepts.
    obs = _obs(scenario, overrides)
    return scenario.reward_fn(
        Transition(prev_obs=obs, obs=obs, action={}, context=_setup(scenario).context)
    )


def test_on_target_scores_one_and_does_not_terminate():
    scenario = make_steady_flight_scenario(SteadyFlightParams(aircraft="737"))
    reward, terminated = _evaluate(scenario)
    assert reward == pytest.approx(1.0)
    assert not terminated


def test_the_corner_of_the_band_scores_zero_and_does_not_terminate():
    scenario = make_steady_flight_scenario()
    reward, terminated = _evaluate(
        scenario,
        {
            V.ALTITUDE: _setup(scenario).initial_conditions.altitude_ft + 500.0,
            V.HEADING: (_setup(scenario).initial_conditions.heading_deg + 10.0) % 360.0,
            V.VERTICAL_SPEED: 1000.0,
        },
    )
    assert reward == pytest.approx(0.0)
    assert not terminated


@pytest.mark.parametrize(
    "overrides",
    [
        # Literal margins: an offset computed from the constant reward_fn uses
        # would move the input and the pass/fail boundary together.
        {V.ALTITUDE: 5501.0},  # 5000 target + 500 ft band + 1
        {V.ALTITUDE: 4499.0},  # 5000 target - 500 ft band - 1
        {V.HEADING: 281.0},  # 270 target + 10 deg band + 1
        {V.VERTICAL_SPEED: 1001.0},  # 1000 fpm band + 1
        {V.ROLL: 76.0},
        {V.PITCH: 46.0},
        {V.AOA: 21.0},
    ],
)
def test_each_limit_terminates_on_its_own(overrides):
    scenario = make_steady_flight_scenario()
    _, terminated = _evaluate(scenario, overrides)
    assert terminated


def test_the_floor_terminates_on_its_own():
    """Against the default 5000 ft target, V.ALTITUDE: 999.0 fires the band
    term (alt = 4001/500) before it ever reaches FLOOR_ALTITUDE_FT, and that
    case would silently duplicate the band's lower-edge case above and never
    exercise the floor clause. A scenario with a low target altitude (the
    case the floor's code comment says it exists for) puts 999.0 comfortably
    inside the band (alt = 201/500) so the floor is what fires, alone."""
    scenario = make_steady_flight_scenario(SteadyFlightParams(altitude_ft=1200.0))
    _, terminated = _evaluate(scenario, {V.ALTITUDE: 999.0})
    assert terminated


def test_in_band_reward_is_never_negative():
    """The property that stops 'terminate immediately' from beating flying.

    Altitude and vertical speed are pinned from both sides by their literal
    band edges. Heading gets the same literal ±10.0 (not `HEADING_BAND_DEG`):
    deriving the offset from the constant under test would make the ratio 1.0
    for any band value, the same insensitivity fixed for the altitude
    parametrize cases above.
    """
    scenario = make_steady_flight_scenario()
    for alt in (-500.0, -250.0, 0.0, 250.0, 500.0):
        for hdg in (-10.0, 0.0, 10.0):
            for vs in (-1000.0, 0.0, 1000.0):
                reward, terminated = _evaluate(
                    scenario,
                    {
                        V.ALTITUDE: _setup(scenario).initial_conditions.altitude_ft + alt,
                        V.HEADING: (_setup(scenario).initial_conditions.heading_deg + hdg) % 360.0,
                        V.VERTICAL_SPEED: vs,
                    },
                )
                assert not terminated
                assert reward >= 0.0


def test_the_agent_sees_errors_and_positions_but_not_absolute_heading():
    scenario = make_steady_flight_scenario()
    names = [f.name for f in scenario.observation_features]
    assert names == [
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
    # Read from the simulator to build the error features, never shown raw:
    # a 0-360 wraparound is poison for an MLP.
    assert V.HEADING in scenario.observation_variables
    assert "heading" not in names


def test_every_feature_is_bounded_by_its_params_band():
    defaults = SteadyFlightParams()
    scenario = make_steady_flight_scenario()
    bounds = {f.name: (f.spec.low, f.spec.high) for f in scenario.observation_features}
    assert bounds["altitude_error_ft"] == (-defaults.altitude_band_ft, defaults.altitude_band_ft)
    assert bounds["heading_error_deg"] == (-defaults.heading_band_deg, defaults.heading_band_deg)
    assert bounds["vertical_speed"] == (
        -defaults.vertical_speed_band_fpm,
        defaults.vertical_speed_band_fpm,
    )
    assert bounds["roll"] == (-defaults.roll_limit_deg, defaults.roll_limit_deg)


def _feature(scenario, name):
    return next(f for f in scenario.observation_features if f.name == name)


def test_altitude_error_feature_computes_actual_minus_target():
    """The lambda, not just its name and bounds: a feature that leaked the
    absolute altitude instead of the error would pass every other test here."""
    scenario = make_steady_flight_scenario()  # c172p cruise target: 5000.0 ft
    altitude_error_ft = _feature(scenario, "altitude_error_ft")
    assert altitude_error_ft.fn(
        {V.ALTITUDE: 5300.0}, _setup(scenario).context
    ) == pytest.approx(300.0)


def test_heading_error_feature_computes_signed_shortest_path_error():
    scenario = make_steady_flight_scenario()  # default target heading: 270.0 deg
    heading_error_deg_feature = _feature(scenario, "heading_error_deg")
    assert heading_error_deg_feature.fn(
        {V.HEADING: 275.0}, _setup(scenario).context
    ) == pytest.approx(-5.0)


def test_heading_error_wraps_the_short_way_across_zero():
    """Naive `target - current` gives -350 here; the short way across 0 is +10.
    heading_error_deg is load-bearing in the feature, the reward normalizer,
    and termination, so a naive (non-wrapping) implementation anywhere in that
    chain must be caught here."""
    assert heading_error_deg(current=355.0, target=5.0) == pytest.approx(10.0)


def test_heading_error_at_the_antipodal_heading():
    assert heading_error_deg(current=0.0, target=180.0) == pytest.approx(-180.0)


def test_the_scenario_is_discoverable_by_name():
    """The whole point of the registry: nothing wires this up by hand."""
    from aero_gym.registry import build_scenario, lookup

    registration = lookup("steady_flight")
    assert registration.params_type is SteadyFlightParams
    assert registration.module == "sandbox.scenarios.steady_flight"
    assert build_scenario("steady_flight", {"aircraft": "737"}).aircraft == "737"


# The golden pin below predates the sandbox refactor: it fixes the implementation
# runs/*/best.pt was trained against. Only the adapter helpers may change, not the constants.

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
    """The episode context the features are evaluated against."""
    return scenario.setup_fn(np.random.default_rng(0)).context


def _feature_values(scenario, obs):
    """The one place a feature-signature change is allowed to land."""
    context = _context(scenario)
    return [float(f.fn(obs, context)) for f in scenario.observation_features]


def _reward(scenario, obs):
    return scenario.reward_fn(
        Transition(prev_obs=obs, obs=obs, action={}, context=_context(scenario))
    )


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
    ic = scenario.setup_fn(np.random.default_rng(0)).initial_conditions
    assert (ic.latitude_deg, ic.longitude_deg) == (37.6191, -122.3816)
    assert (ic.altitude_ft, ic.heading_deg, ic.airspeed_kts) == (5000.0, 270.0, 95.0)
    assert (ic.pitch_deg, ic.roll_deg) == (0.0, 0.0)
    assert scenario.aircraft == "c172p"
    assert scenario.agent_hz == 10.0
    assert scenario.max_episode_steps == 2000
