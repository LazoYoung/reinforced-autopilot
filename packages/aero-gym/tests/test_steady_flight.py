from __future__ import annotations

import pytest

from aero_gym.scenario import Transition
from aero_gym.scenarios import make_steady_flight_scenario
from aero_gym.scenarios.steady_flight import (
    ALTITUDE_BAND_FT,
    HEADING_BAND_DEG,
    VERTICAL_SPEED_BAND_FPM,
    heading_error_deg,
)
from aero_gym.variables import FlightVariable as V


def test_the_flight_condition_follows_the_aircraft():
    """5000 ft at 95 kts is a Cessna cruise; a 737 would stall there."""
    ic = make_steady_flight_scenario(aircraft="737").initial_conditions
    assert ic.altitude_ft == 30000.0
    assert ic.airspeed_kts == 280.0


def test_c172p_keeps_the_flight_condition_it_always_had():
    ic = make_steady_flight_scenario().initial_conditions
    assert ic.altitude_ft == 5000.0
    assert ic.airspeed_kts == 95.0


def test_an_explicit_flight_condition_wins_over_the_aircrafts():
    ic = make_steady_flight_scenario(
        aircraft="737", altitude_ft=12000.0, airspeed_kts=250.0
    ).initial_conditions
    assert ic.altitude_ft == 12000.0
    assert ic.airspeed_kts == 250.0


def test_one_overridden_value_does_not_drag_the_other_along():
    ic = make_steady_flight_scenario(aircraft="737", altitude_ft=12000.0).initial_conditions
    assert ic.altitude_ft == 12000.0
    assert ic.airspeed_kts == 280.0


def _obs(scenario, overrides=None):
    obs = {v: 0.0 for v in scenario.observation_variables}
    obs[V.ALTITUDE] = scenario.initial_conditions.altitude_ft
    obs[V.HEADING] = scenario.initial_conditions.heading_deg
    obs.update(overrides or {})
    return obs


def _evaluate(scenario, overrides=None):
    # A plain dict, not **kwargs: the override keys are FlightVariable members,
    # and CPython's call-unpacking requires str keys, so `**overrides` would
    # raise TypeError here regardless of the callee's signature.
    obs = _obs(scenario, overrides)
    return scenario.reward_fn(Transition(prev_obs=obs, obs=obs, action={}))


def test_on_target_scores_one_and_does_not_terminate():
    scenario = make_steady_flight_scenario(aircraft="737")
    reward, terminated = _evaluate(scenario)
    assert reward == pytest.approx(1.0)
    assert not terminated


def test_the_corner_of_the_band_scores_zero_and_does_not_terminate():
    scenario = make_steady_flight_scenario()
    reward, terminated = _evaluate(
        scenario,
        {
            V.ALTITUDE: scenario.initial_conditions.altitude_ft + ALTITUDE_BAND_FT,
            V.HEADING: (scenario.initial_conditions.heading_deg + HEADING_BAND_DEG) % 360.0,
            V.VERTICAL_SPEED: VERTICAL_SPEED_BAND_FPM,
        },
    )
    assert reward == pytest.approx(0.0)
    assert not terminated


@pytest.mark.parametrize(
    "overrides",
    [
        # Margins are literal, not `ALTITUDE_BAND_FT + 1.0` etc: if the offset
        # were computed from the same constant the reward_fn uses, mutating
        # that constant would move the test input and the pass/fail boundary
        # together, and the test could never notice a wrong band value (the
        # ratio stays > 1 for any positive band). Literal 501/281/1001 pin the
        # input to the *intended* 500/10/1000 bands so a mutated constant
        # actually moves the boundary relative to a fixed input.
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
    term (alt = 4001/500) before it ever reaches FLOOR_ALTITUDE_FT -- that
    case would silently duplicate the band's lower-edge case above and never
    exercise the floor clause. A scenario with a low target altitude (the
    case the floor's code comment says it exists for) puts 999.0 comfortably
    inside the band (alt = 201/500) so the floor is what fires, alone."""
    scenario = make_steady_flight_scenario(altitude_ft=1200.0)
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
                        V.ALTITUDE: scenario.initial_conditions.altitude_ft + alt,
                        V.HEADING: (scenario.initial_conditions.heading_deg + hdg) % 360.0,
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


def test_every_feature_is_bounded_by_the_limit_that_terminates_on_it():
    scenario = make_steady_flight_scenario()
    bounds = {f.name: (f.spec.low, f.spec.high) for f in scenario.observation_features}
    assert bounds["altitude_error_ft"] == (-ALTITUDE_BAND_FT, ALTITUDE_BAND_FT)
    assert bounds["heading_error_deg"] == (-HEADING_BAND_DEG, HEADING_BAND_DEG)
    assert bounds["vertical_speed"] == (-VERTICAL_SPEED_BAND_FPM, VERTICAL_SPEED_BAND_FPM)
    assert bounds["roll"] == (-75.0, 75.0)


def _feature(scenario, name):
    return next(f for f in scenario.observation_features if f.name == name)


def test_altitude_error_feature_computes_actual_minus_target():
    """The lambda, not just its name and bounds -- a feature that leaked the
    absolute altitude instead of the error would pass every other test here."""
    scenario = make_steady_flight_scenario()  # c172p cruise target: 5000.0 ft
    altitude_error_ft = _feature(scenario, "altitude_error_ft")
    assert altitude_error_ft.fn({V.ALTITUDE: 5300.0}) == pytest.approx(300.0)


def test_heading_error_feature_computes_signed_shortest_path_error():
    scenario = make_steady_flight_scenario()  # default target heading: 270.0 deg
    heading_error_deg_feature = _feature(scenario, "heading_error_deg")
    assert heading_error_deg_feature.fn({V.HEADING: 275.0}) == pytest.approx(-5.0)


def test_heading_error_wraps_the_short_way_across_zero():
    """Naive `target - current` gives -350 here; the short way across 0 is +10.
    heading_error_deg is load-bearing in the feature, the reward normalizer,
    and termination, so a naive (non-wrapping) implementation anywhere in that
    chain must be caught here."""
    assert heading_error_deg(current=355.0, target=5.0) == pytest.approx(10.0)


def test_heading_error_at_the_antipodal_heading():
    assert heading_error_deg(current=0.0, target=180.0) == pytest.approx(-180.0)
