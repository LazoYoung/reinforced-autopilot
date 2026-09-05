from __future__ import annotations

import pytest

from aerogym.scenarios import make_steady_flight_scenario


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


def test_the_altitude_target_follows_the_aircrafts_cruise():
    """Reward is measured against the altitude the aircraft actually started at."""
    scenario = make_steady_flight_scenario(aircraft="737")
    obs = {v: 0.0 for v in scenario.observation_variables}
    from aerogym.variables import FlightVariable as V

    obs[V.ALTITUDE] = 30000.0
    obs[V.HEADING] = 270.0
    from aerogym.scenario import Transition

    reward, terminated = scenario.reward_fn(Transition(prev_obs=obs, obs=obs, action={}))
    assert reward == pytest.approx(0.0)
    assert not terminated
