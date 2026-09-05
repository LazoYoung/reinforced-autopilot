from __future__ import annotations

import pytest

from aerogym.simulators.jsbsim.aircraft import PROFILES, profile_for


def test_the_737_is_rendered_by_flightgears_737_300():
    """JSBSim and FlightGear disagree on the name for the same airframe."""
    assert profile_for("737").flightgear == "737-300"


def test_an_unknown_aircraft_keeps_its_jsbsim_name_for_flightgear():
    """Most models share a name; guessing it beats rendering the wrong aircraft."""
    assert profile_for("777-200").flightgear == "777-200"


def test_an_unknown_aircraft_gets_no_aircraft_specific_wiring():
    profile = profile_for("777-200")
    assert profile.cockpit is None
    assert profile.mirrors == ()
    assert profile.republish == ()


def test_the_737_cruises_where_a_737_can_fly():
    profile = profile_for("737")
    assert profile.cruise_altitude_ft == 30000.0
    assert profile.cruise_airspeed_kts == 280.0


def test_c172p_keeps_the_cruise_it_had_before_profiles_existed():
    profile = profile_for("c172p")
    assert profile.cruise_altitude_ft == 5000.0
    assert profile.cruise_airspeed_kts == 95.0


def test_c172p_starts_with_the_piston_controls_it_needs():
    startup = profile_for("c172p").startup
    assert startup["fcs/mixture-cmd-norm"] == 1.0
    assert startup["fcs/throttle-cmd-norm"] == 0.7


def test_the_737_starts_without_a_mixture_control_it_does_not_have():
    """Mixture is a piston control; a CFM56 has no use for one."""
    assert "fcs/mixture-cmd-norm" not in profile_for("737").startup


def test_the_737_starts_with_both_engines_set():
    """A twin needs an index per engine; the bare property only reaches the first."""
    startup = profile_for("737").startup
    assert startup["fcs/throttle-cmd-norm[0]"] == 0.85
    assert startup["fcs/throttle-cmd-norm[1]"] == 0.85


def test_the_737_starts_clean():
    """JSBSim's 737 initialises with the gear down, which is not a cruise state."""
    startup = profile_for("737").startup
    assert startup["gear/gear-cmd-norm"] == 0.0
    assert startup["fcs/flap-cmd-norm"] == 0.0


def test_the_737_starts_trimmed_close_to_level_flight():
    """Untrimmed, JSBSim's 737 pitches down hard and leaves the envelope."""
    startup = profile_for("737").startup
    assert startup["fcs/pitch-trim-cmd-norm"] == pytest.approx(-0.2)


def test_an_unknown_aircraft_still_gets_engine_controls():
    """Nothing is known about it, but idling to the ground helps no one."""
    startup = profile_for("777-200").startup
    assert startup["fcs/mixture-cmd-norm"] == 1.0
    assert startup["fcs/throttle-cmd-norm"] == 0.7


def test_every_profile_names_a_flightgear_aircraft():
    for name, profile in PROFILES.items():
        assert profile.flightgear, f"{name} has no FlightGear aircraft"
