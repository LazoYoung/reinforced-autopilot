from __future__ import annotations

import types

import pytest

from aerogym.simulators.jsbsim.cockpit import CockpitSync


class _FakeFDM:
    """Stands in for JSBSim's FGFDMExec: property lookup plus an engine count."""

    def __init__(self, values: dict[str, float], engines: int = 1) -> None:
        self._values = values
        self._engines = engines

    def __getitem__(self, name: str) -> float:
        try:
            return self._values[name]
        except KeyError:
            raise KeyError(f"No property named {name}") from None

    def get_propulsion(self):
        return types.SimpleNamespace(get_num_engines=lambda: self._engines)


def _fdm(engines: int = 1, **overrides: float) -> _FakeFDM:
    values = {
        "fcs/elevator-cmd-norm": 0.3,
        "fcs/aileron-cmd-norm": -0.2,
        "fcs/rudder-cmd-norm": 0.1,
        "fcs/flap-pos-norm": 0.5,
        "fcs/throttle-cmd-norm[0]": 0.7,
        "fcs/throttle-cmd-norm[1]": 0.4,
        "velocities/vc-kts": 95.0,
        "velocities/mach": 0.15,
        "propulsion/tank[0]/contents-lbs": 90.0,
        "propulsion/tank[0]/pct-full": 50.0,
        "propulsion/tank[0]/density-lbs_per_gal": 6.0,
        "propulsion/tank[1]/contents-lbs": 45.0,
        "propulsion/tank[1]/pct-full": 25.0,
        "propulsion/tank[1]/density-lbs_per_gal": 6.0,
    }
    values.update(overrides)
    return _FakeFDM(values, engines=engines)


def test_controls_are_published_in_the_pilot_convention():
    """JSBSim's elevator and rudder signs are the opposite of FlightGear's."""
    props = CockpitSync(_fdm(), "c172p").properties()
    assert props["/controls/flight/elevator"] == pytest.approx(-0.3)
    assert props["/controls/flight/rudder"] == pytest.approx(-0.1)
    assert props["/controls/flight/aileron"] == pytest.approx(-0.2)
    assert props["/controls/flight/flaps"] == pytest.approx(0.5)


def test_mach_is_published_for_the_pitot_model():
    """FlightGear leaves mach at zero without a flight model of its own."""
    assert CockpitSync(_fdm(), "c172p").properties()["/velocities/mach"] == pytest.approx(0.15)


def test_properties_flightgear_maintains_itself_are_left_alone():
    """Writing these only fights FlightGear, which rewrites them every frame.

    Airspeed already arrives in the native-FDM packet, and c172p's fuel gauge
    needle is driven by a property rule that runs with or without a flight
    model.
    """
    props = CockpitSync(_fdm(), "c172p").properties()
    assert "/velocities/airspeed-kt" not in props
    assert "/consumables/fuel/tank[0]/indicated-level-gal_us" not in props


def test_a_throttle_is_published_per_engine():
    props = CockpitSync(_fdm(engines=2), "c172p").properties()
    assert props["/controls/engines/engine[0]/throttle"] == pytest.approx(0.7)
    assert props["/controls/engines/engine[1]/throttle"] == pytest.approx(0.4)


def test_tanks_are_enumerated_until_jsbsim_runs_out():
    props = CockpitSync(_fdm(), "c172p").properties()
    assert "/consumables/fuel/tank[1]/capacity-gal_us" in props
    assert "/consumables/fuel/tank[2]/capacity-gal_us" not in props


def test_tank_capacity_is_recovered_from_percent_full():
    """JSBSim publishes no capacity, so it comes from contents and percentage."""
    props = CockpitSync(_fdm(), "c172p").properties()
    assert props["/consumables/fuel/tank[0]/capacity-gal_us"] == pytest.approx(180.0)
    assert props["/consumables/fuel/tank[1]/capacity-gal_us"] == pytest.approx(180.0)


def test_capacity_is_measured_once_so_a_draining_tank_keeps_it():
    """Percent-full falls to zero as fuel burns; capacity must not follow it."""
    fdm = _fdm()
    sync = CockpitSync(fdm, "c172p")
    fdm._values["propulsion/tank[0]/contents-lbs"] = 0.0
    fdm._values["propulsion/tank[0]/pct-full"] = 0.0
    props = sync.properties()
    assert props["/consumables/fuel/tank[0]/capacity-gal_us"] == pytest.approx(180.0)


def test_an_already_empty_tank_does_not_divide_by_zero():
    props = CockpitSync(_fdm(**{"propulsion/tank[0]/pct-full": 0.0}), "c172p").properties()
    assert props["/consumables/fuel/tank[0]/capacity-gal_us"] == 0.0


def test_the_aircraft_is_powered_up_once_rather_than_every_tick():
    """A cold, dark panel reads zero however good the data behind it is.

    Writing these once leaves the switches usable in the cockpit afterwards.
    """
    sync = CockpitSync(_fdm(), "c172p")
    assert sync.initial_properties() == {
        "/controls/switches/master-bat": "true",
        "/controls/switches/master-alt": "true",
    }
    assert not [p for p in sync.properties() if p.startswith("/controls/switches/")]


def test_tank_density_keeps_flightgears_derived_units_self_consistent():
    """FlightGear reads the packet's pounds as gallons; 1 lb/gal squares that."""
    props = CockpitSync(_fdm(), "c172p").properties()
    assert props["/consumables/fuel/tank[0]/density-ppg"] == 1.0
    assert props["/consumables/fuel/tank[0]/selected"] == "true"


def test_c172p_yoke_follows_the_controls():
    props = CockpitSync(_fdm(), "c172p").properties()
    assert props["/sim/model/c172p/cockpit/yoke-elevator"] == props["/controls/flight/elevator"]
    assert props["/sim/model/c172p/cockpit/yoke-aileron"] == props["/controls/flight/aileron"]
    assert props["/sim/model/c172p/cockpit/pedals-rudder"] == props["/controls/flight/rudder"]


def test_an_unknown_aircraft_gets_only_the_generic_properties():
    sync = CockpitSync(_fdm(), "777-200")
    props = sync.properties()
    assert sync.mirrors == ()
    assert not [p for p in props if p.startswith("/sim/model/")]
    assert props["/controls/flight/elevator"] == pytest.approx(-0.3)


def test_c172p_mirrors_its_static_source_from_flightgears_own_pressure():
    assert CockpitSync(_fdm(), "c172p").mirrors == (
        ("/systems/static-pressure-inhg-selected", "/environment/pressure-inhg"),
    )


def _b737_fdm(**overrides: float) -> _FakeFDM:
    """A stand-in shaped like JSBSim's stock 737: two engines, three tanks."""
    values = {
        "fcs/elevator-cmd-norm": 0.1,
        "fcs/aileron-cmd-norm": 0.0,
        "fcs/rudder-cmd-norm": 0.0,
        "fcs/flap-pos-norm": 0.25,
        "fcs/flap-cmd-norm": 0.25,
        "fcs/speedbrake-cmd-norm": 0.4,
        "fcs/speedbrake-pos-norm": 0.4,
        "fcs/pitch-trim-cmd-norm": -0.2,
        "fcs/throttle-cmd-norm[0]": 0.85,
        "fcs/throttle-cmd-norm[1]": 0.85,
        "fcs/throttle-pos-norm[0]": 0.85,
        "fcs/throttle-pos-norm[1]": 0.85,
        "velocities/mach": 0.78,
        "velocities/vg-fps": 780.0,
        "atmosphere/a-fps": 995.0,
        "inertia/weight-lbs": 83000.0,
        "position/h-agl-ft": 29500.0,
        "gear/unit[0]/WOW": 0.0,
        "gear/unit[1]/WOW": 0.0,
        "gear/unit[2]/WOW": 0.0,
        "propulsion/tank[0]/contents-lbs": 10000.0,
        "propulsion/tank[0]/pct-full": 98.0,
    }
    values.update(overrides)
    return _FakeFDM(values, engines=2)


def test_the_737_republishes_the_jsbsim_state_its_panel_reads():
    """The 737-300 panel reads /fdm/jsbsim/*, which --fdm=null leaves empty."""
    props = CockpitSync(_b737_fdm(), "737").properties()
    assert props["/fdm/jsbsim/fcs/flap-pos-norm"] == pytest.approx(0.25)
    assert props["/fdm/jsbsim/inertia/weight-lbs"] == pytest.approx(83000.0)
    assert props["/fdm/jsbsim/velocities/vg-fps"] == pytest.approx(780.0)


def test_a_republished_property_the_model_does_not_have_is_skipped():
    """Models vary; a missing property must not take the whole cockpit down."""
    fdm = _b737_fdm()
    del fdm._values["velocities/vg-fps"]
    props = CockpitSync(fdm, "737").properties()
    assert "/fdm/jsbsim/velocities/vg-fps" not in props
    assert props["/fdm/jsbsim/fcs/flap-pos-norm"] == pytest.approx(0.25)


def test_c172p_republishes_nothing_because_its_panel_asks_for_nothing():
    props = CockpitSync(_fdm(), "c172p").properties()
    assert not [p for p in props if p.startswith("/fdm/jsbsim/")]


def test_the_737_stabilizer_moves_with_pitch_trim():
    """Nose-up trim is negative in JSBSim and more units on a 737's trim wheel."""
    nose_up = CockpitSync(_b737_fdm(**{"fcs/pitch-trim-cmd-norm": -0.5}), "737").properties()
    nose_down = CockpitSync(_b737_fdm(**{"fcs/pitch-trim-cmd-norm": 0.5}), "737").properties()
    assert nose_up["/fdm/jsbsim/fcs/stabilizer-pos-unit"] > nose_down["/fdm/jsbsim/fcs/stabilizer-pos-unit"]
    assert 0.0 <= nose_down["/fdm/jsbsim/fcs/stabilizer-pos-unit"] <= 17.0
    assert 0.0 <= nose_up["/fdm/jsbsim/fcs/stabilizer-pos-unit"] <= 17.0


def test_the_737_trim_wheel_and_its_target_agree():
    """The panel animates the wheel from one and the indicator from the other."""
    props = CockpitSync(_b737_fdm(), "737").properties()
    assert props["/fdm/jsbsim/fcs/stabilizer/stab-target"] == pytest.approx(
        props["/fdm/jsbsim/fcs/stabilizer-pos-unit"]
    )


def test_the_737_spoiler_lever_follows_the_speedbrake_command():
    """The only /controls/ path the 737-300 model animates from."""
    props = CockpitSync(_b737_fdm(), "737").properties()
    assert props["/controls/flight/spoilers-lever-pos"] == pytest.approx(0.4)


def test_the_737_gets_the_generic_cockpit_properties_too():
    props = CockpitSync(_b737_fdm(), "737").properties()
    assert props["/controls/flight/elevator"] == pytest.approx(-0.1)
    assert props["/controls/engines/engine[1]/throttle"] == pytest.approx(0.85)


def test_the_cockpit_is_keyed_on_the_jsbsim_model_not_flightgears_name():
    """FlightGear calls it 737-300; the profile registry calls it 737."""
    sync = CockpitSync(_b737_fdm(), "737", flightgear_aircraft="737-300")
    props = sync.properties()
    assert props["/fdm/jsbsim/fcs/flap-pos-norm"] == pytest.approx(0.25)
    assert "/fdm/jsbsim/fcs/stabilizer-pos-unit" in props


def test_a_cockpit_flightgear_did_not_load_falls_back_to_the_generic_one():
    """Writing 737 panel properties into a Cessna's cockpit means nothing."""
    with pytest.warns(UserWarning, match="c172p"):
        sync = CockpitSync(_b737_fdm(), "737", flightgear_aircraft="c172p")
    props = sync.properties()
    assert not [p for p in props if p.startswith("/fdm/jsbsim/")]
    assert sync.mirrors == ()
    assert props["/controls/flight/elevator"] == pytest.approx(-0.1)


def test_the_cockpit_is_built_when_flightgear_does_not_say_what_it_loaded():
    sync = CockpitSync(_b737_fdm(), "737", flightgear_aircraft=None)
    assert "/fdm/jsbsim/fcs/flap-pos-norm" in sync.properties()
