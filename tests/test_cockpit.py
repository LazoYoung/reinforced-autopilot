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
