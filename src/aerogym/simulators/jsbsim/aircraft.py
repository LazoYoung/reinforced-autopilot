"""What aerogym needs to know about an aircraft beyond its JSBSim model.

The JSBSim model name is the key: it is what a :class:`~aerogym.scenario.Scenario`
carries, and everything else an aircraft needs -- what FlightGear calls it, how
its engines and controls start, what its cockpit reads -- hangs off it here, so
adding an aircraft means editing one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

CockpitFn = Callable[[object, dict[str, float]], dict[str, object]]
"""Builds aircraft-specific cockpit writes from the FDM and the pilot controls."""


@dataclass(frozen=True)
class AircraftProfile:
    """Everything aerogym knows about one JSBSim aircraft model."""

    flightgear: str
    cruise_altitude_ft: float = 5000.0
    cruise_airspeed_kts: float = 95.0
    startup: dict[str, float] = {}
    cockpit: CockpitFn | None = None
    mirrors: tuple[tuple[str, str], ...] = ()
    republish: tuple[str, ...] = ()


def _c172p_cockpit(fdm, controls: dict[str, float]) -> dict[str, object]:
    """c172p animates its yoke and pedals from JSBSim FCS outputs."""
    return {
        "/sim/model/c172p/cockpit/yoke-elevator": controls["/controls/flight/elevator"],
        "/sim/model/c172p/cockpit/yoke-aileron": controls["/controls/flight/aileron"],
        "/sim/model/c172p/cockpit/pedals-rudder": controls["/controls/flight/rudder"],
    }


# Half the 737's 0-17 unit trim-wheel range. JSBSim's stock 737 has no
# stabilizer channel at all, so the wheel is driven from pitch trim: nose-up is
# negative to JSBSim and more units to a 737 pilot. The direction is right; the
# calibration is a visual approximation, not a real stab schedule.
_STAB_UNITS_SPAN = 8.5


def _b737_cockpit(fdm, controls: dict[str, float]) -> dict[str, object]:
    stab_unit = _STAB_UNITS_SPAN * (1.0 - fdm["fcs/pitch-trim-cmd-norm"])
    return {
        "/fdm/jsbsim/fcs/stabilizer-pos-unit": stab_unit,
        "/fdm/jsbsim/fcs/stabilizer/stab-target": stab_unit,
        # The one /controls/ path the 737-300's own model animates from.
        "/controls/flight/spoilers-lever-pos": fdm["fcs/speedbrake-cmd-norm"],
    }


# JSBSim properties the 737-300 panel reads back under /fdm/jsbsim, which
# --fdm=null leaves empty. Only names the stock JSBSim 737 actually publishes.
_B737_REPUBLISH = (
    "fcs/flap-pos-norm",
    "fcs/speedbrake-pos-norm",
    "fcs/throttle-pos-norm[0]",
    "fcs/throttle-pos-norm[1]",
    "inertia/weight-lbs",
    "velocities/vg-fps",
    "atmosphere/a-fps",
    "position/h-agl-ft",
    "gear/unit[0]/WOW",
    "gear/unit[1]/WOW",
    "gear/unit[2]/WOW",
)


PROFILES: dict[str, AircraftProfile] = {
    "c172p": AircraftProfile(
        flightgear="c172p",
        cruise_altitude_ft=5000.0,
        cruise_airspeed_kts=95.0,
        startup={
            "fcs/mixture-cmd-norm": 1.0,
            "fcs/throttle-cmd-norm": 0.7,
            "fcs/pitch-trim-cmd-norm": 0.2,
        },
        cockpit=_c172p_cockpit,
        # c172p routes its pressure instruments through a dead JSBSim static
        # source; point them at the pressure FlightGear still computes itself.
        mirrors=(("/systems/static-pressure-inhg-selected", "/environment/pressure-inhg"),),
    ),
    "737": AircraftProfile(
        flightgear="737-300",
        cruise_altitude_ft=30000.0,
        cruise_airspeed_kts=280.0,
        startup={
            "fcs/throttle-cmd-norm[0]": 0.85,
            "fcs/throttle-cmd-norm[1]": 0.85,
            # JSBSim's 737 initialises gear-down, which is not a cruise state.
            "gear/gear-cmd-norm": 0.0,
            "fcs/flap-cmd-norm": 0.0,
            "fcs/speedbrake-cmd-norm": 0.0,
            # Untrimmed the 737 pitches down hard; -0.2 holds FL300 to within a
            # couple of hundred fpm, which is a flyable place to start.
            "fcs/pitch-trim-cmd-norm": -0.2,
        },
        cockpit=_b737_cockpit,
        republish=_B737_REPUBLISH,
    ),
}


def profile_for(aircraft: str) -> AircraftProfile:
    """The profile for a JSBSim model, or a bare one named after the model.

    An unlisted aircraft still flies and still renders: JSBSim and FlightGear
    agree on most model names, so the name itself is the best guess available.
    """
    return PROFILES.get(aircraft) or AircraftProfile(flightgear=aircraft)
