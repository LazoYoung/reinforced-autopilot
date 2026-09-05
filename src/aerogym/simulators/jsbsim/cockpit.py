from __future__ import annotations

import warnings

from .aircraft import profile_for

_MAX_TANKS = 32
_TINY = 1e-9

# FlightGear reads the packet's fuel quantity, which JSBSim fills in pounds, as
# US gallons; 1 lb/gal keeps every unit it derives from that self-consistent.
_FUEL_DENSITY_PPG = 1.0


def _tank_capacities_lbs(fdm) -> list[float]:
    """Each JSBSim tank's capacity in pounds, recovered from contents and
    percent-full, which JSBSim publishes instead. Read once: the percentage goes
    to zero as the tank drains."""
    capacities = []
    for i in range(_MAX_TANKS):
        try:
            contents = fdm[f"propulsion/tank[{i}]/contents-lbs"]
            pct_full = fdm[f"propulsion/tank[{i}]/pct-full"]
        except KeyError:
            break
        capacities.append(100.0 * contents / pct_full if pct_full > _TINY else 0.0)
    return capacities


class CockpitSync:
    """Builds the property writes that keep a FlightGear cockpit in step."""

    def __init__(self, fdm, aircraft: str, flightgear_aircraft: str | None = None) -> None:
        """``aircraft`` is the JSBSim model, which is what profiles are keyed on.

        ``flightgear_aircraft`` is what FlightGear reports it actually loaded.
        When that is not the aircraft the profile describes, only the generic
        properties are published: a 737's panel properties mean nothing in a
        Cessna's cockpit.
        """
        self._fdm = fdm
        self._engines = range(fdm.get_propulsion().get_num_engines())
        self._capacities_lbs = _tank_capacities_lbs(fdm)
        profile = profile_for(aircraft)
        if flightgear_aircraft is not None and flightgear_aircraft != profile.flightgear:
            warnings.warn(
                f"FlightGear loaded {flightgear_aircraft!r} but the simulation is "
                f"flying {aircraft!r}; its cockpit will show only generic state.",
                stacklevel=2,
            )
            profile = profile_for("")
        self._cockpit = profile.cockpit
        self._republish = profile.republish
        self.mirrors = profile.mirrors

    def initial_properties(self) -> dict[str, object]:
        """Power up the panel, which FlightGear otherwise boots cold and dark.

        Written once, so the switches stay usable in the cockpit afterwards.
        """
        return {
            "/controls/switches/master-bat": "true",
            "/controls/switches/master-alt": "true",
        }

    def properties(self) -> dict[str, object]:
        fdm = self._fdm
        # /controls/* is the pilot convention, not JSBSim's command signs.
        controls = {
            "/controls/flight/elevator": -fdm["fcs/elevator-cmd-norm"],
            "/controls/flight/aileron": fdm["fcs/aileron-cmd-norm"],
            "/controls/flight/rudder": -fdm["fcs/rudder-cmd-norm"],
            "/controls/flight/flaps": fdm["fcs/flap-pos-norm"],
        }
        props: dict[str, object] = dict(controls)
        # Absent from the packet, and zero flattens the pitot model. Airspeed is
        # not written: the packet carries it, so writing it fights FlightGear.
        props["/velocities/mach"] = fdm["velocities/mach"]
        for i in self._engines:
            props[f"/controls/engines/engine[{i}]/throttle"] = fdm[f"fcs/throttle-cmd-norm[{i}]"]
        for i, capacity_lbs in enumerate(self._capacities_lbs):
            tank = f"/consumables/fuel/tank[{i}]"
            props[f"{tank}/density-ppg"] = _FUEL_DENSITY_PPG
            props[f"{tank}/capacity-gal_us"] = capacity_lbs
            props[f"{tank}/selected"] = "true"
        for prop in self._republish:
            try:
                props[f"/fdm/jsbsim/{prop}"] = fdm[prop]
            except KeyError:
                pass  # models vary; a missing one must not stop the rest
        if self._cockpit is not None:
            props.update(self._cockpit(fdm, controls))
        return props
