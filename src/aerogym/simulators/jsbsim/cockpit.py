from __future__ import annotations

from typing import Callable

_MAX_TANKS = 32
_TINY = 1e-9

# FlightGear reads the packet's fuel quantity, which JSBSim fills in pounds, as
# US gallons; 1 lb/gal keeps every unit it derives from that self-consistent.
_FUEL_DENSITY_PPG = 1.0


def _c172p_cockpit(controls: dict[str, float]) -> dict[str, float]:
    """c172p animates its yoke and pedals from JSBSim FCS outputs."""
    return {
        "/sim/model/c172p/cockpit/yoke-elevator": controls["/controls/flight/elevator"],
        "/sim/model/c172p/cockpit/yoke-aileron": controls["/controls/flight/aileron"],
        "/sim/model/c172p/cockpit/pedals-rudder": controls["/controls/flight/rudder"],
    }


# Keyed by the aircraft FlightGear has loaded, not the one JSBSim is flying.
COCKPIT_MODELS: dict[str, Callable[[dict[str, float]], dict[str, float]]] = {
    "c172p": _c172p_cockpit,
}

# (destination, source) pairs to copy within FlightGear's own tree
COCKPIT_MIRRORS: dict[str, tuple[tuple[str, str], ...]] = {
    "c172p": (("/systems/static-pressure-inhg-selected", "/environment/pressure-inhg"),),
}


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

    def __init__(self, fdm, aircraft: str) -> None:
        self._fdm = fdm
        self._engines = range(fdm.get_propulsion().get_num_engines())
        self._capacities_lbs = _tank_capacities_lbs(fdm)
        self._cockpit = COCKPIT_MODELS.get(aircraft)
        self.mirrors = COCKPIT_MIRRORS.get(aircraft, ())

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
        if self._cockpit is not None:
            props.update(self._cockpit(controls))
        return props
