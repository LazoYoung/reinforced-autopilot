"""Mapping between canonical flight variables and JSBSim properties.

Conversions bring JSBSim's imperial/radian outputs to the canonical units
(degrees, deg/s, feet, fpm) and JSBSim's control sign conventions to the
pilot convention (+elevator = pitch up, +rudder = yaw right): JSBSim's
positive elevator command pitches the nose down and its positive rudder
command yaws the nose left, so both are negated; ailerons already match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from aerogym.variables import FlightVariable

FPS_TO_FPM = 60.0
RAD_TO_DEG = math.degrees(1.0)


@dataclass(frozen=True)
class ObservationMapping:
    prop: str
    to_canonical: Callable[[float], float]


def _identity(x: float) -> float:
    return x


def _fps_to_fpm(x: float) -> float:
    return x * FPS_TO_FPM


def _rad_s_to_deg_s(x: float) -> float:
    return x * RAD_TO_DEG


def _wrap_360(x: float) -> float:
    return x % 360.0


OBSERVATION_MAP: dict[FlightVariable, ObservationMapping] = {
    FlightVariable.HEADING: ObservationMapping("attitude/psi-deg", _wrap_360),
    FlightVariable.ALTITUDE: ObservationMapping("position/h-sl-ft", _identity),
    FlightVariable.VERTICAL_SPEED: ObservationMapping("velocities/h-dot-fps", _fps_to_fpm),
    FlightVariable.PITCH: ObservationMapping("attitude/theta-deg", _identity),
    FlightVariable.ROLL: ObservationMapping("attitude/phi-deg", _identity),
    FlightVariable.YAW: ObservationMapping("attitude/psi-deg", _wrap_360),
    FlightVariable.PITCH_RATE: ObservationMapping("velocities/q-rad_sec", _rad_s_to_deg_s),
    FlightVariable.ROLL_RATE: ObservationMapping("velocities/p-rad_sec", _rad_s_to_deg_s),
    FlightVariable.YAW_RATE: ObservationMapping("velocities/r-rad_sec", _rad_s_to_deg_s),
    FlightVariable.AOA: ObservationMapping("aero/alpha-deg", _identity),
    FlightVariable.AOS: ObservationMapping("aero/beta-deg", _identity),
}


@dataclass(frozen=True)
class ControlMapping:
    prop: str
    sign: float
    per_engine: bool = False
    """Whether one command in the action space drives every engine.

    JSBSim's bare ``fcs/throttle-cmd-norm`` is engine 0 alone, so on a
    multi-engine aircraft the rest would never hear the agent.
    """


CONTROL_MAP: dict[FlightVariable, ControlMapping] = {
    FlightVariable.ELEVATOR: ControlMapping("fcs/elevator-cmd-norm", -1.0),
    FlightVariable.AILERON: ControlMapping("fcs/aileron-cmd-norm", 1.0),
    FlightVariable.RUDDER: ControlMapping("fcs/rudder-cmd-norm", -1.0),
    FlightVariable.THROTTLE: ControlMapping("fcs/throttle-cmd-norm", 1.0, per_engine=True),
}
