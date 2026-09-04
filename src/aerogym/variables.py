"""Canonical flight variables shared by every simulator backend.

All simulators express these variables in the same canonical units so that
scenarios and trained agents are portable across backends:

- angles in degrees, angular rates in deg/s
- altitude in feet (MSL), vertical speed in feet per minute
- controls normalized to [-1, 1] ([0, 1] for throttle), pilot convention:
  +elevator = pitch up, +aileron = roll right, +rudder = yaw right
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class VariableSpec:
    """Unit and bounds of one canonical variable."""

    unit: str
    low: float
    high: float
    description: str


class FlightVariable(enum.Enum):
    # Controls (actions)
    ELEVATOR = "elevator"
    AILERON = "aileron"
    RUDDER = "rudder"
    THROTTLE = "throttle"
    # State (observations)
    HEADING = "heading"
    ALTITUDE = "altitude"
    VERTICAL_SPEED = "vertical_speed"
    PITCH = "pitch"
    ROLL = "roll"
    YAW = "yaw"
    PITCH_RATE = "pitch_rate"
    ROLL_RATE = "roll_rate"
    YAW_RATE = "yaw_rate"
    AOA = "aoa"
    AOS = "aos"


ACTION_SPECS: dict[FlightVariable, VariableSpec] = {
    FlightVariable.ELEVATOR: VariableSpec("norm", -1.0, 1.0, "+1 = full pitch-up command"),
    FlightVariable.AILERON: VariableSpec("norm", -1.0, 1.0, "+1 = full roll-right command"),
    FlightVariable.RUDDER: VariableSpec("norm", -1.0, 1.0, "+1 = full yaw-right command"),
    FlightVariable.THROTTLE: VariableSpec("norm", 0.0, 1.0, "0 = idle, 1 = full throttle"),
}

OBSERVATION_SPECS: dict[FlightVariable, VariableSpec] = {
    FlightVariable.HEADING: VariableSpec("deg", 0.0, 360.0, "true heading"),
    FlightVariable.ALTITUDE: VariableSpec("ft", -1500.0, 85000.0, "altitude above mean sea level"),
    FlightVariable.VERTICAL_SPEED: VariableSpec("fpm", -40000.0, 40000.0, "climb rate, + = up"),
    FlightVariable.PITCH: VariableSpec("deg", -90.0, 90.0, "pitch attitude, + = nose up"),
    FlightVariable.ROLL: VariableSpec("deg", -180.0, 180.0, "bank angle, + = right wing down"),
    FlightVariable.YAW: VariableSpec("deg", 0.0, 360.0, "yaw (true heading of the body x-axis)"),
    FlightVariable.PITCH_RATE: VariableSpec("deg/s", -360.0, 360.0, "body pitch rate q"),
    FlightVariable.ROLL_RATE: VariableSpec("deg/s", -720.0, 720.0, "body roll rate p"),
    FlightVariable.YAW_RATE: VariableSpec("deg/s", -360.0, 360.0, "body yaw rate r"),
    FlightVariable.AOA: VariableSpec("deg", -90.0, 90.0, "angle of attack (alpha)"),
    FlightVariable.AOS: VariableSpec("deg", -90.0, 90.0, "angle of sideslip (beta)"),
}

ALL_ACTION_VARIABLES: tuple[FlightVariable, ...] = tuple(ACTION_SPECS)
ALL_OBSERVATION_VARIABLES: tuple[FlightVariable, ...] = tuple(OBSERVATION_SPECS)
