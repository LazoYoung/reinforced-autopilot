"""Scenario definition: what the agent controls, sees, and is rewarded for."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .variables import (
    ACTION_SPECS,
    ALL_ACTION_VARIABLES,
    ALL_OBSERVATION_VARIABLES,
    OBSERVATION_SPECS,
    FlightVariable,
)


@dataclass(frozen=True)
class InitialConditions:
    """Initial flight condition in 6 DoF plus airspeed.

    Airspeed is not part of the requested 6-DoF pose but without it the
    aircraft would start at rest in mid-air and simply fall.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_ft: float
    heading_deg: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    airspeed_kts: float = 90.0


@dataclass(frozen=True)
class Transition:
    """One agent step, in canonical units, passed to the reward function."""

    prev_obs: dict[FlightVariable, float]
    obs: dict[FlightVariable, float]
    action: dict[FlightVariable, float]


RewardFn = Callable[[Transition], tuple[float, bool]]
"""Maps a transition to (reward, terminated)."""


@dataclass(frozen=True)
class Scenario:
    """A fully-specified task for :class:`aerogym.env.AeroGymEnv`."""

    name: str
    initial_conditions: InitialConditions
    reward_fn: RewardFn
    action_variables: tuple[FlightVariable, ...] = ALL_ACTION_VARIABLES
    observation_variables: tuple[FlightVariable, ...] = ALL_OBSERVATION_VARIABLES
    aircraft: str = "c172p"
    agent_hz: float = 10.0
    max_episode_steps: int | None = 2000

    def __post_init__(self) -> None:
        if not self.action_variables:
            raise ValueError("scenario needs at least one action variable")
        if not self.observation_variables:
            raise ValueError("scenario needs at least one observation variable")
        for var in self.action_variables:
            if var not in ACTION_SPECS:
                raise ValueError(f"{var} is not a valid action variable")
        for var in self.observation_variables:
            if var not in OBSERVATION_SPECS:
                raise ValueError(f"{var} is not a valid observation variable")
        if len(set(self.action_variables)) != len(self.action_variables):
            raise ValueError("duplicate action variables")
        if len(set(self.observation_variables)) != len(self.observation_variables):
            raise ValueError("duplicate observation variables")
        if self.agent_hz <= 0:
            raise ValueError("agent_hz must be positive")
