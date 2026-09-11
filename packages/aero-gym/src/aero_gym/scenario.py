"""Scenario definition: what the agent controls, sees, and is rewarded for."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .variables import (
    ACTION_SPECS,
    ALL_ACTION_VARIABLES,
    ALL_OBSERVATION_VARIABLES,
    OBSERVATION_SPECS,
    FlightVariable,
    VariableSpec,
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
class EpisodeSetup:
    """One episode's starting situation and the targets that go with it.

    The context is not required to derive from the initial conditions. A hold
    task sets its target to the start; a "go to" task draws the two
    independently, which is why this is a separate mapping rather than the
    reset observation.
    """

    initial_conditions: InitialConditions
    context: Mapping[str, float] = field(default_factory=dict)


SetupFn = Callable[[np.random.Generator], EpisodeSetup]
"""Draws one episode's setup from the environment's own seeded generator."""


def fixed_setup(ic: InitialConditions, **context: float) -> SetupFn:
    """A sampler that always returns the same setup. The constant case."""
    setup = EpisodeSetup(ic, dict(context))
    return lambda rng: setup


@dataclass(frozen=True)
class Transition:
    """One agent step, in canonical units, passed to the reward function."""

    prev_obs: dict[FlightVariable, float]
    obs: dict[FlightVariable, float]
    action: dict[FlightVariable, float]
    context: Mapping[str, float] = field(default_factory=dict)
    """The episode's targets, as sampled by the scenario's setup_fn."""


RewardFn = Callable[[Transition], tuple[float, bool]]
"""Maps a transition to (reward, terminated)."""


@dataclass(frozen=True)
class ObservationFeature:
    """One entry of the observation Box, computed from the canonical readings.

    Separate from `observation_variables` because the canonical bounds are the
    physically valid range of a variable, not the range a task occupies:
    altitude is bounded [-1500, 85000] while a level-flight task lives within a
    few hundred feet of a target. A network fed the raw number sees an input
    that is huge in absolute terms and nearly constant in relative terms, and
    fails silently.

    A feature's bounds are declared by its author, not derived: for a hold task
    they are usually the limit that terminates on it, but a "go to" task starts
    far outside its tolerance band and needs a wider range than its terminator.
    """

    name: str
    spec: VariableSpec
    fn: Callable[[dict[FlightVariable, float], Mapping[str, float]], float]


def raw(
    var: FlightVariable,
    low: float | None = None,
    high: float | None = None,
) -> ObservationFeature:
    """Pass a canonical variable through, optionally narrowing its bounds."""
    base = OBSERVATION_SPECS[var]
    return ObservationFeature(
        var.value,
        VariableSpec(
            base.unit,
            base.low if low is None else low,
            base.high if high is None else high,
            base.description,
        ),
        lambda obs, context: obs[var],
    )


@dataclass(frozen=True)
class Scenario:
    """A fully-specified task for :class:`aero_gym.env.AeroGymEnv`."""

    name: str
    setup_fn: SetupFn
    reward_fn: RewardFn
    action_variables: tuple[FlightVariable, ...] = ALL_ACTION_VARIABLES
    observation_variables: tuple[FlightVariable, ...] = ALL_OBSERVATION_VARIABLES
    observation_features: tuple[ObservationFeature, ...] = ()
    context_keys: tuple[str, ...] = ()
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
        if not self.observation_features:
            # Default: the observed variables, in order, at canonical bounds.
            # Filled here so AeroGymEnv only ever reads observation_features.
            object.__setattr__(
                self,
                "observation_features",
                tuple(raw(v) for v in self.observation_variables),
            )
        names = [f.name for f in self.observation_features]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate observation feature names: {sorted(names)}")
