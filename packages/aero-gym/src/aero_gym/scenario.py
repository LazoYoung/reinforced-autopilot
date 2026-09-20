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


Observation = dict[FlightVariable, float]
"""What the agent sees: one canonical reading per observed variable.

Keys are the scenario's `observation_variables`, values are in that variable's
canonical unit (degrees, deg/s, feet, fpm, normalized control positions). A
feature reads the ones it needs by key, for example
`obs[FlightVariable.ALTITUDE]`.
"""

Action = dict[FlightVariable, float]
"""What the agent commanded: one value per action variable.

The same shape as an Observation, deliberately named apart: the keys come from
the scenario's `action_variables` and the values follow the pilot sign
convention (+elevator is pitch up, +aileron is roll right, +rudder is yaw right).
"""

Context = Mapping[str, float]
"""One episode's targets, fixed for the whole episode.

Drawn alongside the initial conditions by the scenario's `setup_fn`, so a
feature can measure against *this* episode's target rather than a number fixed
when the scenario was built. Keys are the scenario's declared `context_keys`,
which the environment validates at reset, for example
`context["target_altitude_ft"]`.
"""


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
    context: Context = field(default_factory=dict)


SetupFn = Callable[[np.random.Generator], EpisodeSetup]
"""Draws one episode's setup from the environment's own seeded generator."""


def fixed_setup(ic: InitialConditions, **context: float) -> SetupFn:
    """A sampler that always returns the same setup. The constant case."""
    setup = EpisodeSetup(ic, dict(context))
    return lambda rng: setup


@dataclass(frozen=True)
class Transition:
    """One agent step, in canonical units, passed to the reward function.

    `prev_obs` is what the agent saw when it chose `action`, and `obs` is what
    resulted, so a reward function can score the change as well as the state.
    """

    prev_obs: Observation
    obs: Observation
    action: Action
    context: Context = field(default_factory=dict)
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

    `name` labels the entry, `spec` carries its unit and the bounds the value is
    clipped to, and `fn` computes it. `fn` is called once per step with this
    step's Observation and the episode's Context, and returns the single number
    that occupies this slot of the Box:

        ObservationFeature(
            "altitude_error_ft",
            VariableSpec("ft", -500.0, 500.0, "altitude above target"),
            lambda obs, context: obs[FlightVariable.ALTITUDE]
            - context["target_altitude_ft"],
        )
    """

    name: str
    spec: VariableSpec
    fn: Callable[[Observation, Context], float]


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
    action_delta: tuple[float, ...] | None = None
    """Per-axis action step as a fraction of each action variable's declared
    range, positional against action_variables. None poses the task for
    continuous actions, and the training side applies no stepping wrapper."""

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
        if self.action_delta is not None:
            object.__setattr__(
                self, "action_delta", tuple(float(s) for s in self.action_delta)
            )
            if len(self.action_delta) != len(self.action_variables):
                raise ValueError(
                    f"action_delta has {len(self.action_delta)} entries for "
                    f"{len(self.action_variables)} action variables"
                )
            for var, step in zip(self.action_variables, self.action_delta):
                if not 0.0 < step <= 1.0:
                    raise ValueError(
                        f"action_delta for {var.value} must be in (0, 1], got {step}"
                    )
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
