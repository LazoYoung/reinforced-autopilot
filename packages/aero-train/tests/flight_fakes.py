"""A toy aircraft and a hold task flown on it, for the export tests.

Deterministic, with no physics and no JSBSim: enough motion that an untrained
policy's choices change what it sees, which is all the export check needs.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import gymnasium as gym

from aero_gym import AeroGymEnv
from aero_gym.scenario import (
    ContextVariable,
    EpisodeSetup,
    InitialConditions,
    ObservationFeature,
    Scenario,
    Transition,
    raw,
)
from aero_gym.simulator import Simulator
from aero_gym.variables import FlightVariable as V, VariableSpec
from aero_train.envs.delta import DeltaAction
from aero_train.envs.scale import ScaleObservation

ACTIONS = (V.ELEVATOR, V.AILERON, V.THROTTLE)
N_FEATURES = 4
N_ACTIONS = 3 ** len(ACTIONS)


class ToyAircraft(Simulator):
    """Controls integrate into attitude, and attitude into altitude and heading.

    Every axis starts on a different command, so a policy file that mixes up
    two axes' names cannot pass for correct.
    """

    def __init__(self) -> None:
        self.state: dict[V, float] = {}
        self.controls: dict[V, float] = {}

    @property
    def dt(self) -> float:
        return 0.1

    def reset(
        self, aircraft: str, initial_conditions: InitialConditions, settle_steps: int = 1
    ) -> None:
        ic = initial_conditions
        self.state = {
            V.ALTITUDE: ic.altitude_ft,
            V.HEADING: ic.heading_deg,
            V.ROLL: 0.0,
            V.PITCH: 0.0,
            V.AIRSPEED: ic.airspeed_kts,
        }
        self.controls = {V.ELEVATOR: 0.1, V.AILERON: -0.2, V.THROTTLE: 0.6}

    def set_control(self, var: V, value: float) -> None:
        self.controls[var] = value

    def get_control(self, var: V) -> float:
        return self.controls[var]

    def run(self) -> None:
        s, c = self.state, self.controls
        s[V.PITCH] += 2.0 * c[V.ELEVATOR]
        s[V.ROLL] += 3.0 * c[V.AILERON]
        s[V.ALTITUDE] += 5.0 * s[V.PITCH]
        s[V.HEADING] = (s[V.HEADING] + 0.2 * s[V.ROLL]) % 360.0
        s[V.AIRSPEED] += 2.0 * (c[V.THROTTLE] - 0.5) - 0.1 * s[V.PITCH]

    def observe(self, var: V) -> float:
        return self.state[var]


def _setup(rng) -> EpisodeSetup:
    altitude = 5000.0 + rng.uniform(-200.0, 200.0)
    heading = rng.uniform(0.0, 360.0)
    return EpisodeSetup(
        InitialConditions(0.0, 0.0, altitude, heading, airspeed_kts=100.0),
        {
            "target_altitude_ft": altitude,
            "target_heading_deg": heading,
            "target_airspeed_kts": 100.0,
        },
    )


def _reward(t: Transition) -> tuple[float, bool]:
    error = abs(t.obs[V.ALTITUDE] - t.context["target_altitude_ft"])
    return 1.0 - error / 500.0, error > 500.0


def toy_scenario(**changes) -> Scenario:
    """Airspeed is observed and its target declared, but no feature reads
    either, so a policy file for this task must take neither."""
    scenario = Scenario(
        name="toy-hold",
        setup_fn=_setup,
        reward_fn=_reward,
        action_variables=ACTIONS,
        observation_variables=(V.ALTITUDE, V.HEADING, V.ROLL, V.PITCH, V.AIRSPEED),
        observation_features=(
            ObservationFeature(
                "altitude_error_ft",
                VariableSpec("ft", -500.0, 500.0, "altitude minus target"),
                lambda obs, ctx: obs[V.ALTITUDE] - ctx["target_altitude_ft"],
            ),
            ObservationFeature(
                "heading_error_deg",
                VariableSpec("deg", -30.0, 30.0, "target minus heading, wrapped"),
                lambda obs, ctx: (ctx["target_heading_deg"] - obs[V.HEADING] + 180.0)
                % 360.0
                - 180.0,
            ),
            raw(V.ROLL, -45.0, 45.0),
            raw(V.PITCH, -30.0, 30.0),
        ),
        context_variables=(
            ContextVariable(
                "target_altitude_ft", VariableSpec("ft", 4800.0, 5200.0, "altitude to hold")
            ),
            ContextVariable(
                "target_heading_deg", VariableSpec("deg", 0.0, 360.0, "heading to hold")
            ),
            ContextVariable(
                "target_airspeed_kts", VariableSpec("kt", 100.0, 100.0, "airspeed to hold")
            ),
        ),
        max_episode_steps=200,
        action_delta=(0.05, 0.1, 0.2),
    )
    return dataclasses.replace(scenario, **changes)


def toy_env_fn(scenario: Scenario | None = None) -> Callable[[], gym.Env]:
    """DeltaAction(ScaleObservation(AeroGymEnv)), as make_env_fn stacks a stepped scenario."""
    scenario = scenario or toy_scenario()

    def make() -> gym.Env:
        return DeltaAction(
            ScaleObservation(AeroGymEnv(scenario, simulator=ToyAircraft())),
            delta=scenario.action_delta,
        )

    return make
