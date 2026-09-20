"""Reference scenario: hold the initial heading and altitude in level flight."""

from __future__ import annotations

from dataclasses import field
from typing import Annotated

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from aero_gym.registry import scenario
from aero_gym.scenario import (
    EpisodeSetup,
    InitialConditions,
    ObservationFeature,
    Scenario,
    Transition,
    raw,
)
from aero_gym.simulators.jsbsim.aircraft import profile_for
from aero_gym.variables import (
    ACTION_SPECS,
    ALL_ACTION_VARIABLES,
    VariableSpec,
    FlightVariable as V,
)

from ..util import delta_fractions


def heading_error_deg(current: float, target: float) -> float:
    """Signed shortest-path heading error in (-180, 180]."""
    return (target - current + 180.0) % 360.0 - 180.0


OBSERVATION_VARIABLES = (
    V.ALTITUDE,
    V.HEADING,
    V.VERTICAL_SPEED,
    V.ROLL,
    V.PITCH,
    V.AOA,
    V.ROLL_RATE,
    V.PITCH_RATE,
    V.ELEVATOR_POS,
    V.AILERON_POS,
    V.RUDDER_POS,
    V.THROTTLE_POS,
)

CONTROL_POSITIONS = {
    V.ELEVATOR_POS: V.ELEVATOR,
    V.AILERON_POS: V.AILERON,
    V.RUDDER_POS: V.RUDDER,
}
"""Positions the smoothness term watches, each keyed to the command that moves it."""

DEFAULT_DELTA = {"elevator": 0.015, "aileron": 0.015, "rudder": 0.015, "throttle": 0.1}
"""Action step per axis, as a fraction of that axis's declared range."""

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(gt=0, le=1)]


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class SteadyFlightParams:
    """Everything tweakable about the steady-flight task.

    The bands set both the observation bounds and the reward's normalizers.
    """

    latitude_deg: float = 37.6191
    longitude_deg: float = -122.3816
    altitude_ft: float | None = None
    """None: the chosen aircraft's cruise altitude."""
    heading_deg: float = 270.0
    airspeed_kts: float | None = None
    """None: the chosen aircraft's cruise airspeed."""
    aircraft: str = "c172p"
    agent_hz: Positive = 10.0
    altitude_band_ft: Positive = 500.0
    heading_band_deg: Positive = 10.0
    vertical_speed_band_fpm: Positive = 1000.0
    roll_limit_deg: Positive = 75.0
    pitch_limit_deg: Positive = 45.0
    aoa_limit_deg: Positive = 20.0
    floor_altitude_ft: float = 1000.0
    roll_rate_scale_deg_s: Positive = 60.0
    pitch_rate_scale_deg_s: Positive = 30.0
    altitude_jitter_ft: NonNegative = 0.0
    """Half-width of the uniform draw around the target altitude."""
    heading_jitter_deg: NonNegative = 0.0
    """Half-width of the uniform draw around the target heading."""
    airspeed_jitter_kts: NonNegative = 0.0
    """Half-width of the uniform draw around the target airspeed."""
    hold_airspeed: bool = False
    """Whether airspeed joins the task: observed, and a fourth share of accuracy."""
    airspeed_band_kts: Positive = 25.0
    """Airspeed error at which that share reaches zero, and its bound."""
    observe_sideslip: bool = False
    """Whether the agent sees sideslip. Off reproduces the original task."""
    aos_scale_deg: Positive = 5.0
    """Sideslip at which the coordination term reaches zero, and its bound."""
    coordination_weight: NonNegative = 0.0
    """Share of the reward paid for flying coordinated rather than slipping."""
    smoothness_weight: NonNegative = 0.0
    """Share of the reward paid for holding the controls still."""
    delta: Fraction | dict[str, Fraction] | None = field(
        default_factory=lambda: dict(DEFAULT_DELTA)
    )
    """Action step as a fraction of each axis's declared range.

    One number for every axis, a mapping naming each axis, or None for a
    continuous task. Also the unit the smoothness term counts movement in.
    """

    @model_validator(mode="after")
    def check_combinations(self) -> SteadyFlightParams:
        shares = self.coordination_weight + self.smoothness_weight
        if shares >= 1.0:
            raise ValueError(
                "coordination_weight + smoothness_weight must leave the accuracy term "
                f"a positive share, got {shares}"
            )
        if self.coordination_weight > 0 and not self.observe_sideslip:
            raise ValueError("coordination_weight needs observe_sideslip=True")
        if self.smoothness_weight > 0 and self.delta is None:
            raise ValueError("smoothness_weight needs a delta")
        delta_fractions(self.delta)
        return self


@scenario("steady_flight", params=SteadyFlightParams)
def make_steady_flight_scenario(params: SteadyFlightParams | None = None) -> Scenario:
    """Level-flight hold task starting near KSFO by default.

    An episode succeeds by surviving its step limit inside the altitude, heading
    and vertical-speed bands; leaving a band or the flight envelope ends it.
    Airspeed is scored when hold_airspeed is set but never ends an episode.
    Altitude and airspeed default to the aircraft's cruise, and a jittered start
    moves the target with it.
    """
    p = params or SteadyFlightParams()
    profile = profile_for(p.aircraft)
    # `or` would read an explicit 0 as unset and substitute the cruise value.
    base_altitude_ft = (
        profile.cruise_altitude_ft if p.altitude_ft is None else p.altitude_ft
    )
    airspeed_kts = (
        profile.cruise_airspeed_kts if p.airspeed_kts is None else p.airspeed_kts
    )
    base_heading_deg = p.heading_deg % 360.0
    base_airspeed_kts = airspeed_kts
    fractions = delta_fractions(p.delta)
    step_of = dict(zip(ALL_ACTION_VARIABLES, fractions or ()))
    # Absolute travel of one action step on each priced axis: the smoothness unit.
    control_step = {
        pos: step_of[cmd] * (ACTION_SPECS[cmd].high - ACTION_SPECS[cmd].low)
        for pos, cmd in CONTROL_POSITIONS.items()
        if cmd in step_of
    }

    def setup_fn(rng: np.random.Generator) -> EpisodeSetup:
        """Draw one episode's start, with the target following it."""
        altitude_ft = base_altitude_ft + rng.uniform(
            -p.altitude_jitter_ft, p.altitude_jitter_ft
        )
        heading_deg = (
            base_heading_deg + rng.uniform(-p.heading_jitter_deg, p.heading_jitter_deg)
        ) % 360.0
        airspeed_kts = base_airspeed_kts + rng.uniform(
            -p.airspeed_jitter_kts, p.airspeed_jitter_kts
        )
        return EpisodeSetup(
            InitialConditions(
                latitude_deg=p.latitude_deg,
                longitude_deg=p.longitude_deg,
                altitude_ft=altitude_ft,
                heading_deg=heading_deg,
                pitch_deg=0.0,
                roll_deg=0.0,
                airspeed_kts=airspeed_kts,
            ),
            {
                "target_altitude_ft": altitude_ft,
                "target_heading_deg": heading_deg,
                "target_airspeed_kts": airspeed_kts,
            },
        )

    features = (
        ObservationFeature(
            "altitude_error_ft",
            VariableSpec(
                "ft", -p.altitude_band_ft, p.altitude_band_ft, "altitude minus target"
            ),
            # actual - target: positive means above target, negative means below.
            lambda obs, ctx: obs[V.ALTITUDE] - ctx["target_altitude_ft"],
        ),
        ObservationFeature(
            "heading_error_deg",
            VariableSpec(
                "deg",
                -p.heading_band_deg,
                p.heading_band_deg,
                "signed shortest-path heading error",
            ),
            # target - current, wrapped short: positive means turn right to
            # close it, the opposite convention from altitude_error_ft above.
            lambda obs, ctx: heading_error_deg(obs[V.HEADING], ctx["target_heading_deg"]),
        ),
        raw(V.VERTICAL_SPEED, -p.vertical_speed_band_fpm, p.vertical_speed_band_fpm),
        raw(V.ROLL, -p.roll_limit_deg, p.roll_limit_deg),
        raw(V.PITCH, -p.pitch_limit_deg, p.pitch_limit_deg),
        raw(V.AOA, -p.aoa_limit_deg, p.aoa_limit_deg),
        raw(V.ROLL_RATE, -p.roll_rate_scale_deg_s, p.roll_rate_scale_deg_s),
        raw(V.PITCH_RATE, -p.pitch_rate_scale_deg_s, p.pitch_rate_scale_deg_s),
        raw(V.ELEVATOR_POS),
        raw(V.AILERON_POS),
        raw(V.RUDDER_POS),
        raw(V.THROTTLE_POS),
    )
    # Appended, never inserted, and sideslip always before airspeed: the Box
    # layout is positional, so any other order moves every later input.
    if p.observe_sideslip:
        features += (raw(V.AOS, -p.aos_scale_deg, p.aos_scale_deg),)
    if p.hold_airspeed:
        features += (
            ObservationFeature(
                "airspeed_error_kts",
                VariableSpec(
                    "kt",
                    -p.airspeed_band_kts,
                    p.airspeed_band_kts,
                    "airspeed minus target",
                ),
                # actual - target, matching altitude_error_ft: positive is fast.
                lambda obs, ctx: obs[V.AIRSPEED] - ctx["target_airspeed_kts"],
            ),
        )
    observation_variables = (
        OBSERVATION_VARIABLES
        + ((V.AOS,) if p.observe_sideslip else ())
        + ((V.AIRSPEED,) if p.hold_airspeed else ())
    )

    def reward_fn(t: Transition) -> tuple[float, bool]:
        """Reward in [0, 1] inside the band: 1.0 on target, 0.0 at the corner.

        Coordination and smoothness mix in as a convex combination with accuracy;
        hold_airspeed adds airspeed error as a fourth equal share of accuracy.
        """
        obs = t.obs
        alt = abs(obs[V.ALTITUDE] - t.context["target_altitude_ft"]) / p.altitude_band_ft
        hdg = (
            abs(heading_error_deg(obs[V.HEADING], t.context["target_heading_deg"]))
            / p.heading_band_deg
        )
        vs = abs(obs[V.VERTICAL_SPEED]) / p.vertical_speed_band_fpm
        errors = [alt, hdg, vs]
        if p.hold_airspeed:
            # Clamped, unlike the three above: nothing terminates on airspeed,
            # so an unclamped share could sink an in-band step below zero.
            errors.append(
                min(
                    1.0,
                    abs(obs[V.AIRSPEED] - t.context["target_airspeed_kts"])
                    / p.airspeed_band_kts,
                )
            )
        # Not clamped below: a violent excursion scores slightly negative on its
        # final step, which never competes with the forfeited future reward.
        accuracy = 1.0 - sum(errors) / len(errors)
        reward = (1.0 - p.coordination_weight - p.smoothness_weight) * accuracy
        if p.coordination_weight > 0.0:
            # Magnitude, not sign: a signed term would pay for slipping one way.
            slip = min(1.0, abs(obs[V.AOS]) / p.aos_scale_deg)
            reward += p.coordination_weight * (1.0 - slip)
        if p.smoothness_weight > 0.0:
            # Counted in action steps per axis, so a full step on every priced
            # axis costs the whole weight whatever each axis's delta is.
            steps = sum(
                abs(obs[pos] - t.prev_obs[pos]) / control_step[pos]
                for pos in CONTROL_POSITIONS
            ) / len(CONTROL_POSITIONS)
            reward += p.smoothness_weight * (1.0 - min(1.0, steps))
        terminated = (
            alt > 1.0
            or hdg > 1.0
            or vs > 1.0
            # Mostly preempted by the band, but a fast roll can exceed the limit
            # inside it, and the floor matters at a low target altitude.
            or abs(obs[V.ROLL]) > p.roll_limit_deg
            or abs(obs[V.PITCH]) > p.pitch_limit_deg
            or abs(obs[V.AOA]) > p.aoa_limit_deg
            or obs[V.ALTITUDE] < p.floor_altitude_ft
        )
        return reward, terminated

    return Scenario(
        name="steady-flight",
        setup_fn=setup_fn,
        action_variables=ALL_ACTION_VARIABLES,
        observation_variables=observation_variables,
        observation_features=features,
        reward_fn=reward_fn,
        context_keys=(
            "target_altitude_ft",
            "target_heading_deg",
            "target_airspeed_kts",
        ),
        aircraft=p.aircraft,
        agent_hz=p.agent_hz,
        action_delta=fractions,
    )
