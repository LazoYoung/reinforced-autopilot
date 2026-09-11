"""Reference scenario: hold the initial heading and altitude in level flight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aero_gym.scenario import (
    EpisodeSetup,
    InitialConditions,
    ObservationFeature,
    Scenario,
    Transition,
    raw,
)
from aero_gym.simulators.jsbsim.aircraft import profile_for
from aero_gym.variables import ALL_ACTION_VARIABLES, VariableSpec, FlightVariable as V


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

_BANDS = ("altitude_band_ft", "heading_band_deg", "vertical_speed_band_fpm")
_LIMITS = (
    "roll_limit_deg",
    "pitch_limit_deg",
    "aoa_limit_deg",
    "roll_rate_scale_deg_s",
    "pitch_rate_scale_deg_s",
)
_JITTERS = ("altitude_jitter_ft", "heading_jitter_deg")


@dataclass(frozen=True)
class SteadyFlightParams:
    """Everything tweakable about the steady-flight task.

    The bands define both the observation bounds and the reward's normalizers,
    so narrowing one makes the task harder in all three places at once. The
    two rate scales are the only bounds not derived from a termination
    condition; they exist so an excursion saturates instead of wrecking the
    observation scale.

    max_episode_steps is deliberately absent. It is a field every Scenario has,
    so repeating it here would be boilerplate; a caller applies it with
    dataclasses.replace on the built scenario.
    """

    latitude_deg: float = 37.6191
    longitude_deg: float = -122.3816
    altitude_ft: float | None = None
    """None: the chosen aircraft's cruise altitude."""
    heading_deg: float = 270.0
    airspeed_kts: float | None = None
    """None: the chosen aircraft's cruise airspeed."""
    aircraft: str = "c172p"
    agent_hz: float = 10.0
    altitude_band_ft: float = 500.0
    heading_band_deg: float = 10.0
    vertical_speed_band_fpm: float = 1000.0
    roll_limit_deg: float = 75.0
    pitch_limit_deg: float = 45.0
    aoa_limit_deg: float = 20.0
    floor_altitude_ft: float = 1000.0
    roll_rate_scale_deg_s: float = 60.0
    pitch_rate_scale_deg_s: float = 30.0
    altitude_jitter_ft: float = 0.0
    """Half-width of the uniform draw around the target altitude."""
    heading_jitter_deg: float = 0.0
    """Half-width of the uniform draw around the target heading."""

    def __post_init__(self) -> None:
        if self.agent_hz <= 0:
            raise ValueError("agent_hz must be positive")
        for name in _BANDS + _LIMITS:
            # Bands are divisors in reward_fn, so zero produces inf or NaN and
            # a negative one inverts the comparison that terminates on it.
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in _JITTERS:
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} is a half-width and must not be negative, got "
                    f"{getattr(self, name)}"
                )


def make_steady_flight_scenario(params: SteadyFlightParams | None = None) -> Scenario:
    """Level-flight hold task starting near KSFO by default.

    The episode is a success when it survives its step limit inside the
    band: altitude within altitude_band_ft of target, heading within
    heading_band_deg, and vertical speed within vertical_speed_band_fpm.
    Reward is 1.0 on target and 0.0 at the band's corner. Leaving the band,
    or the wider flight envelope, ends the episode.

    Altitude and airspeed default to a cruise the chosen aircraft can hold, so
    that naming an aircraft is enough to get a flyable task. With a non-zero
    jitter the start is drawn per episode and the target follows it, which
    keeps the task a hold rather than turning it into an unreachable go-to.
    """
    p = params if params is not None else SteadyFlightParams()
    profile = profile_for(p.aircraft)
    base_altitude_ft = (
        profile.cruise_altitude_ft if p.altitude_ft is None else p.altitude_ft
    )
    airspeed_kts = (
        profile.cruise_airspeed_kts if p.airspeed_kts is None else p.airspeed_kts
    )
    base_heading_deg = p.heading_deg % 360.0

    def setup_fn(rng: np.random.Generator) -> EpisodeSetup:
        """Draw one episode's start, with the target following it.

        The two draws are unconditional: at the default jitter of 0.0,
        uniform(-0.0, 0.0) returns exactly 0.0, so the no-jitter path needs no
        branch and there is no second code path to rot.
        """
        altitude_ft = base_altitude_ft + rng.uniform(
            -p.altitude_jitter_ft, p.altitude_jitter_ft
        )
        heading_deg = (
            base_heading_deg + rng.uniform(-p.heading_jitter_deg, p.heading_jitter_deg)
        ) % 360.0
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
            {"target_altitude_ft": altitude_ft, "target_heading_deg": heading_deg},
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

    def reward_fn(t: Transition) -> tuple[float, bool]:
        """Reward in [0, 1] inside the band: 1.0 on target, 0.0 at the corner.

        Never negative in band, by design. Leaving the band ends the episode,
        so under a strictly-negative reward quitting on step 1 would score ~0
        and beat any real flying; Transition carries no elapsed-step count,
        so the reward function cannot price a death penalty itself.
        """
        obs = t.obs
        alt = abs(obs[V.ALTITUDE] - t.context["target_altitude_ft"]) / p.altitude_band_ft
        hdg = (
            abs(heading_error_deg(obs[V.HEADING], t.context["target_heading_deg"]))
            / p.heading_band_deg
        )
        vs = abs(obs[V.VERTICAL_SPEED]) / p.vertical_speed_band_fpm
        # Not clamped below: a violent excursion scores slightly negative on its
        # final step, which never competes with the forfeited future reward.
        reward = 1.0 - (alt + hdg + vs) / 3.0
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
        observation_variables=OBSERVATION_VARIABLES,
        observation_features=features,
        reward_fn=reward_fn,
        context_keys=("target_altitude_ft", "target_heading_deg"),
        aircraft=p.aircraft,
        agent_hz=p.agent_hz,
    )
