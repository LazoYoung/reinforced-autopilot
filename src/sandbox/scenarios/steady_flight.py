"""Reference scenario: hold the initial heading and altitude in level flight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

CONTROL_POSITIONS = (V.ELEVATOR_POS, V.AILERON_POS, V.RUDDER_POS)
"""What the smoothness term watches: the three aerodynamic axes.

Throttle is deliberately out. A movement penalty is minimized by pegging an
axis against a stop, and the throttle has one the aircraft can survive sitting
on: with throttle in the mean, two of three seeds held full throttle for most
of an episode and flew 25 kts fast. The aerodynamic axes have no such refuge,
which is why the same term behaves on them. hold_airspeed prices the throttle
instead, and prices it better, because it pays for the resulting speed rather
than for stillness: movement the aircraft needs stays free.
"""

_BANDS = (
    "altitude_band_ft",
    "heading_band_deg",
    "vertical_speed_band_fpm",
    "airspeed_band_kts",
)
_LIMITS = (
    "roll_limit_deg",
    "pitch_limit_deg",
    "aoa_limit_deg",
    "roll_rate_scale_deg_s",
    "pitch_rate_scale_deg_s",
)
_SCALES = ("aos_scale_deg", "control_rate_scale")
_WEIGHTS = ("coordination_weight", "smoothness_weight")
_JITTERS = ("altitude_jitter_ft", "heading_jitter_deg", "airspeed_jitter_kts")


@dataclass(frozen=True)
class SteadyFlightParams:
    """Everything tweakable about the steady-flight task.

    The bands define both the observation bounds and the reward's normalizers,
    so narrowing one makes the task harder in all three places at once. The two
    rate scales and airspeed_band_kts are the only bounds not derived from a
    termination condition; they exist so an excursion saturates instead of
    wrecking the observation scale.

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
    airspeed_jitter_kts: float = 0.0
    """Half-width of the uniform draw around the target airspeed.

    Unlike the other two this one changes the trim the episode needs, not just
    where it starts: a different speed wants a different elevator and throttle,
    so the agent cannot memorize one lever position and pass.
    """
    hold_airspeed: bool = False
    """Whether airspeed joins the task: observed, and a fourth share of accuracy."""
    airspeed_band_kts: float = 25.0
    """Airspeed error at which that share reaches zero, and its bound."""
    observe_sideslip: bool = False
    """Whether the agent sees sideslip. Off reproduces the original task."""
    aos_scale_deg: float = 5.0
    """Sideslip at which the coordination term reaches zero, and its bound."""
    coordination_weight: float = 0.0
    """Share of the reward paid for flying coordinated rather than slipping."""
    smoothness_weight: float = 0.0
    """Share of the reward paid for holding the controls still."""
    control_rate_scale: float = 0.1
    """Per-step control movement at which the smoothness term reaches zero.

    Couples to aero_train's env.delta: at the same value, one full increment
    on every axis costs the whole smoothness weight.
    """

    def __post_init__(self) -> None:
        if self.agent_hz <= 0:
            raise ValueError("agent_hz must be positive")
        for name in _WEIGHTS:
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} is a reward share and must not be negative, got "
                    f"{getattr(self, name)}"
                )
        # Strictly less than 1: accuracy is what makes this a flying task, and
        # at a zero share the agent is paid the same wherever it flies.
        if sum(getattr(self, name) for name in _WEIGHTS) >= 1.0:
            raise ValueError(
                f"{' + '.join(_WEIGHTS)} must leave the accuracy term a positive "
                f"share, got {sum(getattr(self, name) for name in _WEIGHTS)}"
            )
        # Pricing a variable the agent cannot see makes the task partially
        # observable: the same trap that ruled out penalizing yaw rate.
        if self.coordination_weight > 0 and not self.observe_sideslip:
            raise ValueError(
                "coordination_weight prices sideslip, so observe_sideslip must "
                "be True or the agent is penalized for a state it cannot see"
            )
        for name in _BANDS + _LIMITS + _SCALES:
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


@scenario("steady_flight", params=SteadyFlightParams)
def make_steady_flight_scenario(params: SteadyFlightParams | None = None) -> Scenario:
    """Level-flight hold task starting near KSFO by default.

    The episode is a success when it survives its step limit inside the
    band: altitude within altitude_band_ft of target, heading within
    heading_band_deg, and vertical speed within vertical_speed_band_fpm.
    Reward is 1.0 on target and 0.0 at the band's corner. Leaving the band,
    or the wider flight envelope, ends the episode. Airspeed is scored when
    hold_airspeed is set but never ends an episode: the aoa_limit_deg check
    already catches the slow end, and a fourth way to die would change the
    task more than pricing the throttle's effect does.

    Altitude and airspeed default to a cruise the chosen aircraft can hold, so
    that naming an aircraft is enough to get a flyable task. With a non-zero
    jitter the start is drawn per episode and the target follows it, which
    keeps the task a hold rather than turning it into an unreachable go-to.
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

    def setup_fn(rng: np.random.Generator) -> EpisodeSetup:
        """Draw one episode's start, with the target following it.

        All three draws are unconditional: at the default jitter of 0.0,
        uniform(-0.0, 0.0) returns exactly 0.0, so the no-jitter path needs no
        branch and there is no second code path to rot.
        """
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

        Never negative in band, by design. Leaving the band ends the episode,
        so under a strictly-negative reward quitting on step 1 would score ~0
        and beat any real flying; Transition carries no elapsed-step count,
        so the reward function cannot price a death penalty itself.

        The optional coordination and smoothness terms are mixed in as a convex
        combination rather than subtracted, which is what keeps that floor: each
        term is itself in [0, 1] and the shares sum to one, so no combination of
        them can drive an in-band step below zero. Both default to a zero share,
        which reproduces the accuracy-only reward exactly.

        hold_airspeed instead divides accuracy itself, adding airspeed error as
        a fourth equal share alongside altitude, heading and vertical speed.
        That changes the reward's scale, so returns are comparable only between
        runs that set it the same way.
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
            moved = sum(
                abs(obs[pos] - t.prev_obs[pos]) for pos in CONTROL_POSITIONS
            ) / len(CONTROL_POSITIONS)
            reward += p.smoothness_weight * (1.0 - min(1.0, moved / p.control_rate_scale))
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
    )
