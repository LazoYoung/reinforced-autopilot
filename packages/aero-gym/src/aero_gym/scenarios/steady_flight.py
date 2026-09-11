"""Reference scenario: hold the initial heading and altitude in level flight."""

from __future__ import annotations

from aero_gym.scenario import InitialConditions, ObservationFeature, Scenario, Transition, raw
from aero_gym.simulators.jsbsim.aircraft import profile_for
from aero_gym.variables import ALL_ACTION_VARIABLES, VariableSpec, FlightVariable as V


def heading_error_deg(current: float, target: float) -> float:
    """Signed shortest-path heading error in (-180, 180]."""
    return (target - current + 180.0) % 360.0 - 180.0


ALTITUDE_BAND_FT = 500.0
HEADING_BAND_DEG = 10.0
VERTICAL_SPEED_BAND_FPM = 1000.0

ROLL_LIMIT_DEG = 75.0
PITCH_LIMIT_DEG = 45.0
AOA_LIMIT_DEG = 20.0
FLOOR_ALTITUDE_FT = 1000.0

# The only two bounds not derived from a termination condition. _to_array clips,
# so an excursion saturates instead of wrecking the observation scale.
ROLL_RATE_SCALE_DEG_S = 60.0
PITCH_RATE_SCALE_DEG_S = 30.0

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


def make_steady_flight_scenario(
    latitude_deg: float = 37.6191,
    longitude_deg: float = -122.3816,
    altitude_ft: float | None = None,
    heading_deg: float = 270.0,
    airspeed_kts: float | None = None,
    aircraft: str = "c172p",
    agent_hz: float = 10.0,
    max_episode_steps: int | None = 2000,
) -> Scenario:
    """Level-flight hold task starting near KSFO by default.

    The episode is a success when it survives its step limit inside the
    band: altitude within ALTITUDE_BAND_FT of target, heading within
    HEADING_BAND_DEG, and vertical speed within VERTICAL_SPEED_BAND_FPM.
    Reward is 1.0 on target and 0.0 at the band's corner. Leaving the band,
    or the wider flight envelope, ends the episode.

    Altitude and airspeed default to a cruise the chosen aircraft can hold, so
    that naming an aircraft is enough to get a flyable task.
    """
    profile = profile_for(aircraft)
    if altitude_ft is None:
        altitude_ft = profile.cruise_altitude_ft
    if airspeed_kts is None:
        airspeed_kts = profile.cruise_airspeed_kts
    target_altitude_ft = altitude_ft
    target_heading_deg = heading_deg % 360.0

    features = (
        ObservationFeature(
            "altitude_error_ft",
            VariableSpec("ft", -ALTITUDE_BAND_FT, ALTITUDE_BAND_FT, "altitude minus target"),
            # actual - target: positive means above target, negative means below.
            lambda obs: obs[V.ALTITUDE] - target_altitude_ft,
        ),
        ObservationFeature(
            "heading_error_deg",
            VariableSpec(
                "deg", -HEADING_BAND_DEG, HEADING_BAND_DEG, "signed shortest-path heading error"
            ),
            # target - current, wrapped short: positive means turn right to
            # close it, the opposite convention from altitude_error_ft above.
            lambda obs: heading_error_deg(obs[V.HEADING], target_heading_deg),
        ),
        raw(V.VERTICAL_SPEED, -VERTICAL_SPEED_BAND_FPM, VERTICAL_SPEED_BAND_FPM),
        raw(V.ROLL, -ROLL_LIMIT_DEG, ROLL_LIMIT_DEG),
        raw(V.PITCH, -PITCH_LIMIT_DEG, PITCH_LIMIT_DEG),
        raw(V.AOA, -AOA_LIMIT_DEG, AOA_LIMIT_DEG),
        raw(V.ROLL_RATE, -ROLL_RATE_SCALE_DEG_S, ROLL_RATE_SCALE_DEG_S),
        raw(V.PITCH_RATE, -PITCH_RATE_SCALE_DEG_S, PITCH_RATE_SCALE_DEG_S),
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
        alt = abs(obs[V.ALTITUDE] - target_altitude_ft) / ALTITUDE_BAND_FT
        hdg = abs(heading_error_deg(obs[V.HEADING], target_heading_deg)) / HEADING_BAND_DEG
        vs = abs(obs[V.VERTICAL_SPEED]) / VERTICAL_SPEED_BAND_FPM
        # Not clamped below: a violent excursion scores slightly negative on its
        # final step, which never competes with the forfeited future reward.
        reward = 1.0 - (alt + hdg + vs) / 3.0
        terminated = (
            alt > 1.0
            or hdg > 1.0
            or vs > 1.0
            # Mostly preempted by the band, but a fast roll can exceed the limit
            # inside it, and the floor matters at a low target altitude.
            or abs(obs[V.ROLL]) > ROLL_LIMIT_DEG
            or abs(obs[V.PITCH]) > PITCH_LIMIT_DEG
            or abs(obs[V.AOA]) > AOA_LIMIT_DEG
            or obs[V.ALTITUDE] < FLOOR_ALTITUDE_FT
        )
        return reward, terminated

    return Scenario(
        name="steady-flight",
        initial_conditions=InitialConditions(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_ft=altitude_ft,
            heading_deg=heading_deg,
            pitch_deg=0.0,
            roll_deg=0.0,
            airspeed_kts=airspeed_kts,
        ),
        action_variables=ALL_ACTION_VARIABLES,
        observation_variables=OBSERVATION_VARIABLES,
        observation_features=features,
        reward_fn=reward_fn,
        aircraft=aircraft,
        agent_hz=agent_hz,
        max_episode_steps=max_episode_steps,
    )
