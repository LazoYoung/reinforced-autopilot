"""Reference scenario: hold the initial heading and altitude in level flight."""

from __future__ import annotations

from aero_gym.scenario import InitialConditions, Scenario, Transition
from aero_gym.simulators.jsbsim.aircraft import profile_for
from aero_gym.variables import (
    ALL_ACTION_VARIABLES,
    ALL_OBSERVATION_VARIABLES,
    FlightVariable as V,
)


def heading_error_deg(current: float, target: float) -> float:
    """Signed shortest-path heading error in (-180, 180]."""
    return (target - current + 180.0) % 360.0 - 180.0


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

    Reward is highest (0) when the aircraft keeps the target altitude and
    heading with wings level; each error term subtracts proportionally. The
    episode terminates when the aircraft leaves a safe flight envelope.

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

    def reward_fn(t: Transition) -> tuple[float, bool]:
        obs = t.obs
        altitude_err = abs(obs[V.ALTITUDE] - target_altitude_ft)
        heading_err = abs(heading_error_deg(obs[V.HEADING], target_heading_deg))
        reward = -(
            altitude_err / 300.0
            + heading_err / 45.0
            + abs(obs[V.ROLL]) / 60.0
        )
        terminated = (
            obs[V.ALTITUDE] < 1000.0
            or abs(obs[V.ROLL]) > 75.0
            or abs(obs[V.PITCH]) > 45.0
            or abs(obs[V.AOA]) > 20.0
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
        observation_variables=ALL_OBSERVATION_VARIABLES,
        reward_fn=reward_fn,
        aircraft=aircraft,
        agent_hz=agent_hz,
        max_episode_steps=max_episode_steps,
    )
