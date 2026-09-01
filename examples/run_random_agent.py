"""Fly the steady-flight scenario with small random control perturbations.

Run headless:            uv run examples/run_random_agent.py
With FlightGear viewer:  uv run examples/run_random_agent.py --flightgear
"""

from __future__ import annotations

import argparse

import numpy as np

from aerogym import AeroGymEnv, FlightVariable
from aerogym.scenarios import make_steady_flight_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flightgear", action="store_true", help="launch FlightGear viewer")
    parser.add_argument("--seconds", type=float, default=60.0, help="episode length to fly")
    args = parser.parse_args()

    scenario = make_steady_flight_scenario()
    env = AeroGymEnv(scenario, render_mode="flightgear" if args.flightgear else None)
    rng = np.random.default_rng(0)

    obs, info = env.reset(seed=0)
    print(f"scenario: {scenario.name} | aircraft: {scenario.aircraft} | agent rate: {scenario.agent_hz} Hz")

    # Roughly-trimmed baseline controls with small random perturbations on top.
    baseline = {
        FlightVariable.ELEVATOR: 0.05,
        FlightVariable.AILERON: 0.0,
        FlightVariable.RUDDER: 0.0,
        FlightVariable.THROTTLE: 0.7,
    }
    n_steps = int(args.seconds * scenario.agent_hz)
    total_reward = 0.0
    for step in range(n_steps):
        action = np.array(
            [baseline[v] for v in scenario.action_variables], dtype=np.float32
        ) + rng.normal(0.0, 0.02, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % int(scenario.agent_hz) == 0:
            o = info["obs"]
            print(
                f"t={step / scenario.agent_hz:5.1f}s "
                f"alt={o[FlightVariable.ALTITUDE]:7.1f} m "
                f"hdg={o[FlightVariable.HEADING]:6.1f} deg "
                f"vs={o[FlightVariable.VERTICAL_SPEED]:+6.2f} m/s "
                f"pitch={o[FlightVariable.PITCH]:+6.2f} roll={o[FlightVariable.ROLL]:+6.2f} "
                f"aoa={o[FlightVariable.AOA]:+5.2f} reward={reward:+7.2f}"
            )
        if terminated or truncated:
            print(f"episode ended at step {step} (terminated={terminated}, truncated={truncated})")
            break
    print(f"total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
