"""Fly the steady-flight scenario with small random control perturbations.

Run headless:            uv run examples/run_random_agent.py
With FlightGear viewer:  uv run examples/run_random_agent.py --flightgear
Fly a different type:    uv run examples/run_random_agent.py --aircraft 737
"""

from __future__ import annotations

import argparse

import numpy as np

from aero_gym import AeroGymEnv, FlightVariable
from aero_gym.scenarios import make_steady_flight_scenario
from aero_gym.simulators.jsbsim.aircraft import profile_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flightgear", action="store_true", help="launch FlightGear viewer")
    parser.add_argument("--seconds", type=float, default=60.0, help="episode length to fly")
    parser.add_argument(
        "--aircraft", default="c172p", help="JSBSim aircraft model, e.g. c172p or 737"
    )
    args = parser.parse_args()

    scenario = make_steady_flight_scenario(aircraft=args.aircraft)
    env = AeroGymEnv(scenario, render_mode="flightgear" if args.flightgear else None)
    rng = np.random.default_rng(0)

    obs, info = env.reset(seed=0)
    print(f"scenario: {scenario.name} | aircraft: {scenario.aircraft} | agent rate: {scenario.agent_hz} Hz")

    # Roughly-trimmed baseline controls, perturbed. A profile that sets pitch
    # trim needs no standing elevator; one without it holds the nose up itself.
    startup = profile_for(scenario.aircraft).startup
    baseline = {
        FlightVariable.ELEVATOR: 0.0 if "fcs/pitch-trim-cmd-norm" in startup else 0.05,
        FlightVariable.AILERON: 0.0,
        FlightVariable.RUDDER: 0.0,
        FlightVariable.THROTTLE: startup.get(
            "fcs/throttle-cmd-norm", startup.get("fcs/throttle-cmd-norm[0]", 0.7)
        ),
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
                f"alt={o[FlightVariable.ALTITUDE]:7.1f} ft "
                f"hdg={o[FlightVariable.HEADING]:6.1f} deg "
                f"vs={o[FlightVariable.VERTICAL_SPEED]:+7.1f} fpm "
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
