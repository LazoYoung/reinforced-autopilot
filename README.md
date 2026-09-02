# aero-gym

Simulator-agnostic [Gymnasium](https://gymnasium.farama.org/) environments for
fixed-wing flight control, backed by [JSBSim](https://jsbsim.sourceforge.net/)
with optional live [FlightGear](https://www.flightgear.org/) visualization.

## Features

- **Scenario customization** — each `Scenario` selects which control axes
  (elevator, aileron, rudder, throttle) and which observations (heading,
  altitude, vertical speed, pitch, roll, yaw, angular rates, AoA, AoS) the
  agent sees, defines the reward function, the initial flight condition
  (lat, lon, altitude, heading, pitch, bank), and the aircraft model
  (`c172p` by default).
- **Simulator-agnostic core** — the Gymnasium env talks to an abstract
  `Simulator` interface; JSBSim is the first backend, others (e.g. X-Plane)
  can be added without touching the env or scenarios.
- **FlightGear visualization** — while JSBSim runs, simulation frames are
  streamed to FlightGear over UDP (native FDM protocol) so you can watch the
  agent fly in real time. The first `reset()` launches FlightGear at the
  scenario's initial position and blocks until its scenery has finished
  loading (polled via the FlightGear telnet server, port 5401 by default),
  so the episode never starts before you can see it.

## Installation

```sh
uv sync
```

JSBSim ships its aircraft/engine data inside the Python wheel — no separate
install needed. For visualization, install
[FlightGear](https://www.flightgear.org/download/) and, if `fgfs` is not on
your `PATH` or in the default install location, point the `AEROGYM_FGFS`
environment variable at the executable.

## Quick start

```python
from aerogym import AeroGymEnv
from aerogym.scenarios import make_steady_flight_scenario

env = AeroGymEnv(make_steady_flight_scenario())          # headless
# env = AeroGymEnv(..., render_mode="flightgear")        # with visualization

obs, info = env.reset(seed=0)
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
env.close()
```

Or run the bundled example (add `--flightgear` to watch it):

```sh
uv run examples/run_random_agent.py --flightgear
```

## Defining a scenario

```python
from aerogym.scenario import InitialConditions, Scenario
from aerogym.variables import FlightVariable as V

def my_reward(t):  # t: Transition(prev_obs, obs, action) — dicts keyed by FlightVariable
    reward = -abs(t.obs[V.ROLL]) / 60.0
    terminated = t.obs[V.ALTITUDE] < 300.0
    return reward, terminated

scenario = Scenario(
    name="wings-level",
    initial_conditions=InitialConditions(
        latitude_deg=37.62, longitude_deg=-122.38, altitude_m=1500.0,
        heading_deg=270.0, pitch_deg=0.0, roll_deg=20.0, airspeed_kts=95.0,
    ),
    action_variables=(V.AILERON, V.RUDDER),
    observation_variables=(V.ROLL, V.ROLL_RATE, V.YAW_RATE, V.AOS, V.ALTITUDE),
    reward_fn=my_reward,
    aircraft="c172p",
    agent_hz=10.0,
)
```

Units are canonical across all simulator backends: degrees for angles,
deg/s for rates, metres for altitude, m/s for vertical speed. Controls use
the pilot convention: +elevator = pitch up, +aileron = roll right,
+rudder = yaw right, throttle in [0, 1].

## Testing

```sh
uv run pytest
```

## Scope

RL algorithms are intentionally not included yet — this stage is about
verifying the integrity of the simulator interaction loop.
