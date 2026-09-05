# aero-gym

Simulator-agnostic [Gymnasium](https://gymnasium.farama.org/) environments for
fixed-wing flight control, backed by [JSBSim](https://jsbsim.sourceforge.net/)
with optional live [FlightGear](https://www.flightgear.org/) visualization.

## Features

- **Scenario customization**: each `Scenario` selects which control axes and observations the agent sees, defines the reward function, the initial flight condition, and the aircraft model.
- **Simulator-agnostic core**: the Gymnasium env talks to `Simulator` interface which abstracts the backend simulator instance.
- **Aircraft profiles**: everything aerogym needs to know about an aircraft is keyed on its JSBSim model name in one registry, so naming an aircraft is enough to get a flyable task.
- **FlightGear visualization**: JSBSim simulation frames are streamed to FlightGear over UDP, so you can watch the agent fly in real time — with a cockpit that tracks the simulation rather than contradicting it.

## Installation

> This project requires [uv](https://docs.astral.sh/uv/) package manager. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) before you proceed.

```sh
uv sync
```

The python wheel includes the JSBSim simulator backend.

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

Or run the bundled example script:

```sh
uv run examples/run_random_agent.py                  # headless Cessna 172
uv run examples/run_random_agent.py --flightgear     # with visualization
uv run examples/run_random_agent.py --aircraft 737   # fly a different type
```

To use `--flightgear` to visualize the agent, install [FlightGear](https://www.flightgear.org/download/).

## Defining a scenario

```python
from aerogym.scenario import InitialConditions, Scenario
from aerogym.variables import FlightVariable as V

def my_reward(t):  # t: Transition(prev_obs, obs, action) — dicts keyed by FlightVariable
    reward = -abs(t.obs[V.ROLL]) / 60.0
    terminated = t.obs[V.ALTITUDE] < 1000.0
    return reward, terminated

scenario = Scenario(
    name="wings-level",
    initial_conditions=InitialConditions(
        latitude_deg=37.62, longitude_deg=-122.38, altitude_ft=5000.0,
        heading_deg=270.0, pitch_deg=0.0, roll_deg=20.0, airspeed_kts=95.0,
    ),
    action_variables=(V.AILERON, V.RUDDER),
    observation_variables=(V.ROLL, V.ROLL_RATE, V.YAW_RATE, V.AOS, V.ALTITUDE),
    reward_fn=my_reward,
    aircraft="c172p",
    agent_hz=10.0,
)
```

Units are canonical across all simulator backends.

|Flight variable|Unit|
|---------------|----|
| Angle | Degrees |
| Angle rate | deg/s |
| Altitude | feet |
| Vertical speed | fpm |
| Elevator | + for Pitch Up |
| Aileron | + for Roll Right |
| Rudder | + for Yaw Right |
| Throttle | [0, 1] |

## Aircraft

`Scenario.aircraft` is a JSBSim model name. Everything else aerogym needs to know about an
aircraft — what FlightGear calls it, a cruise it can hold, how its engines and controls start,
what its cockpit reads — hangs off that name in a single profile registry
(`src/aerogym/simulators/jsbsim/aircraft.py`). Profiles ship for `c172p` and `737`.

Naming an aircraft is therefore enough to get a flyable task: the scenario takes its initial
altitude and airspeed from the profile when you don't specify them.

```python
make_steady_flight_scenario(aircraft="c172p")  # 5000 ft at 95 kts
make_steady_flight_scenario(aircraft="737")    # 30000 ft at 280 kts
```

An aircraft with no profile still flies and still renders — JSBSim and FlightGear agree on most
model names, so the name itself is the fallback. Adding a profile is one entry in that one file.

## Testing

```sh
uv run pytest
```
