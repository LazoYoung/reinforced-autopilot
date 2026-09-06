# reinforced-autopilot

Reinforcement learning for fixed-wing flight control.

Two packages in one [uv](https://docs.astral.sh/uv/) workspace:

| Package | What it is |
|---|---|
| [`aero-gym`](packages/aero-gym) | Simulator-agnostic [Gymnasium](https://gymnasium.farama.org/) environments for fixed-wing flight, backed by [JSBSim](https://jsbsim.sourceforge.net/) with optional live [FlightGear](https://www.flightgear.org/) visualization. |
| [`aero-train`](packages/aero-train) | The training framework — config, rollout, replay, algorithms, tracking, checkpointing, and the `autopilot` CLI. |

The dependency runs one way only: `aero-train` depends on `aero-gym`, and inside
`aero-train` just one module (`aero_train.envs`) is allowed to import it. That keeps the
learning stack environment-agnostic, which is what lets it train on CartPole — so when a
flight run fails to learn, a bug in the algorithm and a bug in the flight model are
distinguishable. A test enforces the rule rather than trusting it.

## Installation

> Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.
> [Install uv](https://docs.astral.sh/uv/getting-started/installation/) before you proceed.

```sh
uv sync
```

The Python wheel includes the JSBSim simulator backend. FlightGear is optional and only
needed for visualization — [install it separately](https://www.flightgear.org/download/).

## Quick start: train something

```sh
uv run autopilot run configs/dqn_cartpole.yaml
```

That trains DQN on CartPole-v1 — the end-to-end proof that the stack learns, and the
fastest way to check an install works. Checkpoints land under `runs/`, metrics under
`mlruns/`.

```sh
# override any config key without editing the file; applied before validation,
# so an override can't slip past a config check
uv run autopilot run configs/dqn_cartpole.yaml --override algo.lr=1e-3

# extend a finished run past its original budget
uv run autopilot resume runs/dqn-cartpole/checkpoints/latest.pt --total-steps 120000

# watch a trained policy fly, greedily
uv run autopilot eval runs/dqn-cartpole/checkpoints/best.pt --episodes 20
uv run autopilot eval runs/dqn-cartpole/checkpoints/best.pt --flightgear
```

Two checkpoints are kept: `latest.pt` and `best.pt` (by evaluation return). Prefer
`best.pt` for evaluation — DQN can reach a good policy and then lose it, so the last
snapshot is not reliably the best one. Saves are atomic, so an interrupted write cannot
leave a truncated checkpoint behind.

## Quick start: use the environment directly

```python
from aero_gym import AeroGymEnv
from aero_gym.scenarios import make_steady_flight_scenario

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

## The training stack

A run config is YAML that loads into frozen dataclasses, so a misspelled key is a
`TypeError` at load time instead of a setting silently ignored three hours into training.
See [`configs/dqn_cartpole.yaml`](configs/dqn_cartpole.yaml) for a complete worked example.

```yaml
run_name: dqn-cartpole
total_steps: 60000        # frames, not gradient updates
env:      { id: CartPole-v1 }
algo:     { type: dqn, lr: 0.0005, buffer_size: 50000, updates_per_iter: 32 }
rollout:  { engine: local, envs_per_worker: 8, frames_per_iter: 128 }
tracking: { enabled: true, experiment: reinforced-autopilot }
```

What is in place today:

- **Algorithms** — DQN, with n-step returns. `Algorithm` is an ABC, so more slot in beside it.
- **Rollout** — `engine: local` runs a `SyncVectorEnv` in-process. `engine: ray` is
  accepted by config validation but raises `NotImplementedError`; the distributed engine
  is future work.
- **Experience** — collection is lockstep across a vector of environments, carrying time
  as an explicit axis, so `terminated` and `truncated` stay distinct all the way through.
  A truncated episode bootstraps; a terminated one does not.
- **Tracking** — MLflow, defaulting to a local `./mlruns` file store. Point
  `MLFLOW_TRACKING_URI` or `tracking.tracking_uri` at a server to move to one, no code change.
  `tracking.enabled: false` turns it off entirely.
- **Checkpointing** — atomic, latest + best, with the replay buffer optional. Resume is
  exact for the learner: networks, optimizer state, counters and the replay sampler's
  generator all restore. It is *not* exact for the workers — environments cannot be
  restored mid-episode — so a resumed run restarts its environments at an episode boundary.

### Viewing runs

Metrics land in a local `./mlruns` file store. This project depends on **`mlflow-skinny`** —
the client, not the server — because `aero-train` only ever *writes* runs, and full `mlflow`
drags in a tracking server plus ~40 further distributions that a Ray worker image would carry
for nothing. One consequence worth knowing: `mlflow server` is listed by `mlflow --help` but
fails here with `ModuleNotFoundError: No module named 'flask'`.

```sh
autopilot mlflow
```

runs the UI against `./mlruns` on <http://127.0.0.1:5000>; `--port`, `--host` and `--store`
override the defaults, and `--store` accepts either a filesystem path or a full URI
(e.g. `sqlite:///mlflow.db`) unchanged. Under the hood it runs the UI from a throwaway tool
environment, which leaves the project's own dependencies alone:

```sh
MLFLOW_ALLOW_FILE_STORE=true \
  uvx --from mlflow mlflow server --backend-store-uri file:./mlruns --port 5000
```

The `MLFLOW_ALLOW_FILE_STORE=true` is not optional: MLflow 3 treats the file store as
deprecated and refuses it outright without that opt-in. `aero-train` sets the same variable
for its own writes, and `autopilot mlflow` sets it for the server too (without overriding one
you've already set).

To use a real tracking server instead, point `MLFLOW_TRACKING_URI` or `tracking.tracking_uri`
at it — no code change. `tracking.enabled: false` turns tracking off entirely.

## Defining a scenario

```python
from aero_gym.scenario import InitialConditions, Scenario
from aero_gym.variables import FlightVariable as V

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

Each `Scenario` selects which control axes and observations the agent sees, defines the
reward function, the initial flight condition, and the aircraft model. The environment
itself holds no task knowledge — it talks only to the `Simulator` interface, which is what
allows a second backend to be added without touching the RL side.

Units are canonical across all simulator backends. Every backend converts into them and
into the pilot sign convention, so scenarios and trained agents stay portable.

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

`Scenario.aircraft` is a JSBSim model name. Everything else `aero_gym` needs to know about an
aircraft — what FlightGear calls it, a cruise it can hold, how its engines and controls start,
what its cockpit reads — hangs off that name in a single profile registry
([`aircraft.py`](packages/aero-gym/src/aero_gym/simulators/jsbsim/aircraft.py)). Profiles ship
for `c172p` and `737`.

Naming an aircraft is therefore enough to get a flyable task: the scenario takes its initial
altitude and airspeed from the profile when you don't specify them.

```python
make_steady_flight_scenario(aircraft="c172p")  # 5000 ft at 95 kts
make_steady_flight_scenario(aircraft="737")    # 30000 ft at 280 kts
```

An aircraft with no profile still flies and still renders — JSBSim and FlightGear agree on most
model names, so the name itself is the fallback. Adding a profile is one entry in that one file.

## FlightGear visualization

FlightGear is a *viewer only*. It is launched with `--fdm=null` and fed native-FDM frames over
UDP from JSBSim, so there is exactly one authoritative flight model and the cockpit tracks the
simulation rather than contradicting it.

## Testing

```sh
uv run pytest                  # the full suite
uv run pytest -m slow          # the end-to-end learning proof, deselected by default
```

No test needs JSBSim, a socket, a subprocess, an MLflow server, or a GPU. The JSBSim
integration test skips itself when JSBSim is absent; everything else runs against fakes.
