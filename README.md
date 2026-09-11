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
uv run autopilot train configs/dqn_cartpole.yaml
```

That trains DQN on CartPole-v1 — the end-to-end proof that the stack learns, and the
fastest way to check an install works. Checkpoints land under `runs/`, metrics under
`mlruns/`.

```sh
# override any config key without editing the file; applied before validation,
# so an override can't slip past a config check
uv run autopilot train configs/dqn_cartpole.yaml --override algo.lr=1e-3

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
from aero_gym.registry import build_scenario

env = AeroGymEnv(build_scenario("steady_flight"))     # headless
# env = AeroGymEnv(..., render_mode="flightgear")     # with visualization

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

Scenarios live in `src/sandbox/scenarios/`. Drop a Python file there, decorate a builder,
and it is available by name: nothing else is wired up, and no installed package is edited.

```python
# src/sandbox/scenarios/wings_level.py
from dataclasses import dataclass

import numpy as np

from aero_gym.registry import scenario
from aero_gym.scenario import EpisodeSetup, InitialConditions, Scenario, raw
from aero_gym.variables import FlightVariable as V


@dataclass(frozen=True)
class WingsLevelParams:
    """Every knob the task has. Defaults make it flyable as it stands."""

    roll_limit_deg: float = 60.0
    roll_jitter_deg: float = 20.0


@scenario("wings_level", params=WingsLevelParams)
def build(params: WingsLevelParams) -> Scenario:
    def setup_fn(rng: np.random.Generator) -> EpisodeSetup:
        """One episode's start, drawn from the environment's own generator."""
        roll = rng.uniform(-params.roll_jitter_deg, params.roll_jitter_deg)
        return EpisodeSetup(
            InitialConditions(
                latitude_deg=37.62, longitude_deg=-122.38, altitude_ft=5000.0,
                heading_deg=270.0, pitch_deg=0.0, roll_deg=roll, airspeed_kts=95.0,
            ),
            {"target_roll_deg": 0.0},
        )

    def reward_fn(t):
        """t: Transition(prev_obs, obs, action, context), dicts keyed by FlightVariable."""
        error = abs(t.obs[V.ROLL] - t.context["target_roll_deg"])
        return 1.0 - error / params.roll_limit_deg, error > params.roll_limit_deg

    return Scenario(
        name="wings-level",
        setup_fn=setup_fn,
        reward_fn=reward_fn,
        action_variables=(V.AILERON, V.RUDDER),
        observation_variables=(V.ROLL, V.ROLL_RATE, V.YAW_RATE, V.AOS),
        observation_features=(
            raw(V.ROLL, -params.roll_limit_deg, params.roll_limit_deg),
            raw(V.ROLL_RATE, -60.0, 60.0),
            raw(V.YAW_RATE, -30.0, 30.0),
            raw(V.AOS, -20.0, 20.0),
        ),
        context_keys=("target_roll_deg",),
        aircraft="c172p",
        agent_hz=10.0,
    )
```

Train it, and tweak it from the config or the command line:

```yaml
env:
  scenario: wings_level
  params:
    roll_jitter_deg: 30
```

```sh
uv run autopilot train configs/my_run.yaml --override env.params.roll_limit_deg=45
```

Four things carry the design:

- **`setup_fn` runs once per episode**, with the environment's own seeded generator, so one
  scenario can train an agent across many starting situations while a seeded run stays exactly
  reproducible.
- **The context is the episode's targets**, and it does not have to derive from where the
  episode began. A hold task points the target at the start; a "go to" task draws the two
  independently. It reaches the feature functions, the reward function, and `info["context"]`.
- **Observation bounds are declared, not derived.** The `Box` is built once and must stay fixed,
  because the policy's input scaling is part of what it learned. A feature that leaves its bounds
  is clipped and warns once, naming itself, so a wrong bound is visible rather than silent.
- **The params dataclass is the list of what is tweakable.** A name that is not a field on it is
  rejected before any environment is built, and the error names the field and lists the valid ones.

An installed package can ship scenarios too, by declaring the same entry point in its own
`pyproject.toml`:

```toml
[project.entry-points."aero_gym.scenarios"]
my-tasks = "my_package.scenarios"
```

AeroGymEnv talks only to the `Simulator` interface and holds no task knowledge of its own,
which is what lets a second flight simulator be added without touching the RL side;
`Scenario` carries everything task specific; and this repo is that sandbox, with its
scenarios in `src/sandbox/`, its configs in `configs/`, and its examples in `examples/`,
none of it imported by `aero_gym` or `aero_train`.

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
build_scenario("steady_flight", {"aircraft": "c172p"})  # 5000 ft at 95 kts
build_scenario("steady_flight", {"aircraft": "737"})    # 30000 ft at 280 kts
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
