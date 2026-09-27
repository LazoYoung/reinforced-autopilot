# reinforced-autopilot

Reinforcement learning for fixed-wing flight control.

Two packages in one [uv](https://docs.astral.sh/uv/) workspace:

| Package | What it is |
|---|---|
| [`aero-gym`](packages/aero-gym) | Simulator-agnostic [Gymnasium](https://gymnasium.farama.org/) environments for fixed-wing flight, backed by [JSBSim](https://jsbsim.sourceforge.net/) with optional live [FlightGear](https://www.flightgear.org/) visualization. |
| [`aero-train`](packages/aero-train) | The training framework: config, rollout, replay, algorithms, tracking, checkpointing, and the `autopilot` CLI. |

`aero-train` depends on `aero-gym` and never the reverse, and only `aero_train.envs` may
import it. A test enforces both directions.

## Installation

> Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.
> [Install uv](https://docs.astral.sh/uv/getting-started/installation/) first.

```sh
uv sync
```

The Python wheel includes the JSBSim backend. FlightGear is optional and only needed for
visualization: [install it separately](https://www.flightgear.org/download/).

## Quick start: train something

```sh
uv run autopilot train configs/dqn_cartpole.yaml
```

DQN on CartPole-v1, and the fastest check that an install works. Checkpoints land under
`runs/`, metrics under `mlruns/`.

```sh
# override any config key without editing the file
uv run autopilot train configs/dqn_cartpole.yaml --override algo.lr=1e-3

# extend a finished run past its original budget
uv run autopilot resume runs/dqn-cartpole/checkpoints/latest.pt --total-steps 120000

# watch a trained policy fly, greedily
uv run autopilot eval runs/dqn-cartpole/checkpoints/best.pt --episodes 20
uv run autopilot eval runs/dqn-cartpole/checkpoints/best.pt --flightgear
```

Every run keeps `latest.pt` and `best.pt` (by evaluation return). Prefer `best.pt`: DQN can
reach a good policy and then lose it.

A flight policy exports to one self-contained ONNX file, checked against its checkpoint
before it is kept. [`docs/policy-file.md`](docs/policy-file.md) is the file's contract for
a host.

```sh
uv run autopilot export runs/dqn-steady-flight/v4/best.pt   # writes best.onnx beside it
```

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

Or run the bundled example:

```sh
uv run examples/run_random_agent.py                  # headless Cessna 172
uv run examples/run_random_agent.py --flightgear     # with visualization
uv run examples/run_random_agent.py --aircraft 737   # fly a different type
```

## The training stack

A run config is YAML loaded into frozen dataclasses, so a misspelled key fails at load time.
See [`configs/dqn_cartpole.yaml`](configs/dqn_cartpole.yaml) for a complete example, and
[`docs/training-design.md`](docs/training-design.md) for why the stack is shaped this way.

```yaml
run_name: dqn-cartpole
total_steps: 60000        # frames, not gradient updates
env:      { id: CartPole-v1 }
algo:     { type: dqn, lr: 0.0005, buffer_size: 50000, updates_per_iter: 32 }
rollout:  { engine: local, envs_per_worker: 8, frames_per_iter: 128 }
tracking: { enabled: true, experiment: reinforced-autopilot }
```

What is in place today:

- **Algorithms**: DQN with n-step returns. `Algorithm` is an ABC, so more slot in beside it.
- **Rollout**: `engine: local` runs a `SyncVectorEnv` in-process. `engine: ray` passes config
  validation but raises `NotImplementedError`.
- **Experience**: lockstep across a vector of environments, carrying time as an explicit axis.
  A truncated episode bootstraps; a terminated one does not.
- **Tracking**: MLflow, defaulting to a local `./mlruns` file store.
- **Checkpointing**: atomic, latest plus best, replay buffer optional. Resume restores the
  learner exactly; environments restart at an episode boundary.

### Viewing runs

```sh
autopilot mlflow
```

serves the UI against `./mlruns` on <http://127.0.0.1:5000>. `--port`, `--host` and `--store`
override the defaults, and `--store` accepts either a path or a full URI such as
`sqlite:///mlflow.db`.

To use a tracking server instead, point `MLFLOW_TRACKING_URI` or `tracking.tracking_uri` at
it. `tracking.enabled: false` turns tracking off entirely.

## Defining a scenario

Scenarios live in [`src/sandbox/scenarios/`](src/sandbox/scenarios/). Drop a Python file
there, decorate a builder, and it is available by name: nothing else is wired up, and no
installed package is edited.

A builder takes its params, draws each episode's starting situation, and returns a
`Scenario`:

```python
# src/sandbox/scenarios/wings_level.py
@dataclass(frozen=True)
class WingsLevelParams:
    """Every knob the task has. Defaults make it flyable as it stands."""

    roll_limit_deg: float = 60.0
    roll_jitter_deg: float = 20.0


@scenario("wings_level", params=WingsLevelParams)
def build(params: WingsLevelParams) -> Scenario:
    def setup_fn(rng: np.random.Generator) -> EpisodeSetup:
        """This episode's start, drawn from the environment's own generator."""
        roll = rng.uniform(-params.roll_jitter_deg, params.roll_jitter_deg)
        ic = InitialConditions(37.62, -122.38, 5000.0, 270.0, roll_deg=roll)
        return EpisodeSetup(ic, {"target_roll_deg": 0.0})
```

Each entry of the observation vector is an `ObservationFeature`: a name, the bounds its
value is clipped to, and a function computing it from this step's readings and the
episode's targets.

```python
def roll_error_deg(obs: Observation, context: Context) -> float:
    """An Observation is this step's readings keyed by FlightVariable; a Context
    is the episode's targets, the ones setup_fn drew above."""
    return obs[V.ROLL] - context["target_roll_deg"]


ObservationFeature(
    "roll_error_deg",
    VariableSpec("deg", -60.0, 60.0, "roll relative to target"),
    roll_error_deg,
)
```

[`steady_flight.py`](src/sandbox/scenarios/steady_flight.py) is the complete worked example,
and [`docs/glossary.md`](docs/glossary.md) defines the vocabulary it uses.

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

A name that is not a field on the params dataclass is rejected before any environment is
built, and the error lists the valid ones.

An installed package can ship scenarios too, by declaring the same entry point in its own
`pyproject.toml`:

```toml
[project.entry-points."aero_gym.scenarios"]
my-tasks = "my_package.scenarios"
```

## Canonical units

Every backend converts into these units and into the pilot sign convention, so scenarios and
trained agents stay portable across simulators.

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
aircraft hangs off that name in a single profile registry
([`aircraft.py`](packages/aero-gym/src/aero_gym/simulators/jsbsim/aircraft.py)): what
FlightGear calls it, a cruise it can hold, how its engines and controls start, what its
cockpit reads. Profiles ship for `c172p` and `737`.

Naming an aircraft is enough to get a flyable task, since a scenario takes its initial
altitude and airspeed from the profile when you do not specify them:

```python
build_scenario("steady_flight", {"aircraft": "c172p"})  # 5000 ft at 95 kts
build_scenario("steady_flight", {"aircraft": "737"})    # 30000 ft at 280 kts
```

An aircraft with no profile still flies and still renders, because the model name itself is
the fallback. Adding a profile is one entry in that one file.

## FlightGear visualization

FlightGear is a viewer only. It is launched with `--fdm=null` and fed native-FDM frames over
UDP from JSBSim, so there is exactly one authoritative flight model. See
[`docs/cockpit-sync.md`](docs/cockpit-sync.md).

## Testing

```sh
uv run pytest                  # the full suite
uv run pytest -m slow          # the end-to-end learning proof, deselected by default
```

No test needs JSBSim, a socket, a subprocess, an MLflow server, or a GPU. The JSBSim
integration test skips itself when JSBSim is absent; everything else runs against fakes.
