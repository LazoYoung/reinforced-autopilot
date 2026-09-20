# Glossary

Terms used in a specific way in this project. Some are standard RL vocabulary
pinned to one meaning here; some are ours; a few are words that mean two or three
different things depending on which layer you are standing on, and those are
collected at the bottom.

The rule for this document: if a term needs a definition to read the code or a
design doc without guessing, it belongs here. If it means exactly what the
literature means and nothing collides with it, it does not.

## Time and stepping

The most common source of confusion in this codebase is that three different
clocks run at three different rates. They are, from fastest to slowest:

**physics step** — one integration step of the flight dynamics model.
`Simulator.dt` seconds; the JSBSim backend defaults to `1/120 s`. Nothing in the
RL layer ever sees one of these.

**agent step** — one call to `AeroGymEnv.step()`: apply an action, advance
`round(1 / (dt * agent_hz))` physics steps, observe. At the default `dt = 1/120`
and `agent_hz = 10`, that is **12 physics steps per agent step**, or 0.1 s of
simulated time. This is the unit the RL algorithms count in.

**iteration** — one pass of the `Trainer` loop: broadcast weights, collect, update,
log. Covers `frames_per_iter` agent steps spread across all workers.

**global step** — the learner's monotonic count of gradient updates applied. This
is the x-axis of everything logged to MLflow, and the value a checkpoint resumes
from. Not the same as the number of agent steps taken; use `frames` for that.

**frame** — **one agent step's worth of experience** (one transition). "1M frames"
means one million environment transitions. See the collision note below: this word
already means two other things nearby, so it is used in the training layer only.

## Environment layer (`aero_gym`)

`aero_gym` ships no scenarios of its own; it owns the concept and the machinery,
and every task lives outside the package.

**Scenario** — a frozen dataclass holding everything task-specific: a
`setup_fn`, which flight variables are actions and which are observations, the
reward function, the aircraft, `agent_hz`, `max_episode_steps`, and the
`action_delta` a stepped agent acts in. The env itself
holds no task knowledge. A scenario is the closest thing this project has to a
"task" or "level".

**Episode setup** — one episode's starting situation plus the context that goes
with it, drawn by a scenario's `setup_fn` from the environment's own seeded
generator at each `reset()`. The pairing lets a scenario start somewhere new
every episode while a seeded run stays exactly reproducible.

**Context** — the episode's targets, sampled alongside its `EpisodeSetup` and
not required to derive from it: a hold task points the target at the start, a
"go to" task draws the two independently. Reaches the observation features, the
reward function, and `info["context"]`.

**Registry** — `aero_gym.registry`, which resolves a scenario name to its
builder. A scenario registers itself with `@scenario("name", params=...)`; there
is no map to edit and no import to add. `build_scenario` looks up, validates,
and binds params in one call.

**Sandbox** — this repo: a training project built on `aero_gym` and `aero_train`,
with its scenarios in the top-level `src/sandbox/` package, its configs in
`configs/`, and its examples in `examples/`. The scenarios are discovered by
the registry through the `aero_gym.scenarios` entry point; none of it is
imported by `aero_gym` or `aero_train`.

**Simulator** — the flight-model backend ABC (`dt`, `reset`, `set_control`, `run`,
`observe`, `close`). JSBSim is currently the only implementation. Deliberately
narrow so a second flight model can be added without touching the RL side.

**canonical units** — the unit and sign convention every backend must convert to,
defined once in `variables.py` (+elevator = pitch up, +aileron = roll right,
+rudder = yaw right; degrees, deg/s, feet, fpm). This is a hard contract, not a
convention: a trained agent is only portable across backends because of it.

**Euler angles** (roll φ, pitch θ, yaw ψ) — the aircraft's attitude relative to the
local NED frame. `HEADING` is ψ, so it is always *true* heading; no backend here
models magnetic variation.

**body rates** (p, q, r) — angular rates about the aircraft's own x, y and z axes
(`ROLL_RATE`, `PITCH_RATE`, `YAW_RATE`). These are **not** the derivatives of the
matching Euler angles, because the two triads are about different axes:
ψ̇ = (q·sin φ + r·cos φ) / cos θ. In a steep bank most of the heading change comes
from q, not r, and the relation is singular at θ = ±90°.

**profile** — aircraft-specific knowledge (FlightGear's name for the aircraft, its
cruise condition, startup properties, cockpit needs), keyed on the **JSBSim model
name**. One `PROFILES` entry per aircraft, in one file.

**episode** — one `reset()` to one terminal or truncated state. In `steady_flight`
that is at most 2000 agent steps, i.e. 200 s of simulated flight.

**terminated** — the episode ended because the task said so: the aircraft left the
flight envelope. The value function must **not** bootstrap past this.

**truncated** — the episode ended because it hit `max_episode_steps`. The aircraft
was still flying and the value function **must** bootstrap past it. Keeping these
two apart is a correctness requirement, not bookkeeping: `steady_flight` truncates
far more often than it terminates, so conflating them biases every value estimate
in the run. (PTAN predates the distinction and conflates them; we do not.)

## Training core (`aero_train`)

**rollout** — the act of running a policy against environments to produce
experience, and by extension the experience produced. Always sampling, never
learning. A rollout is not necessarily a whole episode and usually is not.

**segment** — the unit of experience this project moves around: contiguous
`[T, B]` arrays (T timesteps × B envs) plus `terminated` / `truncated` masks and
optional agent state. Time is an explicit axis, which is what makes GAE and
recurrent unrolling possible. Off-policy algorithms flatten segments into
individual transitions when inserting into the replay buffer; on-policy
algorithms consume them as-is.

**transition** — a single `(obs, action, reward, next_obs, terminated)` tuple. A
segment flattened along T. Note `aero_gym.scenario.Transition` is a *different,
narrower* thing (`prev_obs`, `obs`, `action`) used only to compute rewards.

**ActionSelector** — the piece that turns network output into an action: argmax,
epsilon-greedy, categorical sample, Gaussian noise, Ornstein-Uhlenbeck. Separated
from the network so exploration strategy is configuration, not code.

**Agent** — network + action selector, running **inside a rollout worker**. Always
called batched: `[B, obs_dim] -> [B, act_dim]`. Holds a *replica* of the policy
weights, refreshed by `sync_weights`.

**ExperienceSource** — the loop that drives envs with an agent and yields segments.
Also runs inside the worker. Owns n-step return accumulation and episode-boundary
bookkeeping.

**delta action** (also **incremental action**) — a discrete action that *changes*
a control deflection by a step rather than *setting* it: per axis, `{-δ, 0, +δ}`,
where δ is a fraction of that axis's declared range. This is how DQN acts in this
project; the `DeltaAction` wrapper accumulates the command and clamps it to the
`ACTION_SPECS` bounds. A scenario sets δ, once for every axis or per axis, or
leaves it unset to pose a continuous task. Contrast **absolute binning**, where
each discrete action names a fixed deflection.

**Algorithm** — the learner-side object: owns the networks, the optimizers, and the
loss. `update(batch, step) -> metrics` performs one complete parameter update,
including any target-network sync. There is deliberately no shared
"compute loss, then step" scaffold, because DQN, A2C and DDPG do genuinely
different things in genuinely different orders.

**actor_state_dict** — the subset of learner state a worker actually needs: the
policy network only. Distinct from `state_dict`, which is everything needed to
resume training (all networks, all optimizers, schedules). DDPG's critic is
learner-only and never crosses to a worker.

**Trainer** — the loop that owns everything cross-cutting and algorithm-independent:
weight broadcast, collection, replay, the update call, logging, checkpointing,
evaluation. Written once.

## Distribution

**learner** — the single process that holds the authoritative weights and applies
gradient updates. Also called the driver, because it is the Ray driver.

**worker** — a process that only ever samples: envs + agent replica +
experience source. Never computes a gradient. Holds `envs_per_worker` environments.

**RolloutEngine** — the abstraction that hides whether sampling is happening
in-process or across Ray actors. `LocalRolloutEngine` imports nothing from Ray;
`RayRolloutEngine` is a fan-out wrapper around the same triple a local engine holds.

**lockstep** — stepping all `envs_per_worker` environments of a worker together, so
their observations form one batch and the policy runs one forward pass over all of
them. The alternative would be stepping each env independently and paying a forward
pass per env. Lockstep's cost is that a slow env (notably one that is resetting,
which re-trims the aircraft) stalls its peers.

**data parallelism** — workers ship *experience* to the learner, and the learner
computes all gradients. What this project does.

**gradient parallelism** — workers compute gradients locally and ship *gradients*
to the learner (classic A3C). Rejected here: the policy is ~5k parameters (~20 KB),
so shipping weights and experience is trivially cheap, while JSBSim is the real
cost; and it does not compose with off-policy replay or with PPO's multiple epochs.

**weight broadcast** — sending `actor_state_dict` to every worker at the start of an
iteration. At 20 KB this costs microseconds, which is why on-policy algorithms here
can be **exactly** on-policy rather than approximately so.

**exactly on-policy** — every sample in a batch was generated by the same weights
that are about to be updated. Achievable here only because broadcast is nearly free;
most distributed RL systems accept policy lag and correct for it with V-trace or
importance ratios instead.

**policy lag** / **staleness** — the gap between the weights that generated a sample
and the weights being updated. Zero under synchronous collection; one iteration
under double buffering; unbounded under fully async collection.

**double buffering** — launching the next iteration's collection before running this
iteration's update, so the learner and the workers overlap. Introduces exactly one
iteration of policy lag, which is harmless for off-policy algorithms (the replay
buffer already holds far staler data) and is therefore enabled for DQN and DDPG and
disabled for A2C.

**Ray hop** — informal shorthand for one round trip across the Ray actor boundary:
a `.remote()` call plus the `ray.get()` that retrieves its result. Involves
serialization, the object store or gRPC, a scheduler queue, and deserialization —
on the order of 200 µs to 1 ms locally, i.e. comparable to a whole agent step. The
reason inference stays inside workers rather than being centralized in the driver.

## Collisions

Words that already mean something else nearby. Prefer the qualified form.

| Word | Meaning A | Meaning B | Rule |
|---|---|---|---|
| **actor** | a Ray actor: a worker process | actor-critic's actor: the policy network | Say **worker** for the process and **policy** for the network. Avoid bare "actor". |
| **step** | agent step (`env.step`) | physics step (FDM integration) | Always qualify. Bare "step" in the training layer means the learner's **global step**. |
| **frame** | one transition (RL) | 0.1 s wall-clock duration (`env.py:108`, real-time pacing) | Also FlightGear's 60 Hz render frame. Use **frame** only in the training layer; the pacing code's local variable is a historical exception. |
| **state** | env state | `agent_state` (recurrent hidden state) and `state_dict` (network weights) | Never bare "state" — say env state, agent state, or state dict. |
| **environment** | one Gymnasium env instance | the `aero_gym` package as a layer | Say **env** for the instance, **environment layer** for the package. |
| **transition** | `aero_train`'s `(obs, action, reward, next_obs, terminated)` | `aero_gym.scenario.Transition` (`prev_obs`, `obs`, `action`), reward-fn input only | Qualify as **scenario transition** for the latter. |
| **reset** | Gymnasium `env.reset()` | JSBSim re-trim, the expensive part of it | Note that a reset is *not* cheap here — it re-trims the aircraft, which is why it stalls a lockstep batch. |
| **yaw rate** | body `r` (`YAW_RATE`), about the body z-axis | ψ̇, the rate at which heading changes | They are equal only wings-level. Say **body yaw rate** or **r** for the observation, **heading rate** for ψ̇. The same trap applies to p and φ̇. |
