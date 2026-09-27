# Training framework design

How `aero_train` is put together and why it is shaped that way.
[README.md](../README.md) says what exists and how to run it;
[glossary.md](glossary.md) defines the vocabulary. This file carries the
reasoning, and the roadmap for what is not built yet.

## Goal

A training stack for `aero_gym` environments: DQN, A2C and DDPG behind one
`Algorithm` ABC, extensible to PPO without redesign; rollout collection that can
fan out across processes; MLflow tracking, local by default. Runs are
reproducible, resumable and observable.

Deliberately out of scope: offline RL, model-based RL and multi-agent training
(the abstractions must not preclude them, but nothing is built for them);
hyperparameter search orchestration, since CLI overrides are enough to drive an
external sweeper; and serving a trained policy: `autopilot export` writes one self-contained
policy file ([policy-file.md](policy-file.md)), and running it is a host's job.

## Two packages, one dependency rule

`aero-train` depends on `aero-gym`. The reverse never holds, and inside
`aero_train` only `aero_train.envs` imports `aero_gym`, function locally.
Everything else talks to Gymnasium's API alone.

This is not purity for its own sake: it is what makes the CartPole integration
test possible, and that test is the only cheap end-to-end proof that the learning
stack is correct. A bug in DQN and a bug in the flight model are otherwise
indistinguishable when a flight run fails to learn.

The split also keeps the environment cheap to install. `aero-train` pulls in
torch and mlflow; `aero-gym` needs only gymnasium, jsbsim and numpy. Export's ONNX packages sit behind an extra, `aero-train[export]`.

## Experience is a segment, not a flat transition

`ExperienceSource` emits `Segment` and nothing else: contiguous `[T, B]` arrays
with time as an explicit axis. Off-policy algorithms flatten to transitions when
inserting into the replay buffer, at exactly one place in the codebase.

Emitting flat n-step transitions instead was rejected because it blocks GAE,
which needs contiguous time and cannot reconstruct it from a shuffled pile of
transitions. Segments are also the prerequisite for recurrent unrolling, and for
V-trace should policy lag ever appear.

Two corrections to the PTAN decomposition the primitives otherwise follow:

- **`terminated` and `truncated` stay distinct.** Bootstrapping must continue
  through a time limit and must stop at a real terminal state. A hold task
  truncates far more often than it terminates, so conflating the two would bias
  every value estimate in the project's flagship scenario.
- **`valid` masks Gymnasium's autoreset step.** Under 1.x next-step autoreset,
  the step after a termination ignores the submitted action and returns the reset
  observation. Recording it would train the agent on a fabricated (final state,
  ignored action, reset state) tuple. This is silent when wrong, so it has a
  dedicated test.

`obs` carries `T + 1` rows so the bootstrap value at the end of a segment needs
no second forward pass.

## Algorithm: observe, then update

`observe` and `update` are separate calls because their cadence differs by
family. Off-policy: `observe` inserts into the buffer, `update` samples from it
and may run several gradient steps per iteration, returning `None` while the
buffer is still filling. On-policy: `observe` stashes the segment, `update`
consumes and discards it.

`actor_state_dict` returns only what a worker needs for inference: the policy
network, not the critic, the target networks or the optimizer state. For DDPG
that halves what crosses the wire each iteration. `update` returns the metrics
dict that goes straight to the tracker, which is what keeps `Trainer` from
knowing anything algorithm-specific.

## DQN acts in increments

`DeltaAction` makes each discrete action a *step* of the control deflection
rather than an absolute position: per axis, one of `{-δ, 0, +δ}`, accumulated
across the episode and clamped to the `ACTION_SPECS` bounds. The step size is
declared by the scenario, per axis and relative to that axis's span; a scenario
that leaves it unset is posing a continuous task. Increments were chosen over
absolute binning for two reasons.

**Exploration is temporally correlated by construction.** An aircraft is a
low-pass filter. Absolute bins let a random policy slam the elevator between
extremes on consecutive 10 Hz steps; the airframe integrates that to roughly
nothing, and the agent explores almost none of the state space. A random walk
over `{-δ, 0, +δ}` is already a smooth control input, with no noise process to
tune.

**Range and precision decouple.** Five bins over `[-1, 1]` means the elevator can
only ever hold one of five deflections, and the trim setting for level flight is
almost certainly not among them. That quantization error is a standing bias no
amount of training removes. Increments integrate to any reachable value at the
step's resolution, which is what a trim-dominated task needs.

The required consequence is that the observation must carry the current control
positions, and `aero_gym` supplies them as canonical observations reading the
simulator's actual surface position rather than the wrapper's accumulated
command. Position is command plus trim, and trim is exactly the hidden state a
hold task is about, so an agent reading back its own commands cannot see the
offset it is fighting. Left unfixed this is a POMDP that trains to a mysteriously
mediocre policy rather than failing loudly.

**Honest limit.** This puts the joint action space at 3^4 = 81, down from 625 at
five absolute bins, which is comfortable for DQN. It does not eliminate the
combinatorics. Doing that needs a factored head, which vanilla DQN cannot
express: Q-learning maximizes over the joint action space, so factored Q-values
need an independence assumption or a mixing network, and that is a different
algorithm. Revisit only if 81 actions prove to be the bottleneck.

## Config, tracking, checkpointing

Config is frozen, slotted dataclasses validating in `__post_init__`, the same
idiom as `aero_gym.scenario.Scenario` so the two packages read alike. YAML loads
into them, never into bare dicts, so a misspelled key is an error at load time
rather than a silently ignored setting discovered three hours into a run.
`AlgoConfig` is a tagged union on a `type:` field, which keeps the file flat and
readable while the code still sees a typed `DQNConfig`. `--override` takes dotted
paths and is applied *before* validation, so an override cannot slip past the
checks.

Validation covers the cross-cutting constraints the type system cannot express: a
discrete-action algorithm against a continuous action space with no wrapper,
`frames_per_iter` not divisible by the worker count, `total_steps` below the
replay buffer's warmup threshold.

`Tracker` is an ABC with an MLflow implementation and a null one, for one
concrete reason: no unit test should need an MLflow server. What is logged is the
flattened config (a seed means nothing without its topology), the per-iteration
metrics straight from `Algorithm.update`, per-episode return, length and
termination reason, and throughput split into rollout time and update time.
Termination reason earns its place: an episode that ends by leaving the band is a
different event from one that hits the step limit, and the mean return alone
hides which.

Checkpoints are written atomically, temp file then rename, so an interrupted save
cannot leave a truncated file that fails on resume. They carry the config, all
learner state, the gradient-update count, the frame count, and the best return so
far; the replay buffer only when `include_buffer` is set, because a large buffer
is expensive and most resumes do not need bit-exact contents. Two are kept:
latest, and best by evaluation return. The second is not a nicety, because a DQN
curve can peak and then collapse, and the final weights are not the best ones.

**Stated limit:** resume restores the learner exactly. It does not restore
environment RNG state, so a resumed run continues from an episode boundary rather
than reproducing the original frame for frame.

## Testing

Unit tests run against a deterministic fake env: no JSBSim, no MLflow, no
sockets, no subprocesses. The correctness tests target the failures that silently
corrupt training rather than raising: n-step return accumulation against a
hand-computed expectation, the `terminated` against `truncated` bootstrap
distinction, the autoreset skip, and segment-to-transition flattening. One slow
integration test, marked and deselected by default, trains DQN on CartPole to a
return threshold; it is the payoff for keeping `aero_train` environment-agnostic.

## Roadmap

Phase 1 is being designed.

**Phase 2 is A2C (with GAE), DDPG, and the Ray rollout engine.** The segment
shape already supports GAE, and DDPG gets Ornstein-Uhlenbeck rather than
uncorrelated noise for the same low-pass-filter reason increments beat absolute
bins. Five things are settled in advance for the engine:

- **The measured agent step is 0.04 ms, and that changes the arithmetic.** The
  original design assumed 0.25 to 0.7 ms and argued that a Ray hop (200
  microseconds to 1 millisecond locally) was affordable next to an environment
  step. JSBSim turned out roughly 17 times faster than that estimate, so a hop
  now costs 5 to 25 agent steps rather than one. Centralized inference is
  therefore firmly out, and fanning out only pays if each hop carries enough
  frames to amortize itself: pick `frames_per_iter` and `envs_per_worker` from a
  measured hop on the target machine before tuning anything else. A reset costs
  3 ms, about 75 agent steps, which is what makes a resetting env stall its
  lockstep peers.
- **Broadcast weights with a single serialization.** `ray.put` the actor state
  dict once and hand every worker the reference; passing the dict into each
  `.remote()` call serializes it once per worker. Waiting on the whole list is
  what makes the iteration a synchronization barrier, and therefore what lets an
  on-policy run be exactly on-policy.
- **Seed by spawning, not by adding.** Each worker derives its environment seeds
  from `SeedSequence(seed).spawn(n_workers)` and spawns again for its own envs;
  `seed + worker_index` produces correlated streams. DQN today seeds its selector
  from a single root, which is invisible with one worker and means identical
  exploration in every worker once there are several.
- **One compute thread per worker.** Set `OMP_NUM_THREADS=1` and the torch
  equivalent in the worker's runtime environment, not in the user's shell. Torch
  otherwise starts a thread per core in every worker, and N workers on an N-core
  machine thrash N^2 threads. For the same reason `AsyncVectorEnv` is never used
  inside a worker: Ray owns process-level parallelism, `SyncVectorEnv` owns the
  in-process batch.
- **Engine equivalence is a test, not a hope.** The local and Ray engines run
  against the same assertions with identical seeds and topology and must produce
  identical segments. Without that test the local path quietly becomes a
  different algorithm from the distributed one.

Two known seams will be hit on the way in. `Algorithm`'s surface is too narrow,
so `Trainer`, `cli` and `config` reach around it with `getattr` for `buffer` and
`warmup_frames`; widening it is a call worth making with A2C's and DDPG's real
requirements in view rather than guessed at. And `ExperienceSource` cannot
accumulate n-step returns across segment boundaries, which GAE will want.

**Open question:** whether DDPG should also train through `DeltaAction` or
against the raw `Box`. Going through the wrapper would make the DQN and DDPG
curves comparable, since otherwise the two are solving different MDPs, and rate
control is a plausible exploration aid in its own right that partly subsumes OU
noise. It costs an extra config axis.

**Recorded flip trigger.** Inference runs inside the workers because the policy
is a few thousand parameters, so a replica per worker is nearly free. If
observations ever become images, a camera view or a rendered panel, the per-frame
payload grows by orders of magnitude and centralizing inference starts to win.
Written down so that revisit is a deliberate act rather than a rediscovery.
