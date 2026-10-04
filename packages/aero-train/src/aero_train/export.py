"""Export a trained policy as one self-contained ONNX file, and check it.

The file's contract is docs/policy-file.md. This module composes parts the
training path already has (the scenario's features, ScaleObservation, the
algorithm's greedy policy, DeltaAction's step rule) into one graph, and keeps
the file only once it has flown an episode the way the checkpoint does. The
ONNX packages are the optional `export` extra, so they are imported inside
functions.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .algo.base import Algorithm
from .envs.delta import DeltaAction
from .envs.features import ColumnSpec, TracedScenario, scenario_of, trace_scenario
from .envs.scale import ScaleObservation
from .tensors import Column, column_shapes, example_columns

FORMAT = "1"
VALUE_RTOL = 1e-4
VALUE_ATOL = 1e-5
COMMAND_ATOL = 1e-6
CHECK_MAX_STEPS = 2000

CHECKPOINT_METADATA = ("aero.config", "aero.frames", "aero.scenario")
"""What only the checkpoint knows, so the caller of export_policy supplies it."""
OWN_METADATA = ("aero.format", "aero.agent_hz", "aero.aircraft", "aero.inputs", "aero.outputs")
"""What export_policy writes itself, which a caller may not overwrite."""

VALUE = ColumnSpec(
    "value", "value", "the policy's estimate of this state's worth, comparable only within this file"
)


class PolicyCheckError(ValueError):
    """The exported file does not fly the way its checkpoint does."""


@dataclass(frozen=True)
class CheckReport:
    """How the checked episode went."""

    steps: int
    flips: int
    """Steps where near-equal Q-values picked a different, still reachable, command."""
    max_value_error: float


class _PolicyGraph(nn.Module):
    """Readings, targets and prev.* columns in; a column per action, and value, out."""

    def __init__(
        self,
        features: nn.Module,
        n_feature_inputs: int,
        scale: nn.Module,
        policy: nn.Module,
        step: nn.Module,
    ) -> None:
        super().__init__()
        self.features = features
        self._n_feature_inputs = n_feature_inputs
        self.scale = scale
        self.policy = policy
        self.step = step

    def forward(self, *inputs: Column) -> tuple[Column, ...]:
        split = self._n_feature_inputs
        action, value = self.policy(self.scale(self.features(*inputs[:split])))
        command = self.step(action, torch.stack(inputs[split:], dim=1))
        return (*command.unbind(1), value)


def export_policy(
    path: Path,
    algo: Algorithm,
    env_fn: Callable[[], gym.Env],
    *,
    seed: int,
    metadata: Mapping[str, str],
) -> CheckReport:
    """Write `algo`'s greedy policy for `env_fn`'s scenario to `path`, checked.

    `env_fn` builds the env the policy was trained through; `metadata` adds
    what only the checkpoint knows (CHECKPOINT_METADATA). The file is
    written beside `path` and renamed into place only once the check passes,
    so a failure leaves any earlier file untouched. Traces the scenario code
    in the current checkout, so export a checkpoint with the code it was
    trained on.
    """
    missing = [key for key in CHECKPOINT_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"export_policy needs metadata {missing}, which only the checkpoint knows")
    clashing = [key for key in OWN_METADATA if key in metadata]
    if clashing:
        raise ValueError(f"export_policy writes {clashing} itself; do not pass them as metadata")
    _require_onnx()
    policy = algo.policy_module()
    path = Path(path)
    env = env_fn()
    try:
        delta, scale = _layers(env)
        traced = trace_scenario(scenario_of(env))
        graph = _PolicyGraph(
            traced.features, len(traced.inputs), scale.to_torch(), policy, delta.to_torch()
        ).eval()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            _write(tmp, graph, traced, metadata)
            report = check_policy_file(tmp, policy, env, traced, seed=seed)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return report
    finally:
        env.close()


def check_policy_file(
    path: Path,
    policy: nn.Module,
    env: gym.Env,
    traced: TracedScenario,
    *,
    seed: int,
) -> CheckReport:
    """Fly one greedy episode through `env`, comparing the file with the training path.

    The env steps on `policy`'s actions. The file is fed the same readings and
    targets by name, and prev.* from the wrapper's command, re-synced every
    step so one flip cannot compound. Raises PolicyCheckError at the first step
    no near-tie explains, or after the episode ends if flips exceeded
    max(1, steps // 100).
    """
    # Read when onnxruntime loads; its telemetry upload thread can abort the
    # process at exit (libc++ "recursive_mutex lock failed").
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime as ort

    delta, _ = _layers(env)
    low, high = delta.env.action_space.low, delta.env.action_space.high
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = [o.name for o in session.get_outputs()]
    obs, info = env.reset(seed=seed)
    steps = flips = 0
    first_flip: tuple[int, str, float, float] | None = None
    worst = 0.0
    done = False
    while not done and steps < CHECK_MAX_STEPS:
        prev = delta.command
        feed = {name: np.array([v], np.float32) for name, v in traced.feed(info).items()}
        feed |= {
            f"prev.{column.name}": np.array([c], np.float32)
            for column, c in zip(traced.actions, prev)
        }
        got = {name: float(v[0]) for name, v in zip(outputs, session.run(None, feed))}
        with torch.no_grad():
            action, value = policy(torch.as_tensor(obs, dtype=torch.float32)[None])
        want = float(value[0])
        worst = max(worst, abs(got["value"] - want))
        if not math.isclose(got["value"], want, rel_tol=VALUE_RTOL, abs_tol=VALUE_ATOL):
            raise PolicyCheckError(
                f"step {steps}: value expected {want:.6g}, got {got['value']:.6g}"
            )
        obs, _, terminated, truncated, info = env.step(int(action[0]))
        expected = delta.command
        reachable = np.clip(
            prev[:, None] + np.array([-1.0, 0.0, 1.0], np.float32) * delta.step_sizes[:, None],
            low[:, None],
            high[:, None],
        )
        flipped = False
        for axis, column in enumerate(traced.actions):
            name = column.name
            if abs(got[name] - expected[axis]) <= COMMAND_ATOL:
                continue
            if np.abs(reachable[axis] - got[name]).min() > COMMAND_ATOL:
                raise PolicyCheckError(
                    f"step {steps}: {name} expected {expected[axis]:.6g}, got "
                    f"{got[name]:.6g}, which no action reaches from "
                    f"prev.{name} {prev[axis]:.6g}"
                )
            flipped = True
            first_flip = first_flip or (steps, name, float(expected[axis]), got[name])
        flips += int(flipped)
        steps += 1
        done = terminated or truncated
    allowed = max(1, steps // 100)
    if flips > allowed:
        at, name, want_command, got_command = first_flip
        raise PolicyCheckError(
            f"{flips} of {steps} steps chose a different command than the checkpoint "
            f"(at most {allowed} allowed); first at step {at}: {name} expected "
            f"{want_command:.6g}, got {got_command:.6g}"
        )
    return CheckReport(steps=steps, flips=flips, max_value_error=worst)


def _write(
    path: Path, graph: nn.Module, traced: TracedScenario, metadata: Mapping[str, str]
) -> None:
    """Export `graph` as one file, with documented columns and the metadata."""
    import onnx

    prev = tuple(
        ColumnSpec(
            f"prev.{a.name}",
            "prev",
            f"{a.description}; the command applied now, read back every call",
            a.unit,
            a.low,
            a.high,
        )
        for a in traced.actions
    )
    inputs = (*traced.inputs, *prev)
    outputs = (*traced.actions, VALUE)
    with warnings.catch_warnings():
        # Every input and output shares one dim name ("batch") on purpose; this
        # warning is about that, not about anything wrong with the graph.
        warnings.filterwarnings(
            "ignore", message=r"# The axis name: batch will not be used.*", category=UserWarning
        )
        program = torch.onnx.export(
            graph,
            example_columns(len(inputs)),
            dynamo=True,
            input_names=[p.name for p in inputs],
            output_names=[p.name for p in outputs],
            dynamic_shapes=column_shapes(len(inputs)),
            verbose=False,
        )
    program.save(str(path), external_data=False)
    model = onnx.load(str(path))
    docs = {p.name: p.doc for p in (*inputs, *outputs)}
    for column in (*model.graph.input, *model.graph.output):
        column.doc_string = docs[column.name]
    columns = {
        "aero.inputs": json.dumps({c.name: c.entry() for c in inputs}, allow_nan=False),
        "aero.outputs": json.dumps({c.name: c.entry() for c in outputs}, allow_nan=False),
    }
    for key, value in {"aero.format": FORMAT, **traced.metadata, **columns, **metadata}.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
    onnx.save(model, str(path), save_as_external_data=False)
    onnx.checker.check_model(str(path), full_check=True)


def _layers(env: gym.Env) -> tuple[DeltaAction, ScaleObservation]:
    """The two wrappers the policy was trained through, as make_env_fn stacks them."""
    if not (isinstance(env, DeltaAction) and isinstance(env.env, ScaleObservation)):
        raise ValueError(
            "export needs DeltaAction(ScaleObservation(env)), which make_env_fn "
            "builds for a stepped scenario; a continuous-action policy cannot be "
            "exported yet"
        )
    return env, env.env


def _require_onnx() -> None:
    missing = [
        name
        for name in ("onnx", "onnxruntime", "onnxscript")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ModuleNotFoundError(
            f"policy export needs {', '.join(missing)}: install the aero-train[export] extra"
        )
