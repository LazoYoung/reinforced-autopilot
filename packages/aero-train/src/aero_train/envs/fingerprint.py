"""A scenario's fingerprint: what a policy file takes from it, recorded when training starts.

Export traces the scenario code in the current checkout, and its check compares
the file with that same code, so neither can see a change made since training.
The fingerprint can: `autopilot train` stores it in every checkpoint, and
`resume` and `export` refuse a scenario that no longer matches it. Like every
module in aero_train.envs, it imports aero_gym inside functions only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

FINGERPRINT_VERSION = 1
PROBE_ROWS = 64
PROBE_SEED = 0
PROBE_RTOL = 1e-9

Fingerprint = dict[str, Any]
"""Plain str, float and list values, so a checkpoint and JSON both keep it as is."""


def scenario_fingerprint(env: gym.Env) -> Fingerprint | None:
    """The fingerprint of the scenario under `env`, or None for a Gymnasium env.

    Records the variables, rate, aircraft and action steps, each feature's and
    each target's name and spec, and a probe: PROBE_ROWS seeded rows (each
    reading uniform over its canonical bounds, each target over its declared
    range) with every feature's unclipped value on each, from the float path.
    The probe is what catches a changed feature body.
    """
    declared = _declarations(env)
    if declared is None:
        return None
    scenario = env.unwrapped.scenario
    from aero_gym.variables import OBSERVATION_SPECS

    rng = np.random.default_rng(PROBE_SEED)
    inputs = {
        var.value: rng.uniform(
            OBSERVATION_SPECS[var].low, OBSERVATION_SPECS[var].high, PROBE_ROWS
        ).tolist()
        for var in scenario.observation_variables
    }
    inputs |= {
        c.name: rng.uniform(c.spec.low, c.spec.high, PROBE_ROWS).tolist()
        for c in scenario.context_variables
    }
    return {**declared, "probe": {"inputs": inputs, "values": _evaluate(scenario, inputs)}}


def fingerprint_difference(stored: Fingerprint, env: gym.Env) -> str | None:
    """The first way the scenario under `env` differs from `stored`, or None.

    Declarations are compared exactly. The probe is compared by running today's
    features on the stored probe rows, to PROBE_RTOL.
    """
    current = _declarations(env)
    if current is None:
        return "the env has no aero_gym scenario"
    if stored.get("version") != FINGERPRINT_VERSION:
        return f"fingerprint version {stored.get('version')!r} is not {FINGERPRINT_VERSION}"
    for key in (
        "aircraft",
        "agent_hz",
        "observation_variables",
        "action_variables",
        "action_steps",
    ):
        if stored[key] != current[key]:
            return f"{key} was {stored[key]!r}, now {current[key]!r}"
    for kind in ("features", "context_variables"):
        was = [d["name"] for d in stored[kind]]
        now = [d["name"] for d in current[kind]]
        if was != now:
            return f"{kind} were {was}, now {now}"
        for old, new in zip(stored[kind], current[kind]):
            if old != new:
                return f"{old['name']} range was {_range(old)}, now {_range(new)}"
    values = _evaluate(env.unwrapped.scenario, stored["probe"]["inputs"])
    for name, was in stored["probe"]["values"].items():
        for row, (old, new) in enumerate(zip(was, values[name])):
            if not math.isclose(old, new, rel_tol=PROBE_RTOL, abs_tol=PROBE_RTOL):
                return f"{name} changed: probe row {row} was {old:.6g}, now {new:.6g}"
    return None


def _declarations(env: gym.Env) -> Fingerprint | None:
    """Everything in the fingerprint but its probe, or None for a Gymnasium env."""
    from aero_gym import AeroGymEnv

    base = env.unwrapped
    if not isinstance(base, AeroGymEnv):
        return None
    scenario = base.scenario
    return {
        "version": FINGERPRINT_VERSION,
        "aircraft": scenario.aircraft,
        "agent_hz": float(scenario.agent_hz),
        "observation_variables": [v.value for v in scenario.observation_variables],
        "action_variables": [v.value for v in scenario.action_variables],
        "action_steps": _action_steps(env),
        "features": [_declared(f.name, f.spec) for f in scenario.observation_features],
        "context_variables": [_declared(c.name, c.spec) for c in scenario.context_variables],
    }


def _declared(name: str, spec: Any) -> dict[str, Any]:
    return {"name": name, "unit": spec.unit, "low": float(spec.low), "high": float(spec.high)}


def _range(declared: Mapping[str, Any]) -> str:
    return f"[{declared['low']:g}, {declared['high']:g}] {declared['unit']}"


def _action_steps(env: gym.Env) -> list[float] | None:
    from .delta import DeltaAction

    while isinstance(env, gym.Wrapper):
        if isinstance(env, DeltaAction):
            return env.step_sizes.tolist()
        env = env.env
    return None


class _Unread(ValueError):
    """A feature read a key its probe row does not hold."""


class _Row(Mapping):
    """One probe row by key, naming the feature that reads a key it was not given."""

    def __init__(self, feature: str, values: Mapping[Any, float], why: str) -> None:
        self._feature, self._values, self._why = feature, values, why

    def __getitem__(self, key: Any) -> float:
        if key not in self._values:
            raise _Unread(
                f"{self._feature} reads {getattr(key, 'value', key)!r}, which {self._why}"
            )
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _evaluate(scenario: Any, inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Every feature's value on every probe row, as the env computes it.

    Raises ValueError naming the feature when it reads a key the scenario does
    not give it or fails on a row.
    """
    values: dict[str, list[float]] = {f.name: [] for f in scenario.observation_features}
    for row in range(PROBE_ROWS):
        readings = {var: inputs[var.value][row] for var in scenario.observation_variables}
        targets = {c.name: inputs[c.name][row] for c in scenario.context_variables}
        for feature in scenario.observation_features:
            obs = _Row(feature.name, readings, "the scenario does not observe")
            context = _Row(
                feature.name, targets, "the scenario does not declare in context_variables"
            )
            try:
                values[feature.name].append(float(feature.fn(obs, context)))
            except _Unread:
                raise
            except Exception as exc:
                raise _cannot(feature.name, row, exc) from exc
    return values


def _cannot(name: str, row: int, exc: Exception) -> ValueError:
    return ValueError(f"{name} cannot be fingerprinted on probe row {row}: {exc}")
