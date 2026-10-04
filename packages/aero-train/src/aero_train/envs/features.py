"""A scenario's observation features as a torch module, for the policy file.

Tracing runs the scenario's own `ObservationFeature.fn`s on tensors, so a policy
file keeps the features it was trained on after the scenario's code changes.
This is the only export code that reads a Scenario, and like every module in
aero_train.envs it imports aero_gym inside functions only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from torch import nn

from ..tensors import Column, FeatureBatch, column_shapes, example_columns

FeatureFn = Callable[[Mapping[Any, Column], Mapping[str, Column]], Column]
"""An ObservationFeature.fn as export calls it: readings and targets as columns."""


@dataclass(frozen=True)
class ColumnSpec:
    """One named input or output of a policy file: what it is, in which unit and range.

    `role` is one of reading, target, prev, command or value; a value column has
    no unit or range. `doc` is the ONNX doc string a person reads, and `entry()`
    the same facts as aero.inputs and aero.outputs carry them for a host.
    """

    name: str
    role: str
    description: str
    unit: str | None = None
    low: float | None = None
    high: float | None = None

    @property
    def doc(self) -> str:
        if self.unit is None:
            return self.description
        return f"{self.unit}, {self.low:g} to {self.high:g}: {self.description}"

    def entry(self) -> dict[str, Any]:
        """This column's aero.inputs or aero.outputs entry."""
        entry: dict[str, Any] = {"role": self.role, "description": self.description}
        if self.unit is not None:
            entry |= {"unit": self.unit, "low": self.low, "high": self.high}
        return entry


class FeatureModule(nn.Module):
    """A scenario's features over columns of readings and targets.

    Takes one Column per reading, then one per target, in the order its
    TracedScenario lists them, and returns what the env would have computed
    for the same values.
    """

    def __init__(
        self,
        fns: Sequence[FeatureFn],
        readings: Sequence[Any],
        targets: Sequence[str],
        low: Sequence[float],
        high: Sequence[float],
    ) -> None:
        super().__init__()
        self._fns = tuple(fns)
        self._readings = tuple(readings)
        self._targets = tuple(targets)
        self.register_buffer("low", torch.tensor(list(low), dtype=torch.float32))
        self.register_buffer("high", torch.tensor(list(high), dtype=torch.float32))

    def values(self, *inputs: Column) -> FeatureBatch:
        """The features before they are clipped to their bounds."""
        split = len(self._readings)
        obs = dict(zip(self._readings, inputs[:split]))
        context = dict(zip(self._targets, inputs[split:]))
        return torch.stack([fn(obs, context) for fn in self._fns], dim=1)

    def forward(self, *inputs: Column) -> FeatureBatch:
        return torch.clamp(self.values(*inputs), self.low, self.high)


@dataclass(frozen=True)
class TracedScenario:
    """Everything a policy file takes from its scenario: one module, the rest plain data."""

    features: FeatureModule
    inputs: tuple[ColumnSpec, ...]
    """The readings, then the targets: the order `features` takes them in."""
    actions: tuple[ColumnSpec, ...]
    """One command column per action variable, in the action Box's order."""
    metadata: dict[str, str]
    """aero.agent_hz and aero.aircraft."""
    readings: tuple[Any, ...]
    """The FlightVariables the features read, in observation_variables order."""
    targets: tuple[str, ...]
    """The context variables the features read, in declaration order."""

    def feed(self, info: Mapping[str, Any]) -> dict[str, float]:
        """This step's readings and targets by input name, from an AeroGymEnv
        reset or step `info`."""
        obs, context = info["obs"], info["context"]
        return {
            **{var.value: float(obs[var]) for var in self.readings},
            **{name: float(context[name]) for name in self.targets},
        }


def scenario_of(env: gym.Env) -> Any:
    """The Scenario under a wrapped AeroGymEnv."""
    from aero_gym import AeroGymEnv

    base = env.unwrapped
    if not isinstance(base, AeroGymEnv):
        raise ValueError(
            f"export needs an aero_gym scenario env; {type(base).__name__} has no "
            "canonical names for its observations"
        )
    return base.scenario


def trace_scenario(scenario: Any) -> TracedScenario:
    """Traces each of the scenario's features alone, then builds them into one module.

    Tracing alone first is what lets an error name the feature. The file takes
    only the readings and targets some feature reads.
    """
    from aero_gym.variables import ACTION_SPECS, OBSERVATION_SPECS

    declared = {c.name: c.spec for c in scenario.context_variables}
    read: set[Any] = set()
    for feature in scenario.observation_features:
        read.update(_trace_one(feature, scenario.observation_variables, tuple(declared)))
    readings = tuple(v for v in scenario.observation_variables if v in read)
    targets = tuple(name for name in declared if name in read)
    features = scenario.observation_features
    return TracedScenario(
        features=FeatureModule(
            [f.fn for f in features],
            readings,
            targets,
            [f.spec.low for f in features],
            [f.spec.high for f in features],
        ),
        inputs=tuple(_column(v.value, "reading", OBSERVATION_SPECS[v]) for v in readings)
        + tuple(_column(name, "target", declared[name]) for name in targets),
        actions=tuple(
            _column(v.value, "command", ACTION_SPECS[v]) for v in scenario.action_variables
        ),
        metadata={
            "aero.agent_hz": str(scenario.agent_hz),
            "aero.aircraft": scenario.aircraft,
        },
        readings=readings,
        targets=targets,
    )


def _column(name: str, role: str, spec: Any) -> ColumnSpec:
    return ColumnSpec(name, role, spec.description, spec.unit, float(spec.low), float(spec.high))


class _Unavailable(KeyError):
    def __init__(self, key: Any, why: str) -> None:
        super().__init__(key)
        self.message = f"{getattr(key, 'value', key)!r}, which {why}"


class _Lookups(Mapping):
    """Sample columns by key, remembering which keys were read."""

    def __init__(self, keys: Sequence[Any], why: str) -> None:
        self._values = dict(zip(keys, example_columns(len(keys))))
        self._why = why
        self.read: list[Any] = []

    def __getitem__(self, key: Any) -> Column:
        if key not in self._values:
            raise _Unavailable(key, self._why)
        if key not in self.read:
            self.read.append(key)
        return self._values[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def samples(self) -> list[Column]:
        """Every sample column handed out, whether or not the feature read it."""
        return list(self._values.values())


def _trace_one(feature: Any, observed: Sequence[Any], declared: Sequence[str]) -> list[Any]:
    """Which readings and targets one feature reads, raising if it cannot be traced."""
    obs = _Lookups(observed, "the scenario does not observe")
    context = _Lookups(declared, "the scenario does not declare in context_variables")
    rule = "a feature may use arithmetic operators and abs only; see ObservationFeature"
    versions = [c._version for c in (*obs.samples(), *context.samples())]
    try:
        out = feature.fn(obs, context)
    except _Unavailable as exc:
        raise ValueError(f"{feature.name} reads {exc.message}") from None
    except Exception as exc:
        raise ValueError(f"{feature.name} is not traceable: {exc}; {rule}") from exc
    if versions != [c._version for c in (*obs.samples(), *context.samples())]:
        raise ValueError(f"{feature.name} is not traceable: it changes its input in place; {rule}")
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != (2,):
        raise ValueError(
            f"{feature.name} is not traceable: it returned {type(out).__name__}, not "
            f"one value per row computed from its inputs; {rule}"
        )
    module = FeatureModule(
        [feature.fn], obs.read, context.read, [feature.spec.low], [feature.spec.high]
    )
    n = len(obs.read) + len(context.read)
    try:
        torch.export.export(module, example_columns(n), dynamic_shapes=column_shapes(n))
    except Exception as exc:
        raise ValueError(f"{feature.name} is not traceable: {exc}; {rule}") from exc
    return obs.read + context.read
