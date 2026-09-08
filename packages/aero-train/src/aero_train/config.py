"""Run configuration: frozen dataclasses outside, YAML inside.

YAML never reaches the rest of the codebase as a dict. It is validated into
these dataclasses at load time, so a misspelled key fails immediately instead
of being silently ignored for the length of a run.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import yaml


DEFAULT_DELTA = 0.1


@dataclass(frozen=True, slots=True)
class EnvConfig:
    id: str | None = None
    """A Gymnasium id. Mutually exclusive with `scenario`."""

    scenario: str | None = None
    """An aero_gym scenario name. Mutually exclusive with `id`.

    A field rather than a prefix on `id` for two reasons. Gymnasium's make()
    already documents and implements "module:Env-v0" as "import module, then
    make Env-v0", so an id like "aero_gym:steady_flight" is a valid Gymnasium
    id meaning something else -- reusing it would mean a removed branch check
    silently does something almost-right instead of failing. And a prefix would
    force this validation to parse a string to know which rules apply, when the
    point of this module is that nothing downstream parses.
    """

    max_episode_steps: int | None = None
    scenario_kwargs: dict[str, Any] = field(default_factory=dict)
    render_mode: str | None = None
    delta: float = DEFAULT_DELTA

    def __post_init__(self) -> None:
        if (self.id is None) == (self.scenario is None):
            raise ValueError(
                "env needs exactly one of `id` (a Gymnasium id) or `scenario` "
                "(an aero_gym scenario name)"
            )
        if self.id is not None and not self.id:
            raise ValueError("env.id must be a non-empty Gymnasium id")
        if self.scenario is not None and not self.scenario:
            raise ValueError("env.scenario must be a non-empty scenario name")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("env.max_episode_steps must be positive or null")
        if self.render_mode not in (None, "flightgear"):
            raise ValueError(
                f"env.render_mode must be null or 'flightgear', got {self.render_mode!r}"
            )
        if not 0.0 < self.delta <= 1.0:
            raise ValueError("env.delta must be in (0, 1]")
        if self.scenario is None:
            # Rejected rather than ignored: a setting that silently does nothing
            # is the failure mode this whole module exists to prevent.
            for name, value in (
                ("render_mode", self.render_mode),
                ("scenario_kwargs", self.scenario_kwargs),
                ("delta", self.delta != DEFAULT_DELTA or None),
            ):
                if value:
                    raise ValueError(
                        f"env.{name} only applies to an aero_gym scenario env; "
                        "set env.scenario instead of env.id"
                    )
        if "max_episode_steps" in self.scenario_kwargs:
            raise ValueError(
                "set the episode limit with env.max_episode_steps, not inside "
                "env.scenario_kwargs -- one setting, one place"
            )


@dataclass(frozen=True, slots=True)
class DQNConfig:
    type: Literal["dqn"] = "dqn"
    hidden_sizes: tuple[int, ...] = (128, 128)
    lr: float = 5e-4
    gamma: float = 0.99
    n_step: int = 1
    buffer_size: int = 50_000
    warmup_frames: int = 1_000
    batch_size: int = 64
    target_sync_interval: int = 500      # in global steps
    updates_per_iter: int = 4
    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    epsilon_decay_frames: int = 20_000

    def __post_init__(self) -> None:
        # Reject non-ints before coercing, so a float like 3.7 fails loudly instead
        # of being silently truncated into a network with the wrong layer widths.
        # bool is a subclass of int in Python, so True/False must be excluded too.
        if any(isinstance(h, bool) or not isinstance(h, int) for h in self.hidden_sizes):
            raise ValueError("algo.hidden_sizes must be a non-empty list of positive ints")
        # YAML gives a list; the rest of the code wants an immutable tuple
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if not self.hidden_sizes or any(h <= 0 for h in self.hidden_sizes):
            raise ValueError("algo.hidden_sizes must be a non-empty list of positive ints")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("algo.gamma must be in (0, 1]")
        if self.n_step < 1:
            raise ValueError("algo.n_step must be >= 1")
        if self.batch_size > self.buffer_size:
            raise ValueError("algo.batch_size cannot exceed algo.buffer_size")
        if self.warmup_frames < self.batch_size:
            raise ValueError("algo.warmup_frames must be at least algo.batch_size")
        if self.warmup_frames > self.buffer_size:
            # the ring buffer caps at buffer_size, so len(buffer) could never
            # reach a larger warmup_frames -- update() would return None forever
            # and the run would silently never take a gradient step
            raise ValueError("algo.warmup_frames cannot exceed algo.buffer_size")
        if not self.epsilon_final <= self.epsilon_start:
            raise ValueError("algo.epsilon_final must not exceed algo.epsilon_start")
        if self.epsilon_decay_frames < 1:
            raise ValueError("algo.epsilon_decay_frames must be >= 1")
        for name in ("lr", "buffer_size", "batch_size", "target_sync_interval", "updates_per_iter"):
            if getattr(self, name) <= 0:
                raise ValueError(f"algo.{name} must be positive")


# Widens to `DQNConfig | A2CConfig | DDPGConfig` in Phase 2. The registry is the
# discriminator for the tagged union, keyed on the YAML `type:` field.
AlgoConfig = DQNConfig
ALGO_CONFIGS: dict[str, type] = {"dqn": DQNConfig}


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    engine: Literal["local", "ray"] = "local"
    n_workers: int = 1
    envs_per_worker: int = 8
    frames_per_iter: int = 128

    def __post_init__(self) -> None:
        if self.engine not in ("local", "ray"):
            raise ValueError(f"rollout.engine must be 'local' or 'ray', got {self.engine!r}")
        if self.n_workers < 1 or self.envs_per_worker < 1 or self.frames_per_iter < 1:
            raise ValueError("rollout counts must all be >= 1")
        if self.engine == "local" and self.n_workers != 1:
            raise ValueError("rollout.n_workers must be 1 for the local engine; use engine: ray")
        batch = self.n_workers * self.envs_per_worker
        if self.frames_per_iter % batch:
            raise ValueError(
                f"rollout.frames_per_iter ({self.frames_per_iter}) must be divisible by "
                f"n_workers * envs_per_worker ({batch}); otherwise workers collect "
                "unequal segment lengths and cannot be concatenated"
            )

    @property
    def steps_per_iter(self) -> int:
        """Segment length T each worker collects per iteration."""
        return self.frames_per_iter // (self.n_workers * self.envs_per_worker)


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    enabled: bool = True
    experiment: str = "reinforced-autopilot"
    tracking_uri: str | None = None      # None -> local ./mlruns file store


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    dir: str = "runs/checkpoints"      # gitignored; see "Where experiments and artifacts go"
    interval_iters: int = 50
    include_buffer: bool = False
    eval_episodes: int = 5
    eval_interval_iters: int = 50

    def __post_init__(self) -> None:
        if self.interval_iters < 1 or self.eval_interval_iters < 1:
            raise ValueError("checkpoint intervals must be >= 1")
        if self.eval_episodes < 1:
            raise ValueError("checkpoint.eval_episodes must be >= 1")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    run_name: str
    env: EnvConfig
    algo: AlgoConfig
    seed: int = 0
    total_steps: int = 100_000           # total FRAMES to collect, not gradient updates
    device: str = "cpu"
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if self.total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        warmup = getattr(self.algo, "warmup_frames", 0)
        if self.total_steps <= warmup:
            raise ValueError(
                f"total_steps ({self.total_steps}) is at or below the replay warmup "
                f"threshold ({warmup}); the run would never take a gradient step"
            )


def _build(cls: type, data: dict[str, Any], path: str | None):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        where = f"{path}." if path else ""
        raise TypeError(
            f"unknown config key(s): {', '.join(where + k for k in unknown)}; "
            f"known keys are {sorted(known)}"
        )
    return cls(**data)


def config_from_dict(d: dict[str, Any]) -> TrainConfig:
    """Build a validated TrainConfig from a plain (YAML-shaped) dict."""
    d = copy.deepcopy(d)
    algo_raw = dict(d.pop("algo", {}) or {})
    algo_type = algo_raw.get("type", "dqn")
    if algo_type not in ALGO_CONFIGS:
        raise ValueError(
            f"unknown algo.type {algo_type!r}; known types are {sorted(ALGO_CONFIGS)}"
        )
    nested = {
        "env": _build(EnvConfig, dict(d.pop("env", {}) or {}), "env"),
        "algo": _build(ALGO_CONFIGS[algo_type], algo_raw, "algo"),
        "rollout": _build(RolloutConfig, dict(d.pop("rollout", {}) or {}), "rollout"),
        "tracking": _build(TrackingConfig, dict(d.pop("tracking", {}) or {}), "tracking"),
        "checkpoint": _build(CheckpointConfig, dict(d.pop("checkpoint", {}) or {}), "checkpoint"),
    }
    return _build(TrainConfig, {**d, **nested}, None)


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Inverse of config_from_dict. Used to embed the config in a checkpoint."""
    return dataclasses.asdict(cfg)


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``dotted.path=value`` strings to a raw config dict.

    Applied *before* the dataclasses are built, so an override cannot bypass
    __post_init__ validation. Values are parsed as YAML scalars, which gives
    ints, floats, bools, nulls and lists their natural types for free.
    """
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key.path=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        parts = [p for p in dotted.split(".") if p]
        if not parts:
            raise ValueError(f"override must be key.path=value, got {item!r}")
        node: Any = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"override path {dotted!r} traverses a non-mapping value")
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, str):
            # PyYAML implements YAML 1.1, whose float grammar requires a decimal
            # point before the exponent -- so "1e-3" parses as the string "1e-3",
            # not the float 0.001. That spelling is exactly how anyone types a
            # learning-rate override (the DQN default is 5e-4), so without this
            # fallback the value reaches __post_init__ as a str and fails an
            # unrelated numeric comparison instead of being accepted normally.
            try:
                parsed = float(parsed)
            except (TypeError, ValueError):
                pass
        node[parts[-1]] = parsed
    return out


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> TrainConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a run config must be a mapping")
    return config_from_dict(apply_overrides(data, overrides))


def flatten_config(cfg: TrainConfig) -> dict[str, str]:
    """Dotted, stringified config for MLflow params.

    Topology (n_workers, envs_per_worker, frames_per_iter) is included because a
    seed alone does not identify a run — changing the topology changes how the
    random streams are partitioned.
    """

    def walk(obj: Any, prefix: str) -> Iterator[tuple[str, str]]:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in fields(obj):
                yield from walk(getattr(obj, f.name), f"{prefix}{f.name}.")
        else:
            yield prefix.rstrip("."), str(obj)

    return dict(walk(cfg, ""))
