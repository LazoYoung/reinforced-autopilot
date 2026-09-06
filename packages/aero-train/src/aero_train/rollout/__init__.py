"""Rollout engines. The Ray engine is imported lazily so the local path never
pulls Ray into the process."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym

from ..config import RolloutConfig
from ..primitives import Agent
from .base import RolloutEngine
from .local import LocalRolloutEngine

__all__ = ["RolloutEngine", "LocalRolloutEngine", "build_engine"]


def build_engine(
    cfg: RolloutConfig,
    env_fn: Callable[[], gym.Env],
    agent: Agent,
    seed: int,
) -> RolloutEngine:
    if cfg.engine == "local":
        return LocalRolloutEngine(env_fn, agent, cfg.envs_per_worker, seed)
    if cfg.engine == "ray":
        raise NotImplementedError(
            "the Ray rollout engine arrives in Phase 3; use rollout.engine: local"
        )
    raise ValueError(f"unknown rollout engine {cfg.engine!r}")
