"""Learner-side algorithms."""

from __future__ import annotations

import gymnasium as gym

from ..config import DQNConfig, TrainConfig
from .base import Algorithm
from .dqn import DQN, DQNAgent

__all__ = ["Algorithm", "DQN", "DQNAgent", "build_algorithm"]


def _env_label(cfg: TrainConfig) -> str:
    """Whichever of the two env identifiers this config actually set."""
    return cfg.env.scenario or cfg.env.id


def build_algorithm(
    cfg: TrainConfig, obs_space: gym.Space, act_space: gym.Space
) -> Algorithm:
    """Construct the configured algorithm, validating it against the env's spaces.

    This is where "DQN against a continuous action space" is caught -- the check
    needs the env, so it cannot live in config validation.
    """
    if isinstance(cfg.algo, DQNConfig):
        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError(
                f"DQN needs a discrete action space, but env {_env_label(cfg)!r} exposes "
                f"{type(act_space).__name__}. Use a discretizing wrapper (DeltaAction, "
                "Phase 2) or choose a continuous-native algorithm."
            )
        return DQN(
            obs_shape=tuple(obs_space.shape),
            n_actions=int(act_space.n),
            cfg=cfg.algo,
            device=cfg.device,
            seed=cfg.seed,
        )
    raise ValueError(f"no builder for algo config {type(cfg.algo).__name__}")
