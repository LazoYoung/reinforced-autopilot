"""Learner-side algorithms."""

from __future__ import annotations

import gymnasium as gym

from ..config import ConstantConfig, DQNConfig, RandomConfig, TrainConfig
from .base import Algorithm
from .baseline import Constant, Random
from .dqn import DQN, DQNAgent

__all__ = [
    "Algorithm",
    "Constant",
    "DQN",
    "DQNAgent",
    "Random",
    "build_algorithm",
]


def _env_label(cfg: TrainConfig) -> str:
    """Whichever of the two env identifiers this config actually set."""
    return cfg.env.scenario or cfg.env.id


def _require_discrete(cfg: TrainConfig, act_space: gym.Space, name: str) -> int:
    if not isinstance(act_space, gym.spaces.Discrete):
        raise ValueError(
            f"{name} needs a discrete action space, but env "
            f"{_env_label(cfg)!r} exposes {type(act_space).__name__}. Use a "
            "discretizing wrapper (DeltaAction) or a continuous-native algorithm."
        )
    return int(act_space.n)


def build_algorithm(
    cfg: TrainConfig, obs_space: gym.Space, act_space: gym.Space
) -> Algorithm:
    """Construct the configured algorithm, validating it against the env's spaces.

    This is where "DQN against a continuous action space" is caught -- the check
    needs the env, so it cannot live in config validation.
    """
    if isinstance(cfg.algo, DQNConfig):
        n_actions = _require_discrete(cfg, act_space, "DQN")
        return DQN(
            obs_shape=tuple(obs_space.shape),
            n_actions=n_actions,
            cfg=cfg.algo,
            device=cfg.device,
            seed=cfg.seed,
        )
    if isinstance(cfg.algo, RandomConfig):
        return Random(_require_discrete(cfg, act_space, "the random baseline"), seed=cfg.seed)
    if isinstance(cfg.algo, ConstantConfig):
        n_actions = _require_discrete(cfg, act_space, "the constant baseline")
        action = cfg.algo.action if cfg.algo.action is not None else (n_actions - 1) // 2
        if not 0 <= action < n_actions:
            raise ValueError(
                f"algo.action {action} is outside the env's {n_actions} actions"
            )
        return Constant(n_actions, action, seed=cfg.seed)
    raise ValueError(f"no builder for algo config {type(cfg.algo).__name__}")
