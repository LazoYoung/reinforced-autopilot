from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

from aero_train.envs import make_env_fn


@dataclass(frozen=True)
class _EnvCfg:
    id: str
    max_episode_steps: int | None = None


def test_make_env_fn_builds_a_gymnasium_env():
    env = make_env_fn(_EnvCfg(id="CartPole-v1"))()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    env.close()


def test_make_env_fn_applies_max_episode_steps():
    env = make_env_fn(_EnvCfg(id="CartPole-v1", max_episode_steps=3))()
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    env.close()


def test_make_env_fn_returns_independent_envs():
    fn = make_env_fn(_EnvCfg(id="CartPole-v1"))
    a, b = fn(), fn()
    assert a is not b
    a.close()
    b.close()


def test_unknown_env_id_raises():
    with pytest.raises(gym.error.Error):
        make_env_fn(_EnvCfg(id="NoSuchEnv-v0"))()
