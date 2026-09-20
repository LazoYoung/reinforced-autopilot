"""Build environment factories from an EnvConfig.

Returns a *factory* rather than an env because SyncVectorEnv needs one callable
per environment, and because a Ray worker must construct its envs in its own
process rather than receive them pickled.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import gymnasium as gym
from aero_train.config import EnvConfig


def make_env_fn(cfg: EnvConfig) -> Callable[[], gym.Env]:
    """Return a zero-argument callable that builds one environment.

    Two branches, selected by which of `cfg.scenario` and `cfg.id` is set.
    """
    if getattr(cfg, "scenario", None) is not None:
        return _make_scenario_env_fn(cfg)

    env_id = cfg.id
    max_episode_steps = getattr(cfg, "max_episode_steps", None)

    def _make() -> gym.Env:
        return gym.make(env_id, max_episode_steps=max_episode_steps)

    return _make


def _make_scenario_env_fn(cfg: EnvConfig) -> Callable[[], gym.Env]:
    """Build an AeroGymEnv for the configured scenario.

    aero_gym is imported here, not at module scope, so the Gymnasium branch
    never loads it and the CartPole integration test stays honest proof that
    the learning stack runs with no flight model near it. This module is the
    only place in aero_train allowed the import (tests/test_dependency_rules).

    The registration and the params object are resolved eagerly, here, and only
    registration.build() is deferred into the returned callable, because each
    worker must construct its own environments in its own process.
    """
    from aero_gym import AeroGymEnv
    from aero_gym.registry import build_params, lookup

    from .delta import DeltaAction
    from .scale import ScaleObservation

    # Resolved before any environment exists, so a misspelled parameter fails
    # once here rather than eight times inside workers.
    registration = lookup(cfg.scenario)
    params = build_params(registration, cfg.params)
    max_episode_steps = cfg.max_episode_steps
    render_mode = cfg.render_mode

    def _make() -> gym.Env:
        scenario = (
            registration.build() if params is None else registration.build(params)
        )
        if max_episode_steps is not None:
            # Applied here rather than as a params field every scenario would
            # have to repeat; replace re-runs the frozen dataclass validation.
            scenario = dataclasses.replace(scenario, max_episode_steps=max_episode_steps)
        env = ScaleObservation(AeroGymEnv(scenario, render_mode=render_mode))
        if scenario.action_delta is None:
            # Posed for continuous actions: the algorithm acts on the Box itself.
            return env
        return DeltaAction(env, delta=scenario.action_delta)

    return _make
