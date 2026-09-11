"""Build environment factories from an EnvConfig.

Returns a *factory* rather than an env because SyncVectorEnv needs one callable
per environment, and because a Ray worker must construct its envs in its own
process rather than receive them pickled.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import gymnasium as gym


def make_env_fn(cfg) -> Callable[[], gym.Env]:
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


def _make_scenario_env_fn(cfg) -> Callable[[], gym.Env]:
    """Build an AeroGymEnv for the configured scenario.

    aero_gym is imported here, not at module scope, so the Gymnasium branch
    never loads it and the CartPole integration test stays honest proof that
    the learning stack runs with no flight model near it. This module is the
    only place in aero_train allowed the import (tests/test_dependency_rules).
    """
    from aero_gym import AeroGymEnv
    from aero_gym.scenarios import SteadyFlightParams, make_steady_flight_scenario

    from .delta import DeltaAction
    from .scale import ScaleObservation

    scenarios = {"steady_flight": (make_steady_flight_scenario, SteadyFlightParams)}
    entry = scenarios.get(cfg.scenario)
    if entry is None:
        raise ValueError(
            f"unknown env.scenario {cfg.scenario!r}; known scenarios are "
            f"{sorted(scenarios)}"
        )
    factory, params_type = entry
    try:
        # Bound once, here, so a misspelled scenario parameter fails at factory
        # construction rather than eight environments later inside a worker.
        params = params_type(**dict(cfg.scenario_kwargs))
    except TypeError as exc:
        raise ValueError(
            f"env.scenario_kwargs does not fit scenario {cfg.scenario!r}: {exc}"
        ) from exc

    max_episode_steps = cfg.max_episode_steps
    render_mode = cfg.render_mode
    delta = cfg.delta

    def _make() -> gym.Env:
        scenario = factory(params)
        if max_episode_steps is not None:
            scenario = dataclasses.replace(scenario, max_episode_steps=max_episode_steps)
        env = AeroGymEnv(scenario, render_mode=render_mode)
        return DeltaAction(ScaleObservation(env), delta=delta)

    return _make
