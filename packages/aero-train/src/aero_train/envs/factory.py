"""Build environment factories from an EnvConfig.

Returns a *factory* rather than an env because SyncVectorEnv needs one callable
per environment, and because a Ray worker must construct its envs in its own
process rather than receive them pickled.
"""

from __future__ import annotations

import inspect
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
    # Imported here rather than at module scope so the Gymnasium branch never
    # loads aero_gym at all. That is what keeps the CartPole integration test an
    # honest proof that the learning stack works with no flight model anywhere
    # near it. This module is the only place in aero_train permitted to import
    # aero_gym (see tests/test_dependency_rules.py).
    from aero_gym import AeroGymEnv
    from aero_gym.scenarios import make_steady_flight_scenario

    from .delta import DeltaAction
    from .scale import ScaleObservation

    scenarios = {"steady_flight": make_steady_flight_scenario}
    factory = scenarios.get(cfg.scenario)
    if factory is None:
        raise ValueError(
            f"unknown env.scenario {cfg.scenario!r}; known scenarios are "
            f"{sorted(scenarios)}"
        )

    kwargs = dict(cfg.scenario_kwargs)
    if cfg.max_episode_steps is not None:
        kwargs["max_episode_steps"] = cfg.max_episode_steps
    try:
        # Bound once, here, so a misspelled scenario parameter fails at config
        # load with the same force as a misspelled config key -- not eight
        # environments later inside a worker.
        inspect.signature(factory).bind_partial(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"env.scenario_kwargs does not fit scenario {cfg.scenario!r}: {exc}"
        ) from exc

    render_mode = cfg.render_mode
    delta = cfg.delta

    def _make() -> gym.Env:
        env = AeroGymEnv(factory(**kwargs), render_mode=render_mode)
        return DeltaAction(ScaleObservation(env), delta=delta)

    return _make
