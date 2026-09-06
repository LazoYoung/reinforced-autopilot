"""Build environment factories from an EnvConfig.

Returns a *factory* rather than an env because SyncVectorEnv needs one callable
per environment, and because a Ray worker must construct its envs in its own
process rather than receive them pickled.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym


def make_env_fn(cfg) -> Callable[[], gym.Env]:
    """Return a zero-argument callable that builds one environment.

    ``cfg.id`` is a Gymnasium id. Phase 2 adds an ``aero_gym:`` branch here that
    builds an AeroGymEnv from a named scenario and wraps it in DeltaAction; this
    module is the only place in aero_train permitted to import aero_gym.
    """
    env_id = cfg.id
    max_episode_steps = getattr(cfg, "max_episode_steps", None)

    def _make() -> gym.Env:
        return gym.make(env_id, max_episode_steps=max_episode_steps)

    return _make
