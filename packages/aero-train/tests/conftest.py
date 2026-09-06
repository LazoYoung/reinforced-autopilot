from __future__ import annotations

import gymnasium as gym
import numpy as np


class CountingEnv(gym.Env):
    """Fully deterministic env whose every quantity is hand-computable.

    Observation is the step index as a 1-vector; reward is always 1.0; the
    episode ends after ``episode_len`` steps, by termination or by truncation
    depending on ``truncate``. A test can therefore predict segment contents,
    episode returns and n-step sums exactly, with no physics and no RNG.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 4, truncate: bool = False) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.episode_len = episode_len
        self.truncate = truncate
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        return (
            np.array([float(self.t)], np.float32),
            1.0,
            done and not self.truncate,
            done and self.truncate,
            {},
        )


def counting_env_fn(episode_len: int = 4, truncate: bool = False):
    """Env factory for SyncVectorEnv, which needs a zero-argument callable."""
    return lambda: CountingEnv(episode_len=episode_len, truncate=truncate)
