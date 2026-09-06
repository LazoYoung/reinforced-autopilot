"""In-process rollout. Imports nothing from Ray, by design and by test.

This is the debugging path, the unit-test path, and what the CartPole
integration test uses -- which is the only cheap end-to-end proof that the
learning stack is correct. A bug in DQN and a bug in the flight model are
otherwise indistinguishable when a steady_flight run fails to learn.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from ..experience import ExperienceSource
from ..primitives import Agent, EpisodeResult, Segment
from .base import RolloutEngine


class LocalRolloutEngine(RolloutEngine):
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        agent: Agent,
        envs_per_worker: int,
        seed: int,
    ) -> None:
        self.agent = agent
        env = gym.vector.SyncVectorEnv([env_fn] * envs_per_worker)
        self.source = ExperienceSource(env, agent, seed)

    def sync_weights(self, actor_state_dict: dict[str, np.ndarray]) -> None:
        self.agent.sync_weights(actor_state_dict)

    def collect(self, n_frames: int) -> tuple[Segment, list[EpisodeResult]]:
        envs = self.source.num_envs
        if n_frames % envs:
            raise ValueError(
                f"n_frames ({n_frames}) must be divisible by the {envs} environments "
                "stepped in lockstep"
            )
        return self.source.collect(n_frames // envs)

    def close(self) -> None:
        self.source.close()
