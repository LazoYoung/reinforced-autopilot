"""Drives environments with an agent and yields segments.

Runs inside a rollout worker. Steps all B environments in lockstep -- every env
steps once, then a single batched forward pass produces all B actions. The
alternative, stepping each env independently, pays a forward pass per env.
Lockstep's cost is that a slow env (notably one that is resetting, which
re-trims the aircraft) stalls its peers.

SyncVectorEnv is used here and AsyncVectorEnv never is: async vector envs spawn
their own process pool, which nests inside Ray's and fights its scheduler for
cores. Ray owns process-level parallelism; SyncVectorEnv owns the in-process
batch.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .primitives import Agent, EpisodeResult, Segment


class ExperienceSource:
    """A vector env plus an agent, producing segments on demand."""

    def __init__(self, env: gym.vector.VectorEnv, agent: Agent, seed: int) -> None:
        self.env = env
        self.agent = agent
        self.num_envs = int(env.num_envs)
        self._obs_shape = env.single_observation_space.shape
        self._act_space = env.single_action_space
        self._act_dtype = np.int64 if isinstance(self._act_space, gym.spaces.Discrete) else np.float32
        self._act_shape = () if isinstance(self._act_space, gym.spaces.Discrete) else self._act_space.shape
        # SeedSequence rather than seed + index: independent streams from one root
        seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(self.num_envs)]
        obs, _ = env.reset(seed=seeds)
        self._obs = np.asarray(obs, np.float32)
        self._agent_state = None
        self._autoreset = np.zeros(self.num_envs, bool)
        self._ep_return = np.zeros(self.num_envs, np.float64)
        self._ep_len = np.zeros(self.num_envs, np.int64)

    def collect(self, n_steps: int) -> tuple[Segment, list[EpisodeResult]]:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        B = self.num_envs
        obs_buf = np.empty((n_steps + 1, B, *self._obs_shape), np.float32)
        act_buf = np.empty((n_steps, B, *self._act_shape), self._act_dtype)
        rewards = np.zeros((n_steps, B), np.float32)
        terminated = np.zeros((n_steps, B), bool)
        truncated = np.zeros((n_steps, B), bool)
        valid = np.zeros((n_steps, B), bool)
        episodes: list[EpisodeResult] = []

        for t in range(n_steps):
            obs_buf[t] = self._obs
            actions, self._agent_state = self.agent(self._obs, self._agent_state)
            act_buf[t] = actions
            next_obs, reward, term, trunc, _ = self.env.step(actions)
            live = ~self._autoreset
            valid[t] = live
            rewards[t] = np.where(live, reward, 0.0)
            terminated[t] = term & live
            truncated[t] = trunc & live
            self._record_episodes(reward, term, trunc, live, episodes)
            self._obs = np.asarray(next_obs, np.float32)

        obs_buf[n_steps] = self._obs
        return (
            Segment(obs_buf, act_buf, rewards, terminated, truncated, valid),
            episodes,
        )

    def _record_episodes(self, reward, term, trunc, live, episodes) -> None:
        """Completed in the episode-boundary task."""
        self._autoreset = np.zeros(self.num_envs, bool)

    def close(self) -> None:
        self.env.close()
