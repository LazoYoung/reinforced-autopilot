"""Drives environments with an agent and yields segments.

Runs inside a rollout worker. Steps all B environments in lockstep: every env
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

from .primitives import Agent, EpisodeResult, Segment, Transitions


class ExperienceSource:
    """A vector env plus an agent, producing segments on demand."""

    def __init__(self, env: gym.vector.VectorEnv, agent: Agent, seed: int) -> None:
        self.env = env
        self.agent = agent
        self.num_envs = int(env.num_envs)
        self._obs_shape = env.single_observation_space.shape
        act_space = env.single_action_space
        self._act_dtype = np.int64 if isinstance(act_space, gym.spaces.Discrete) else np.float32
        self._act_shape = () if isinstance(act_space, gym.spaces.Discrete) else act_space.shape
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
            # Terminated dominates, since calling a both-flags step truncated
            # bootstraps past a terminal state. Here, as no env fix reaches TimeLimit.
            truncated[t] = trunc & live & ~term
            self._record_episodes(reward, term, trunc, live, episodes)
            self._obs = np.asarray(next_obs, np.float32)

        obs_buf[n_steps] = self._obs
        return (
            Segment(obs_buf, act_buf, rewards, terminated, truncated, valid),
            episodes,
        )

    def _record_episodes(self, reward, term, trunc, live, episodes) -> None:
        """Accumulate per-env returns and arm the autoreset skip.

        Gymnasium 1.x autoresets on the step *after* the one that ended the
        episode: that next step ignores the action and returns the reset
        observation. Arming `_autoreset` here is what makes the following
        collect() row invalid, so a fabricated (final state, ignored action,
        reset state) tuple never reaches the buffer.
        """
        self._ep_return[live] += reward[live]
        self._ep_len[live] += 1
        done = (term | trunc) & live
        for index in np.flatnonzero(done):
            episodes.append(
                EpisodeResult(
                    ret=float(self._ep_return[index]),
                    length=int(self._ep_len[index]),
                    terminated=bool(term[index]),
                )
            )
            self._ep_return[index] = 0.0
            self._ep_len[index] = 0
        self._autoreset = done

    def close(self) -> None:
        self.env.close()


def to_transitions(segment: Segment, gamma: float, n_step: int = 1) -> Transitions:
    """Flatten a segment into n-step transitions, dropping the autoreset rows.

    This is the single place in the codebase where a segment becomes flat
    transitions. Off-policy algorithms call it on insert into the replay buffer;
    on-policy algorithms consume segments as they are.

    Accumulation for a row stops at whichever comes first: n steps, a terminal
    state, a time limit, the end of the segment, or an autoreset row. `steps`
    records how many were actually accumulated, so the caller bootstraps with
    gamma ** steps rather than a fixed gamma ** n.

    A time limit sets `terminated=False` and leaves `next_obs` at the true final
    observation, so the value function bootstraps straight through it. Under
    Gymnasium's next-step autoreset that observation is already sitting at
    obs[t + 1]; nothing has to be stashed.

    Cost is O(T * B * n_step), not just O(T * B): each of the T * B valid rows
    walks up to n_step rows ahead. That is fine at this scale (128 x 8 x 3 is a
    few thousand inner iterations per collect(), against a flight model that
    costs milliseconds per step); vectorize only if a profile says to.
    """
    if n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {n_step}")

    obs, actions, returns, next_obs, terminated, steps = [], [], [], [], [], []
    for b in range(segment.B):
        for t in range(segment.T):
            if not segment.valid[t, b]:
                continue
            total = 0.0
            taken = 0
            hit_terminal = False
            last = t
            j = t
            while taken < n_step and j < segment.T and segment.valid[j, b]:
                total += (gamma**taken) * float(segment.rewards[j, b])
                taken += 1
                last = j
                if segment.terminated[j, b]:
                    hit_terminal = True
                    break
                if segment.truncated[j, b]:
                    break
                j += 1
            obs.append(segment.obs[t, b])
            actions.append(segment.actions[t, b])
            returns.append(total)
            next_obs.append(segment.obs[last + 1, b])
            terminated.append(hit_terminal)
            steps.append(taken)

    # np.asarray([], dtype) collapses to (0,), dropping the trailing axes an
    # all-invalid segment still owns, and breaks concatenation downstream.
    obs_shape = segment.obs.shape[2:]
    act_shape = segment.actions.shape[2:]

    def _stack(rows: list, trailing_shape: tuple[int, ...], dtype) -> np.ndarray:
        if rows:
            return np.asarray(rows, dtype)
        return np.empty((0, *trailing_shape), dtype)

    return Transitions(
        obs=_stack(obs, obs_shape, np.float32),
        actions=_stack(actions, act_shape, segment.actions.dtype),
        returns=np.asarray(returns, np.float32),
        next_obs=_stack(next_obs, obs_shape, np.float32),
        terminated=np.asarray(terminated, bool),
        steps=np.asarray(steps, np.int32),
    )
