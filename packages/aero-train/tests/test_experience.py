from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from tests.conftest import CountingEnv, counting_env_fn

from aero_train.experience import ExperienceSource
from aero_train.primitives import Agent, Segment


class ConstantAgent(Agent):
    """Always emits action 0. Keeps segment contents fully predictable."""

    def __init__(self, action: int = 0) -> None:
        self.action = action
        self.calls = 0

    def __call__(self, obs, agent_state=None):
        self.calls += 1
        return np.full(len(obs), self.action, np.int64), agent_state

    def sync_weights(self, actor_state_dict):
        pass


def make_source(n_envs=2, episode_len=100, truncate=False, agent=None):
    env = gym.vector.SyncVectorEnv([counting_env_fn(episode_len, truncate)] * n_envs)
    return ExperienceSource(env, agent or ConstantAgent(), seed=0)


def test_collect_returns_a_segment_of_the_requested_length():
    src = make_source(n_envs=3)
    seg, _ = src.collect(5)
    assert isinstance(seg, Segment)
    assert (seg.T, seg.B) == (5, 3)
    src.close()


def test_segment_obs_carries_the_bootstrap_row():
    src = make_source(n_envs=2)
    seg, _ = src.collect(4)
    assert seg.obs.shape == (5, 2, 1)
    src.close()


def test_obs_are_the_counting_env_step_indices():
    src = make_source(n_envs=1)
    seg, _ = src.collect(4)
    assert seg.obs[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    src.close()


def test_rewards_and_actions_are_recorded():
    src = make_source(n_envs=2, agent=ConstantAgent(action=1))
    seg, _ = src.collect(3)
    assert seg.rewards.tolist() == [[1.0, 1.0]] * 3
    assert seg.actions.tolist() == [[1, 1]] * 3
    src.close()


def test_all_steps_are_valid_when_no_episode_ends():
    src = make_source(n_envs=2, episode_len=100)
    seg, episodes = src.collect(6)
    assert seg.valid.all()
    assert not seg.terminated.any() and not seg.truncated.any()
    assert episodes == []
    src.close()


def test_one_batched_forward_pass_per_timestep_not_one_per_env():
    agent = ConstantAgent()
    src = make_source(n_envs=8, agent=agent)
    src.collect(5)
    assert agent.calls == 5      # lockstep: 5, not 40
    src.close()


def test_consecutive_collects_are_contiguous():
    src = make_source(n_envs=1)
    first, _ = src.collect(3)
    second, _ = src.collect(3)
    # the bootstrap row of one segment is the opening row of the next
    assert first.obs[-1, 0, 0] == second.obs[0, 0, 0]
    src.close()


def test_num_envs_is_exposed():
    src = make_source(n_envs=4)
    assert src.num_envs == 4
    src.close()


def test_collect_rejects_a_non_positive_length():
    src = make_source()
    with pytest.raises(ValueError, match="n_steps"):
        src.collect(0)
    src.close()
