from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from tests.conftest import counting_env_fn
from tests.test_experience import ConstantAgent

from aero_train.experience import ExperienceSource, to_transitions


def segment_of(episode_len=4, truncate=False, n_steps=6):
    env = gym.vector.SyncVectorEnv([counting_env_fn(episode_len, truncate)])
    src = ExperienceSource(env, ConstantAgent(), seed=0)
    seg, _ = src.collect(n_steps)
    src.close()
    return seg


def test_one_step_transitions_skip_the_autoreset_row():
    tr = to_transitions(segment_of(), gamma=0.5, n_step=1)
    assert len(tr) == 5                       # 6 rows, 1 fabricated
    assert tr.obs[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]
    assert tr.next_obs[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 1.0]
    assert tr.returns.tolist() == [1.0] * 5
    assert tr.steps.tolist() == [1] * 5
    assert tr.terminated.tolist() == [False, False, False, True, False]


def test_three_step_returns_match_the_hand_computation():
    # rewards are 1.0 each; gamma 0.5; the episode terminates on segment row 3
    tr = to_transitions(segment_of(), gamma=0.5, n_step=3)
    assert len(tr) == 5
    assert tr.returns.tolist() == pytest.approx([1.75, 1.75, 1.5, 1.0, 1.0])
    assert tr.steps.tolist() == [3, 3, 2, 1, 1]
    assert tr.terminated.tolist() == [False, True, True, True, False]
    assert tr.next_obs[:, 0].tolist() == [3.0, 4.0, 4.0, 4.0, 1.0]


def test_n_step_accumulation_stops_at_a_terminal_state():
    # row 2 can only reach 2 steps before terminating, not 3
    tr = to_transitions(segment_of(), gamma=0.5, n_step=3)
    assert tr.steps[2] == 2
    assert tr.terminated[2]


def test_truncation_bootstraps_from_the_true_final_observation():
    tr = to_transitions(segment_of(truncate=True), gamma=0.5, n_step=3)
    # terminated stays False through a time limit, so the value function
    # bootstraps -- and it bootstraps from the real final state, 4.0
    assert tr.terminated.tolist() == [False] * len(tr)
    assert tr.next_obs[1, 0] == 4.0


def test_accumulation_truncates_at_the_segment_edge():
    tr = to_transitions(segment_of(episode_len=100, n_steps=4), gamma=1.0, n_step=3)
    assert tr.steps.tolist() == [3, 3, 2, 1]
    assert tr.returns.tolist() == pytest.approx([3.0, 3.0, 2.0, 1.0])


def test_gamma_one_sums_rewards_undiscounted():
    tr = to_transitions(segment_of(episode_len=100, n_steps=6), gamma=1.0, n_step=2)
    assert tr.returns[:4].tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_transitions_from_all_envs_are_concatenated():
    env = gym.vector.SyncVectorEnv([counting_env_fn(100)] * 3)
    src = ExperienceSource(env, ConstantAgent(), seed=0)
    seg, _ = src.collect(5)
    src.close()
    assert len(to_transitions(seg, gamma=0.99, n_step=1)) == 15


def test_n_step_must_be_at_least_one():
    with pytest.raises(ValueError, match="n_step"):
        to_transitions(segment_of(), gamma=0.99, n_step=0)
