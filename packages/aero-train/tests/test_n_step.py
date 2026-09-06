from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from tests.conftest import counting_env_fn
from tests.test_experience import ConstantAgent

from aero_train.experience import ExperienceSource, to_transitions
from aero_train.primitives import Segment


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


def _hand_built_segment(terminated_row: int, truncated_row: int | None = None):
    """A 3-row, single-env segment with a *valid* row after the terminal/
    truncation row.

    ExperienceSource never produces this: it always arms the autoreset skip,
    so the row after any terminal or truncation is masked invalid, and
    to_transitions' segment.valid[j, b] check alone would stop accumulation
    there regardless of the terminated/truncated break statements below it.
    But Segment.__post_init__ does not forbid a valid row after a terminal
    one -- only an *invalid* terminal/truncated row is rejected -- so a
    hand-built (or, per Task 12, replay-buffer-concatenated) segment can have
    one. Rewards are 1.0, 1.0, 100.0 so that a cross-episode n-step sum
    (2.0 -> 102.0) is unmistakable rather than off by a hard-to-notice amount.
    """
    obs = np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]], np.float32)
    actions = np.zeros((3, 1), np.int64)
    rewards = np.array([[1.0], [1.0], [100.0]], np.float32)
    terminated = np.zeros((3, 1), bool)
    truncated = np.zeros((3, 1), bool)
    if truncated_row is None:
        terminated[terminated_row, 0] = True
    else:
        truncated[truncated_row, 0] = True
    valid = np.ones((3, 1), bool)
    return Segment(obs, actions, rewards, terminated, truncated, valid)


def test_n_step_return_does_not_cross_a_terminal_boundary():
    # row 1 terminates; row 2 is a valid row from the next episode. n_step=3
    # is large enough to reach row 2 if the terminal break is missing.
    seg = _hand_built_segment(terminated_row=1)
    tr = to_transitions(seg, gamma=1.0, n_step=3)
    assert tr.returns[0] == 2.0
    assert tr.steps[0] == 2
    assert tr.terminated[0]


def test_n_step_return_does_not_cross_a_truncation_boundary():
    # same shape, but row 1 is a time limit rather than a real terminal.
    seg = _hand_built_segment(terminated_row=-1, truncated_row=1)
    tr = to_transitions(seg, gamma=1.0, n_step=3)
    assert tr.returns[0] == 2.0
    assert tr.steps[0] == 2
    assert not tr.terminated[0]


def test_to_transitions_on_an_all_invalid_segment_keeps_trailing_dims():
    # every row masked invalid (e.g. a pending autoreset never followed by a
    # real step); the resulting arrays must still carry obs_dim / act_shape
    # so Task 12 can concatenate this Transitions with a non-empty one.
    obs = np.zeros((2, 1, 3), np.float32)
    actions = np.zeros((1, 1), np.int64)
    rewards = np.zeros((1, 1), np.float32)
    terminated = np.zeros((1, 1), bool)
    truncated = np.zeros((1, 1), bool)
    valid = np.zeros((1, 1), bool)
    seg = Segment(obs, actions, rewards, terminated, truncated, valid)

    tr = to_transitions(seg, gamma=0.99, n_step=1)
    assert len(tr) == 0
    assert tr.obs.shape == (0, 3)
    assert tr.next_obs.shape == (0, 3)
    assert tr.actions.shape == (0,)
