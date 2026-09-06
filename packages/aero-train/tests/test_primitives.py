from __future__ import annotations

import numpy as np
import pytest

from aero_train.primitives import EpisodeResult, Segment, Transitions


def make_segment(T=3, B=2, obs_dim=1):
    return Segment(
        obs=np.zeros((T + 1, B, obs_dim), np.float32),
        actions=np.zeros((T, B), np.int64),
        rewards=np.ones((T, B), np.float32),
        terminated=np.zeros((T, B), bool),
        truncated=np.zeros((T, B), bool),
        valid=np.ones((T, B), bool),
    )


def test_segment_reports_its_dimensions():
    seg = make_segment(T=5, B=4)
    assert (seg.T, seg.B) == (5, 4)


def test_obs_carries_one_extra_timestep_for_bootstrapping():
    seg = make_segment(T=5, B=4)
    assert seg.obs.shape[0] == seg.T + 1


def test_wrong_obs_length_raises():
    seg = make_segment()
    with pytest.raises(ValueError, match="T \\+ 1"):
        Segment(seg.obs[:-1], seg.actions, seg.rewards, seg.terminated, seg.truncated, seg.valid)


def test_mismatched_batch_axis_raises():
    seg = make_segment(T=3, B=2)
    with pytest.raises(ValueError, match="batch"):
        Segment(
            np.zeros((4, 3, 1), np.float32),
            seg.actions, seg.rewards, seg.terminated, seg.truncated, seg.valid,
        )


def test_a_step_cannot_be_both_terminated_and_truncated():
    seg = make_segment()
    term = seg.terminated.copy(); term[0, 0] = True
    trunc = seg.truncated.copy(); trunc[0, 0] = True
    with pytest.raises(ValueError, match="both terminated and truncated"):
        Segment(seg.obs, seg.actions, seg.rewards, term, trunc, seg.valid)


def test_invalid_steps_cannot_carry_episode_ends():
    seg = make_segment()
    term = seg.terminated.copy(); term[1, 0] = True
    valid = seg.valid.copy(); valid[1, 0] = False
    with pytest.raises(ValueError, match="invalid step"):
        Segment(seg.obs, seg.actions, seg.rewards, term, seg.truncated, valid)


def test_non_boolean_masks_raise():
    seg = make_segment()
    with pytest.raises(ValueError, match="bool"):
        Segment(
            seg.obs, seg.actions, seg.rewards,
            seg.terminated.astype(np.int8), seg.truncated, seg.valid,
        )


def test_transitions_length_is_the_number_of_rows():
    n = 7
    tr = Transitions(
        obs=np.zeros((n, 1), np.float32),
        actions=np.zeros(n, np.int64),
        returns=np.zeros(n, np.float32),
        next_obs=np.zeros((n, 1), np.float32),
        terminated=np.zeros(n, bool),
        steps=np.ones(n, np.int32),
    )
    assert len(tr) == n


def test_transitions_reject_ragged_fields():
    with pytest.raises(ValueError, match="same length"):
        Transitions(
            obs=np.zeros((3, 1), np.float32),
            actions=np.zeros(2, np.int64),
            returns=np.zeros(3, np.float32),
            next_obs=np.zeros((3, 1), np.float32),
            terminated=np.zeros(3, bool),
            steps=np.ones(3, np.int32),
        )


def test_episode_result_records_why_the_episode_ended():
    assert EpisodeResult(ret=12.0, length=4, terminated=True).terminated is True
    assert EpisodeResult(ret=12.0, length=4, terminated=False).terminated is False
