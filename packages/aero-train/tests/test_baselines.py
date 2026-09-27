from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from aero_train.algo import build_algorithm
from aero_train.config import config_from_dict


def _cfg(algo: dict):
    return config_from_dict(
        {"run_name": "b", "env": {"id": "CartPole-v1"}, "algo": algo, "total_steps": 100}
    )


def _spaces(n: int):
    return gym.spaces.Box(-1.0, 1.0, (4,), np.float32), gym.spaces.Discrete(n)


def test_constant_defaults_to_the_middle_action():
    """For any Discrete that is the middle index, and for DeltaAction the middle
    index is the all-zero-delta action by construction, which is how the
    baseline stays a DeltaAction fact without aero_train.algo knowing about it."""
    algo = build_algorithm(_cfg({"type": "constant"}), *_spaces(81))
    assert algo.action == 40


def test_the_middle_index_really_is_delta_actions_null_action():
    from aero_train.envs.delta import DeltaAction

    class _Box(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

    env = DeltaAction(_Box())
    assert env.null_action == (int(env.action_space.n) - 1) // 2


def test_an_out_of_range_constant_action_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        build_algorithm(_cfg({"type": "constant", "action": 999}), *_spaces(81))


def test_a_baseline_never_learns_but_still_counts_frames():
    from aero_train.primitives import Segment

    algo = build_algorithm(_cfg({"type": "random"}), *_spaces(81))
    segment = Segment(
        obs=np.zeros((3, 2, 4), np.float32),
        actions=np.zeros((2, 2), np.int64),
        rewards=np.zeros((2, 2), np.float32),
        terminated=np.zeros((2, 2), bool),
        truncated=np.zeros((2, 2), bool),
        valid=np.ones((2, 2), bool),
    )
    algo.observe(segment)
    assert algo.update() is None
    assert algo.global_step == 4


def test_a_baseline_round_trips_through_its_state_dict():
    """restored is freshly built, so its global_step is already 0 before
    load_state_dict runs: asserting equality against an algo that never
    observed anything would pass on a load_state_dict that does nothing.
    Advancing algo's frames first, so the round trip has something to lose,
    is what makes this a real check of load_state_dict rather than 0 == 0."""
    from aero_train.primitives import Segment

    algo = build_algorithm(_cfg({"type": "random"}), *_spaces(81))
    algo.observe(
        Segment(
            obs=np.zeros((3, 2, 4), np.float32),
            actions=np.zeros((2, 2), np.int64),
            rewards=np.zeros((2, 2), np.float32),
            terminated=np.zeros((2, 2), bool),
            truncated=np.zeros((2, 2), bool),
            valid=np.ones((2, 2), bool),
        )
    )
    restored = build_algorithm(_cfg({"type": "random"}), *_spaces(81))
    restored.load_state_dict(algo.state_dict())
    assert restored.global_step == algo.global_step != 0


def test_a_baseline_rejects_a_continuous_action_space():
    with pytest.raises(ValueError, match="discrete"):
        build_algorithm(
            _cfg({"type": "random"}),
            gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
        )


@pytest.mark.parametrize("algo", [{"type": "random"}, {"type": "constant"}])
def test_a_baseline_cannot_be_exported(algo):
    baseline = build_algorithm(_cfg(algo), *_spaces(81))
    with pytest.raises(NotImplementedError, match=f"{type(baseline).__name__} cannot be exported"):
        baseline.policy_module()
