from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from aero_train.envs.scale import ScaleObservation


class BoxEnv(gym.Env):
    """Returns whatever observation it is told to, so scaling is hand-checkable."""

    def __init__(self, low, high, value=None) -> None:
        self.observation_space = gym.spaces.Box(
            np.array(low, np.float32), np.array(high, np.float32), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.value = np.array(low if value is None else value, np.float32)

    def reset(self, *, seed=None, options=None):
        return self.value.copy(), {}

    def step(self, action):
        return self.value.copy(), 0.0, False, False, {}


def test_bounds_map_to_the_ends_of_the_unit_interval():
    env = ScaleObservation(BoxEnv([-500.0, 0.0], [500.0, 1.0], [500.0, 1.0]))
    obs, _ = env.reset()
    assert obs.tolist() == pytest.approx([1.0, 1.0])
    assert env.observation_space.low.tolist() == [-1.0, -1.0]


def test_the_low_bound_maps_to_minus_one_and_the_midpoint_to_zero():
    env = ScaleObservation(BoxEnv([-500.0, 0.0], [500.0, 1.0], [-500.0, 0.5]))
    obs, _ = env.reset()
    assert obs.tolist() == pytest.approx([-1.0, 0.0])


def test_an_unbounded_feature_is_rejected_by_index():
    with pytest.raises(ValueError, match=r"\[1\]"):
        ScaleObservation(BoxEnv([-1.0, -np.inf], [1.0, np.inf]))


def test_a_degenerate_bound_is_rejected():
    with pytest.raises(ValueError, match="low < high"):
        ScaleObservation(BoxEnv([0.0, 1.0], [0.0, 2.0]))
