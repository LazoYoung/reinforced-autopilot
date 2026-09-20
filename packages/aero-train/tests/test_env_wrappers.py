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


from aero_train.envs.delta import DeltaAction


class RecordingBoxEnv(BoxEnv):
    """Keeps the last action so the accumulator can be read back."""

    def __init__(self, low, high, initial=None) -> None:
        super().__init__([-1.0] * 4, [1.0] * 4)
        self.action_space = gym.spaces.Box(
            np.array(low, np.float32), np.array(high, np.float32), dtype=np.float32
        )
        self.last_action = None
        self._initial = initial

    def initial_action(self):
        if self._initial is None:
            return np.zeros(self.action_space.shape, np.float32)
        return np.array(self._initial, np.float32)

    def step(self, action):
        self.last_action = np.asarray(action, np.float32).copy()
        return super().step(action)


def _four_axis_env(initial=None):
    return RecordingBoxEnv([-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0], initial)


def test_the_action_space_is_three_to_the_number_of_axes():
    env = DeltaAction(_four_axis_env())
    assert env.action_space == gym.spaces.Discrete(81)


def test_the_null_action_moves_nothing():
    env = DeltaAction(_four_axis_env(initial=[0.1, -0.2, 0.3, 0.7]))
    env.reset()
    env.step(env.null_action)
    assert env.unwrapped.last_action.tolist() == pytest.approx([0.1, -0.2, 0.3, 0.7])
    assert env.null_action == 40


def test_each_axis_gets_its_own_base_three_digit():
    env = DeltaAction(_four_axis_env(), delta=0.1)
    env.reset()
    # digit 0 is axis 0: index 41 == null (40) + 1 -> axis 0 moves one step up
    env.step(41)
    assert env.unwrapped.last_action.tolist() == pytest.approx([0.2, 0.0, 0.0, 0.0])
    # index 40 + 3 -> axis 1 moves one step up, axis 0 unchanged
    env.step(43)
    assert env.unwrapped.last_action.tolist() == pytest.approx([0.2, 0.2, 0.0, 0.0])


def test_the_accumulator_integrates_and_clips_at_the_axis_bounds():
    env = DeltaAction(_four_axis_env(), delta=0.5)
    env.reset()
    for _ in range(5):
        env.step(41)  # axis 0 up by 0.5 each time
    assert env.unwrapped.last_action[0] == pytest.approx(1.0)


def test_reset_seeds_the_accumulator_from_the_aircrafts_startup_state():
    """Otherwise the first step chops the throttle to idle, undoing the
    flyable state the aircraft profile establishes."""
    env = DeltaAction(_four_axis_env(initial=[0.0, 0.0, 0.0, 0.7]))
    env.reset()
    env.step(env.null_action)
    assert env.unwrapped.last_action[3] == pytest.approx(0.7)


def test_reset_reseeds_between_episodes():
    env = DeltaAction(_four_axis_env(initial=[0.0, 0.0, 0.0, 0.7]), delta=0.5)
    env.reset()
    for _ in range(4):
        env.step(41)
    env.reset()
    env.step(env.null_action)
    assert env.unwrapped.last_action.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.7])


def test_it_reaches_initial_action_through_an_intervening_wrapper():
    env = DeltaAction(ScaleObservation(_four_axis_env(initial=[0.0, 0.0, 0.0, 0.7])))
    env.reset()
    env.step(env.null_action)
    assert env.unwrapped.last_action[3] == pytest.approx(0.7)


def test_delta_is_a_fraction_of_each_axis_declared_range():
    """[-1, 1] is twice as wide as [0, 1], so the same fraction moves it twice as
    far. An absolute delta would move both by 0.1 and fail the first entry."""
    env = DeltaAction(_four_axis_env(), delta=0.1)
    env.reset()
    env.step(40 + 1 + 27)  # axis 0 and axis 3 both up one step
    assert env.unwrapped.last_action.tolist() == pytest.approx([0.2, 0.0, 0.0, 0.1])


def test_each_axis_can_take_its_own_delta():
    env = DeltaAction(_four_axis_env(), delta=[0.015, 0.05, 0.1, 0.1])
    env.reset()
    env.step(40 + 1 + 3 + 9 + 27)  # every axis up one step
    assert env.unwrapped.last_action.tolist() == pytest.approx([0.03, 0.1, 0.2, 0.1])
    assert env.delta == pytest.approx((0.015, 0.05, 0.1, 0.1))


def test_a_delta_per_axis_must_match_the_axis_count():
    with pytest.raises(ValueError, match="one value per axis"):
        DeltaAction(_four_axis_env(), delta=[0.1, 0.1, 0.1])


def test_a_bad_delta_is_rejected():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        DeltaAction(_four_axis_env(), delta=0.0)
