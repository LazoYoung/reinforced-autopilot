from __future__ import annotations

import numpy as np

from aero_train.primitives import ArgmaxSelector, EpsilonGreedySelector

SCORES = np.array([[0.0, 1.0, 0.0], [5.0, 0.0, 0.0]], np.float32)


def test_argmax_picks_the_highest_scoring_action_per_row():
    assert ArgmaxSelector()(SCORES).tolist() == [1, 0]


def test_argmax_returns_one_action_per_batch_row():
    assert ArgmaxSelector()(SCORES).shape == (2,)


def test_epsilon_zero_is_greedy():
    sel = EpsilonGreedySelector(epsilon=0.0, rng=np.random.default_rng(0))
    assert sel(SCORES).tolist() == [1, 0]


def test_epsilon_one_ignores_the_scores():
    sel = EpsilonGreedySelector(epsilon=1.0, rng=np.random.default_rng(0))
    counts = np.zeros(3, int)
    for _ in range(500):
        for a in sel(SCORES):
            counts[a] += 1
    # every action must occur; a greedy selector would leave two of them at zero
    assert (counts > 0).all()


def test_epsilon_greedy_actions_are_always_in_range():
    sel = EpsilonGreedySelector(epsilon=0.5, rng=np.random.default_rng(1))
    for _ in range(100):
        actions = sel(SCORES)
        assert actions.shape == (2,)
        assert ((actions >= 0) & (actions < 3)).all()


def test_epsilon_greedy_is_reproducible_from_its_generator():
    a = EpsilonGreedySelector(epsilon=0.5, rng=np.random.default_rng(7))
    b = EpsilonGreedySelector(epsilon=0.5, rng=np.random.default_rng(7))
    for _ in range(20):
        assert a(SCORES).tolist() == b(SCORES).tolist()


def test_epsilon_is_mutable_so_a_schedule_can_drive_it():
    sel = EpsilonGreedySelector(epsilon=1.0, rng=np.random.default_rng(0))
    sel.epsilon = 0.0
    assert sel(SCORES).tolist() == [1, 0]
