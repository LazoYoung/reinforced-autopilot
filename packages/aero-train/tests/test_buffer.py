from __future__ import annotations

import numpy as np
import pytest

from aero_train.buffer import ReplayBuffer
from aero_train.primitives import Transitions


def transitions(values):
    n = len(values)
    arr = np.asarray(values, np.float32).reshape(n, 1)
    return Transitions(
        obs=arr,
        actions=np.zeros(n, np.int64),
        returns=np.asarray(values, np.float32),
        next_obs=arr + 1.0,
        terminated=np.zeros(n, bool),
        steps=np.ones(n, np.int32),
    )


def test_buffer_starts_empty():
    assert len(ReplayBuffer(10, (1,))) == 0


def test_add_grows_the_buffer():
    buf = ReplayBuffer(10, (1,))
    buf.add(transitions([1.0, 2.0, 3.0]))
    assert len(buf) == 3


def test_length_is_capped_at_capacity():
    buf = ReplayBuffer(4, (1,))
    buf.add(transitions([float(i) for i in range(10)]))
    assert len(buf) == 4


def test_length_is_capped_at_capacity_across_incremental_adds():
    """An oversized add() is truncated on its own, which can mask a buffer that
    fails to cap itself when the overflow arrives across several small adds."""
    buf = ReplayBuffer(4, (1,))
    for _ in range(5):
        buf.add(transitions([1.0, 2.0]))
    assert len(buf) == 4


def _enumerate_sampled_values(buf, draws=200):
    """Every distinct value the buffer can hand back, via repeated valid draws.

    sample() refuses batch_size > len(buf), so coverage has to be accumulated
    rather than taken in one oversized call. Many draws of len(buf) reach full
    coverage by the birthday paradox, deterministically under the injected rng.
    """
    sampled: set[float] = set()
    for _ in range(draws):
        sampled |= {float(v) for v in buf.sample(len(buf)).obs[:, 0]}
    return sampled


def test_wrapping_keeps_the_most_recent_transitions():
    buf = ReplayBuffer(4, (1,), rng=np.random.default_rng(0))
    buf.add(transitions([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert _enumerate_sampled_values(buf) == {2.0, 3.0, 4.0, 5.0}


def test_a_single_add_larger_than_capacity_keeps_the_tail():
    buf = ReplayBuffer(3, (1,), rng=np.random.default_rng(0))
    buf.add(transitions([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert _enumerate_sampled_values(buf) == {2.0, 3.0, 4.0}


def test_sampling_never_returns_unwritten_rows():
    buf = ReplayBuffer(100, (1,), rng=np.random.default_rng(0))
    buf.add(transitions([7.0, 8.0, 9.0]))
    assert _enumerate_sampled_values(buf) <= {7.0, 8.0, 9.0}


def test_sample_returns_the_requested_batch_shape():
    buf = ReplayBuffer(50, (1,), rng=np.random.default_rng(0))
    buf.add(transitions([float(i) for i in range(20)]))
    batch = buf.sample(8)
    assert len(batch) == 8
    assert batch.obs.shape == (8, 1)
    assert batch.actions.dtype == np.int64
    assert batch.terminated.dtype == np.bool_


def test_sampling_more_than_is_stored_raises():
    buf = ReplayBuffer(50, (1,))
    buf.add(transitions([1.0, 2.0]))
    with pytest.raises(ValueError, match="batch_size"):
        buf.sample(8)


def test_sampling_is_reproducible_from_its_generator():
    def draw(seed):
        buf = ReplayBuffer(50, (1,), rng=np.random.default_rng(seed))
        buf.add(transitions([float(i) for i in range(20)]))
        return buf.sample(8).obs[:, 0].tolist()

    assert draw(3) == draw(3)


def test_adding_an_empty_batch_is_a_no_op():
    buf = ReplayBuffer(10, (1,))
    buf.add(transitions([]))
    assert len(buf) == 0


def test_state_dict_round_trip_restores_contents_and_position():
    buf = ReplayBuffer(10, (1,), rng=np.random.default_rng(0))
    buf.add(transitions([1.0, 2.0, 3.0]))
    restored = ReplayBuffer(10, (1,), rng=np.random.default_rng(0))
    restored.load_state_dict(buf.state_dict())
    assert len(restored) == 3
    # 3 is len(restored) == len(buf); sample() refuses anything larger.
    assert restored.sample(3).obs[:, 0].tolist() == buf.sample(3).obs[:, 0].tolist()


def test_state_dict_round_trip_preserves_ring_wrap_position():
    """sample() never reads _pos, so sampling cannot tell whether it came back.

    A buffer restored mid-ring must overwrite from the same slot as the live
    one, or the next add corrupts a different still-valid transition. This
    holds only because test_state_dict_round_trip_does_not_alias proves the two
    buffers are independent; shared storage would pass it vacuously.
    """
    buf = ReplayBuffer(5, (1,))
    buf.add(transitions([1.0, 2.0, 3.0]))  # pos=3: not 0, not a capacity boundary
    restored = ReplayBuffer(5, (1,))
    restored.load_state_dict(buf.state_dict())

    buf.add(transitions([9.0]))
    restored.add(transitions([9.0]))

    assert restored.state_dict()["pos"] == buf.state_dict()["pos"]
    assert np.array_equal(restored.state_dict()["obs"], buf.state_dict()["obs"])


def test_state_dict_round_trip_does_not_alias():
    """A snapshot must freeze, and a restore must not retain the dict it was
    given, or add() on one buffer corrupts its checkpoint or a sibling
    restored from it. Checked in both directions."""
    buf = ReplayBuffer(5, (1,))
    buf.add(transitions([1.0, 2.0, 3.0]))

    sd = buf.state_dict()
    buf.add(transitions([99.0]))  # mutate the source after snapshotting
    assert 99.0 not in sd["obs"][:, 0]

    restored = ReplayBuffer(5, (1,))
    restored.load_state_dict(sd)
    restored.add(transitions([77.0]))  # mutate the restored copy
    assert 77.0 not in sd["obs"][:, 0]
    assert 77.0 not in buf.state_dict()["obs"][:, 0]
