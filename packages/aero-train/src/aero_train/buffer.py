"""Uniform experience replay over flat transitions.

Segments arrive from the rollout engine and are flattened on insert by
to_transitions -- the one place in the codebase where that reshape happens.
Prioritized replay is a Phase 2+ concern.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .primitives import Transitions


class ReplayBuffer:
    """Fixed-capacity ring buffer, sampled uniformly with replacement."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...] = (),
        act_dtype: Any = np.int64,
        rng: np.random.Generator | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._obs = np.zeros((capacity, *obs_shape), np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), np.float32)
        self._actions = np.zeros((capacity, *act_shape), act_dtype)
        self._returns = np.zeros(capacity, np.float32)
        self._terminated = np.zeros(capacity, bool)
        self._steps = np.ones(capacity, np.int32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, tr: Transitions) -> None:
        n = len(tr)
        if n == 0:
            return
        if n > self.capacity:
            # keep the most recent; the older rows would be overwritten anyway
            keep = slice(n - self.capacity, n)
            tr = Transitions(
                obs=tr.obs[keep],
                actions=tr.actions[keep],
                returns=tr.returns[keep],
                next_obs=tr.next_obs[keep],
                terminated=tr.terminated[keep],
                steps=tr.steps[keep],
            )
            n = self.capacity
        idx = (self._pos + np.arange(n)) % self.capacity
        # Copy (not alias) into the preallocated arrays: fancy indexing with an
        # ndarray index already copies, so the caller's Transitions stays
        # independent of what the buffer stores.
        self._obs[idx] = tr.obs
        self._next_obs[idx] = tr.next_obs
        self._actions[idx] = tr.actions
        self._returns[idx] = tr.returns
        self._terminated[idx] = tr.terminated
        self._steps[idx] = tr.steps
        self._pos = int((self._pos + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> Transitions:
        if batch_size > self._size:
            raise ValueError(
                f"batch_size {batch_size} exceeds the {self._size} transitions stored"
            )
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Transitions(
            obs=self._obs[idx],
            actions=self._actions[idx],
            returns=self._returns[idx],
            next_obs=self._next_obs[idx],
            terminated=self._terminated[idx],
            steps=self._steps[idx],
        )

    def state_dict(self) -> dict[str, Any]:
        # Copy every array out: the caller may go on to add() to this buffer,
        # or to another buffer restored from this same dict, and neither of
        # those should be able to mutate what's meant to be a frozen snapshot.
        return {
            "obs": self._obs.copy(),
            "next_obs": self._next_obs.copy(),
            "actions": self._actions.copy(),
            "returns": self._returns.copy(),
            "terminated": self._terminated.copy(),
            "steps": self._steps.copy(),
            "pos": self._pos,
            "size": self._size,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        # np.asarray is a no-op (returns the same object) when the input is
        # already an ndarray, so this must copy explicitly -- otherwise this
        # buffer and the dict's source end up sharing storage, and add() on
        # either one silently corrupts the other.
        self._obs = np.array(sd["obs"], copy=True)
        self._next_obs = np.array(sd["next_obs"], copy=True)
        self._actions = np.array(sd["actions"], copy=True)
        self._returns = np.array(sd["returns"], copy=True)
        self._terminated = np.array(sd["terminated"], copy=True)
        self._steps = np.array(sd["steps"], copy=True)
        self._pos = int(sd["pos"])
        self._size = int(sd["size"])
        self.capacity = len(self._obs)
