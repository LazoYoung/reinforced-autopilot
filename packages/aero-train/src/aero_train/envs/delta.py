"""Discrete per-axis increments over a continuous control Box."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DeltaAction(gym.ActionWrapper):
    """Turn an n-axis control Box into Discrete(3 ** n) of per-axis increments.

    Each action index decodes to n base-3 digits, least significant digit
    first; digit d in {0, 1, 2} moves its axis by {-delta, 0, +delta}. The
    accumulated command is clipped to the inner action space and passed
    through.

    Deltas rather than absolute bins for two reasons. Exploration becomes
    temporally correlated by construction: an aircraft is a low-pass filter, so
    independent uniform absolute commands at 10 Hz average out to nothing and
    the aircraft barely moves, while a random walk over commands produces the
    sustained deflections that actually explore the state space. And range and
    precision decouple -- full deflection stays reachable, and delta only sets
    how finely it is approached, where absolute binning trades one against the
    other.

    The observation is deliberately not augmented with the accumulator. The
    scenario already observes actual control positions, which carry the
    command plus a constant trim offset and converge to the command within one
    agent step, so the position determines the command and the problem stays
    Markov. See docs/adr/0003.
    """

    def __init__(self, env: gym.Env, delta: float = 0.1) -> None:
        super().__init__(env)
        space = env.action_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            raise TypeError(
                f"DeltaAction needs a 1-D Box action space, got {space}"
            )
        if not 0.0 < delta <= 1.0:
            raise ValueError(f"delta must be in (0, 1], got {delta}")
        self.delta = float(delta)
        self._n_axes = int(space.shape[0])
        self._low = np.asarray(space.low, np.float32)
        self._high = np.asarray(space.high, np.float32)
        self.action_space = gym.spaces.Discrete(3**self._n_axes)
        self._command = np.zeros(self._n_axes, np.float32)

    @property
    def null_action(self) -> int:
        """The index whose every digit is 1, i.e. every axis unchanged."""
        return (int(self.action_space.n) - 1) // 2

    def _deltas(self, index: int) -> np.ndarray:
        out = np.empty(self._n_axes, np.float32)
        for axis in range(self._n_axes):
            out[axis] = (index % 3 - 1) * self.delta
            index //= 3
        return out

    def action(self, action):
        self._command = np.clip(
            self._command + self._deltas(int(action)), self._low, self._high
        )
        # A copy, not the live array: the inner env keeps whatever it is given,
        # and handing out the accumulator would let the next step mutate it.
        return self._command.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Seeded from the aircraft's post-reset control state rather than zero.
        # PROFILES["c172p"].startup commands 0.7 throttle; an accumulator
        # starting at zero would chop it to idle on the very first step.
        self._command = np.clip(
            np.asarray(self.get_wrapper_attr("initial_action")(), np.float32),
            self._low,
            self._high,
        )
        return obs, info
