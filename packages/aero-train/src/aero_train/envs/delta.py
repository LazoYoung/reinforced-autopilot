"""Discrete per-axis increments over a continuous control Box."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ..tensors import ActionIndex, Command


class DeltaAction(gym.ActionWrapper):
    """Turn an n-axis control Box into Discrete(3 ** n) of per-axis increments.

    Each action index decodes to n base-3 digits, least significant digit
    first; digit d in {0, 1, 2} moves its axis by {-step, 0, +step}, where the
    step is that axis's delta times the width of its declared range. The
    accumulated command is clipped to the inner action space and passed
    through.

    delta is a fraction rather than an absolute amount so that one number means
    the same thing on every axis: [-1, 1] is twice as wide as [0, 1], and an
    absolute delta would move it half as far in proportion. Give one value to
    apply everywhere, or one value per axis.

    Deltas rather than absolute bins for two reasons. Exploration becomes
    temporally correlated by construction: an aircraft is a low-pass filter, so
    independent uniform absolute commands at 10 Hz average out to nothing and
    the aircraft barely moves, while a random walk over commands produces the
    sustained deflections that actually explore the state space. And range and
    precision decouple: full deflection stays reachable, and delta only sets
    how finely it is approached, where absolute binning trades one against the
    other.

    The observation is deliberately not augmented with the accumulator. The
    scenario already observes actual control positions, which carry the
    command plus a constant trim offset and converge to the command within one
    agent step, so the position determines the command and the problem stays
    Markov. See docs/adr/0003.
    """

    def __init__(self, env: gym.Env, delta: float | Sequence[float] = 0.1) -> None:
        super().__init__(env)
        space = env.action_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            raise TypeError(
                f"DeltaAction needs a 1-D Box action space, got {space}"
            )
        self._n_axes = int(space.shape[0])
        self._low = np.asarray(space.low, np.float32)
        self._high = np.asarray(space.high, np.float32)
        fractions = np.asarray(delta, np.float64)
        if fractions.ndim == 0:
            fractions = np.full(self._n_axes, float(fractions))
        if fractions.shape != (self._n_axes,):
            raise ValueError(
                f"delta needs one value per axis, {self._n_axes} here, got shape "
                f"{fractions.shape}"
            )
        if not np.all((fractions > 0.0) & (fractions <= 1.0)):
            raise ValueError(f"every delta must be in (0, 1], got {fractions.tolist()}")
        self.delta = tuple(float(f) for f in fractions)
        self._step = (fractions * (self._high - self._low)).astype(np.float32)
        self.action_space = gym.spaces.Discrete(3**self._n_axes)
        self._command = np.zeros(self._n_axes, np.float32)

    @property
    def null_action(self) -> int:
        """The index whose every digit is 1, i.e. every axis unchanged."""
        return (int(self.action_space.n) - 1) // 2

    @property
    def command(self) -> np.ndarray:
        """The accumulated command the next action steps from, as a copy.

        What a policy file's prev.* inputs carry while this wrapper is flying.
        """
        return self._command.copy()

    @property
    def step_sizes(self) -> np.ndarray:
        """One step on each axis in that axis's units: its delta times its span."""
        return self._step.copy()

    def to_torch(self) -> nn.Module:
        """The step rule as a torch module, for export: (ActionIndex, Command) to Command.

        Built from the arrays `action` uses, so the two cannot drift apart.
        """
        return _Step(self._low, self._high, self._step)

    def _deltas(self, index: int) -> np.ndarray:
        out = np.empty(self._n_axes, np.float32)
        for axis in range(self._n_axes):
            out[axis] = (index % 3 - 1) * self._step[axis]
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
        # Seeded from post-reset control state, not zero: c172p startup commands
        # 0.7 throttle, which an accumulator at zero chops to idle on step one.
        self._command = np.clip(
            np.asarray(self.get_wrapper_attr("initial_action")(), np.float32),
            self._low,
            self._high,
        )
        return obs, info


class _Step(nn.Module):
    """DeltaAction.action over a batch: decode each index, step, clip."""

    def __init__(self, low: np.ndarray, high: np.ndarray, step: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("low", torch.as_tensor(low.copy()))
        self.register_buffer("high", torch.as_tensor(high.copy()))
        self.register_buffer("step", torch.as_tensor(step.copy()))
        # Least significant digit first, as DeltaAction._deltas reads them.
        self.register_buffer("radix", 3 ** torch.arange(len(step)))

    def forward(self, action: ActionIndex, command: Command) -> Command:
        digits = (action.unsqueeze(1) // self.radix) % 3 - 1
        return torch.clamp(command + digits.to(command.dtype) * self.step, self.low, self.high)
