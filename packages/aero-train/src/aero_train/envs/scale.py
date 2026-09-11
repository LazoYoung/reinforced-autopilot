"""Affine observation scaling onto [-1, 1]."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ScaleObservation(gym.ObservationWrapper):
    """Map a bounded Box observation affinely onto [-1, 1].

    Stateless on purpose. Running mean/std normalization would adapt to
    whatever the policy currently visits, but it introduces non-stationary
    statistics that have to be checkpointed, and getting that wrong makes a
    resumed run differ from an uninterrupted one. Here the transform is a pure
    function of the space, so resume is exact by construction.

    This only works because the scenario's features carry bounds that mean
    something: the raw canonical bound on altitude is [-1500, 85000], against
    which a level-flight task is a rounding error.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                f"ScaleObservation needs a Box observation space, got "
                f"{type(space).__name__}"
            )
        low = np.asarray(space.low, np.float64)
        high = np.asarray(space.high, np.float64)
        unbounded = np.flatnonzero(~(np.isfinite(low) & np.isfinite(high)))
        if unbounded.size:
            raise ValueError(
                f"ScaleObservation needs finite bounds; indices "
                f"{unbounded.tolist()} are infinite. A feature with no "
                "meaningful range cannot be scaled; bound it in the scenario."
            )
        degenerate = np.flatnonzero(high <= low)
        if degenerate.size:
            raise ValueError(
                f"ScaleObservation needs low < high; indices "
                f"{degenerate.tolist()} do not satisfy it"
            )
        self._low = low.astype(np.float32)
        self._span = (high - low).astype(np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, space.shape, np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        scaled = 2.0 * (np.asarray(observation, np.float32) - self._low) / self._span - 1.0
        return scaled.astype(np.float32)
