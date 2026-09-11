"""The learner-side interface every algorithm implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..primitives import Agent, Segment


class Algorithm(ABC):
    """Owns the networks, the optimizers and the loss.

    observe() and update() are separate because their cadence differs by family.
    Off-policy: observe inserts into a replay buffer, update samples from it and
    may run several gradient steps, returning None while the buffer fills.
    On-policy: observe stashes the segment, update consumes and discards it.
    """

    @property
    @abstractmethod
    def global_step(self) -> int:
        """Monotonic count of gradient updates applied.

        The x-axis of everything logged, and the value a checkpoint resumes
        from. Not the number of agent steps taken; that is `frames`.
        """

    @abstractmethod
    def make_agent(self) -> Agent:
        """Build a worker-side agent: a weight replica plus a selector."""

    @abstractmethod
    def actor_state_dict(self) -> dict[str, np.ndarray]:
        """Only what a worker needs for inference.

        The policy network and the exploration parameters (under the
        `selector.` prefix), never the critic, the target networks or the
        optimizer state.
        """

    @abstractmethod
    def observe(self, segment: Segment) -> None:
        """Take in one iteration's experience."""

    @abstractmethod
    def update(self) -> dict[str, float] | None:
        """Perform one complete parameter update, target sync included.

        Returns metrics for the tracker, or None if it is not yet time to learn
        (an off-policy buffer still warming up). Returning the metrics dict is
        what keeps the Trainer free of algorithm-specific knowledge.
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume: all networks, optimizers, counters."""

    @abstractmethod
    def load_state_dict(self, sd: dict[str, Any]) -> None: ...
