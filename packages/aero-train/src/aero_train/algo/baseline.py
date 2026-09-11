"""Weightless baselines, run through the same Trainer as a learning algorithm.

They are algorithms rather than a CLI subcommand so that a baseline run lands
in the same MLflow experiment, on the same metrics, as the run it is compared
against, which is the whole thing a comparison needs. Nothing else about the
stack has to know they are special.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..primitives import Agent, ConstantAgent, RandomAgent, Segment
from .base import Algorithm


class _Baseline(Algorithm):
    """No parameters, no gradients, no learning."""

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.n_actions = int(n_actions)
        self._seed = int(seed)
        self._frames = 0

    @property
    def global_step(self) -> int:
        """Frames, not gradient updates: a baseline takes none.

        A deliberate departure from the Algorithm contract, because global_step
        is the x-axis everything is logged against: returning a constant 0
        would collapse a whole baseline run to a single point in MLflow and
        make it uncomparable with the DQN run it exists to sit beside.
        """
        return self._frames

    def make_agent(self) -> Agent:
        raise NotImplementedError

    def actor_state_dict(self) -> dict[str, np.ndarray]:
        return {}

    def observe(self, segment: Segment) -> None:
        self._frames += int(segment.valid.sum())

    def update(self) -> dict[str, float] | None:
        # None is the "not time to learn" cadence Trainer already handles for
        # DQN's replay warmup. For a baseline it is never time.
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"frames": self._frames}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._frames = int(sd["frames"])


class Random(_Baseline):
    def make_agent(self) -> Agent:
        """A fresh agent on a fixed derived seed, per the Agent contract.

        +1 decorrelates this action stream from the other consumers seeded off
        the same run config. Because evaluate_policy calls this fresh each time
        and eval fixes its per-episode seeds, every periodic evaluation replays
        one identical trajectory: a flat eval/return_mean curve here is by
        construction. Training rollouts build the agent once and are unaffected.
        """
        return RandomAgent(self.n_actions, np.random.default_rng(self._seed + 1))


class Constant(_Baseline):
    def __init__(self, n_actions: int, action: int, seed: int = 0) -> None:
        super().__init__(n_actions, seed)
        self.action = int(action)

    def make_agent(self) -> Agent:
        return ConstantAgent(self.action)
