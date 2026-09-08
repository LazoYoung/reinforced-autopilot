"""Weightless baselines, run through the same Trainer as a learning algorithm.

They are algorithms rather than a CLI subcommand so that a baseline run lands
in the same MLflow experiment, on the same metrics, as the run it is compared
against -- which is the whole thing a comparison needs. Nothing else about the
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
        """Frames, not gradient updates -- a baseline takes none.

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
        # +1, not cfg.seed itself: decorrelates this action stream from the
        # other consumers seeded off the same run config (env resets, the
        # rollout engine) so the baseline's randomness isn't a shadow of
        # theirs.
        #
        # evaluate_policy calls make_agent() fresh each time it evaluates, so
        # this reseeds from the same fixed value on every call. Combined with
        # eval's fixed per-episode seeds, every periodic evaluation of this
        # baseline replays an identical trajectory -- its eval/return_mean
        # curve in MLflow is flat by construction, not a sign anything is
        # broken. Training-rollout metrics are unaffected: there the agent is
        # built once and its RNG advances normally across the whole run.
        return RandomAgent(self.n_actions, np.random.default_rng(self._seed + 1))


class Constant(_Baseline):
    def __init__(self, n_actions: int, action: int, seed: int = 0) -> None:
        super().__init__(n_actions, seed)
        self.action = int(action)

    def make_agent(self) -> Agent:
        return ConstantAgent(self.action)
