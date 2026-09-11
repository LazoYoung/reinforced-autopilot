"""The vocabulary shared by every algorithm: what experience looks like, and
what turns network output into an action.

This is the PTAN decomposition (ActionSelector -> Agent -> ExperienceSource ->
ReplayBuffer) with two corrections. Gymnasium's `terminated` and `truncated`
stay distinct (PTAN predates the distinction and conflates them into `done`),
and the payload is a [T, B] segment rather than a flat transition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

SELECTOR_PREFIX = "selector."
"""Reserved key prefix inside an actor_state_dict.

Keys under it are exploration parameters (epsilon, sigma) applied to the
agent's ActionSelector rather than loaded into the network. It exists so that
RolloutEngine.sync_weights stays a single payload and a single method: the
learner owns the exploration schedule, and the current value has to reach every
worker's selector each iteration alongside the weights.
"""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One finished episode. `terminated=False` means it was truncated.

    Which of the two ended the episode is worth logging on its own: an episode
    that ends by leaving the flight envelope is a different event from one that
    hits the step limit, and a mean return alone hides which happened.
    """

    ret: float
    length: int
    terminated: bool


@dataclass(frozen=True, slots=True)
class Segment:
    """Contiguous experience with time as an explicit axis.

    obs         [T + 1, B, *obs_shape]  the extra row is the bootstrap obs
    actions     [T, B, *act_shape]
    rewards     [T, B]  float32
    terminated  [T, B]  bool  a real terminal state; do NOT bootstrap past it
    truncated   [T, B]  bool  a time limit; you MUST bootstrap past it
    valid       [T, B]  bool  False on Gymnasium's fabricated autoreset steps

    `obs` carries T + 1 entries so V(s_T) can be computed without a second
    forward pass or a stashed observation.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.rewards.ndim != 2:
            raise ValueError(f"rewards must be [T, B], got shape {self.rewards.shape}")
        T, B = self.rewards.shape
        if self.obs.shape[0] != T + 1:
            raise ValueError(
                f"obs must have T + 1 = {T + 1} timesteps for bootstrapping, "
                f"got {self.obs.shape[0]}"
            )
        if self.obs.shape[1] != B:
            raise ValueError(
                f"obs batch axis ({self.obs.shape[1]}) does not match rewards ({B})"
            )
        if self.actions.shape[:2] != (T, B):
            raise ValueError(
                f"actions must start with [T, B] = [{T}, {B}], got {self.actions.shape}"
            )
        for name in ("terminated", "truncated", "valid"):
            mask = getattr(self, name)
            if mask.shape != (T, B):
                raise ValueError(f"{name} must be [T, B] = [{T}, {B}], got {mask.shape}")
            if mask.dtype != np.bool_:
                raise ValueError(f"{name} must have dtype bool, got {mask.dtype}")
        if (self.terminated & self.truncated).any():
            raise ValueError("a step cannot be both terminated and truncated")
        if ((self.terminated | self.truncated) & ~self.valid).any():
            raise ValueError("an invalid step (autoreset) cannot end an episode")

    @property
    def T(self) -> int:
        """Segment length in agent steps."""
        return self.rewards.shape[0]

    @property
    def B(self) -> int:
        """Number of environments stepped in lockstep."""
        return self.rewards.shape[1]


@dataclass(frozen=True, slots=True)
class Transitions:
    """A flat batch of n-step transitions, as stored in the replay buffer.

    returns     discounted sum of the rewards actually collected
    steps       how many steps `returns` covers; fewer than n at a segment
                edge or an episode end, so the bootstrap discount is gamma**steps
    terminated  True only if a real terminal state was reached within those steps

    There is no `truncated` field on purpose. A truncated transition is stored
    with terminated=False and next_obs set to the true final observation, so the
    value function bootstraps straight through the time limit, which is correct.
    """

    obs: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    steps: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.obs)
        for name in ("actions", "returns", "next_obs", "terminated", "steps"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"all Transitions fields must have the same length; "
                    f"obs has {n}, {name} has {len(getattr(self, name))}"
                )

    def __len__(self) -> int:
        return len(self.obs)


class ActionSelector(ABC):
    """Turns network output into an action. Holds any exploration state.

    Separated from the network so that exploration strategy is configuration
    rather than code: the same Q-network is greedy under ArgmaxSelector and
    exploring under EpsilonGreedySelector.
    """

    @abstractmethod
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        """[B, n_actions] scores -> [B] (or [B, act_dim]) actions."""


class Agent(ABC):
    """Network plus action selector, running inside a rollout worker.

    Always called batched. Holds a *replica* of the policy weights, refreshed
    each iteration by sync_weights.
    """

    @abstractmethod
    def __call__(
        self, obs: np.ndarray, agent_state: Any | None = None
    ) -> tuple[np.ndarray, Any | None]:
        """[B, *obs_shape] -> ([B, *act_shape], next agent_state).

        `agent_state` carries recurrent hidden state. It is None for every
        algorithm in Phase 1-2; threading it now means adding an LSTM policy
        later does not require changing every call site.
        """

    @abstractmethod
    def sync_weights(self, actor_state_dict: dict[str, np.ndarray]) -> None:
        """Load an actor_state_dict produced by Algorithm.actor_state_dict().

        Keys prefixed `selector.` are exploration parameters (e.g. epsilon) and
        are applied to the selector rather than the network. This convention is
        what keeps RolloutEngine.sync_weights to a single payload.
        """


class ArgmaxSelector(ActionSelector):
    """Greedy. Used for evaluation and as the deterministic baseline."""

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores).argmax(axis=-1)


class EpsilonGreedySelector(ActionSelector):
    """Greedy with probability 1 - epsilon, uniform random otherwise.

    `epsilon` is a plain mutable attribute rather than a constructor-fixed value
    because the learner owns the schedule and pushes the current value to each
    worker every iteration, under the `selector.epsilon` key of the
    actor_state_dict.
    """

    def __init__(
        self, epsilon: float = 1.0, rng: np.random.Generator | None = None
    ) -> None:
        self.epsilon = float(epsilon)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        batch, n_actions = scores.shape
        actions = scores.argmax(axis=-1)
        explore = self.rng.random(batch) < self.epsilon
        n_explore = int(explore.sum())
        if n_explore:
            actions[explore] = self.rng.integers(0, n_actions, size=n_explore)
        return actions


class _WeightlessAgent(Agent):
    """Shared base for agents with no network behind them."""

    def sync_weights(self, actor_state_dict: dict[str, np.ndarray]) -> None:
        """Accepts and discards the payload. It arrives at all only because
        evaluate_policy and RolloutEngine speak one protocol to every
        algorithm, and there is genuinely no state here to put it in."""
        return


class RandomAgent(_WeightlessAgent):
    """Uniform over a Discrete action space. The exploration floor."""

    def __init__(self, n_actions: int, rng: np.random.Generator | None = None) -> None:
        self.n_actions = int(n_actions)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self, obs: np.ndarray, agent_state: Any | None = None
    ) -> tuple[np.ndarray, Any | None]:
        actions = self.rng.integers(0, self.n_actions, size=len(obs))
        return actions.astype(np.int64), agent_state


class ConstantAgent(_WeightlessAgent):
    """The same action every step. Under DeltaAction's null action, that is a
    do-nothing pilot holding whatever the aircraft started trimmed for."""

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def __call__(
        self, obs: np.ndarray, agent_state: Any | None = None
    ) -> tuple[np.ndarray, Any | None]:
        return np.full(len(obs), self.action, np.int64), agent_state
