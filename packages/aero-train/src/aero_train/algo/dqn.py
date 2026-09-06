"""DQN: the worker-side agent here, the learner in a later task."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ..primitives import SELECTOR_PREFIX, ActionSelector, Agent


class DQNAgent(Agent):
    """A Q-network replica plus a selector, running inside a rollout worker.

    Inference happens here, next to the environments, rather than centrally in
    the driver: one round trip across a Ray actor boundary costs 200 us - 1 ms,
    comparable to a whole JSBSim agent step, so centralizing inference would put
    a Ray hop in the inner loop of every environment step.
    """

    def __init__(
        self, net: nn.Module, selector: ActionSelector, device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.net.eval()
        self.selector = selector

    def __call__(
        self, obs: np.ndarray, agent_state: Any | None = None
    ) -> tuple[np.ndarray, Any | None]:
        with torch.no_grad():
            batch = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
            q_values = self.net(batch).cpu().numpy()
        return np.asarray(self.selector(q_values), dtype=np.int64), agent_state

    def sync_weights(self, actor_state_dict: dict[str, np.ndarray]) -> None:
        weights: dict[str, torch.Tensor] = {}
        for key, value in actor_state_dict.items():
            if key.startswith(SELECTOR_PREFIX):
                attr = key[len(SELECTOR_PREFIX) :]
                if not hasattr(self.selector, attr):
                    raise AttributeError(
                        f"{type(self.selector).__name__} has no exploration parameter "
                        f"{attr!r} (from actor_state_dict key {key!r})"
                    )
                setattr(self.selector, attr, float(value))
            else:
                weights[key] = torch.as_tensor(np.asarray(value), device=self.device)
        self.net.load_state_dict(weights)
