"""DQN: the worker-side agent and the learner."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from ..buffer import ReplayBuffer
from ..config import DQNConfig
from ..experience import to_transitions
from ..nets import TargetNet, mlp
from ..primitives import (
    SELECTOR_PREFIX,
    ActionSelector,
    Agent,
    EpsilonGreedySelector,
    Segment,
    Transitions,
)
from .base import Algorithm


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


GRAD_CLIP = 10.0


class DQN(Algorithm):
    """Deep Q-learning with a target network, n-step returns and uniform replay."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        cfg: DQNConfig,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self._seed = int(seed)
        torch.manual_seed(seed)
        self.net = self._build_net().to(self.device)
        self.target = TargetNet(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(
            cfg.buffer_size, self.obs_shape, rng=np.random.default_rng(seed)
        )
        self._frames = 0
        self._global_step = 0

    def _build_net(self) -> nn.Module:
        """The one place the Q-network architecture is named.

        Both __init__ and make_agent go through it, so a variant that changes
        only the architecture (Dueling DQN's value/advantage heads, say) is
        a subclass overriding this method and nothing else.
        """
        return mlp(int(np.prod(self.obs_shape)), self.cfg.hidden_sizes, self.n_actions)

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def frames(self) -> int:
        """Agent steps observed. Drives the epsilon schedule."""
        return self._frames

    @property
    def epsilon(self) -> float:
        """Linear decay in frames, floored at epsilon_final."""
        progress = min(self._frames / self.cfg.epsilon_decay_frames, 1.0)
        return self.cfg.epsilon_start + progress * (
            self.cfg.epsilon_final - self.cfg.epsilon_start
        )

    def make_agent(self) -> Agent:
        replica = self._build_net()
        replica.load_state_dict(self.net.state_dict())
        selector: ActionSelector = EpsilonGreedySelector(
            self.epsilon, np.random.default_rng(self._seed + 1)
        )
        return DQNAgent(replica, selector, str(self.device))

    def actor_state_dict(self) -> dict[str, np.ndarray]:
        sd: dict[str, np.ndarray] = {
            k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()
        }
        # a 0-d ndarray, not np.float32: the actor_state_dict contract is
        # dict[str, np.ndarray], and a numpy scalar is not an ndarray
        sd[SELECTOR_PREFIX + "epsilon"] = np.asarray(self.epsilon, np.float32)
        return sd

    def observe(self, segment: Segment) -> None:
        self.buffer.add(to_transitions(segment, self.cfg.gamma, self.cfg.n_step))
        self._frames += int(segment.valid.sum())

    def update(self) -> dict[str, float] | None:
        if len(self.buffer) < self.cfg.warmup_frames:
            return None
        losses, q_means, grad_norms = [], [], []
        for _ in range(self.cfg.updates_per_iter):
            batch = self.buffer.sample(self.cfg.batch_size)
            q_taken = self._q_taken(batch)
            target = self._bellman_target(batch)
            loss = nn.functional.smooth_l1_loss(q_taken, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
            self.optimizer.step()
            self._global_step += 1
            if self._global_step % self.cfg.target_sync_interval == 0:
                self.target.sync()
            # .item(), not float(): both still require grad here, and float()
            # warns on such a tensor though backward() already freed the graph.
            losses.append(loss.item())
            q_means.append(q_taken.mean().item())
            grad_norms.append(float(grad_norm))
        return {
            "loss": float(np.mean(losses)),
            "q_mean": float(np.mean(q_means)),
            "grad_norm": float(np.mean(grad_norms)),
            "epsilon": float(self.epsilon),
            "buffer_size": float(len(self.buffer)),
        }

    def _q_taken(self, batch: Transitions) -> torch.Tensor:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        return self.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    def _bellman_target(self, batch: Transitions) -> torch.Tensor:
        """R_n + gamma**steps * max_a Q_target(s') , zeroed at real terminals.

        `steps` rather than a fixed n because accumulation stops early at an
        episode end or a segment edge. `terminated` rather than any notion of
        `done`: a time-limited transition arrives here with terminated=False and
        next_obs set to the true final state, so it bootstraps, which is the
        correct treatment and the one PTAN's conflated `done` gets wrong.
        """
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=self.device)
        steps = torch.as_tensor(batch.steps, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_q = self.target.target_model(next_obs).max(dim=1).values
            next_q = next_q.masked_fill(terminated, 0.0)
            return returns + (self.cfg.gamma**steps) * next_q

    def state_dict(self) -> dict[str, Any]:
        """A frozen snapshot, deepcopied because nn.Module.state_dict() detaches
        each tensor without cloning it and so aliases live parameter storage."""
        return {
            "net": copy.deepcopy(self.net.state_dict()),
            "target": copy.deepcopy(self.target.target_model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "frames": self._frames,
            "global_step": self._global_step,
            # without the sampler state a resumed run draws a different replay
            # sequence from the same buffer and the curve diverges
            "buffer_rng": copy.deepcopy(self.buffer.rng.bit_generator.state),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.net.load_state_dict(sd["net"])
        self.target.target_model.load_state_dict(sd["target"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self._frames = int(sd["frames"])
        self._global_step = int(sd["global_step"])
        self.buffer.rng.bit_generator.state = sd["buffer_rng"]
