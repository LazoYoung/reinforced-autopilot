"""Network builders shared by every algorithm.

The policies here are small -- an 11-observation, 4-action flight task with a
2x64 MLP is roughly 5k parameters (~20 KB in fp32). That number is load-bearing
for the whole distribution design: at 20 KB a weight broadcast costs
microseconds, which is why on-policy algorithms in this project can be exactly
on-policy rather than approximately so.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from torch import nn


def mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    """Fully-connected ReLU stack with a *linear* output layer.

    No output activation: Q-values and value estimates are unbounded, and a
    final ReLU would silently clamp every negative one to zero -- which is most
    of them under steady_flight's negative-error reward.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for size in hidden_sizes:
        layers += [nn.Linear(prev, size), nn.ReLU()]
        prev = size
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TargetNet:
    """A frozen copy of a network, synced on demand.

    Q-learning bootstraps from its own estimates; using the live network for
    both sides of the Bellman target makes it chase a moving goal. The copy
    holds the target fixed between syncs.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.target_model = copy.deepcopy(model)
        for param in self.target_model.parameters():
            param.requires_grad_(False)

    def sync(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
