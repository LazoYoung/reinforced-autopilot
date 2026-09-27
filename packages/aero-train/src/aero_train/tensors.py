"""Named tensor shapes for the policy graph that aero_train.export builds.

Each is a plain torch.Tensor. The alias says what the rows and columns are,
which a bare Tensor in a signature cannot.
"""

from __future__ import annotations

import torch

Column = torch.Tensor
"""[batch] float32: one named input or output of a policy file, a value per row."""

FeatureBatch = torch.Tensor
"""[batch, n_features] float32: a scenario's observation features in their
declared units, as AeroGymEnv computes them."""

ScaledObservation = torch.Tensor
"""[batch, n_features] float32: features mapped onto [-1, 1] by
ScaleObservation, which is what a policy network is trained on."""

ActionIndex = torch.Tensor
"""[batch] int64: an index into DeltaAction's Discrete(3 ** n_axes)."""

StateValue = torch.Tensor
"""[batch] float32: the policy's estimate of a state's worth."""

Command = torch.Tensor
"""[batch, n_axes] float32: one control command per action variable, in
canonical units and the action Box's order."""


def example_columns(n: int) -> tuple[Column, ...]:
    """Sample inputs for tracing a forward(*inputs: Column).

    Two rows, so no trace can mistake the batch axis for a constant 1.
    """
    return tuple(torch.zeros(2) for _ in range(n))


def column_shapes(n: int) -> dict[str, tuple[dict[int, torch.export.Dim], ...]]:
    """torch.export dynamic_shapes for a forward(*inputs: Column): one shared batch axis."""
    batch = torch.export.Dim("batch")
    return {"inputs": tuple({0: batch} for _ in range(n))}
