"""Atomic checkpoint save/load and a two-slot retention policy.

Resume is exact for the learner: `algo.state_dict()` carries every number the
learner's own math depends on (network weights, optimizer state, the replay
sampler's generator), so nothing here needs to duplicate an RNG snapshot on
top of it. It is approximate for the workers if the worker count changed
between save and resume, because SeedSequence partitions the random streams by
topology: same seed plus same topology reproduces a run exactly, a different
topology does not. This is stated here rather than left to be discovered from
a diverging learning curve.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .algo.base import Algorithm
from .buffer import ReplayBuffer
from .config import CheckpointConfig, TrainConfig, config_to_dict


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write to a temp file and rename, so an interrupted save cannot leave a
    truncated checkpoint that only fails hours later, on resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    # weights_only=False: the payload carries the config dict and (when
    # include_buffer is set) a numpy-backed buffer snapshot -- not just tensors --
    # so torch 2.6+'s weights_only=True default would refuse to unpickle it.
    # Safe here because these checkpoints are only ever written by this project's
    # own save_checkpoint, never loaded from an untrusted or third-party source.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class CheckpointManager:
    """Keeps two checkpoints: the latest, and the best by evaluation return."""

    def __init__(
        self,
        directory: Path | str,
        cfg: CheckpointConfig,
        *,
        best_return: float = float("-inf"),
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        # Callers resuming a run pass the prior run's best_return (read back out
        # of the checkpoint payload) here. Left at the default, a fresh manager
        # pointed at a directory that already has a best.pt from an earlier run
        # would otherwise treat -inf as the bar to beat and clobber it with the
        # first eval, however much worse than the checkpoint it replaces.
        self.best_return = float(best_return)

    @property
    def latest_path(self) -> Path:
        return self.dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.dir / "best.pt"

    def save(
        self,
        *,
        config: TrainConfig,
        algo: Algorithm,
        frames: int,
        buffer: ReplayBuffer | None = None,
        eval_return: float | None = None,
    ) -> list[Path]:
        # Decided before building the payload so best_return in the payload is
        # always this save's up-to-date value, and the improvement check below
        # (which needs the *prior* value) doesn't compare it against itself.
        improved = eval_return is not None and eval_return > self.best_return
        if improved:
            self.best_return = float(eval_return)
        payload = {
            "config": config_to_dict(config),
            "algo": algo.state_dict(),
            "global_step": algo.global_step,
            "frames": int(frames),
            # persisted so a resumed CheckpointManager can be seeded with it --
            # otherwise resume starts back at -inf and the first eval after
            # resuming unconditionally overwrites best.pt
            "best_return": self.best_return,
            # a 1e5-frame buffer is large and most resumes do not need bit-exact
            # buffer contents, so it is opt-in
            "buffer": buffer.state_dict()
            if (buffer is not None and self.cfg.include_buffer)
            else None,
        }
        written = [self.latest_path]
        save_checkpoint(self.latest_path, payload)
        if improved:
            save_checkpoint(self.best_path, payload)
            written.append(self.best_path)
        return written
