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
    """Load a checkpoint written by :func:`save_checkpoint`, and only one.

    Unpickling is unrestricted, so the path must be trusted. That is the whole
    trust boundary: these payloads carry a config dict and an optional
    numpy-backed buffer snapshot alongside the tensors, which torch 2.6+ will
    not load under its `weights_only=True` default.
    """
    # weights_only=False: the payload is more than tensors. See the docstring
    # for why that is safe here.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class CheckpointManager:
    """Keeps two checkpoints: the latest, and the best by evaluation return."""

    def __init__(
        self,
        directory: Path | str,
        cfg: CheckpointConfig,
        *,
        best_return: float = float("-inf"),
        scenario_fingerprint: dict[str, Any] | None = None,
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        # Resuming callers pass the prior run's value. At the default, a manager
        # pointed at an existing best.pt clobbers it on the first eval.
        self.best_return = float(best_return)
        self.scenario_fingerprint = scenario_fingerprint

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
        # Decided before the payload is built: the check needs the *prior*
        # best_return, while the payload must carry this save's.
        improved = eval_return is not None and eval_return > self.best_return
        if improved:
            self.best_return = float(eval_return)
        payload = {
            "config": config_to_dict(config),
            "algo": algo.state_dict(),
            "global_step": algo.global_step,
            "frames": int(frames),
            # Persisted to seed a resumed manager; without it resume restarts at
            # -inf and the first eval overwrites best.pt.
            "best_return": self.best_return,
            "scenario_fingerprint": self.scenario_fingerprint,
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
