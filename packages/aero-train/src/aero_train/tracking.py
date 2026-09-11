"""Experiment tracking behind an ABC.

The ABC earns its place for one concrete reason: no unit test should need an
MLflow server. NullTracker makes the Trainer testable without one, and the
MLflow tests point at a tmp_path file store.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, IO, Sequence

from .config import TrainConfig, flatten_config


class Tracker(ABC):
    @abstractmethod
    def log_params(self, cfg: TrainConfig) -> None:
        """Log the flattened config once, at run start.

        Topology (n_workers, envs_per_worker, frames_per_iter) is part of it:
        a seed alone does not identify a run, because changing the topology
        changes how the random streams are partitioned.
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log one iteration's metrics against the learner's global step."""

    @abstractmethod
    def log_artifact(self, path: Path) -> None:
        """Store a file (a checkpoint) with the run, so a run is reproducible
        from the tracking server alone."""

    @abstractmethod
    def close(self) -> None: ...


class NullTracker(Tracker):
    """Discards everything. The default for tests and for `tracking.enabled: false`."""

    def log_params(self, cfg: TrainConfig) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass

    def close(self) -> None:
        pass


class ConsoleTracker(Tracker):
    """Progress on stdout: without it a 300k-frame run is 35 silent minutes.

    Throttled by wall clock rather than an iteration stride, since an iteration
    is ~30 ms of CartPole and ~2 s of JSBSim. The first iteration, every eval,
    and the final iteration (flushed by close) bypass the throttle.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        stream: IO[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._clock = clock
        self._interval = cfg.tracking.console_interval_s
        self._total_steps = cfg.total_steps
        self._started = clock()
        self._last_print: float | None = None
        self._withheld: dict[str, float] | None = None
        self._frames = 0

    def log_params(self, cfg: TrainConfig) -> None:
        env = cfg.env.scenario or cfg.env.id
        self._write(
            f"{cfg.run_name} | {cfg.algo.type} on {env} | {cfg.total_steps} frames "
            f"| {cfg.device} | {cfg.rollout.engine} x{cfg.rollout.envs_per_worker} envs, "
            f"{cfg.rollout.frames_per_iter} frames/iter"
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return
        if any(k.startswith("eval/") for k in metrics):
            self._write(self._eval_line(metrics))
            return
        self._frames = int(metrics.get("progress/frames", self._frames))
        now = self._clock()
        if self._last_print is not None and now - self._last_print < self._interval:
            self._withheld = metrics
            return
        self._last_print = now
        self._withheld = None
        self._write(self._iteration_line(metrics))

    def log_artifact(self, path: Path) -> None:
        self._write(f"  saved {path}")

    def close(self) -> None:
        if self._withheld is not None:
            self._write(self._iteration_line(self._withheld))
            self._withheld = None
        self._write(f"done | {self._frames} frames | {_duration(self._clock() - self._started)}")

    def _iteration_line(self, m: dict[str, float]) -> str:
        frames = int(m.get("progress/frames", 0))
        parts = [f"iter {int(m.get('progress/iteration', 0)):>5}"]
        if self._total_steps:
            parts.append(f"frames {frames:>8}/{self._total_steps} ({frames / self._total_steps:>4.0%})")
        else:
            parts.append(f"frames {frames:>8}")
        # No episode finished this iteration: the mean keys are absent, not zero.
        if m.get("episode/count", 0.0):
            parts.append(f"ret {m['episode/return_mean']:>7.1f}")
            parts.append(f"len {m['episode/length_mean']:>5.0f}")
            parts.append(f"term {m['episode/terminated_frac']:>4.0%}")
        # Likewise absent for the baselines, which never call update().
        for key, label, places in (("loss", "loss", 4), ("epsilon", "eps", 2)):
            if key in m:
                parts.append(f"{label} {m[key]:.{places}f}")
        if "throughput/fps" in m:
            parts.append(f"{m['throughput/fps']:>6.0f} fps")
        return " | ".join(parts)

    def _eval_line(self, m: dict[str, float]) -> str:
        parts = [f"  eval @ {self._frames}"]
        for key, label, places in (
            ("eval/return_mean", "return", 2),
            ("eval/length_mean", "length", 1),
            ("eval/return_per_step", "return/step", 3),
        ):
            if key in m:
                parts.append(f"{label} {m[key]:.{places}f}")
        if "eval/terminated_frac" in m:
            parts.append(f"terminated {m['eval/terminated_frac']:.0%}")
        return " | ".join(parts)

    def _write(self, line: str) -> None:
        print(line, file=self._stream, flush=True)


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


class CompositeTracker(Tracker):
    """Fan every call out to several trackers, so console and MLflow coexist
    without Trainer knowing how many backends there are."""

    def __init__(self, trackers: Sequence[Tracker]) -> None:
        self.trackers = list(trackers)

    def log_params(self, cfg: TrainConfig) -> None:
        for t in self.trackers:
            t.log_params(cfg)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for t in self.trackers:
            t.log_metrics(metrics, step)

    def log_artifact(self, path: Path) -> None:
        for t in self.trackers:
            t.log_artifact(path)

    def close(self) -> None:
        # Every tracker closes before the first failure is re-raised: one dead
        # backend must not leave MLflow's run open.
        errors: list[BaseException] = []
        for t in self.trackers:
            try:
                t.close()
            except BaseException as exc:      # noqa: BLE001 - re-raised below
                errors.append(exc)
        if errors:
            raise errors[0]


class MLflowTracker(Tracker):
    """Local file store by default; a tracking URI redirects to a server.

    That redirect is the whole migration story for a future cluster: setting
    MLFLOW_TRACKING_URI or tracking.tracking_uri changes where runs land with no
    code change.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        # mlflow>=3 refuses the filesystem backend ("maintenance mode") unless
        # explicitly opted into, but the local `./mlruns` file store is this
        # project's documented default backing store -- see the module and
        # class docstrings on the migration story. These are process-wide
        # environment defaults, set lazily on first tracker construction
        # rather than at import time; `setdefault` is deliberate so an
        # explicit environment setting (e.g. a real MLFLOW_ALLOW_FILE_STORE
        # policy on a shared host) always wins over this default.
        #
        # mlflow also logs a one-time "agent hint" on import/run-creation when
        # it detects it is being driven by a coding agent. It is emitted via
        # `logging`, not `warnings.warn`, so it never reaches pytest's warning
        # summary -- but it is still noise with nothing to act on here.
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

        import mlflow

        self._mlflow = mlflow
        if cfg.tracking.tracking_uri:
            mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        elif not os.environ.get("MLFLOW_TRACKING_URI"):
            # Neither the config nor the environment redirected us, so pin the
            # documented default explicitly. mlflow's own hard-coded fallback
            # is no longer a `./mlruns` file store in the installed version --
            # it is a `sqlite:///mlflow.db` file in the cwd, which is not
            # gitignored (only `mlruns/` is) and would land an untracked
            # binary in the repo root the first time this runs unconfigured.
            mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(cfg.tracking.experiment)
        self._run = mlflow.start_run(run_name=cfg.run_name)

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    def log_params(self, cfg: TrainConfig) -> None:
        self._mlflow.log_params({**flatten_config(cfg), **library_versions()})

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Guard against an empty dict: a caller that logs whatever metrics an
        # iteration happened to produce (e.g. none, before warmup) shouldn't
        # have to check for that itself before calling in.
        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        self._mlflow.end_run()


def library_versions() -> dict[str, str]:
    """Versions of the libraries a run's results depend on.

    aero_gym owns the canonical units contract -- the bounds in ACTION_SPECS and
    the pilot sign convention -- so a run recorded without its version cannot be
    compared against one taken after a release that changed either. Same
    reasoning as logging the topology beside the seed: the seed alone does not
    identify a run.
    """
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for dist in ("aero-gym", "aero-train"):
        try:
            out[f"version.{dist}"] = version(dist)
        except PackageNotFoundError:
            out[f"version.{dist}"] = "unknown"
    return out


def build_tracker(cfg: TrainConfig) -> Tracker:
    """`enabled` governs MLflow, `console` governs stdout -- independently,
    because a run with MLflow off is the one whose only output is the terminal."""
    trackers: list[Tracker] = []
    if cfg.tracking.console:
        trackers.append(ConsoleTracker(cfg))
    if cfg.tracking.enabled:
        trackers.append(MLflowTracker(cfg))
    if not trackers:
        return NullTracker()
    return trackers[0] if len(trackers) == 1 else CompositeTracker(trackers)
