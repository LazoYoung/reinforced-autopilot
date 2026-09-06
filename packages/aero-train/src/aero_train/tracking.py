"""Experiment tracking behind an ABC.

The ABC earns its place for one concrete reason: no unit test should need an
MLflow server. NullTracker makes the Trainer testable without one, and the
MLflow tests point at a tmp_path file store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

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
    return MLflowTracker(cfg) if cfg.tracking.enabled else NullTracker()
