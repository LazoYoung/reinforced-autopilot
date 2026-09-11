from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from aero_train.config import DQNConfig, EnvConfig, TrackingConfig, TrainConfig
from aero_train.tracking import (
    MLflowTracker,
    NullTracker,
    Tracker,
    build_tracker,
    library_versions,
)


def make_cfg(**tracking):
    return TrainConfig(
        run_name="tracked-run",
        env=EnvConfig(id="CartPole-v1"),
        algo=DQNConfig(),
        seed=5,
        tracking=TrackingConfig(**tracking),
    )


def test_null_tracker_satisfies_the_abc_and_does_nothing(tmp_path, monkeypatch):
    """No side effect anywhere, not merely "raises nothing": a
    tracker quietly writing to ./mlruns passes a no-exception check and leaves
    Trainer tests dirty. Run from an empty cwd, which must stay empty."""
    monkeypatch.chdir(tmp_path)
    tracker = NullTracker()
    assert isinstance(tracker, Tracker)
    tracker.log_params(make_cfg())
    tracker.log_metrics({"loss": 1.0}, step=1)
    tracker.log_artifact(Path("nonexistent.pt"))
    tracker.close()
    assert list(tmp_path.iterdir()) == []


def test_build_tracker_builds_no_mlflow_when_disabled():
    """`enabled` governs MLflow only. Console output is a separate switch, so
    what disabling tracking must guarantee is that no run is created, not
    that the terminal goes quiet."""
    tracker = build_tracker(make_cfg(enabled=False))
    assert not isinstance(tracker, MLflowTracker)
    assert isinstance(build_tracker(make_cfg(enabled=False, console=False)), NullTracker)


def test_mlflow_tracker_defaults_to_a_local_mlruns_file_store(tmp_path, monkeypatch):
    """The documented default is a file store under the gitignored `mlruns/`.
    mlflow's own fallback is `sqlite:///mlflow.db` in cwd, which is not
    gitignored, so the default has to be pinned rather than inherited."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = make_cfg(enabled=True, experiment="unit")
    tracker = MLflowTracker(cfg)
    tracker.close()

    assert (tmp_path / "mlruns").is_dir()
    assert not (tmp_path / "mlflow.db").exists()


def test_mlflow_tracker_writes_params_to_a_file_store(tmp_path):
    cfg = make_cfg(enabled=True, experiment="unit", tracking_uri=tmp_path.as_uri())
    tracker = MLflowTracker(cfg)
    tracker.log_params(cfg)
    run_id = tracker.run_id
    tracker.close()

    mlflow.set_tracking_uri(tmp_path.as_uri())
    params = mlflow.get_run(run_id).data.params
    assert params["seed"] == "5"
    assert params["env.id"] == "CartPole-v1"
    # topology is logged because a seed alone does not identify a run
    assert params["rollout.n_workers"] == "1"
    assert params["rollout.envs_per_worker"] == "8"


def test_library_versions_are_logged_beside_the_config(tmp_path):
    # aero_gym owns the canonical units contract, so a run that does not record
    # its version cannot be compared against one taken after a bounds change
    cfg = make_cfg(enabled=True, experiment="unit", tracking_uri=tmp_path.as_uri())
    tracker = MLflowTracker(cfg)
    tracker.log_params(cfg)
    run_id = tracker.run_id
    tracker.close()

    mlflow.set_tracking_uri(tmp_path.as_uri())
    params = mlflow.get_run(run_id).data.params
    assert params["version.aero-gym"] not in ("", None)
    assert "version.aero-train" in params


def test_mlflow_tracker_writes_metrics_against_the_global_step(tmp_path):
    cfg = make_cfg(enabled=True, experiment="unit", tracking_uri=tmp_path.as_uri())
    tracker = MLflowTracker(cfg)
    tracker.log_metrics({"loss": 0.5}, step=10)
    tracker.log_metrics({"loss": 0.25}, step=20)
    run_id = tracker.run_id
    tracker.close()

    mlflow.set_tracking_uri(tmp_path.as_uri())
    history = mlflow.MlflowClient().get_metric_history(run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(10, 0.5), (20, 0.25)]


def test_mlflow_tracker_stores_artifacts(tmp_path):
    cfg = make_cfg(enabled=True, experiment="unit", tracking_uri=tmp_path.as_uri())
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"weights")
    tracker = MLflowTracker(cfg)
    tracker.log_artifact(artifact)
    run_id = tracker.run_id
    tracker.close()

    mlflow.set_tracking_uri(tmp_path.as_uri())
    names = [f.path for f in mlflow.MlflowClient().list_artifacts(run_id)]
    assert "checkpoint.pt" in names
