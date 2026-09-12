"""One flow, not six tests: start, checkpoint, kill, resume, finish, eval.

Phase 1's only Critical defect lived exactly here: a run's final iteration was
never checkpointed, which every component-interface test passed straight over.
"""

from __future__ import annotations

import yaml
import pytest

jsbsim = pytest.importorskip("jsbsim")

from aero_train import cli
from aero_train.checkpoint import load_checkpoint

CONFIG = {
    "run_name": "lifecycle",
    "seed": 0,
    "total_steps": 512,
    "device": "cpu",
    "env": {"scenario": "steady_flight", "max_episode_steps": 40},
    "algo": {
        "type": "dqn",
        "hidden_sizes": [32, 32],
        "buffer_size": 2000,
        "warmup_frames": 128,
        "batch_size": 32,
        "updates_per_iter": 1,
        "epsilon_decay_frames": 500,
    },
    "rollout": {"engine": "local", "n_workers": 1, "envs_per_worker": 4, "frames_per_iter": 64},
    "tracking": {"enabled": False},
}


def _assert_eval_report_printed(out: str) -> None:
    """An exit code alone is the weakest evidence cmd_eval did real work; a
    degenerate report that never raises still returns 0. Structure itself is
    covered by test_cli.test_eval_reports_a_mean_return, which this mirrors."""
    assert "mean return" in out
    assert "length" in out
    assert "terminated" in out


def test_a_flight_run_survives_start_checkpoint_resume_and_eval(tmp_path, capsys):
    """Under this config, no field the checkpoint carries distinguishes a
    real resume from a restart. Measured, not assumed: cli._train(cfg_1024,
    resume_from=None) run against a snapshot of the post-first-run checkpoint
    directory reaches the same global_step (15, delta 8 from first's 7) as a
    real cmd_resume. best_return differs between the two (31.911 vs 32.387)
    only as a coincidence of eval scheduling, not as a property, so it is not
    asserted below either. What is asserted is what is actually true: the
    final iteration was checkpointed, frames advance past the first run's own
    count, the config round-trips, and the resumed run still evals cleanly.

    DQN's exploratory early actions fly steady_flight well outside its pinned
    bounds vertical_speed and heading_error_deg (checked deterministic under
    this config's seed=0/cpu by running twice); that saturation is expected,
    and the end-of-run record is where it surfaces.
    """
    config = dict(CONFIG)
    config["checkpoint"] = {
        "dir": str(tmp_path / "checkpoints"),
        "interval_iters": 2,
        "eval_interval_iters": 4,
        "eval_episodes": 1,
        "include_buffer": True,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))

    assert cli.main(["train", str(path)]) == 0
    latest = tmp_path / "checkpoints" / "latest.pt"
    assert latest.exists(), "the final iteration was not checkpointed"
    first = load_checkpoint(latest)
    assert first["frames"] >= config["total_steps"]

    # kill and resume: a fresh process would reload exactly this payload
    assert cli.main(["resume", str(latest), "--total-steps", "1024"]) == 0
    second = load_checkpoint(latest)
    assert second["frames"] > first["frames"]
    assert second["config"]["env"]["scenario"] == "steady_flight"

    assert cli.main(["eval", str(latest), "--episodes", "2"]) == 0
    _assert_eval_report_printed(capsys.readouterr().out)


def test_the_best_checkpoint_reloads_through_the_same_path_eval_uses(tmp_path, capsys):
    """DQN's exploratory early actions fly steady_flight well outside its
    pinned bounds vertical_speed and heading_error_deg; expected saturation,
    recorded for the end-of-run summary rather than warned about per step."""
    config = dict(CONFIG)
    config["checkpoint"] = {
        "dir": str(tmp_path / "checkpoints"),
        "interval_iters": 2,
        "eval_interval_iters": 2,
        "eval_episodes": 1,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    assert cli.main(["train", str(path)]) == 0

    best = tmp_path / "checkpoints" / "best.pt"
    assert best.exists()
    assert cli.main(["eval", str(best), "--episodes", "1"]) == 0
    _assert_eval_report_printed(capsys.readouterr().out)
