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


@pytest.mark.filterwarnings(
    "ignore:observation feature '.*' (saturated at|is NaN):RuntimeWarning"
)
def test_a_flight_run_survives_start_checkpoint_resume_and_eval(tmp_path, capsys):
    """Resume is checked on global_step, because `frames` cannot see it.

    DQN's exploratory early actions fly steady_flight well outside its pinned
    bounds (vertical_speed, heading_error_deg); that saturation is expected.

    Whether cmd_resume seeds Trainer.frames from the checkpoint or leaves it at
    0, the loop is `while frames < total_steps`, so both paths stop at the same
    ceiling for any total_steps. global_step separates them: updates_per_iter=1
    and a buffer reloaded already past warmup mean one gradient step per
    iteration, so its delta is the iteration count each path actually ran,
    and a real resume, starting past halfway, runs about half of a restart's.
    So a moved constant that is still roughly half a restart's is the scenario's
    dynamics drifting, not a resume regression.
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

    # 8 tracks the flight dynamics, not just the config: autoreset rows are
    # masked out of the frame count. A restart measures 17 here.
    assert second["global_step"] - first["global_step"] == 8, (
        "resume did not continue from the saved global_step; frames alone "
        "can't tell a real resume from a restart, since both converge on the "
        "same total_steps ceiling"
    )

    assert cli.main(["eval", str(latest), "--episodes", "2"]) == 0
    _assert_eval_report_printed(capsys.readouterr().out)


@pytest.mark.filterwarnings(
    "ignore:observation feature '.*' (saturated at|is NaN):RuntimeWarning"
)
def test_the_best_checkpoint_reloads_through_the_same_path_eval_uses(tmp_path, capsys):
    """DQN's exploratory early actions fly steady_flight well outside its
    pinned bounds (vertical_speed, heading_error_deg); expected saturation."""
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
