"""One flow, not six tests: start, checkpoint, kill, resume, finish, eval.

Phase 1's only Critical defect lived exactly here -- a run's final iteration was
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
    # Component tests already cover the report's structure; this is the
    # end-to-end test, and an exit code alone is the weakest possible evidence
    # that cmd_eval did real work -- a degenerate report (or none at all) that
    # never raises would still return 0. Mirrors the substring checks
    # test_eval_reports_a_mean_return uses in test_cli.py.
    assert "mean return" in out
    assert "length" in out
    assert "terminated" in out


def test_a_flight_run_survives_start_checkpoint_resume_and_eval(tmp_path, capsys):
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

    assert cli.main(["run", str(path)]) == 0
    latest = tmp_path / "checkpoints" / "latest.pt"
    assert latest.exists(), "the final iteration was not checkpointed"
    first = load_checkpoint(latest)
    assert first["frames"] >= config["total_steps"]

    # kill and resume: a fresh process would reload exactly this payload
    assert cli.main(["resume", str(latest), "--total-steps", "1024"]) == 0
    second = load_checkpoint(latest)
    assert second["frames"] > first["frames"]
    assert second["config"]["env"]["scenario"] == "steady_flight"

    # `frames` cannot distinguish a real resume from a silent restart: whether
    # cmd_resume seeds Trainer.frames from the checkpoint (512-ish) or leaves it
    # at Trainer.__init__'s default of 0, the run loop is `while frames <
    # total_steps`, so *both* paths stop as soon as frames crosses the same
    # 1024 ceiling -- no choice of total_steps changes that, since it's the
    # very thing both paths converge on. global_step does distinguish them: with
    # updates_per_iter=1 and include_buffer=True (the buffer is already past
    # warmup_frames the moment it's reloaded), every iteration during resume
    # takes exactly one gradient step regardless of which path is taken, so
    # global_step's delta equals the iteration count each path actually runs --
    # and a genuine resume, starting already past the halfway point, runs about
    # half as many new iterations as a restart from zero would. 9 is this run's
    # exact, deterministic (fixed seed, CPU-only) iteration count; a restart
    # measures 19 on this same config (see the mutation report and this task's
    # fix-round report for both numbers, and the demonstration that deleting
    # `trainer.frames = frames` in cmd_resume turns this line red).
    #
    # The constant is a function of this scenario's flight dynamics, not just of
    # the config: iterations are counted in frames, autoreset rows are masked out
    # of the frame count, so longer episodes mean fewer wasted rows and fewer
    # iterations to reach total_steps. It moved from 10 to 9 when the action seed
    # started reading control commands instead of positions and the aircraft
    # stopped sinking out of the band in eight steps. A change here is a signal
    # to re-measure the dynamics, not evidence of a resume regression -- the way
    # to tell them apart is that a restart still measures roughly twice this.
    assert second["global_step"] - first["global_step"] == 9, (
        "resume did not continue from the saved global_step -- frames alone "
        "can't tell a real resume from a restart, since both converge on the "
        "same total_steps ceiling"
    )

    assert cli.main(["eval", str(latest), "--episodes", "2"]) == 0
    _assert_eval_report_printed(capsys.readouterr().out)


def test_the_best_checkpoint_reloads_through_the_same_path_eval_uses(tmp_path, capsys):
    config = dict(CONFIG)
    config["checkpoint"] = {
        "dir": str(tmp_path / "checkpoints"),
        "interval_iters": 2,
        "eval_interval_iters": 2,
        "eval_episodes": 1,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    assert cli.main(["run", str(path)]) == 0

    best = tmp_path / "checkpoints" / "best.pt"
    assert best.exists()
    assert cli.main(["eval", str(best), "--episodes", "1"]) == 0
    _assert_eval_report_printed(capsys.readouterr().out)
