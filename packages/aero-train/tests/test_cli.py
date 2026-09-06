from __future__ import annotations

import textwrap

import pytest

from aero_train import cli
from aero_train.checkpoint import load_checkpoint, save_checkpoint

CONFIG = """
run_name: cli-run
seed: 0
total_steps: 64
env:
  id: CartPole-v1
algo:
  type: dqn
  hidden_sizes: [8]
  buffer_size: 200
  warmup_frames: 16
  batch_size: 8
  target_sync_interval: 5
  updates_per_iter: 2
  epsilon_decay_frames: 50
rollout:
  engine: local
  envs_per_worker: 2
  frames_per_iter: 16
tracking:
  enabled: false
checkpoint:
  dir: {ckpt}
  interval_iters: 1
  eval_interval_iters: 100
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(textwrap.dedent(CONFIG).format(ckpt=tmp_path / "ckpt"))
    return path


def test_run_trains_and_exits_zero(config_path, tmp_path):
    assert cli.main(["run", str(config_path)]) == 0
    assert (tmp_path / "ckpt" / "latest.pt").exists()


def test_override_is_applied(config_path, tmp_path):
    cli.main(["run", str(config_path), "--override", "run_name=overridden"])
    assert load_checkpoint(tmp_path / "ckpt" / "latest.pt")["config"]["run_name"] == "overridden"


def test_override_cannot_bypass_validation(config_path):
    with pytest.raises(ValueError, match="frames_per_iter"):
        cli.main(["run", str(config_path), "--override", "rollout.frames_per_iter=17"])


def test_resume_continues_from_the_saved_global_step(config_path, tmp_path):
    cli.main(["run", str(config_path)])
    ckpt = tmp_path / "ckpt" / "latest.pt"
    before = load_checkpoint(ckpt)["global_step"]
    assert cli.main(["resume", str(ckpt), "--total-steps", "128"]) == 0
    assert load_checkpoint(ckpt)["global_step"] > before


def test_resume_extends_the_frame_budget(config_path, tmp_path):
    cli.main(["run", str(config_path)])
    ckpt = tmp_path / "ckpt" / "latest.pt"
    cli.main(["resume", str(ckpt), "--total-steps", "128"])
    payload = load_checkpoint(ckpt)
    assert payload["config"]["total_steps"] == 128
    assert payload["frames"] >= 128


def test_resume_preserves_the_prior_run_s_best_return(config_path, tmp_path):
    # This config's eval_interval_iters (100) never fires within 4 iterations,
    # so plant a best_return as if an earlier eval already had -- exactly what
    # a real `autopilot run` that found a good policy leaves behind.
    cli.main(["run", str(config_path)])
    ckpt = tmp_path / "ckpt" / "latest.pt"
    payload = load_checkpoint(ckpt)
    payload["best_return"] = 500.0
    save_checkpoint(ckpt, payload)

    cli.main(["resume", str(ckpt), "--total-steps", "128"])
    # cmd_resume's CheckpointManager must be seeded from the checkpoint's own
    # best_return. If it instead defaults to -inf, this resumed run's own
    # saves (which never eval, so never improve on a true best) still carry
    # "best_return": -inf in every payload, silently erasing the 500.0 a real
    # best.pt would have represented.
    assert load_checkpoint(ckpt)["best_return"] == 500.0


def test_eval_reports_a_mean_return(config_path, tmp_path, capsys):
    cli.main(["run", str(config_path)])
    assert cli.main(["eval", str(tmp_path / "ckpt" / "latest.pt"), "--episodes", "2"]) == 0
    assert "mean return" in capsys.readouterr().out


def test_flightgear_is_only_offered_on_eval(config_path):
    with pytest.raises(SystemExit):
        cli.main(["run", str(config_path), "--flightgear"])


def test_flightgear_eval_reports_that_it_lands_in_phase_2(config_path, tmp_path):
    cli.main(["run", str(config_path)])
    with pytest.raises(NotImplementedError, match="Phase 2"):
        cli.main(["eval", str(tmp_path / "ckpt" / "latest.pt"), "--flightgear"])


def test_an_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.main(["fly"])
