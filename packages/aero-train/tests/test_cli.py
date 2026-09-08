from __future__ import annotations

import os
import textwrap

import gymnasium as gym
import numpy as np
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


def test_override_accepts_bare_exponent_learning_rate(config_path, tmp_path):
    # The most natural spelling for a learning-rate sweep -- `algo.lr=1e-3` --
    # used to reach DQNConfig as the str "1e-3" (PyYAML's YAML-1.1 float
    # grammar requires a decimal point before the exponent), raising a bare
    # "'<=' not supported between instances of 'str' and 'int'" from
    # __post_init__ instead of building a working config.
    cli.main(["run", str(config_path), "--override", "algo.lr=1e-3"])
    cfg = load_checkpoint(tmp_path / "ckpt" / "latest.pt")["config"]
    assert cfg["algo"]["lr"] == pytest.approx(0.001)


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


class _ScriptedEnv(gym.Env):
    """One-step stand-in with the flight stack's spaces, so a checkpoint built
    against them reloads. No aero_gym, no JSBSim: this test is about the flag."""

    def __init__(self, n_obs: int = 12, n_actions: int = 81) -> None:
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (n_obs,), np.float32)
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, np.float32), 1.0, True, False, {}


def _write_checkpoint(tmp_path, env: dict, n_obs: int = 12, n_actions: int = 81):
    """A checkpoint for an arbitrary env config, without running a training loop."""
    from aero_train.algo import build_algorithm
    from aero_train.config import config_from_dict, config_to_dict

    cfg = config_from_dict(
        {
            "run_name": "fg",
            "seed": 0,
            "total_steps": 64,
            "env": env,
            "algo": {"type": "dqn", "hidden_sizes": [8], "buffer_size": 200,
                     "warmup_frames": 16, "batch_size": 8},
            "rollout": {"engine": "local", "envs_per_worker": 1, "frames_per_iter": 8},
            "tracking": {"enabled": False},
        }
    )
    algo = build_algorithm(
        cfg,
        gym.spaces.Box(-1.0, 1.0, (n_obs,), np.float32),
        gym.spaces.Discrete(n_actions),
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        {
            "config": config_to_dict(cfg),
            "algo": algo.state_dict(),
            "global_step": 0,
            "frames": 0,
            "best_return": float("-inf"),
            "buffer": None,
        },
    )
    return path


def test_eval_with_flightgear_renders_through_the_scenario_branch(monkeypatch, tmp_path):
    """The flag was accepted and unimplemented since Phase 1; it now sets the
    render mode on a copy of the env config rather than raising."""
    seen = {}

    def fake_make_env_fn(env_cfg):
        seen["render_mode"] = env_cfg.render_mode
        return _ScriptedEnv

    monkeypatch.setattr(cli, "make_env_fn", fake_make_env_fn)
    path = _write_checkpoint(tmp_path, env={"scenario": "steady_flight"})
    assert cli.main(["eval", str(path), "--episodes", "1", "--flightgear"]) == 0
    assert seen["render_mode"] == "flightgear"


def test_eval_without_flightgear_leaves_the_env_config_alone(monkeypatch, tmp_path):
    seen = {}

    def fake_make_env_fn(env_cfg):
        seen["render_mode"] = env_cfg.render_mode
        return _ScriptedEnv

    monkeypatch.setattr(cli, "make_env_fn", fake_make_env_fn)
    path = _write_checkpoint(tmp_path, env={"scenario": "steady_flight"})
    assert cli.main(["eval", str(path), "--episodes", "1"]) == 0
    assert seen["render_mode"] is None


def test_eval_with_flightgear_on_a_gymnasium_env_says_why_not(tmp_path):
    """It must fail before probing spaces, so no env is ever constructed."""
    path = _write_checkpoint(tmp_path, env={"id": "CartPole-v1"}, n_obs=4, n_actions=2)
    with pytest.raises(ValueError, match="aero_gym scenario"):
        cli.main(["eval", str(path), "--flightgear"])


def test_an_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.main(["fly"])


# --- autopilot mlflow -------------------------------------------------------
#
# No test here may spawn a subprocess, open a socket, or launch a real MLflow
# server (project-wide constraint). build_mlflow_server_argv is tested as a
# pure function; cmd_mlflow (reached through main, so the argparse wiring is
# covered too) is tested with os.execvp and shutil.which monkeypatched so
# nothing actually execs.


class _FakeExec:
    """Records an os.execvp call instead of replacing the process image."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, file, args) -> None:
        self.calls.append((file, list(args)))


@pytest.fixture
def fake_execvp(monkeypatch):
    fake = _FakeExec()
    monkeypatch.setattr(cli.os, "execvp", fake)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/uvx")
    return fake


def test_build_mlflow_server_argv_default_store_and_port():
    argv = cli.build_mlflow_server_argv("./mlruns", "127.0.0.1", 5000)
    assert argv == [
        "uvx",
        "--from",
        "mlflow",
        "mlflow",
        "server",
        "--backend-store-uri",
        "file:./mlruns",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
    ]


def test_build_mlflow_server_argv_explicit_path_gets_file_scheme():
    argv = cli.build_mlflow_server_argv("/data/mlruns", "127.0.0.1", 5000)
    assert argv[argv.index("--backend-store-uri") + 1] == "file:/data/mlruns"


def test_build_mlflow_server_argv_uri_store_passed_through_unchanged():
    # A value containing "://" is already a URI (sqlite file, remote server)
    # and must not be re-wrapped in "file:".
    argv = cli.build_mlflow_server_argv("sqlite:///mlflow.db", "127.0.0.1", 5000)
    assert argv[argv.index("--backend-store-uri") + 1] == "sqlite:///mlflow.db"


def test_build_mlflow_server_argv_honours_host_and_port():
    argv = cli.build_mlflow_server_argv("./mlruns", "0.0.0.0", 8080)
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv[argv.index("--port") + 1] == "8080"


def test_mlflow_default_argv_via_main(fake_execvp, monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    assert cli.main(["mlflow"]) == 0
    ((_, args),) = fake_execvp.calls
    assert args == cli.build_mlflow_server_argv("./mlruns", "127.0.0.1", 5000)


def test_mlflow_port_and_host_are_honoured(fake_execvp, monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    assert cli.main(["mlflow", "--port", "6001", "--host", "0.0.0.0"]) == 0
    ((_, args),) = fake_execvp.calls
    assert args == cli.build_mlflow_server_argv("./mlruns", "0.0.0.0", 6001)


def test_mlflow_store_path_via_main(fake_execvp, monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    cli.main(["mlflow", "--store", "/tmp/store"])
    ((_, args),) = fake_execvp.calls
    assert args[args.index("--backend-store-uri") + 1] == "file:/tmp/store"


def test_mlflow_store_uri_via_main(fake_execvp, monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    cli.main(["mlflow", "--store", "sqlite:///mlflow.db"])
    ((_, args),) = fake_execvp.calls
    assert args[args.index("--backend-store-uri") + 1] == "sqlite:///mlflow.db"


def test_mlflow_sets_allow_file_store_when_absent(fake_execvp, monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    cli.main(["mlflow"])
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"


def test_mlflow_respects_an_existing_allow_file_store_value(fake_execvp, monkeypatch):
    # setdefault semantics: an explicit value the caller already set (e.g. a
    # shared-host policy) must win over our default, not be overwritten.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    cli.main(["mlflow"])
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"


def test_mlflow_missing_uvx_returns_nonzero_and_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    exec_calls = []
    monkeypatch.setattr(cli.os, "execvp", lambda *a: exec_calls.append(a))
    assert cli.main(["mlflow"]) == 1
    assert exec_calls == []
    err = capsys.readouterr().err
    assert "uvx" in err
    assert "uv" in err
