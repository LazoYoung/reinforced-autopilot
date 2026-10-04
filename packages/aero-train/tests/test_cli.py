from __future__ import annotations

import dataclasses
import json
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
    assert cli.main(["train", str(config_path)]) == 0
    assert (tmp_path / "ckpt" / "latest.pt").exists()


def test_override_is_applied(config_path, tmp_path):
    cli.main(["train", str(config_path), "--override", "run_name=overridden"])
    assert load_checkpoint(tmp_path / "ckpt" / "latest.pt")["config"]["run_name"] == "overridden"


def test_override_accepts_bare_exponent_learning_rate(config_path, tmp_path):
    """`algo.lr=1e-3`, the natural sweep spelling, used to reach DQNConfig as a
    str (PyYAML's YAML-1.1 grammar wants a decimal point before the exponent)
    and __post_init__ raised a bare str/int comparison error."""
    cli.main(["train", str(config_path), "--override", "algo.lr=1e-3"])
    cfg = load_checkpoint(tmp_path / "ckpt" / "latest.pt")["config"]
    assert cfg["algo"]["lr"] == pytest.approx(0.001)


def test_override_cannot_bypass_validation(config_path):
    with pytest.raises(ValueError, match="frames_per_iter"):
        cli.main(["train", str(config_path), "--override", "rollout.frames_per_iter=17"])


def test_resume_continues_from_the_saved_global_step(config_path, tmp_path):
    cli.main(["train", str(config_path)])
    ckpt = tmp_path / "ckpt" / "latest.pt"
    before = load_checkpoint(ckpt)["global_step"]
    assert cli.main(["resume", str(ckpt), "--total-steps", "128"]) == 0
    assert load_checkpoint(ckpt)["global_step"] > before


def test_resume_extends_the_frame_budget(config_path, tmp_path):
    cli.main(["train", str(config_path)])
    ckpt = tmp_path / "ckpt" / "latest.pt"
    cli.main(["resume", str(ckpt), "--total-steps", "128"])
    payload = load_checkpoint(ckpt)
    assert payload["config"]["total_steps"] == 128
    assert payload["frames"] >= 128


def test_resume_preserves_the_prior_run_s_best_return(config_path, tmp_path):
    # eval_interval_iters (100) never fires within 4 iterations, so plant the
    # best_return a real `autopilot train` would have left behind.
    cli.main(["train", str(config_path)])
    ckpt = tmp_path / "ckpt" / "latest.pt"
    payload = load_checkpoint(ckpt)
    payload["best_return"] = 500.0
    save_checkpoint(ckpt, payload)

    cli.main(["resume", str(ckpt), "--total-steps", "128"])
    # Seeded from the checkpoint, not -inf: this run never evals, so every save
    # would carry -inf and silently erase the prior best.
    assert load_checkpoint(ckpt)["best_return"] == 500.0


def test_eval_reports_a_mean_return(config_path, tmp_path, capsys):
    cli.main(["train", str(config_path)])
    assert cli.main(["eval", str(tmp_path / "ckpt" / "latest.pt"), "--episodes", "2"]) == 0
    out = capsys.readouterr().out
    # "mean return" alone would still match a report whose tail was dropped or
    # garbled, so the length and terminated portions are checked too.
    assert "mean return" in out
    assert "length" in out
    assert "terminated" in out


def test_flightgear_is_only_offered_on_eval(config_path):
    with pytest.raises(SystemExit):
        cli.main(["train", str(config_path), "--flightgear"])


def test_training_subcommand_is_train_not_run(config_path):
    """`run` said nothing about what it did, next to `resume` and `eval`."""
    assert cli.main(["train", str(config_path)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["run", str(config_path)])


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


def _write_checkpoint(
    tmp_path,
    env: dict,
    n_obs: int = 12,
    n_actions: int = 81,
    algo: dict | None = None,
    device: str | None = None,
    global_step: int = 0,
    frames: int = 0,
    fingerprint: dict | None = None,
):
    """A checkpoint for an arbitrary env config, without running a training loop."""
    from aero_train.algo import build_algorithm
    from aero_train.config import config_from_dict, config_to_dict

    cfg = config_from_dict(
        {
            "run_name": "fg",
            "seed": 0,
            "total_steps": 64,
            "env": env,
            "algo": algo
            or {"type": "dqn", "hidden_sizes": [8], "buffer_size": 200,
                "warmup_frames": 16, "batch_size": 8},
            "rollout": {"engine": "local", "envs_per_worker": 1, "frames_per_iter": 8},
            "tracking": {"enabled": False},
            **({"device": device} if device is not None else {}),
        }
    )
    # Built on cpu regardless of cfg.device: load_checkpoint always maps
    # tensors back to cpu, so a saved state dict is cpu-shaped either way.
    algo = build_algorithm(
        dataclasses.replace(cfg, device="cpu"),
        gym.spaces.Box(-1.0, 1.0, (n_obs,), np.float32),
        gym.spaces.Discrete(n_actions),
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        {
            "config": config_to_dict(cfg),
            "algo": algo.state_dict(),
            "global_step": global_step,
            "frames": frames,
            "best_return": float("-inf"),
            "buffer": None,
            "scenario_fingerprint": fingerprint,
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
# No subprocess, socket or real server: os.execvp and shutil.which are patched.


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


def test_training_with_a_viewer_builds_its_eval_env_without_one(monkeypatch, tmp_path):
    """Trainer._checkpoint -> evaluate_policy -> eval_env_fn() builds a whole
    second AeroGymEnv. With the training factory reused, that second env also
    asks for render_mode='flightgear' and launches a second FlightGear on the
    same default ports while the training viewer is still alive. The eval env is
    not the one the user is watching, so it gets no viewer.
    """
    render_modes = []
    captured = {}

    def fake_make_env_fn(env_cfg):
        render_modes.append(env_cfg.render_mode)

        def _make():
            return _ScriptedEnv()

        _make.render_mode = env_cfg.render_mode
        return _make

    class FakeTrainer:
        def __init__(self, *args, eval_env_fn=None, **kwargs):
            captured["eval_env_fn"] = eval_env_fn
            self.frames = 0

        def run(self):
            pass

    monkeypatch.setattr(cli, "make_env_fn", fake_make_env_fn)
    monkeypatch.setattr(cli, "Trainer", FakeTrainer)

    path = tmp_path / "run.yaml"
    path.write_text(
        textwrap.dedent(
            """
            run_name: viewer
            seed: 0
            total_steps: 64
            env:
              scenario: steady_flight
              render_mode: flightgear
            algo:
              type: dqn
              hidden_sizes: [8]
              buffer_size: 200
              warmup_frames: 16
              batch_size: 8
            rollout:
              engine: local
              n_workers: 1
              envs_per_worker: 1
              frames_per_iter: 8
            tracking:
              enabled: false
            checkpoint:
              dir: {ckpt}
            """
        ).format(ckpt=tmp_path / "ckpt")
    )

    assert cli.main(["train", str(path)]) == 0
    assert "flightgear" in render_modes, "the training env must keep its viewer"
    assert captured["eval_env_fn"].render_mode is None


# --- autopilot export --------------------------------------------------------


@pytest.fixture
def toy_checkpoint(monkeypatch, tmp_path):
    """A flight checkpoint whose env is the toy hold task rather than JSBSim."""
    from tests.flight_fakes import N_ACTIONS, N_FEATURES, toy_env_fn, toy_fingerprint

    monkeypatch.setattr(cli, "make_env_fn", lambda env_cfg: toy_env_fn())
    return _write_checkpoint(
        tmp_path,
        env={"scenario": "steady_flight"},
        n_obs=N_FEATURES,
        n_actions=N_ACTIONS,
        global_step=3,
        frames=7,
        fingerprint=toy_fingerprint(),
    )


def test_export_writes_beside_the_checkpoint_and_says_what_it_checked(toy_checkpoint, capsys):
    import onnx
    from aero_gym.registry import build_scenario

    assert cli.main(["export", str(toy_checkpoint)]) == 0
    written = toy_checkpoint.with_suffix(".onnx")
    meta = {p.key: p.value for p in onnx.load(str(written)).metadata_props}
    config = load_checkpoint(toy_checkpoint)["config"]
    # Through JSON on both sides: the checkpoint keeps tuples, JSON reads back lists.
    assert json.loads(meta["aero.config"]) == json.loads(json.dumps(config))
    assert meta["aero.frames"] == "7"
    assert meta["aero.scenario"] == "steady_flight"
    params = json.loads(meta["aero.config"])["env"]["params"]
    assert build_scenario(meta["aero.scenario"], params).name == "steady-flight"
    assert "steps checked" in capsys.readouterr().out


def test_export_builds_a_headless_cpu_env_and_algorithm(monkeypatch, tmp_path):
    """A checkpoint trained with a viewer and on CUDA must still export
    headlessly and on the CPU: the check only flies an episode, it never
    needs a display or a GPU."""
    from tests.flight_fakes import N_ACTIONS, N_FEATURES, toy_env_fn, toy_fingerprint

    seen = {}

    def fake_make_env_fn(env_cfg):
        seen["render_mode"] = env_cfg.render_mode
        return toy_env_fn()

    monkeypatch.setattr(cli, "make_env_fn", fake_make_env_fn)
    path = _write_checkpoint(
        tmp_path,
        env={"scenario": "steady_flight", "render_mode": "flightgear"},
        n_obs=N_FEATURES,
        n_actions=N_ACTIONS,
        device="cuda",
        fingerprint=toy_fingerprint(),
    )
    assert cli.main(["export", str(path)]) == 0
    assert seen["render_mode"] is None


def test_export_writes_where_output_says_creating_the_directory(toy_checkpoint, tmp_path):
    target = tmp_path / "exports" / "v1.onnx"
    assert cli.main(["export", str(toy_checkpoint), "-o", str(target)]) == 0
    assert target.exists()
    assert not toy_checkpoint.with_suffix(".onnx").exists()


def test_export_writes_the_checkpoint_stem_into_an_existing_output_directory(
    toy_checkpoint, tmp_path
):
    out_dir = tmp_path / "exports"
    out_dir.mkdir()
    assert cli.main(["export", str(toy_checkpoint), "-o", str(out_dir)]) == 0
    assert (out_dir / "ckpt.onnx").exists()


def test_export_refuses_a_gymnasium_checkpoint_before_building_anything(monkeypatch, tmp_path):
    def no_env(env_cfg):
        raise AssertionError("no env should be built")

    monkeypatch.setattr(cli, "make_env_fn", no_env)
    path = _write_checkpoint(tmp_path, env={"id": "CartPole-v1"}, n_obs=4, n_actions=2)
    with pytest.raises(ValueError, match="canonical names"):
        cli.main(["export", str(path)])


def test_export_refuses_a_baseline(monkeypatch, tmp_path):
    from tests.flight_fakes import N_ACTIONS, toy_env_fn, toy_fingerprint

    monkeypatch.setattr(cli, "make_env_fn", lambda env_cfg: toy_env_fn())
    path = _write_checkpoint(
        tmp_path,
        env={"scenario": "steady_flight"},
        n_actions=N_ACTIONS,
        algo={"type": "random"},
        fingerprint=toy_fingerprint(),
    )
    with pytest.raises(NotImplementedError, match="Random cannot be exported"):
        cli.main(["export", str(path)])


# --- scenario fingerprint ----------------------------------------------------


@pytest.fixture
def toy_run(monkeypatch, tmp_path):
    """Train the toy hold task through the CLI; returns (latest checkpoint, set_scenario).

    `set_scenario(s)` makes every later CLI command build toy envs from `s`.
    """
    from tests.flight_fakes import toy_env_fn

    def set_scenario(scenario):
        monkeypatch.setattr(cli, "make_env_fn", lambda env_cfg: toy_env_fn(scenario))

    set_scenario(None)
    path = tmp_path / "toy.yaml"
    config = textwrap.dedent(CONFIG).format(ckpt=tmp_path / "ckpt")
    path.write_text(config.replace("id: CartPole-v1", "scenario: steady_flight"))
    assert cli.main(["train", str(path)]) == 0
    return tmp_path / "ckpt" / "latest.pt", set_scenario


def test_train_stores_the_scenario_fingerprint_in_its_checkpoints(toy_run):
    from tests.flight_fakes import toy_fingerprint

    latest, _ = toy_run
    assert load_checkpoint(latest)["scenario_fingerprint"] == toy_fingerprint()


def test_a_gymnasium_run_stores_no_fingerprint(config_path, tmp_path):
    assert cli.main(["train", str(config_path)]) == 0
    assert load_checkpoint(tmp_path / "ckpt" / "latest.pt")["scenario_fingerprint"] is None


def test_resume_refuses_a_scenario_whose_feature_changed(toy_run):
    from tests.flight_fakes import flipped_toy_scenario

    latest, set_scenario = toy_run
    before = latest.read_bytes()
    set_scenario(flipped_toy_scenario())
    with pytest.raises(ValueError, match=r"cannot resume.*altitude_error_ft changed"):
        cli.main(["resume", str(latest), "--total-steps", "128"])
    assert latest.read_bytes() == before


def test_resume_accepts_a_reward_change_and_keeps_the_fingerprint(toy_run):
    from tests.flight_fakes import toy_fingerprint, toy_scenario

    latest, set_scenario = toy_run
    set_scenario(toy_scenario(reward_fn=lambda t: (0.5, False)))
    assert cli.main(["resume", str(latest), "--total-steps", "128"]) == 0
    payload = load_checkpoint(latest)
    assert payload["frames"] > 64
    assert payload["scenario_fingerprint"] == toy_fingerprint()


def test_a_resumed_checkpoint_without_a_fingerprint_stays_without_one(toy_run, capsys):
    """Resume must not vouch for a scenario nobody recorded at training time."""
    latest, _ = toy_run
    payload = load_checkpoint(latest)
    payload.pop("scenario_fingerprint")
    save_checkpoint(latest, payload)
    assert cli.main(["resume", str(latest), "--total-steps", "128"]) == 0
    assert load_checkpoint(latest)["scenario_fingerprint"] is None
    assert "carries no scenario fingerprint" in capsys.readouterr().err


def test_resuming_a_gymnasium_run_prints_no_fingerprint_note(config_path, tmp_path, capsys):
    cli.main(["train", str(config_path)])
    capsys.readouterr()
    cli.main(["resume", str(tmp_path / "ckpt" / "latest.pt"), "--total-steps", "128"])
    assert "fingerprint" not in capsys.readouterr().err


def test_export_refuses_a_scenario_changed_since_training_and_writes_nothing(toy_run):
    from tests.flight_fakes import flipped_toy_scenario

    latest, set_scenario = toy_run
    set_scenario(flipped_toy_scenario())
    with pytest.raises(ValueError, match=r"cannot export.*altitude_error_ft changed"):
        cli.main(["export", str(latest)])
    assert not latest.with_suffix(".onnx").exists()


def test_export_accepts_a_reward_change(toy_run):
    from tests.flight_fakes import toy_scenario

    latest, set_scenario = toy_run
    set_scenario(toy_scenario(reward_fn=lambda t: (0.5, False)))
    assert cli.main(["export", str(latest)]) == 0
    assert latest.with_suffix(".onnx").exists()


def test_export_refuses_a_checkpoint_without_a_fingerprint_naming_the_flag(
    toy_checkpoint, tmp_path
):
    payload = load_checkpoint(toy_checkpoint)
    payload.pop("scenario_fingerprint")
    save_checkpoint(toy_checkpoint, payload)
    with pytest.raises(ValueError, match="--unverified-scenario"):
        cli.main(["export", str(toy_checkpoint)])
    assert not toy_checkpoint.with_suffix(".onnx").exists()


def test_unverified_scenario_exports_a_checkpoint_without_a_fingerprint_with_a_warning(
    toy_checkpoint, capsys
):
    payload = load_checkpoint(toy_checkpoint)
    payload.pop("scenario_fingerprint")
    save_checkpoint(toy_checkpoint, payload)
    assert cli.main(["export", str(toy_checkpoint), "--unverified-scenario"]) == 0
    assert toy_checkpoint.with_suffix(".onnx").exists()
    assert "no scenario fingerprint" in capsys.readouterr().err
