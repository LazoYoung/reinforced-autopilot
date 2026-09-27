"""Train, export, and fly the file the way a host does: one flow on JSBSim.

The export check feeds prev.* from the wrapper's accumulator. A host has no
wrapper and reads the applied commands back from its simulator, so this flies
the exported file on exactly that read-back, beside the checkpoint's own path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

jsbsim = pytest.importorskip("jsbsim")

import onnxruntime as ort

from aero_train import cli
from aero_train.algo import build_algorithm
from aero_train.checkpoint import load_checkpoint
from aero_train.config import config_from_dict
from aero_train.envs import make_env_fn

CONFIG = {
    "run_name": "export-lifecycle",
    "seed": 0,
    "total_steps": 256,
    "device": "cpu",
    "env": {
        "scenario": "steady_flight",
        "max_episode_steps": 60,
        "params": {"altitude_jitter_ft": 200.0, "heading_jitter_deg": 5.0},
    },
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


def tick(session, readings, targets, applied):
    """One call of the policy file.

    readings: this step's canonical readings, by name ("altitude", ...).
    targets: the targets the pilot set, by name ("target_altitude_ft", ...).
    applied: the commands the aircraft is flying with now, read back from the
    simulator, by action name ("elevator", ...).
    Returns the commands to apply next, by action name.
    """
    values = readings | targets | {f"prev.{name}": v for name, v in applied.items()}
    feed = {i.name: np.array([values[i.name]], np.float32) for i in session.get_inputs()}
    names = [o.name for o in session.get_outputs()]
    result = dict(zip(names, session.run(None, feed)))
    return {name: float(column[0]) for name, column in result.items() if name != "value"}


def test_a_trained_policy_exports_and_flies_on_commands_read_back_from_the_simulator(
    tmp_path, capsys
):
    config = dict(CONFIG, checkpoint={"dir": str(tmp_path / "checkpoints"), "interval_iters": 2})
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    assert cli.main(["train", str(path)]) == 0
    latest = tmp_path / "checkpoints" / "latest.pt"
    assert cli.main(["export", str(latest)]) == 0
    assert "steps checked" in capsys.readouterr().out

    session = ort.InferenceSession(str(latest.with_suffix(".onnx")))
    assert float(session.get_modelmeta().custom_metadata_map["aero.agent_hz"]) == 10.0
    payload = load_checkpoint(latest)
    cfg = config_from_dict(payload["config"])
    env = make_env_fn(cfg.env)()
    algo = build_algorithm(cfg, env.observation_space, env.action_space)
    algo.load_state_dict(payload["algo"])
    policy = algo.policy_module()
    aircraft = env.unwrapped
    actions = aircraft.scenario.action_variables
    steps = agreed = 0
    try:
        obs, info = env.reset(seed=cfg.seed)
        done = False
        while not done:
            readings = {var.value: value for var, value in info["obs"].items()}
            applied = {var.value: aircraft.simulator.get_control(var) for var in actions}
            commands = tick(session, readings, dict(info["context"]), applied)
            with torch.no_grad():
                action, _ = policy(torch.as_tensor(obs)[None])
            obs, _, terminated, truncated, info = env.step(int(action[0]))
            flown = [aircraft.simulator.get_control(var) for var in actions]
            agreed += all(
                abs(commands[var.value] - value) <= 1e-6 for var, value in zip(actions, flown)
            )
            steps += 1
            done = terminated or truncated
    finally:
        env.close()
    assert steps > 0
    assert agreed >= steps - max(1, steps // 100)
