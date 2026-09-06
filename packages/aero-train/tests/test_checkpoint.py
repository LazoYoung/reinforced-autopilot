from __future__ import annotations

import numpy as np
import pytest
import torch

from aero_train.algo.dqn import DQN
from aero_train.buffer import ReplayBuffer
from aero_train.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint
from aero_train.config import CheckpointConfig, DQNConfig, EnvConfig, TrainConfig
from aero_train.primitives import Transitions

ALGO_CFG = DQNConfig(hidden_sizes=(8,), buffer_size=100, warmup_frames=8, batch_size=8)


def make_cfg(**ckpt):
    return TrainConfig(
        run_name="ckpt-run",
        env=EnvConfig(id="CartPole-v1"),
        algo=ALGO_CFG,
        checkpoint=CheckpointConfig(**ckpt),
    )


def make_algo():
    return DQN(obs_shape=(4,), n_actions=2, cfg=ALGO_CFG, seed=0)


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "c.pt"
    save_checkpoint(path, {"global_step": 7, "note": "hello"})
    assert load_checkpoint(path)["global_step"] == 7


def test_save_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "c.pt"
    save_checkpoint(path, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["c.pt"]


def test_a_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    path = tmp_path / "c.pt"
    save_checkpoint(path, {"global_step": 1})

    def boom(obj, f, *args, **kwargs):
        # A disk-full failure doesn't necessarily fail cleanly before writing
        # anything -- it can fail mid-write, leaving a truncated file at
        # whatever path it was writing to. Simulate that instead of raising
        # before any bytes move, which a naive `torch.save(payload, path)`
        # straight to the destination (no tmp file) would also survive
        # trivially, since the mock would never touch `path` either way.
        with open(f, "wb") as fh:
            fh.write(b"\x00" * 4)
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        save_checkpoint(path, {"global_step": 2})
    # the truncated write landed on the tmp file, which the rename never
    # promoted -- so the destination still holds the previous good checkpoint
    assert load_checkpoint(path)["global_step"] == 1


def test_manager_writes_latest_on_every_save(tmp_path):
    mgr = CheckpointManager(tmp_path, CheckpointConfig())
    algo = make_algo()
    written = mgr.save(config=make_cfg(), algo=algo, frames=100)
    assert written == [mgr.latest_path]
    assert mgr.latest_path.exists()


def test_checkpoint_carries_the_config_and_the_counters(tmp_path):
    mgr = CheckpointManager(tmp_path, CheckpointConfig())
    algo = make_algo()
    mgr.save(config=make_cfg(), algo=algo, frames=123)
    payload = load_checkpoint(mgr.latest_path)
    assert payload["frames"] == 123
    assert payload["global_step"] == algo.global_step
    assert payload["config"]["run_name"] == "ckpt-run"
    # nothing in this codebase reads torch's global RNG state back out of a
    # checkpoint -- DQN.state_dict() already carries everything the learner's
    # own math depends on -- so the payload must not carry it either
    assert "rng" not in payload


def test_best_is_written_only_when_the_return_improves(tmp_path):
    mgr = CheckpointManager(tmp_path, CheckpointConfig())
    algo, cfg = make_algo(), make_cfg()
    assert mgr.best_path in mgr.save(config=cfg, algo=algo, frames=1, eval_return=10.0)
    assert mgr.best_path not in mgr.save(config=cfg, algo=algo, frames=2, eval_return=5.0)
    assert mgr.best_path in mgr.save(config=cfg, algo=algo, frames=3, eval_return=20.0)
    assert mgr.best_return == 20.0


def test_best_return_is_persisted_in_the_payload(tmp_path):
    mgr = CheckpointManager(tmp_path, CheckpointConfig())
    algo, cfg = make_algo(), make_cfg()
    mgr.save(config=cfg, algo=algo, frames=1, eval_return=42.0)
    assert load_checkpoint(mgr.latest_path)["best_return"] == 42.0


def test_a_manager_seeded_with_a_prior_best_return_does_not_overwrite_a_better_best(tmp_path):
    # Simulates cmd_resume: a fresh CheckpointManager pointed at a directory
    # whose best.pt already reflects a return of 50.0. Without seeding
    # best_return from the checkpoint, this constructor defaults to -inf and
    # the very next eval -- however bad -- would beat it and overwrite best.pt.
    mgr = CheckpointManager(tmp_path, CheckpointConfig(), best_return=50.0)
    algo, cfg = make_algo(), make_cfg()
    written = mgr.save(config=cfg, algo=algo, frames=1, eval_return=10.0)
    assert mgr.best_path not in written
    assert mgr.best_return == 50.0

    # a genuine improvement still updates it
    written = mgr.save(config=cfg, algo=algo, frames=2, eval_return=75.0)
    assert mgr.best_path in written
    assert mgr.best_return == 75.0


def test_only_two_checkpoints_are_ever_retained(tmp_path):
    mgr = CheckpointManager(tmp_path, CheckpointConfig())
    algo, cfg = make_algo(), make_cfg()
    for i in range(5):
        mgr.save(config=cfg, algo=algo, frames=i, eval_return=float(i))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt", "latest.pt"]


def test_buffer_is_omitted_unless_include_buffer_is_set(tmp_path):
    buf = ReplayBuffer(100, (4,))
    buf.add(
        Transitions(
            obs=np.zeros((3, 4), np.float32),
            actions=np.zeros(3, np.int64),
            returns=np.zeros(3, np.float32),
            next_obs=np.zeros((3, 4), np.float32),
            terminated=np.zeros(3, bool),
            steps=np.ones(3, np.int32),
        )
    )
    mgr = CheckpointManager(tmp_path / "a", CheckpointConfig(include_buffer=False))
    mgr.save(config=make_cfg(), algo=make_algo(), frames=1, buffer=buf)
    assert load_checkpoint(mgr.latest_path)["buffer"] is None

    mgr2 = CheckpointManager(tmp_path / "b", CheckpointConfig(include_buffer=True))
    mgr2.save(config=make_cfg(include_buffer=True), algo=make_algo(), frames=1, buffer=buf)
    assert load_checkpoint(mgr2.latest_path)["buffer"]["size"] == 3


def test_a_checkpoint_restores_a_fresh_algorithm(tmp_path):
    mgr = CheckpointManager(tmp_path, CheckpointConfig())
    algo = make_algo()
    with torch.no_grad():
        for p in algo.net.parameters():
            p.fill_(0.25)
    algo._global_step = 42
    mgr.save(config=make_cfg(), algo=algo, frames=99)

    restored = make_algo()
    restored.load_state_dict(load_checkpoint(mgr.latest_path)["algo"])
    assert restored.global_step == 42
    assert all(torch.allclose(p, torch.full_like(p, 0.25)) for p in restored.net.parameters())
