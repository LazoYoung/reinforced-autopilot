from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.conftest import CountingEnv, RecordingTracker

from aero_train.algo.dqn import DQN
from aero_train.checkpoint import CheckpointManager, load_checkpoint
from aero_train.config import (
    CheckpointConfig,
    DQNConfig,
    EnvConfig,
    RolloutConfig,
    TrackingConfig,
    TrainConfig,
)
from aero_train.envs import make_env_fn
from aero_train.rollout import LocalRolloutEngine
from aero_train.trainer import Trainer, evaluate_policy

SMALL = DQNConfig(
    hidden_sizes=(8,), buffer_size=200, warmup_frames=16, batch_size=8,
    target_sync_interval=5, updates_per_iter=2, epsilon_decay_frames=50,
)


def segment_batch(seed=0, T=10, B=2):
    from aero_train.primitives import Segment

    rng = np.random.default_rng(seed)
    return Segment(
        obs=rng.normal(size=(T + 1, B, 1)).astype(np.float32),
        actions=rng.integers(0, 2, (T, B)).astype(np.int64),
        rewards=np.ones((T, B), np.float32),
        terminated=np.zeros((T, B), bool),
        truncated=np.zeros((T, B), bool),
        valid=np.ones((T, B), bool),
    )


def test_a_restored_learner_produces_bit_identical_updates(tmp_path):
    """Resume is exact for the learner. This asserts exactly that claim."""
    original = DQN(obs_shape=(1,), n_actions=2, cfg=SMALL, seed=0)
    original.observe(segment_batch(seed=1))
    original.update()

    restored = DQN(obs_shape=(1,), n_actions=2, cfg=SMALL, seed=999)
    restored.observe(segment_batch(seed=1))
    restored.load_state_dict(original.state_dict())

    fresh = segment_batch(seed=2)
    original.observe(fresh)
    restored.observe(fresh)
    a, b = original.update(), restored.update()

    assert a["loss"] == pytest.approx(b["loss"])
    assert a["q_mean"] == pytest.approx(b["q_mean"])
    assert original.global_step == restored.global_step
    for p, q in zip(original.net.parameters(), restored.net.parameters()):
        assert torch.equal(p, q)


def test_a_checkpoint_round_trip_preserves_learner_state(tmp_path):
    cfg = TrainConfig(
        run_name="resume", env=EnvConfig(id="CartPole-v1"), algo=SMALL,
        total_steps=64,
        rollout=RolloutConfig(envs_per_worker=2, frames_per_iter=16),
        tracking=TrackingConfig(enabled=False),
        checkpoint=CheckpointConfig(
            dir=str(tmp_path), interval_iters=1, eval_interval_iters=100,
            include_buffer=True,
        ),
    )
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=SMALL, seed=0)
    engine = LocalRolloutEngine(lambda: CountingEnv(5), algo.make_agent(), 2, 0)
    mgr = CheckpointManager(tmp_path, cfg.checkpoint)
    Trainer(cfg, algo, engine, RecordingTracker(), mgr).run()

    payload = load_checkpoint(mgr.latest_path)
    restored = DQN(obs_shape=(1,), n_actions=2, cfg=SMALL, seed=42)
    restored.load_state_dict(payload["algo"])
    restored.buffer.load_state_dict(payload["buffer"])
    assert restored.global_step == algo.global_step
    assert restored.frames == algo.frames

    # both buffers now hold the same transitions AND the same sampler state, so
    # the next update must be identical -- which is the "exact for the learner"
    # claim, asserted rather than assumed
    fresh = segment_batch(seed=7)
    algo.observe(fresh)
    restored.observe(fresh)
    assert algo.update()["loss"] == pytest.approx(restored.update()["loss"])


@pytest.mark.slow
def test_dqn_solves_cartpole(tmp_path):
    """The end-to-end proof that the learning stack works at all.

    Asserts on the run's *best* checkpoint, not the final in-memory learner.
    Vanilla DQN on CartPole finds a high-return policy and then does not hold
    it -- that is why CheckpointManager retains a best-by-eval-return
    checkpoint alongside the latest one, and it is why the real workflow is
    `autopilot run` followed by `autopilot eval <best checkpoint>`. Asserting
    on the final learner would test "vanilla DQN converges stably on CartPole
    on a single seed," which is false and not this test's claim. The claim is
    that the best checkpoint the run produces, reloaded the way `autopilot
    eval` reloads it, clears the bar -- so this test exercises save -> load ->
    eval, not just train -> eval.

    Deselected by default (`-m 'not slow'`). Run it with:
        uv run pytest -m slow packages/aero-train/tests/test_integration.py
    """
    algo_cfg = DQNConfig(
        hidden_sizes=(128, 128), lr=5e-4, gamma=0.99, n_step=1,
        buffer_size=50_000, warmup_frames=1_000, batch_size=64,
        target_sync_interval=500, updates_per_iter=32,
        epsilon_start=1.0, epsilon_final=0.02, epsilon_decay_frames=20_000,
    )
    cfg = TrainConfig(
        run_name="cartpole-integration",
        env=EnvConfig(id="CartPole-v1"),
        algo=algo_cfg,
        seed=0,
        total_steps=60_000,
        rollout=RolloutConfig(engine="local", envs_per_worker=8, frames_per_iter=128),
        tracking=TrackingConfig(enabled=False),
        checkpoint=CheckpointConfig(dir=str(tmp_path), interval_iters=50, eval_interval_iters=50),
    )
    env_fn = make_env_fn(cfg.env)
    probe = env_fn()
    algo = DQN(
        obs_shape=probe.observation_space.shape,
        n_actions=int(probe.action_space.n),
        cfg=algo_cfg, device=cfg.device, seed=cfg.seed,
    )
    probe.close()

    engine = LocalRolloutEngine(env_fn, algo.make_agent(), cfg.rollout.envs_per_worker, cfg.seed)
    tracker = RecordingTracker()
    checkpoints = CheckpointManager(tmp_path, cfg.checkpoint)
    Trainer(cfg, algo, engine, tracker, checkpoints, eval_env_fn=env_fn).run()

    assert checkpoints.best_path.exists(), (
        "no best checkpoint was written -- eval_interval_iters never fired during "
        f"the run (total_steps={cfg.total_steps}, frames_per_iter={cfg.rollout.frames_per_iter})"
    )
    payload = load_checkpoint(checkpoints.best_path)
    best = DQN(
        obs_shape=algo.obs_shape, n_actions=algo.n_actions,
        cfg=algo_cfg, device=cfg.device, seed=cfg.seed,
    )
    best.load_state_dict(payload["algo"])

    # 20 episodes, independent of the 5-episode eval that selected this
    # checkpoint -- selection was made on a noisy small sample, so this is
    # what actually makes the reported number trustworthy
    final = evaluate_policy(best, env_fn, episodes=20, seed=1234)
    assert final >= 195.0, (
        f"DQN's best checkpoint reached only {final:.1f} on CartPole-v1 after "
        f"{cfg.total_steps} frames; the learning stack is not working end to end"
    )
