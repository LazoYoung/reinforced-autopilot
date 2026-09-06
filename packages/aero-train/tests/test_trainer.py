from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from tests.conftest import CountingEnv, RecordingTracker

from aero_train.algo.dqn import DQN
from aero_train.checkpoint import CheckpointManager, load_checkpoint
from aero_train.config import (
    CheckpointConfig,
    DQNConfig,
    EnvConfig,
    RolloutConfig,
    TrainConfig,
)
from aero_train.primitives import SELECTOR_PREFIX
from aero_train.rollout import LocalRolloutEngine, RolloutEngine
from aero_train.trainer import Trainer, evaluate_policy

ALGO = DQNConfig(
    hidden_sizes=(8,), buffer_size=200, warmup_frames=16, batch_size=8,
    target_sync_interval=5, updates_per_iter=2, epsilon_decay_frames=50,
)


def env_fn(episode_len=5):
    return lambda: CountingEnv(episode_len=episode_len)


def make_cfg(total_steps=64, **ckpt):
    return TrainConfig(
        run_name="trainer-run",
        env=EnvConfig(id="CartPole-v1"),
        algo=ALGO,
        seed=0,
        total_steps=total_steps,
        rollout=RolloutConfig(engine="local", envs_per_worker=2, frames_per_iter=16),
        checkpoint=CheckpointConfig(**ckpt),
    )


class SpyEngine(RolloutEngine):
    """Wraps a real engine and records every sync_weights payload.

    A test asserting only that the run completed cannot tell a broadcast every
    iteration from a broadcast that happened once (or never): the run still
    finishes, logs metrics and writes checkpoints either way, it just never
    improves. This spy makes the broadcast itself observable.
    """

    def __init__(self, inner: RolloutEngine) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    def sync_weights(self, actor_state_dict) -> None:
        self.calls.append(actor_state_dict)
        self.inner.sync_weights(actor_state_dict)

    def collect(self, n_frames: int):
        return self.inner.collect(n_frames)

    def close(self) -> None:
        self.inner.close()


def make_trainer(tmp_path, cfg=None, episode_len=5, **kwargs):
    cfg = cfg or make_cfg()
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=cfg.algo, seed=cfg.seed)
    engine = LocalRolloutEngine(
        env_fn(episode_len), algo.make_agent(), cfg.rollout.envs_per_worker, cfg.seed
    )
    tracker = RecordingTracker()
    mgr = CheckpointManager(tmp_path, cfg.checkpoint)
    return Trainer(cfg, algo, engine, tracker, mgr, **kwargs), algo, tracker, mgr


def test_run_stops_once_the_frame_budget_is_met(tmp_path):
    # long episodes, so no autoreset rows are dropped and the arithmetic is exact
    trainer, _, _, _ = make_trainer(tmp_path, episode_len=1000)
    trainer.run()
    assert trainer.frames == 64
    assert trainer.iteration == 4          # 64 frames / 16 per iteration


def test_weights_are_broadcast_to_the_engine_every_iteration(tmp_path):
    # long episodes, so iteration count is the exact, predictable 4
    cfg = make_cfg()
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=cfg.algo, seed=cfg.seed)
    inner = LocalRolloutEngine(
        env_fn(episode_len=1000), algo.make_agent(), cfg.rollout.envs_per_worker, cfg.seed
    )
    engine = SpyEngine(inner)
    tracker = RecordingTracker()
    mgr = CheckpointManager(tmp_path, cfg.checkpoint)
    trainer = Trainer(cfg, algo, engine, tracker, mgr)
    trainer.run()

    assert len(engine.calls) == trainer.iteration == 4
    epsilons = [float(call["selector.epsilon"]) for call in engine.calls]
    # a stale (broadcast-once) weight payload would repeat the same epsilon
    # every call; the real schedule decays frame-by-frame, so consecutive
    # broadcasts must differ
    assert len(set(epsilons)) == len(epsilons)
    assert epsilons == sorted(epsilons, reverse=True)
    for call in engine.calls:
        assert call.keys() == algo.actor_state_dict().keys()


def test_params_are_logged_exactly_once(tmp_path):
    trainer, _, tracker, _ = make_trainer(tmp_path)
    trainer.run()
    assert tracker.params["run_name"] == "trainer-run"


def test_algorithm_metrics_reach_the_tracker(tmp_path):
    trainer, _, tracker, _ = make_trainer(tmp_path)
    trainer.run()
    assert tracker.metric_series("loss")
    assert tracker.metric_series("epsilon")


def test_episode_metrics_are_logged_with_the_termination_reason(tmp_path):
    trainer, _, tracker, _ = make_trainer(tmp_path)
    trainer.run()
    assert tracker.metric_series("episode/return_mean")
    assert tracker.metric_series("episode/count")
    # CountingEnv terminates rather than truncating, so every episode is terminal
    assert all(v == 1.0 for v in tracker.metric_series("episode/terminated_frac"))


def test_throughput_is_split_into_rollout_and_update_time(tmp_path, monkeypatch):
    # Presence alone doesn't prove the two are split rather than double-counted
    # (e.g. rollout_s wrongly measured as collect-through-update time): a fake,
    # strictly increasing clock makes the exact split checkable. Three
    # perf_counter() calls per iteration -- started, collected, updated -- times
    # four iterations (long episodes, so the count is the exact, predictable 4).
    times = iter([0, 1, 3, 3, 4, 7, 7, 8, 12, 12, 13, 18])
    monkeypatch.setattr("aero_train.trainer.time.perf_counter", lambda: next(times))
    trainer, _, tracker, _ = make_trainer(tmp_path, episode_len=1000)
    trainer.run()
    assert tracker.metric_series("throughput/rollout_s") == [1.0, 1.0, 1.0, 1.0]
    assert tracker.metric_series("throughput/update_s") == [2.0, 3.0, 4.0, 5.0]
    assert tracker.metric_series("throughput/fps") == pytest.approx(
        [16 / 3, 16 / 4, 16 / 5, 16 / 6]
    )


def test_metrics_are_logged_against_the_learners_global_step(tmp_path):
    trainer, algo, tracker, _ = make_trainer(tmp_path)
    trainer.run()
    assert tracker.metrics[-1][1] == algo.global_step


def test_iterations_before_warmup_still_log_episodes(tmp_path):
    # update() returns None while the buffer fills; the loop must not skip logging.
    # total_steps must exceed warmup_frames (16) or TrainConfig.__post_init__
    # rejects the config outright -- 17 is the smallest value that still keeps
    # the run mostly inside warmup.
    cfg = make_cfg(total_steps=17)
    trainer, _, tracker, _ = make_trainer(tmp_path, cfg)
    trainer.run()
    assert tracker.metric_series("episode/count")


def test_only_valid_frames_count_towards_the_budget(tmp_path):
    # short episodes produce autoreset rows, which must not count as frames
    trainer, algo, _, _ = make_trainer(tmp_path, episode_len=5)
    trainer.run()
    assert trainer.frames == algo.frames
    assert trainer.frames < trainer.iteration * 16


def test_a_checkpoint_is_written_on_the_configured_interval(tmp_path):
    # long episodes, so the iteration count divides evenly by interval_iters and
    # the last checkpoint written coincides with the run's final global_step
    cfg = make_cfg(interval_iters=2, eval_interval_iters=100)
    trainer, algo, tracker, mgr = make_trainer(tmp_path, cfg, episode_len=1000)
    trainer.run()
    assert mgr.latest_path.exists()
    assert load_checkpoint(mgr.latest_path)["global_step"] == algo.global_step
    assert mgr.latest_path in tracker.artifacts
    # exactly at iterations 2 and 4, not on every iteration -- a save-every-
    # iteration bug still leaves the final checkpoint's global_step matching
    # (it is always the last write that wins), so the count is what catches it
    assert tracker.artifacts.count(mgr.latest_path) == 2


def test_evaluation_drives_the_best_checkpoint(tmp_path):
    # long episodes, so the iteration count (4) is exact and predictable
    cfg = make_cfg(interval_iters=100, eval_interval_iters=2)
    trainer, _, tracker, mgr = make_trainer(
        tmp_path, cfg, episode_len=1000, eval_env_fn=env_fn()
    )
    trainer.run()
    assert mgr.best_path.exists()
    assert mgr.best_return > float("-inf")
    # exactly at iterations 2 and 4 -- an eval-every-iteration bug would still
    # produce a best checkpoint, so the count is what catches the wrong cadence
    assert len(tracker.metric_series("eval/return_mean")) == 2


def test_tracker_and_engine_are_closed(tmp_path):
    trainer, _, tracker, _ = make_trainer(tmp_path)
    trainer.run()
    assert tracker.closed
    # LocalRolloutEngine.close() forwards to the underlying SyncVectorEnv,
    # which is the only observable signal that the engine half of run()'s
    # `finally` actually fired
    assert trainer.engine.source.env.closed


class _RecordingDQN(DQN):
    """Captures the actor_state_dict handed to the worker-side agent.

    CountingEnv's reward is action-independent (always 1.0, whatever the
    action), so a mean-return assertion alone cannot tell a greedy evaluation
    from an exploring one -- both average 5.0 over 5-step episodes. Recording
    the sync payload is what actually verifies evaluate_policy forces epsilon
    to 0 rather than reusing the training schedule's current value.
    """

    def make_agent(self):
        agent = super().make_agent()
        self.last_sync: dict | None = None
        real_sync = agent.sync_weights

        def sync_weights(actor_state_dict):
            self.last_sync = actor_state_dict
            real_sync(actor_state_dict)

        agent.sync_weights = sync_weights
        return agent


def test_evaluate_policy_is_greedy_and_returns_a_mean_return():
    # epsilon_start defaults to 1.0 and this algo never trains, so a fully
    # random (non-greedy) evaluation would also average 5.0 here -- the
    # explicit sync-payload check below is what actually proves greediness
    cfg = ALGO
    algo = _RecordingDQN(obs_shape=(1,), n_actions=2, cfg=cfg, seed=0)
    mean_return = evaluate_policy(algo, env_fn(episode_len=5), episodes=3)
    assert mean_return == pytest.approx(5.0)   # CountingEnv pays 1.0 per step
    assert float(algo.last_sync[SELECTOR_PREFIX + "epsilon"]) == 0.0
