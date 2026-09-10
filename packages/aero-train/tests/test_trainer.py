from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from tests.conftest import CountingEnv, RecordingTracker, counting_env_fn

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


class _CountingParamsTracker(RecordingTracker):
    """RecordingTracker.params is a dict that a repeated call simply
    overwrites, so it cannot tell "logged once" from "logged every iteration".
    This subclass counts calls instead, without touching conftest.py.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_params_calls = 0

    def log_params(self, cfg) -> None:
        self.log_params_calls += 1
        super().log_params(cfg)


def test_params_are_logged_exactly_once(tmp_path):
    cfg = make_cfg()
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=cfg.algo, seed=cfg.seed)
    engine = LocalRolloutEngine(
        env_fn(), algo.make_agent(), cfg.rollout.envs_per_worker, cfg.seed
    )
    tracker = _CountingParamsTracker()
    mgr = CheckpointManager(tmp_path, cfg.checkpoint)
    trainer = Trainer(cfg, algo, engine, tracker, mgr)
    trainer.run()
    assert tracker.log_params_calls == 1
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
    # iteration 1 inside warmup.
    cfg = make_cfg(total_steps=17)
    trainer, algo, tracker, _ = make_trainer(tmp_path, cfg)
    trainer.run()
    # Scope the assertion to the warmup iteration itself. A filter over the
    # whole run (metric_series) is satisfied as soon as ANY iteration logs
    # episode/count -- with this config the run has 2 iterations and only the
    # first is in warmup, so a bug that skips logging during warmup specifically
    # would still leave a whole-run filter green via iteration 2 alone.
    warmup_metrics, warmup_step = tracker.metrics[0]
    assert warmup_step == 0          # no gradient step taken yet -- confirms warmup
    assert "loss" not in warmup_metrics  # confirms update() really returned None here
    assert warmup_metrics.get("episode/count")


def test_only_valid_frames_count_towards_the_budget(tmp_path):
    # short episodes produce autoreset rows, which must not count as frames
    trainer, algo, _, _ = make_trainer(tmp_path, episode_len=5)
    trainer.run()
    assert trainer.frames == algo.frames
    assert trainer.frames < trainer.iteration * 16


def test_progress_metrics_track_frames_and_iteration(tmp_path):
    # long episodes, so frames/iteration progression is exact and predictable
    trainer, _, tracker, _ = make_trainer(tmp_path, episode_len=1000)
    trainer.run()
    assert tracker.metric_series("progress/frames") == [16.0, 32.0, 48.0, 64.0]
    assert tracker.metric_series("progress/iteration") == [1.0, 2.0, 3.0, 4.0]


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


def test_the_final_iteration_is_checkpointed_even_off_cadence(tmp_path):
    # long episodes, so the iteration count (4) is exact and predictable.
    # interval_iters=3 divides neither into nor evenly by 4, so cadence alone
    # writes a checkpoint at iteration 3 and never again -- without an
    # unconditional end-of-run save, latest.pt would be stuck at iteration 3's
    # state (48 frames) even though the run actually collected 64.
    cfg = make_cfg(interval_iters=3, eval_interval_iters=100)
    trainer, algo, tracker, mgr = make_trainer(tmp_path, cfg, episode_len=1000)
    trainer.run()
    assert mgr.latest_path.exists()
    payload = load_checkpoint(mgr.latest_path)
    assert payload["frames"] == trainer.frames == 64
    assert payload["global_step"] == algo.global_step
    # one write at the cadence hit (iteration 3), one forced at the run's end
    # (iteration 4) -- a bug that skips the forced save leaves this at 1
    assert tracker.artifacts.count(mgr.latest_path) == 2


def test_a_run_shorter_than_the_checkpoint_interval_still_saves_one(tmp_path):
    # interval_iters (10) exceeds the run's own iteration count (2 -- 17 frames
    # at 16/iteration), so cadence never fires at all: before the end-of-run
    # save this configuration wrote nothing and exited 0.
    cfg = make_cfg(total_steps=17, interval_iters=10, eval_interval_iters=100)
    trainer, algo, tracker, mgr = make_trainer(tmp_path, cfg, episode_len=1000)
    trainer.run()
    assert mgr.latest_path.exists()
    assert load_checkpoint(mgr.latest_path)["frames"] == trainer.frames
    assert mgr.latest_path in tracker.artifacts


def test_evaluation_drives_the_best_checkpoint(tmp_path):
    # long episodes, so the iteration count (4) is exact and predictable
    cfg = make_cfg(interval_iters=100, eval_interval_iters=2)
    trainer, algo, tracker, mgr = make_trainer(
        tmp_path, cfg, episode_len=1000, eval_env_fn=env_fn()
    )
    trainer.run()
    assert mgr.best_path.exists()
    assert mgr.best_return > float("-inf")
    # exactly at iterations 2 and 4 -- an eval-every-iteration bug would still
    # produce a best checkpoint, so the count is what catches the wrong cadence
    assert len(tracker.metric_series("eval/return_mean")) == 2
    # the eval log call is a second, independent log_metrics call site from the
    # main per-iteration one; it must also key off the learner's global_step
    # (2 updates_per_iter x iterations 2 and 4 = 4 and 8), not the iteration
    # index -- the two call sites can drift independently, so both need pinning
    eval_steps = [step for m, step in tracker.metrics if "eval/return_mean" in m]
    assert eval_steps == [4, 8]


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
    report = evaluate_policy(algo, env_fn(episode_len=5), episodes=3)
    assert report.mean_return == pytest.approx(5.0)   # CountingEnv pays 1.0 per step
    assert float(algo.last_sync[SELECTOR_PREFIX + "epsilon"]) == 0.0


class _NeverEndingEnv(gym.Env):
    """An env with no time limit, purely to prove evaluate_policy's step cap
    trips. Not in conftest.py: every real scenario has a time limit, so no
    other test needs an env that lacks one.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], np.float32), {}

    def step(self, action):
        return np.array([0.0], np.float32), 0.0, False, False, {}


def test_evaluate_policy_caps_a_runaway_episode():
    # evaluate_policy is called from Trainer._maybe_checkpoint, so an eval env
    # missing a time limit must not hang the whole training run silently
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=ALGO, seed=0)
    with pytest.raises(RuntimeError, match="time limit"):
        evaluate_policy(algo, _NeverEndingEnv, episodes=1, max_steps=5)


def test_eval_reports_length_and_termination_not_just_return():
    """Return alone conflates 'flew a long time badly' with 'flew briefly well',
    because return is roughly length times quality."""
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=ALGO, seed=0)
    report = evaluate_policy(algo, env_fn(episode_len=4), episodes=3, seed=0)
    assert report.mean_return == pytest.approx(4.0)
    assert report.mean_length == pytest.approx(4.0)
    assert report.terminated_frac == pytest.approx(1.0)
    assert report.returns == (4.0, 4.0, 4.0)
    assert report.lengths == (4, 4, 4)


def test_a_truncated_episode_is_not_counted_as_terminated():
    """terminated_frac is a distinct fact from length -- for a flight scenario
    it inverts, since terminated means the objective band was left."""
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=ALGO, seed=0)
    report = evaluate_policy(
        algo, counting_env_fn(episode_len=4, truncate=True), episodes=2, seed=0
    )
    assert report.terminated_frac == pytest.approx(0.0)
    assert report.mean_length == pytest.approx(4.0)


class _DoubleRewardEnv(gym.Env):
    """CountingEnv pays 1.0/step, so mean_return == mean_length whenever it is
    the eval env -- a build that swaps the two EvalReport fields (or swaps
    which field a value is logged under) would pass unnoticed. This env pays
    2.0/step over a fixed length instead, so return and length land on
    different numbers and a swap is visible. Not in conftest.py: no other
    test needs a non-unit reward.
    """

    metadata = {"render_modes": []}
    episode_len = 3

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        return np.array([float(self.t)], np.float32), 2.0, done, False, {}


def test_the_trainer_logs_every_eval_metric(tmp_path):
    # _DoubleRewardEnv (not CountingEnv): CountingEnv's constant 1.0/step
    # reward makes eval/return_mean == eval/length_mean by coincidence, so a
    # build that logs each under the other's key would pass this test anyway.
    # Distinct 6.0 vs 3.0 values catch that.
    cfg = make_cfg(eval_interval_iters=1, eval_episodes=2)
    trainer, _, tracker, _ = make_trainer(
        tmp_path, cfg, episode_len=1000, eval_env_fn=_DoubleRewardEnv
    )
    trainer.run()
    assert tracker.metric_series("eval/return_mean") == pytest.approx([6.0] * 4)
    assert tracker.metric_series("eval/length_mean") == pytest.approx([3.0] * 4)
    assert tracker.metric_series("eval/terminated_frac") == pytest.approx([1.0] * 4)


def test_mean_return_and_mean_length_land_on_the_right_fields():
    """Distinct return (6.0) and length (3.0) values, so a build that writes
    each into the other's EvalReport field fails this test specifically."""
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=ALGO, seed=0)
    report = evaluate_policy(algo, _DoubleRewardEnv, episodes=2, seed=0)
    assert report.mean_return == pytest.approx(6.0)
    assert report.mean_length == pytest.approx(3.0)


class _VaryingQualityEnv(gym.Env):
    """Episodes that differ in BOTH length and per-step reward.

    Every other eval env here is fixed-length, which makes the mean of the
    per-episode ratios and the ratio of the means agree -- so neither could
    tell the two definitions of return-per-step apart. This one pays 1.0/step
    for 4 steps, then 0.2 for a single step:

        mean of ratios   = (4.0/4 + 0.2/1) / 2      = 0.6
        ratio of means   = ((4.0 + 0.2)/2) / ((4 + 1)/2) = 0.84

    Not in conftest.py: no other test needs episodes to vary between resets.
    """

    metadata = {"render_modes": []}
    plan = ((4, 1.0), (1, 0.2))   # (length, reward per step) per episode

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.episode = -1
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        length, reward = self.plan[self.episode % len(self.plan)]
        self.t += 1
        return np.array([float(self.t)], np.float32), reward, self.t >= length, False, {}


def test_return_per_step_averages_episodes_not_steps():
    """Return is roughly length times quality, so return-per-step is the
    quality half -- but only if each episode counts once. Averaging over steps
    instead (sum of returns over sum of lengths) lets one long episode drown
    out a short one, and gives 0.84 here rather than 0.6.
    """
    algo = DQN(obs_shape=(1,), n_actions=2, cfg=ALGO, seed=0)
    report = evaluate_policy(algo, _VaryingQualityEnv, episodes=2, seed=0)
    assert report.mean_return == pytest.approx(2.1)
    assert report.mean_length == pytest.approx(2.5)
    assert report.return_per_step == pytest.approx(0.6)
    assert report.return_per_step != pytest.approx(0.84)


def test_the_trainer_logs_return_per_step(tmp_path):
    # _DoubleRewardEnv pays 2.0/step over 3 steps, so return (6.0), length
    # (3.0) and return-per-step (2.0) are three distinct numbers: a build that
    # logs any one of them under another's key fails here.
    cfg = make_cfg(eval_interval_iters=1, eval_episodes=2)
    trainer, _, tracker, _ = make_trainer(
        tmp_path, cfg, episode_len=1000, eval_env_fn=_DoubleRewardEnv
    )
    trainer.run()
    assert tracker.metric_series("eval/return_per_step") == pytest.approx([2.0] * 4)
