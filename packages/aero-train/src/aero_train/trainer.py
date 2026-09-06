"""The training loop: everything cross-cutting and algorithm-independent.

Broadcast weights, collect, observe, update, log, checkpoint, evaluate. Written
once. It learns nothing algorithm-specific because Algorithm.update() hands back
a metrics dict that goes straight to the tracker.
"""

from __future__ import annotations

import time
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from .algo.base import Algorithm
from .checkpoint import CheckpointManager
from .config import TrainConfig
from .primitives import SELECTOR_PREFIX, EpisodeResult
from .rollout import RolloutEngine
from .tracking import Tracker


DEFAULT_EVAL_MAX_STEPS = 10_000
"""Generous cap on a single evaluation episode.

Every scenario in this phase carries its own time limit, so this should never
trip. It exists because evaluate_policy is called from
Trainer._maybe_checkpoint: an eval env missing a time limit would otherwise
spin `while not done` forever and silently hang the whole training run, with
no diagnostic pointing at the eval env being the cause.
"""


def evaluate_policy(
    algo: Algorithm,
    env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int = 0,
    max_steps: int = DEFAULT_EVAL_MAX_STEPS,
) -> float:
    """Mean return over `episodes` greedy episodes in a single environment.

    Greediness is requested through the same actor_state_dict convention the
    workers use -- exploration parameters under the `selector.` prefix -- rather
    than by reaching into the agent, so it keeps working as algorithms are added.
    """
    agent = algo.make_agent()
    agent.sync_weights(
        {**algo.actor_state_dict(), SELECTOR_PREFIX + "epsilon": np.asarray(0.0, np.float32)}
    )
    env = env_fn()
    returns: list[float] = []
    try:
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            total, done, steps = 0.0, False, 0
            while not done:
                actions, _ = agent(np.asarray(obs, np.float32)[None, ...])
                obs, reward, terminated, truncated, _ = env.step(actions[0].item())
                total += float(reward)
                steps += 1
                done = bool(terminated or truncated)
                if not done and steps >= max_steps:
                    raise RuntimeError(
                        f"evaluate_policy: episode {i} ran {steps} steps without "
                        f"terminating or truncating (cap max_steps={max_steps}); "
                        "the eval env likely has no time limit"
                    )
            returns.append(total)
    finally:
        env.close()
    return float(np.mean(returns))


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        algo: Algorithm,
        engine: RolloutEngine,
        tracker: Tracker,
        checkpoints: CheckpointManager,
        eval_env_fn: Callable[[], gym.Env] | None = None,
    ) -> None:
        self.cfg = cfg
        self.algo = algo
        self.engine = engine
        self.tracker = tracker
        self.checkpoints = checkpoints
        self.eval_env_fn = eval_env_fn
        self.frames = 0
        self.iteration = 0

    def run(self) -> None:
        self.tracker.log_params(self.cfg)
        try:
            # True until the loop runs at least one iteration whose own cadence
            # (interval_iters / eval_interval_iters) did NOT already checkpoint --
            # tracks whether the final iteration's state still needs saving below.
            final_iteration_checkpointed = True
            while self.frames < self.cfg.total_steps:
                metrics = self._iterate()
                self.tracker.log_metrics(metrics, step=self.algo.global_step)
                final_iteration_checkpointed = self._maybe_checkpoint()
            if not final_iteration_checkpointed:
                # interval_iters has no reason to land on a run's last iteration,
                # so without this the tail of every run -- and the entirety of a
                # run shorter than interval_iters -- is silently discarded. Force
                # an eval alongside it (when there is an eval env) so the
                # best-checkpoint bookkeeping gets a chance to see the final
                # policy too, not just whichever interval iteration preceded it.
                self._checkpoint(eval_due=self.eval_env_fn is not None)
        finally:
            self.engine.close()
            self.tracker.close()

    def _iterate(self) -> dict[str, float]:
        started = time.perf_counter()
        # ray.put-style single broadcast happens inside the engine; at ~20 KB it
        # costs microseconds, which is what makes exactly-on-policy affordable
        self.engine.sync_weights(self.algo.actor_state_dict())
        segment, episodes = self.engine.collect(self.cfg.rollout.frames_per_iter)
        collected = time.perf_counter()

        self.algo.observe(segment)
        algo_metrics = self.algo.update() or {}
        updated = time.perf_counter()

        self.frames += int(segment.valid.sum())
        self.iteration += 1
        rollout_s = collected - started
        update_s = updated - collected
        return {
            **algo_metrics,
            **self._episode_metrics(episodes),
            "throughput/rollout_s": rollout_s,
            "throughput/update_s": update_s,
            "throughput/fps": float(segment.valid.sum()) / max(updated - started, 1e-9),
            "progress/frames": float(self.frames),
            "progress/iteration": float(self.iteration),
        }

    @staticmethod
    def _episode_metrics(episodes: list[EpisodeResult]) -> dict[str, float]:
        """Termination reason is logged, not just the mean return.

        An episode that ends by leaving the flight envelope is a different event
        from one that hits the step limit, and a mean return alone hides which.
        """
        if not episodes:
            return {"episode/count": 0.0}
        returns = [e.ret for e in episodes]
        return {
            "episode/count": float(len(episodes)),
            "episode/return_mean": float(np.mean(returns)),
            "episode/return_min": float(np.min(returns)),
            "episode/return_max": float(np.max(returns)),
            "episode/length_mean": float(np.mean([e.length for e in episodes])),
            "episode/terminated_frac": float(np.mean([e.terminated for e in episodes])),
        }

    def _maybe_checkpoint(self) -> bool:
        """Save if this iteration's cadence calls for it. Returns whether it did.

        The return value is how run() knows whether the final iteration already
        covered the end-of-run save, so that save is not repeated on top of an
        identical, just-written state.
        """
        ckpt = self.cfg.checkpoint
        due = self.iteration % ckpt.interval_iters == 0
        eval_due = self.eval_env_fn is not None and self.iteration % ckpt.eval_interval_iters == 0
        if not (due or eval_due):
            return False
        self._checkpoint(eval_due=eval_due)
        return True

    def _checkpoint(self, *, eval_due: bool) -> None:
        eval_return = None
        if eval_due:
            eval_return = evaluate_policy(
                self.algo, self.eval_env_fn, self.cfg.checkpoint.eval_episodes, seed=self.cfg.seed
            )
            self.tracker.log_metrics(
                {"eval/return_mean": eval_return}, step=self.algo.global_step
            )
        written = self.checkpoints.save(
            config=self.cfg,
            algo=self.algo,
            frames=self.frames,
            rng_state={"torch": torch.get_rng_state(), "seed": self.cfg.seed},
            buffer=getattr(self.algo, "buffer", None),
            eval_return=eval_return,
        )
        for path in written:
            self.tracker.log_artifact(path)
