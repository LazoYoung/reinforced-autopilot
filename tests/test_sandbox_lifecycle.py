"""One flow, not six tests: start, checkpoint, kill, resume, finish, eval,
with a scenario that draws a different starting situation every episode.

Mirrors packages/aero-train/tests/test_flight_lifecycle.py, which flies the
zero-jitter default. This one exists because jitter is the feature most likely
to break a boundary the component tests do not own: eight sub-envs drawing
from one seeded root, an eval whose episodes must stay comparable across a
whole run, and a resume that must not silently re-randomize.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import yaml

jsbsim = pytest.importorskip("jsbsim")

from aero_train import cli
from aero_train.checkpoint import load_checkpoint
from aero_train.config import EnvConfig
from aero_train.envs import make_env_fn
from aero_train.experience import ExperienceSource
from aero_train.primitives import Agent

JITTER = {"altitude_jitter_ft": 200.0, "heading_jitter_deg": 5.0}

CONFIG = {
    "run_name": "sandbox-lifecycle",
    "seed": 0,
    "total_steps": 512,
    "device": "cpu",
    "env": {
        "scenario": "steady_flight",
        "max_episode_steps": 40,
        "params": JITTER,
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


def _config(tmp_path, **checkpoint):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG)
    config["checkpoint"] = {"dir": str(tmp_path / "checkpoints"), **checkpoint}
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_seeded_reset_draws_a_different_situation_per_seed_and_reproduces_it(tmp_path):
    """steady_flight's setup_fn draws deterministically from the env's own
    seeded generator: distinct seeds give distinct starts, and the same seed
    reproduces its own. This exercises one bare env at a time through
    env.reset(seed=...), the same path make_env_fn hands a single-env caller
    (eval, or this test); it says nothing about whether ExperienceSource's
    per-worker SeedSequence.spawn gives sub-envs independent streams from one
    root seed, which is a property of a vector env under one ExperienceSource
    construction, not of this loop. See
    test_experience_source_spawns_one_independent_stream_per_sub_env for that."""
    env_fn = make_env_fn(
        EnvConfig(scenario="steady_flight", max_episode_steps=40, params=dict(JITTER))
    )
    targets = []
    for seed in (0, 1, 2, 3, 0):
        env = env_fn()
        try:
            _, info = env.reset(seed=seed)
            targets.append(info["context"]["target_altitude_ft"])
        finally:
            env.close()
    assert len(set(targets[:4])) == 4, "four seeds gave fewer than four situations"
    assert targets[4] == targets[0], "the same seed did not reproduce its situation"
    assert all(4800.0 <= t <= 5200.0 for t in targets)


class _NoOpAgent(Agent):
    """Satisfies ExperienceSource's constructor signature without acting.

    ExperienceSource.__init__ only resets the vector env; it never calls the
    agent, so this exists purely to construct one.
    """

    def __call__(self, obs, agent_state=None):
        raise NotImplementedError("not called by ExperienceSource.__init__")

    def sync_weights(self, actor_state_dict):
        pass


def _experience_source_contexts(seed: int, num_envs: int = 4) -> np.ndarray:
    """Build the same vector env + ExperienceSource pair LocalRolloutEngine
    does, and recover the per-worker reset context it discards via a spy on
    the vector env's own reset.

    Driving the real ExperienceSource, rather than recomputing
    SeedSequence(seed).spawn(num_envs) by hand, is what makes collapsing that
    spawn call in production code actually fail this test: a hand-rolled copy
    of the seeding would test the copy, not the seam a real rollout worker uses.
    """
    env_fn = make_env_fn(
        EnvConfig(scenario="steady_flight", max_episode_steps=40, params=dict(JITTER))
    )
    vec_env = gym.vector.SyncVectorEnv([env_fn] * num_envs)
    captured: dict = {}
    original_reset = vec_env.reset

    def _reset_and_capture(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        captured["context"] = info["context"]
        return obs, info

    vec_env.reset = _reset_and_capture
    src = ExperienceSource(vec_env, _NoOpAgent(), seed)
    src.close()
    return captured["context"]["target_altitude_ft"]


def test_experience_source_spawns_one_independent_stream_per_sub_env():
    """SeedSequence(seed).spawn(num_envs) is what gives a rollout's sub-envs
    independent starting-situation streams from one root seed; collapsing it
    to a repeated seed, as production mutation row 1 does, makes every
    sub-env draw the identical situation instead.

    Reproducible too: the same root seed must redraw the same per-worker
    streams, or reconstructing ExperienceSource on resume would re-randomize
    every worker's starting situation.
    """
    first = _experience_source_contexts(seed=0, num_envs=4)
    again = _experience_source_contexts(seed=0, num_envs=4)
    assert len(set(first)) == 4, "four sub-envs from one root seed gave fewer than four situations"
    assert np.array_equal(first, again), "the same root seed did not reproduce its per-worker streams"
    assert all(4800.0 <= t <= 5200.0 for t in first)


def test_a_jittered_run_survives_start_checkpoint_resume_and_eval(tmp_path, capsys):
    """Under this config, no field this checkpoint carries distinguishes a
    resume from a restart, measured rather than assumed: cmd_resume reseeds
    ExperienceSource from cfg.seed at construction (the fact behind Deviation
    1), so a from-scratch restart to the same seed and total_steps
    deterministically replays the exact early trajectory a real resume's first
    run already lived through. Both global_step (measured 7 / 15 / 15 for
    first / real resume / row-3 restart, identical at every --total-steps
    extension swept from 1 to 520+ frames past first's) and best_return
    (measured 31.914803038284724 for all three, bit-identical) converge to the
    same value either way, because the restart's deterministic prefix already
    contains the first run's own peak eval. frames itself separates by one
    (1047 vs 1046), a coincidence of eval scheduling rather than a property,
    which is why only its direction is asserted below. What is asserted
    instead is what is actually true: frames advance past the first run's
    own count, the jittered params round-trip through the checkpoint, and
    the resumed run still evals cleanly.
    """
    path = _config(
        tmp_path,
        interval_iters=2,
        eval_interval_iters=4,
        eval_episodes=2,
        include_buffer=True,
    )
    assert cli.main(["train", str(path)]) == 0
    latest = tmp_path / "checkpoints" / "latest.pt"
    assert latest.exists(), "the final iteration was not checkpointed"
    first = load_checkpoint(latest)
    assert first["frames"] >= CONFIG["total_steps"]
    assert first["config"]["env"]["params"] == JITTER

    assert cli.main(["resume", str(latest), "--total-steps", "1024"]) == 0
    second = load_checkpoint(latest)
    assert second["frames"] > first["frames"]
    assert second["config"]["env"]["params"] == JITTER

    assert cli.main(["eval", str(latest), "--episodes", "3"]) == 0
    out = capsys.readouterr().out
    assert "mean return" in out and "length" in out and "terminated" in out


def test_two_runs_of_the_same_jittered_config_are_bit_identical(tmp_path, capsys):
    """The sampler draws from the env's own generator and adds no other source
    of randomness, so seeding a run still pins it completely. Compared on
    best_return at full float precision, not on a rounded log line: a sampler
    that reached for the global numpy RNG would diverge in the last digits
    long before it changed anything a formatted report shows.
    """
    results = []
    for run in ("a", "b"):
        path = _config(
            tmp_path / run,
            interval_iters=2,
            eval_interval_iters=2,
            eval_episodes=2,
        )
        assert cli.main(["train", str(path)]) == 0
        payload = load_checkpoint(tmp_path / run / "checkpoints" / "latest.pt")
        results.append(
            (payload["frames"], payload["global_step"], payload["best_return"])
        )
    assert results[0] == results[1]
    assert np.isfinite(results[0][2])


def test_the_best_checkpoint_reloads_through_the_same_path_eval_uses(tmp_path, capsys):
    path = _config(tmp_path, interval_iters=2, eval_interval_iters=2, eval_episodes=1)
    assert cli.main(["train", str(path)]) == 0
    best = tmp_path / "checkpoints" / "best.pt"
    assert best.exists()
    assert cli.main(["eval", str(best), "--episodes", "1"]) == 0
    out = capsys.readouterr().out
    assert "mean return" in out and "terminated" in out
