from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np
import pytest
from tests.conftest import CountingEnv
from tests.test_experience import ConstantAgent

from aero_train.config import RolloutConfig
from aero_train.rollout import LocalRolloutEngine, RolloutEngine, build_engine


def env_fn(episode_len=100):
    return lambda: CountingEnv(episode_len=episode_len)


class RandomEnv(gym.Env):
    """Observation is a draw from the env's seeded RNG.

    CountingEnv's obs is the step index and never touches the RNG, so a test
    that reuses it to check seed reproducibility would pass even if the seed
    were silently dropped or corrupted before reaching env.reset: there is
    nothing in CountingEnv's output that could disagree. RandomEnv exists only
    to give the reproducibility test something that would actually change if
    LocalRolloutEngine forwarded the wrong seed.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 100) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.episode_len = episode_len
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([self.np_random.random()], np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        return np.array([self.np_random.random()], np.float32), 1.0, done, False, {}


def test_collect_returns_the_requested_number_of_frames():
    eng = LocalRolloutEngine(env_fn(), ConstantAgent(), envs_per_worker=4, seed=0)
    seg, _ = eng.collect(32)
    assert (seg.T, seg.B) == (8, 4)
    assert seg.T * seg.B == 32
    eng.close()


def test_frames_must_divide_the_env_count():
    eng = LocalRolloutEngine(env_fn(), ConstantAgent(), envs_per_worker=4, seed=0)
    with pytest.raises(ValueError, match="divisible"):
        eng.collect(30)
    eng.close()


def test_episode_results_reach_the_caller():
    eng = LocalRolloutEngine(env_fn(episode_len=3), ConstantAgent(), envs_per_worker=2, seed=0)
    _, episodes = eng.collect(16)
    assert episodes and all(e.length == 3 for e in episodes)
    eng.close()


def test_sync_weights_is_forwarded_to_the_agent():
    class RecordingAgent(ConstantAgent):
        def __init__(self):
            super().__init__()
            self.synced = None

        def sync_weights(self, actor_state_dict):
            self.synced = actor_state_dict

    agent = RecordingAgent()
    eng = LocalRolloutEngine(env_fn(), agent, envs_per_worker=2, seed=0)
    payload = {"selector.epsilon": np.float32(0.25)}
    eng.sync_weights(payload)
    assert agent.synced is payload
    eng.close()


def test_the_same_seed_and_topology_reproduce_the_same_segment():
    def run():
        eng = LocalRolloutEngine(lambda: RandomEnv(), ConstantAgent(), envs_per_worker=3, seed=11)
        seg, _ = eng.collect(12)
        eng.close()
        return seg

    a, b = run(), run()
    assert np.array_equal(a.obs, b.obs)
    assert np.array_equal(a.rewards, b.rewards)


def test_build_engine_dispatches_on_the_config():
    cfg = RolloutConfig(engine="local", n_workers=1, envs_per_worker=2, frames_per_iter=8)
    eng = build_engine(cfg, env_fn(), ConstantAgent(), seed=0)
    assert isinstance(eng, LocalRolloutEngine)
    eng.close()


def test_build_engine_reports_the_ray_engine_as_not_yet_available():
    cfg = RolloutConfig(engine="ray", n_workers=2, envs_per_worker=2, frames_per_iter=8)
    with pytest.raises(NotImplementedError, match="Phase 3"):
        build_engine(cfg, env_fn(), ConstantAgent(), seed=0)


def test_the_local_path_never_imports_ray():
    # the spec's LocalRolloutEngine requirement, enforced rather than asserted in prose
    eng = LocalRolloutEngine(env_fn(), ConstantAgent(), envs_per_worker=2, seed=0)
    eng.sync_weights({})
    eng.collect(4)
    eng.close()
    assert "ray" not in sys.modules


def test_local_rollout_engine_satisfies_the_abc():
    assert issubclass(LocalRolloutEngine, RolloutEngine)
