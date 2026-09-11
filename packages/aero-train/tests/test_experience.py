from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from tests.conftest import CountingEnv, counting_env_fn

from aero_train.experience import ExperienceSource
from aero_train.primitives import Agent, Segment


class ConstantAgent(Agent):
    """Always emits action 0. Keeps segment contents fully predictable."""

    def __init__(self, action: int = 0) -> None:
        self.action = action
        self.calls = 0

    def __call__(self, obs, agent_state=None):
        self.calls += 1
        return np.full(len(obs), self.action, np.int64), agent_state

    def sync_weights(self, actor_state_dict):
        pass


def make_source(n_envs=2, episode_len=100, truncate=False, agent=None):
    env = gym.vector.SyncVectorEnv([counting_env_fn(episode_len, truncate)] * n_envs)
    return ExperienceSource(env, agent or ConstantAgent(), seed=0)


def test_collect_returns_a_segment_of_the_requested_length():
    src = make_source(n_envs=3)
    seg, _ = src.collect(5)
    assert isinstance(seg, Segment)
    assert (seg.T, seg.B) == (5, 3)
    src.close()


def test_segment_obs_carries_the_bootstrap_row():
    src = make_source(n_envs=2)
    seg, _ = src.collect(4)
    assert seg.obs.shape == (5, 2, 1)
    src.close()


def test_obs_are_the_counting_env_step_indices():
    src = make_source(n_envs=1)
    seg, _ = src.collect(4)
    assert seg.obs[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    src.close()


class VaryingActionAgent(Agent):
    """Action = obs value mod 2, so the recorded actions vary over time.

    A stub that hardcodes a constant action, or one that feeds the agent the
    wrong observation, produces a different sequence than the one asserted
    below, unlike a constant agent, whose output can't distinguish either
    bug from a correct implementation.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, obs, agent_state=None):
        self.calls += 1
        actions = np.asarray(obs)[:, 0].astype(np.int64) % 2
        return actions, agent_state

    def sync_weights(self, actor_state_dict):
        pass


class VaryingRewardEnv(CountingEnv):
    """Reward equals the post-step counter, not CountingEnv's constant 1.0.

    CountingEnv's own reward is deliberately constant (Tasks 9, 10, 15 and
    17 rely on it for hand-computable n-step sums), so the variation needed
    to make a reward-recording test discriminating lives here instead of in
    conftest.py.
    """

    def step(self, action):
        obs, _, term, trunc, info = super().step(action)
        return obs, float(self.t), term, trunc, info


def varying_reward_env_fn(episode_len: int = 100):
    return lambda: VaryingRewardEnv(episode_len=episode_len)


def test_rewards_and_actions_are_recorded():
    env = gym.vector.SyncVectorEnv([varying_reward_env_fn()] * 2)
    src = ExperienceSource(env, VaryingActionAgent(), seed=0)
    seg, _ = src.collect(3)
    # obs before each step is 0, 1, 2 -> action = obs % 2 -> 0, 1, 0
    assert seg.actions.tolist() == [[0, 0], [1, 1], [0, 0]]
    # reward is the post-step counter -> 1, 2, 3
    assert seg.rewards.tolist() == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    src.close()


def test_all_steps_are_valid_when_no_episode_ends():
    src = make_source(n_envs=2, episode_len=100)
    seg, episodes = src.collect(6)
    assert seg.valid.all()
    assert not seg.terminated.any() and not seg.truncated.any()
    assert episodes == []
    src.close()


def test_one_batched_forward_pass_per_timestep_not_one_per_env():
    agent = ConstantAgent()
    src = make_source(n_envs=8, agent=agent)
    src.collect(5)
    assert agent.calls == 5      # lockstep: 5, not 40
    src.close()


def test_consecutive_collects_are_contiguous():
    src = make_source(n_envs=1)
    first, _ = src.collect(3)
    second, _ = src.collect(3)
    # the bootstrap row of one segment is the opening row of the next
    assert first.obs[-1, 0, 0] == second.obs[0, 0, 0]
    src.close()


def test_num_envs_is_exposed():
    src = make_source(n_envs=4)
    assert src.num_envs == 4
    src.close()


def test_collect_rejects_a_non_positive_length():
    src = make_source()
    with pytest.raises(ValueError, match="n_steps"):
        src.collect(0)
    src.close()


def test_terminating_step_is_recorded_and_the_autoreset_step_is_not():
    src = make_source(n_envs=1, episode_len=4)
    seg, _ = src.collect(10)
    assert seg.valid[:, 0].tolist() == [True] * 4 + [False] + [True] * 4 + [False]
    assert seg.terminated[:, 0].tolist() == [
        False, False, False, True, False, False, False, False, True, False,
    ]
    src.close()


def test_bootstrap_obs_of_a_terminating_step_is_the_true_final_obs():
    src = make_source(n_envs=1, episode_len=4)
    seg, _ = src.collect(10)
    # obs[3] -> obs[4] is the terminating transition; obs[4] must be the final
    # state (4.0), not the reset state (0.0)
    assert seg.obs[3, 0, 0] == 3.0
    assert seg.obs[4, 0, 0] == 4.0
    assert seg.obs[5, 0, 0] == 0.0
    src.close()


def test_the_autoreset_row_contributes_no_reward_to_the_segment():
    """Mostly pins Gymnasium's NEXT_STEP autoreset, which already returns
    reward=0.0 on the fabricated step, rather than collect()'s own masking.
    That masking stays as the local statement of the invariant: it covers a
    consumer reading seg.rewards unmasked, and turns a change in Gymnasium's
    behaviour into a loud Segment.__post_init__ error instead of corruption."""
    src = make_source(n_envs=1, episode_len=4)
    seg, _ = src.collect(6)
    assert seg.rewards[:, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    src.close()


def test_episode_results_are_emitted_on_termination():
    src = make_source(n_envs=1, episode_len=4)
    _, episodes = src.collect(10)
    assert len(episodes) == 2
    assert all(e.ret == 4.0 and e.length == 4 and e.terminated for e in episodes)
    src.close()


def test_truncation_is_reported_as_not_terminated():
    src = make_source(n_envs=1, episode_len=4, truncate=True)
    seg, episodes = src.collect(6)
    assert seg.truncated[3, 0] and not seg.terminated[3, 0]
    assert len(episodes) == 1 and episodes[0].terminated is False
    src.close()


def test_envs_ending_at_different_steps_are_masked_independently():
    env = gym.vector.SyncVectorEnv([counting_env_fn(3), counting_env_fn(5)])
    src = ExperienceSource(env, ConstantAgent(), seed=0)
    seg, episodes = src.collect(7)
    assert seg.valid[:, 0].tolist() == [True, True, True, False, True, True, True]
    assert seg.valid[:, 1].tolist() == [True, True, True, True, True, False, True]
    assert [(e.length, e.terminated) for e in episodes] == [(3, True), (5, True), (3, True)]
    src.close()


def test_a_pending_autoreset_carries_across_a_collect_boundary():
    src = make_source(n_envs=1, episode_len=4)
    first, first_eps = src.collect(4)
    second, second_eps = src.collect(2)
    assert first.terminated[3, 0] and len(first_eps) == 1
    # the fabricated step is the first row of the NEXT segment
    assert second.valid[:, 0].tolist() == [False, True]
    assert second_eps == []
    src.close()


class BothFlagsEnv(gym.Env):
    """Ends its episode with terminated AND truncated on the same step.

    Not a contrived shape: AeroGymEnv computes the two independently, so a step
    that both leaves the reward band and hits max_episode_steps sets both, and
    gymnasium's own TimeLimit wrapper ORs `truncated` onto a terminating step
    the same way.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 3) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.episode_len = episode_len
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        return np.array([float(self.t)], np.float32), 1.0, done, done, {}


def test_a_step_that_both_terminates_and_truncates_is_recorded_as_terminated():
    """Segment forbids both flags on one step, and rightly so: `truncated`
    means "bootstrap past this", which is exactly wrong for a terminal state.
    The collector is where the two are reconciled, because an env-level fix
    would not reach gymnasium's TimeLimit.
    """
    env = gym.vector.SyncVectorEnv([lambda: BothFlagsEnv(episode_len=3)])
    src = ExperienceSource(env, ConstantAgent(), seed=0)
    seg, episodes = src.collect(4)
    assert seg.terminated[2, 0], "the ending step must be recorded as terminated"
    assert not seg.truncated[2, 0], "and must not also be recorded as truncated"
    assert [e.terminated for e in episodes] == [True]
    src.close()
