from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import pytest

from aero_train.envs import make_env_fn


@dataclass(frozen=True)
class _EnvCfg:
    id: str | None = None
    scenario: str | None = None
    max_episode_steps: int | None = None
    scenario_kwargs: dict | None = None
    render_mode: str | None = None
    delta: float = 0.1

    def __post_init__(self):
        if self.scenario_kwargs is None:
            object.__setattr__(self, "scenario_kwargs", {})


def test_make_env_fn_builds_a_gymnasium_env():
    env = make_env_fn(_EnvCfg(id="CartPole-v1"))()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    env.close()


def test_make_env_fn_applies_max_episode_steps():
    env = make_env_fn(_EnvCfg(id="CartPole-v1", max_episode_steps=3))()
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    env.close()


def test_make_env_fn_returns_independent_envs():
    fn = make_env_fn(_EnvCfg(id="CartPole-v1"))
    a, b = fn(), fn()
    assert a is not b
    a.close()
    b.close()


def test_unknown_env_id_raises():
    with pytest.raises(gym.error.Error):
        make_env_fn(_EnvCfg(id="NoSuchEnv-v0"))()


def test_a_gymnasium_id_that_looks_like_a_scenario_is_still_a_gymnasium_id():
    """Gymnasium's make() already defines "module:Env-v0" as "import module,
    then make Env-v0". The branch is a field, never a prefix on the id, so this
    string can only ever mean what Gymnasium says it means."""
    with pytest.raises(gym.error.Error):
        make_env_fn(_EnvCfg(id="aero_gym:steady_flight"))()


def test_an_unknown_scenario_lists_the_known_ones():
    with pytest.raises(ValueError, match="steady_flight"):
        make_env_fn(_EnvCfg(scenario="loop_the_loop"))


def test_a_misspelled_scenario_kwarg_fails_before_any_env_is_built():
    with pytest.raises(ValueError, match="scenario_kwargs"):
        make_env_fn(_EnvCfg(scenario="steady_flight", scenario_kwargs={"aircrafft": "737"}))


def test_the_scenario_branch_wraps_the_env_in_both_wrappers():
    pytest.importorskip("jsbsim")
    from aero_train.envs.delta import DeltaAction

    env = make_env_fn(_EnvCfg(scenario="steady_flight", max_episode_steps=5))()
    try:
        assert isinstance(env, DeltaAction)
        assert env.action_space == gym.spaces.Discrete(81)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert obs.min() >= -1.0 and obs.max() <= 1.0
    finally:
        env.close()


def test_the_scenario_branch_applies_max_episode_steps():
    """The scenario branch has no gym.make TimeLimit wrapper, so
    cfg.max_episode_steps must reach the *scenario* or nothing truncates.
    Mirrors test_make_env_fn_applies_max_episode_steps on the other branch."""
    pytest.importorskip("jsbsim")
    env = make_env_fn(_EnvCfg(scenario="steady_flight", max_episode_steps=5))()
    try:
        env.reset(seed=0)
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(env.null_action)
        assert truncated and not terminated
    finally:
        env.close()


def test_the_scenario_branch_plumbs_delta_through_to_the_wrapper():
    # Discrete(81) follows from the axis count alone and cannot show whether
    # cfg.delta reached DeltaAction, so the attribute is read directly.
    pytest.importorskip("jsbsim")
    env = make_env_fn(_EnvCfg(scenario="steady_flight", delta=0.5))()
    try:
        assert env.delta == 0.5
    finally:
        env.close()


def test_the_scenario_branch_plumbs_render_mode_through():
    # AeroGymEnv only stores render_mode in __init__ and acts on it at reset(),
    # so this is checkable without ever launching FlightGear.
    pytest.importorskip("jsbsim")
    env = make_env_fn(_EnvCfg(scenario="steady_flight", render_mode="flightgear"))()
    try:
        assert env.render_mode == "flightgear"
    finally:
        env.close()
