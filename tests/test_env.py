from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aerogym import AeroGymEnv
from aerogym.scenario import Scenario, Transition
from aerogym.variables import FlightVariable as V


def test_spaces_follow_scenario_selection(initial_conditions, fake_simulator):
    scenario = Scenario(
        name="subset",
        initial_conditions=initial_conditions,
        reward_fn=lambda t: (0.0, False),
        action_variables=(V.AILERON, V.RUDDER),
        observation_variables=(V.ROLL, V.ALTITUDE, V.AOS),
    )
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (3,)
    assert env.action_space.low.tolist() == [-1.0, -1.0]


def test_reset_applies_scenario_and_returns_obs(scenario, fake_simulator):
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    obs, info = env.reset(seed=0)
    assert fake_simulator.reset_calls == [scenario]
    assert obs.shape == env.observation_space.shape
    alt_index = scenario.observation_variables.index(V.ALTITUDE)
    assert obs[alt_index] == pytest.approx(1500.0)
    assert info["obs"][V.HEADING] == pytest.approx(270.0)


def test_step_runs_physics_and_applies_controls(scenario, fake_simulator):
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, np.float32)
    action[0] = 5.0  # out of bounds on purpose: must be clipped to +1
    obs, reward, terminated, truncated, info = env.step(action)
    # 120 Hz physics at 10 Hz agent rate -> 12 physics steps per agent step
    assert fake_simulator.run_count == 12
    assert fake_simulator.controls[scenario.action_variables[0]] == 1.0
    assert reward == 1.0
    assert not terminated


def test_truncation_at_max_episode_steps(scenario, fake_simulator):
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, np.float32)
    truncated = False
    for _ in range(scenario.max_episode_steps):
        *_, truncated, _ = env.step(action)
    assert truncated


def test_reward_fn_receives_transition(initial_conditions, fake_simulator):
    seen: list[Transition] = []

    def spy_reward(t: Transition) -> tuple[float, bool]:
        seen.append(t)
        return -2.5, True

    scenario = Scenario(
        name="spy",
        initial_conditions=initial_conditions,
        reward_fn=spy_reward,
    )
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert reward == -2.5 and terminated
    (t,) = seen
    assert set(t.obs) == set(scenario.observation_variables)
    assert set(t.action) == set(scenario.action_variables)
    assert t.prev_obs[V.ALTITUDE] == pytest.approx(1500.0)


def test_step_before_reset_raises(scenario, fake_simulator):
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(env.action_space.shape, np.float32))


def test_close_closes_simulator(scenario, fake_simulator):
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    env.close()
    assert fake_simulator.closed


def test_flightgear_render_mode_enables_visualization(scenario, fake_simulator):
    scenario = dataclasses.replace(scenario, max_episode_steps=None)
    env = AeroGymEnv(scenario, simulator=fake_simulator, render_mode="flightgear")
    env.reset(seed=0)
    assert fake_simulator.visualization_enabled


def test_invalid_render_mode_rejected(scenario, fake_simulator):
    with pytest.raises(ValueError):
        AeroGymEnv(scenario, simulator=fake_simulator, render_mode="human")
