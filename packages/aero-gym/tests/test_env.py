from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aero_gym import AeroGymEnv
from aero_gym.scenario import Scenario, Transition
from aero_gym.variables import FlightVariable as V


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
    assert fake_simulator.reset_calls == [(scenario.aircraft, scenario.initial_conditions, 12)]
    assert obs.shape == env.observation_space.shape
    alt_index = scenario.observation_variables.index(V.ALTITUDE)
    assert obs[alt_index] == pytest.approx(5000.0)
    assert info["obs"][V.HEADING] == pytest.approx(270.0)


def test_the_env_hands_the_simulator_only_the_aircraft_and_the_conditions(
    scenario, fake_simulator
):
    """The backend is handed what it needs, not the whole task definition.

    settle_steps is asserted too: it replaces the agent_hz the backend used to
    read off the scenario, and a backend given 1 instead would return a stale
    first observation on every episode.
    """
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    env.reset(seed=0)
    (aircraft, conditions, settle_steps), = fake_simulator.reset_calls
    assert aircraft == "c172p"
    assert conditions is scenario.initial_conditions
    # 120 Hz physics at 10 Hz agent rate
    assert settle_steps == 12


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
    assert t.prev_obs[V.ALTITUDE] == pytest.approx(5000.0)


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


def test_the_box_follows_the_features_not_the_variables(initial_conditions, fake_simulator):
    """Altitude is read from the simulator but never shown raw: the agent sees
    the error against a target, bounded by the band that terminates on it."""
    from aero_gym.scenario import ObservationFeature, raw
    from aero_gym.variables import VariableSpec

    error = ObservationFeature(
        "altitude_error_ft",
        VariableSpec("ft", -500.0, 500.0, "altitude minus target"),
        lambda obs: obs[V.ALTITUDE] - 5000.0,
    )
    scenario = Scenario(
        name="features",
        initial_conditions=initial_conditions,
        reward_fn=lambda t: (0.0, False),
        observation_variables=(V.ALTITUDE, V.ROLL),
        observation_features=(error, raw(V.ROLL, -75.0, 75.0)),
    )
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    assert env.observation_space.low.tolist() == [-500.0, -75.0]

    # reset() first: it re-applies the scenario's initial altitude every call,
    # while step() reads the state back without re-establishing it.
    env.reset(seed=0)
    fake_simulator.state[V.ALTITUDE] = 5200.0
    fake_simulator.state[V.ROLL] = 3.0
    obs, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert obs.tolist() == pytest.approx([200.0, 3.0])


def test_a_feature_reading_an_unobserved_variable_names_it(initial_conditions, fake_simulator):
    from aero_gym.scenario import ObservationFeature
    from aero_gym.variables import VariableSpec

    scenario = Scenario(
        name="missing",
        initial_conditions=initial_conditions,
        reward_fn=lambda t: (0.0, False),
        observation_variables=(V.ROLL,),
        observation_features=(
            ObservationFeature(
                "aoa", VariableSpec("deg", -20.0, 20.0, "x"), lambda obs: obs[V.AOA]
            ),
        ),
    )
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    with pytest.raises(KeyError, match="AOA"):
        env.reset(seed=0)
