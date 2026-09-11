from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from aero_gym import AeroGymEnv
from aero_gym.scenario import InitialConditions, Scenario, Transition, fixed_setup
from aero_gym.variables import FlightVariable as V
from conftest import FakeSimulator


def test_spaces_follow_scenario_selection(initial_conditions, fake_simulator):
    scenario = Scenario(
        name="subset",
        setup_fn=fixed_setup(initial_conditions),
        reward_fn=lambda t: (0.0, False),
        action_variables=(V.AILERON, V.RUDDER),
        observation_variables=(V.ROLL, V.ALTITUDE, V.AOS),
    )
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (3,)
    assert env.action_space.low.tolist() == [-1.0, -1.0]


def test_reset_applies_scenario_and_returns_obs(scenario, initial_conditions, fake_simulator):
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    obs, info = env.reset(seed=0)
    assert fake_simulator.reset_calls == [(scenario.aircraft, initial_conditions, 12)]
    assert obs.shape == env.observation_space.shape
    alt_index = scenario.observation_variables.index(V.ALTITUDE)
    assert obs[alt_index] == pytest.approx(5000.0)
    assert info["obs"][V.HEADING] == pytest.approx(270.0)


def test_the_env_hands_the_simulator_only_the_aircraft_and_the_conditions(
    scenario, initial_conditions, fake_simulator
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
    assert conditions is initial_conditions
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
        setup_fn=fixed_setup(initial_conditions),
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
        lambda obs, ctx: obs[V.ALTITUDE] - 5000.0,
    )
    scenario = Scenario(
        name="features",
        setup_fn=fixed_setup(initial_conditions),
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
        setup_fn=fixed_setup(initial_conditions),
        reward_fn=lambda t: (0.0, False),
        observation_variables=(V.ROLL,),
        observation_features=(
            ObservationFeature(
                "aoa", VariableSpec("deg", -20.0, 20.0, "x"), lambda obs, ctx: obs[V.AOA]
            ),
        ),
    )
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    with pytest.raises(KeyError, match="AOA"):
        env.reset(seed=0)


def test_the_setup_is_drawn_from_the_envs_own_generator(fake_simulator):
    """Same seed, same situation; different seed, a different one.

    Both halves: without the second a constant sampler passes, which is the
    exact defect that would make every environment fly the same episode.
    """
    from aero_gym.scenario import EpisodeSetup, InitialConditions, Scenario

    def sample(rng):
        altitude = 5000.0 + rng.uniform(-500.0, 500.0)
        return EpisodeSetup(
            InitialConditions(0.0, 0.0, altitude, 270.0),
            {"target_altitude_ft": altitude},
        )

    scenario = Scenario(name="sampled", setup_fn=sample, reward_fn=lambda t: (0.0, False))
    env = AeroGymEnv(scenario, simulator=fake_simulator)
    env.reset(seed=0)
    env.reset(seed=0)
    env.reset(seed=1)
    altitudes = [conditions.altitude_ft for _, conditions, _ in fake_simulator.reset_calls]
    assert altitudes[0] == altitudes[1]
    assert altitudes[0] != altitudes[2]


def test_the_context_reaches_the_features_and_the_reward(fake_simulator):
    """A stale or empty context is the failure: assert the value, not that the
    reward is merely finite."""
    from aero_gym.scenario import (
        EpisodeSetup,
        InitialConditions,
        ObservationFeature,
        Scenario,
    )
    from aero_gym.variables import VariableSpec

    contexts = iter([{"target_altitude_ft": 5000.0}, {"target_altitude_ft": 4000.0}])

    def sample(rng):
        return EpisodeSetup(InitialConditions(0.0, 0.0, 5200.0, 270.0), next(contexts))

    feature = ObservationFeature(
        "altitude_error_ft",
        VariableSpec("ft", -2000.0, 2000.0, "altitude minus target"),
        lambda obs, context: obs[V.ALTITUDE] - context["target_altitude_ft"],
    )
    scenario = Scenario(
        name="context",
        setup_fn=sample,
        reward_fn=lambda t: (t.context["target_altitude_ft"], False),
        observation_variables=(V.ALTITUDE,),
        observation_features=(feature,),
    )
    env = AeroGymEnv(scenario, simulator=fake_simulator)

    obs, info = env.reset(seed=0)
    assert obs[0] == pytest.approx(200.0)
    assert info["context"] == {"target_altitude_ft": 5000.0}
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert reward == pytest.approx(5000.0)
    assert info["context"] == {"target_altitude_ft": 5000.0}

    obs, info = env.reset(seed=0)
    assert obs[0] == pytest.approx(1200.0)
    _, reward, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert reward == pytest.approx(4000.0)


def test_a_setup_missing_a_declared_context_key_names_the_key_and_the_scenario():
    """The failure this catches is delayed: a sampler with a conditional branch
    can return a complete context for four hundred episodes and a short one on
    the next, and a bare KeyError from inside a lambda names neither."""
    from aero_gym.scenario import EpisodeSetup, InitialConditions, Scenario

    scenario = Scenario(
        name="declared",
        setup_fn=lambda rng: EpisodeSetup(InitialConditions(0.0, 0.0, 5000.0, 270.0)),
        reward_fn=lambda t: (0.0, False),
        context_keys=("target_altitude_ft",),
    )
    env = AeroGymEnv(scenario, simulator=FakeSimulator())
    with pytest.raises(KeyError, match=r"declared.*target_altitude_ft"):
        env.reset(seed=0)


def _bounded_scenario(low=-100.0, high=100.0):
    from aero_gym.scenario import ObservationFeature, Scenario, fixed_setup
    from aero_gym.variables import VariableSpec

    return Scenario(
        name="bounded",
        setup_fn=fixed_setup(InitialConditions(0.0, 0.0, 5000.0, 270.0)),
        reward_fn=lambda t: (0.0, False),
        observation_variables=(V.ROLL,),
        observation_features=(
            ObservationFeature(
                "roll", VariableSpec("deg", low, high, "roll"), lambda obs, ctx: obs[V.ROLL]
            ),
        ),
    )


def test_a_saturated_observation_warns_and_names_the_feature(fake_simulator):
    """np.clip makes an out-of-range feature look in-range, which is a silently
    mis-scaled input rather than a crash."""
    env = AeroGymEnv(_bounded_scenario(), simulator=fake_simulator)
    env.reset(seed=0)
    fake_simulator.state[V.ROLL] = 250.0
    with pytest.warns(RuntimeWarning, match=r"'roll'.*250"):
        obs, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert obs[0] == pytest.approx(100.0)


def test_a_saturated_observation_below_the_low_bound_also_warns(fake_simulator):
    """The compare in _to_array is direction-symmetric; this mirrors the
    over-high case above so the low side of the clip is covered too."""
    env = AeroGymEnv(_bounded_scenario(), simulator=fake_simulator)
    env.reset(seed=0)
    fake_simulator.state[V.ROLL] = -250.0
    with pytest.warns(RuntimeWarning, match=r"'roll'.*-250"):
        obs, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert obs[0] == pytest.approx(-100.0)


def test_the_saturation_warning_fires_once_per_feature_not_once_per_step(fake_simulator):
    """At 10 Hz an excursion would otherwise emit a warning every 100 ms for
    the rest of the run, which is how a useful warning becomes noise.

    recwarn cannot tell this apart from a code path that warns on every
    step: pytest applies its own "default" filter ahead of recwarn's, which
    dedups by (message, category, lineno) regardless of our own guard, since
    every step() call here comes from the same source line with the same
    message. simplefilter("always") in an explicit context bypasses that.
    """
    env = AeroGymEnv(_bounded_scenario(), simulator=fake_simulator)
    env.reset(seed=0)
    fake_simulator.state[V.ROLL] = 250.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            env.step(np.zeros(env.action_space.shape, np.float32))
    assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 1


def test_an_in_bounds_observation_stays_silent(fake_simulator):
    """The other half: a warning that always fires proves nothing.

    Same explicit "always" filter as the sibling test, so this stays a fair
    comparison rather than one relying on pytest's own dedup to look silent.
    """
    env = AeroGymEnv(_bounded_scenario(), simulator=fake_simulator)
    env.reset(seed=0)
    fake_simulator.state[V.ROLL] = 12.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            env.step(np.zeros(env.action_space.shape, np.float32))
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_a_nan_feature_is_reported_as_nan_not_as_saturation(fake_simulator):
    """NaN trips the same comparison, since NaN != NaN, and calling it
    saturation would send the reader looking for an excursion that never was."""
    env = AeroGymEnv(_bounded_scenario(), simulator=fake_simulator)
    env.reset(seed=0)
    fake_simulator.state[V.ROLL] = float("nan")
    with pytest.warns(RuntimeWarning, match=r"'roll'.*NaN"):
        env.step(np.zeros(env.action_space.shape, np.float32))
