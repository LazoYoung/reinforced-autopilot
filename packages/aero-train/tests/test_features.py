"""The scenario half of export: features traced into torch, and what they read."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from aero_gym.scenario import ObservationFeature
from aero_gym.variables import FlightVariable as V, VariableSpec
from aero_train.envs.features import scenario_of, trace_scenario
from tests.flight_fakes import toy_env_fn, toy_scenario


def _stateful_roll_delta():
    """A feature closed over its own mutable state: this step's roll minus last step's."""
    state: dict[str, torch.Tensor] = {}

    def fn(obs, ctx):
        prev = state.get("r", obs[V.ROLL])
        state["r"] = obs[V.ROLL]
        return obs[V.ROLL] - prev

    return fn


def _in_place_roll(obs, ctx):
    x = obs[V.ROLL]
    x -= 10.0
    return x


def _with_feature(fn, name="extra"):
    base = toy_scenario()
    feature = ObservationFeature(name, VariableSpec("deg", -45.0, 45.0, "test feature"), fn)
    return toy_scenario(observation_features=base.observation_features + (feature,))


def test_the_inputs_are_what_the_features_read_and_nothing_else():
    """Airspeed is observed and its target declared, but no feature reads them."""
    traced = trace_scenario(toy_scenario())
    assert [p.name for p in traced.inputs] == [
        "altitude", "heading", "roll", "pitch", "target_altitude_ft", "target_heading_deg",
    ]


def test_readings_come_in_observation_order_whatever_order_the_features_read_them():
    base = toy_scenario()
    traced = trace_scenario(
        toy_scenario(observation_features=tuple(reversed(base.observation_features)))
    )
    assert [p.name for p in traced.inputs][:4] == ["altitude", "heading", "roll", "pitch"]


def test_each_port_carries_its_unit_range_and_meaning():
    traced = trace_scenario(toy_scenario())
    docs = {p.name: p.doc for p in (*traced.inputs, *traced.actions)}
    assert docs["altitude"] == "ft, -1500 to 85000: altitude above mean sea level"
    assert docs["target_altitude_ft"] == "ft, 4800 to 5200: altitude to hold"
    assert docs["throttle"] == "norm, 0 to 1: 0 = idle, 1 = full throttle"
    assert [p.name for p in traced.actions] == ["elevator", "aileron", "throttle"]


def test_the_metadata_names_the_rate_the_aircraft_and_the_scenario():
    assert trace_scenario(toy_scenario()).metadata == {
        "aero.agent_hz": "10.0",
        "aero.aircraft": "c172p",
        "aero.scenario": "toy-hold",
    }


def test_the_module_computes_what_the_env_computes_and_clips_the_same():
    """Both regimes: inside the bounds, and far outside where the clip decides."""
    scenario = toy_scenario()
    traced = trace_scenario(scenario)
    rng = np.random.default_rng(0)
    rows = {
        "altitude": rng.uniform(3000.0, 7000.0, 512),
        "heading": rng.uniform(0.0, 360.0, 512),
        "roll": rng.uniform(-90.0, 90.0, 512),
        "pitch": rng.uniform(-60.0, 60.0, 512),
        "target_altitude_ft": rng.uniform(4800.0, 5200.0, 512),
        "target_heading_deg": rng.uniform(0.0, 360.0, 512),
    }
    rows = {name: values.astype(np.float32) for name, values in rows.items()}
    got = traced.features(*(torch.as_tensor(rows[p.name]) for p in traced.inputs)).numpy()
    low = [f.spec.low for f in scenario.observation_features]
    high = [f.spec.high for f in scenario.observation_features]
    want = np.clip(
        [
            [
                f.fn(
                    {v: float(rows[v.value][i]) for v in traced.readings},
                    {k: float(rows[k][i]) for k in traced.targets},
                )
                for f in scenario.observation_features
            ]
            for i in range(512)
        ],
        low,
        high,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-3)


def test_feed_reads_an_env_step_by_the_names_the_file_takes():
    env = toy_env_fn()()
    try:
        traced = trace_scenario(scenario_of(env))
        _, info = env.reset(seed=0)
        feed = traced.feed(info)
    finally:
        env.close()
    assert list(feed) == [p.name for p in traced.inputs]
    assert feed["altitude"] == info["obs"][V.ALTITUDE]
    assert feed["target_heading_deg"] == info["context"]["target_heading_deg"]


def test_a_gymnasium_env_has_no_scenario_to_export():
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="canonical names"):
            scenario_of(env)
    finally:
        env.close()


@pytest.mark.parametrize(
    "fn",
    [
        lambda obs, ctx: math.fabs(obs[V.ROLL]),
        lambda obs, ctx: float(obs[V.ROLL]),
        lambda obs, ctx: obs[V.ROLL] if obs[V.ROLL] > 0 else -obs[V.ROLL],
        lambda obs, ctx: 0.5,
        _stateful_roll_delta(),
        _in_place_roll,
    ],
    ids=["math", "float", "branch", "constant", "stateful", "in-place"],
)
def test_a_feature_that_cannot_be_traced_is_named(fn):
    with pytest.raises(ValueError, match="roll_magnitude is not traceable"):
        trace_scenario(_with_feature(fn, "roll_magnitude"))


def test_arithmetic_and_abs_trace():
    """Arithmetic and abs are traceable operators; the traced column must match
    the float path for both a negative and a positive roll."""
    fn = lambda obs, ctx: abs(obs[V.ROLL]) * 0.5 - obs[V.PITCH] / 2.0
    traced = trace_scenario(_with_feature(fn))
    assert "roll" in [p.name for p in traced.inputs]

    rows = {
        "altitude": [5000.0, 5000.0],
        "heading": [10.0, 10.0],
        "roll": [-30.0, 30.0],
        "pitch": [5.0, -5.0],
        "target_altitude_ft": [5000.0, 5000.0],
        "target_heading_deg": [10.0, 10.0],
    }
    columns = [torch.as_tensor(rows[p.name], dtype=torch.float32) for p in traced.inputs]
    got = traced.features.values(*columns)[:, -1]
    want = torch.as_tensor(
        [fn({V.ROLL: rows["roll"][i], V.PITCH: rows["pitch"][i]}, {}) for i in range(2)],
        dtype=torch.float32,
    )
    torch.testing.assert_close(got, want)


def test_a_feature_reading_an_undeclared_target_names_both():
    with pytest.raises(ValueError, match="gust reads 'target_wind_kts'.*context_variables"):
        trace_scenario(
            _with_feature(lambda obs, ctx: obs[V.AIRSPEED] - ctx["target_wind_kts"], "gust")
        )


def test_a_feature_reading_an_unobserved_variable_names_both():
    with pytest.raises(ValueError, match="slip reads 'aos'.*does not observe"):
        trace_scenario(_with_feature(lambda obs, ctx: obs[V.AOS], "slip"))
