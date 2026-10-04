"""The scenario fingerprint: what it records, and every change it must notice."""

from __future__ import annotations

import json
import math
import re

import gymnasium as gym
import pytest

from aero_gym.scenario import ContextVariable, ObservationFeature
from aero_gym.variables import FlightVariable as V, VariableSpec
from aero_train.checkpoint import load_checkpoint, save_checkpoint
from aero_train.envs import fingerprint_difference, scenario_fingerprint
from tests.flight_fakes import flipped_toy_scenario, toy_env_fn, toy_fingerprint, toy_scenario


def _difference(stored, scenario):
    env = toy_env_fn(scenario)()
    try:
        return fingerprint_difference(stored, env)
    finally:
        env.close()


def _replace_feature(index, feature):
    features = list(toy_scenario().observation_features)
    features[index] = feature
    return toy_scenario(observation_features=tuple(features))


def test_a_scenario_matches_its_own_fingerprint():
    assert _difference(toy_fingerprint(), toy_scenario()) is None


def test_the_fingerprint_is_the_same_every_time_it_is_taken():
    assert toy_fingerprint() == toy_fingerprint()


def test_the_fingerprint_is_plain_data_a_checkpoint_and_json_keep(tmp_path):
    fingerprint = toy_fingerprint()
    save_checkpoint(tmp_path / "c.pt", {"scenario_fingerprint": fingerprint})
    assert load_checkpoint(tmp_path / "c.pt")["scenario_fingerprint"] == fingerprint
    assert json.loads(json.dumps(fingerprint, allow_nan=False)) == fingerprint


def test_the_probe_covers_every_feature_and_every_reading_and_target():
    probe = toy_fingerprint()["probe"]
    assert list(probe["values"]) == [f.name for f in toy_scenario().observation_features]
    assert set(probe["inputs"]) == {
        "altitude", "heading", "roll", "pitch", "airspeed",
        "target_altitude_ft", "target_heading_deg", "target_airspeed_kts",
    }
    assert {len(rows) for rows in (*probe["inputs"].values(), *probe["values"].values())} == {64}


@pytest.mark.parametrize(
    "scenario, match",
    [
        (flipped_toy_scenario(), r"altitude_error_ft changed: probe row \d+ was -?[\d.e+]+, now"),
        (
            _replace_feature(
                3, ObservationFeature("pitch", VariableSpec("deg", -15.0, 15.0, "pitch"),
                                      lambda obs, ctx: obs[V.PITCH])
            ),
            r"pitch range was \[-30, 30\] deg, now \[-15, 15\] deg",
        ),
        (
            toy_scenario(
                observation_features=toy_scenario().observation_features
                + (ObservationFeature("airspeed_error_kts", VariableSpec("kt", -20.0, 20.0, "x"),
                                      lambda obs, ctx: obs[V.AIRSPEED] - ctx["target_airspeed_kts"]),)
            ),
            r"features were \[.*'pitch'\], now \[.*'pitch', 'airspeed_error_kts'\]",
        ),
        (
            toy_scenario(context_variables=(
                ContextVariable("target_altitude_ft", VariableSpec("ft", 4000.0, 6000.0, "a")),
                *toy_scenario().context_variables[1:],
            )),
            r"target_altitude_ft range was \[4800, 5200\] ft, now \[4000, 6000\] ft",
        ),
        (toy_scenario(action_delta=(0.05, 0.1, 0.1)), r"action_steps was"),
        (toy_scenario(agent_hz=5.0), r"agent_hz was 10.0, now 5.0"),
        (toy_scenario(aircraft="737"), r"aircraft was 'c172p', now '737'"),
    ],
    ids=["feature-body", "feature-bound", "feature-added", "declared-range", "action-step",
         "rate", "aircraft"],
)
def test_every_change_a_policy_file_would_carry_is_named(scenario, match):
    difference = _difference(toy_fingerprint(), scenario)
    assert difference is not None and re.search(match, difference), difference


def test_a_feature_reading_an_undeclared_context_key_is_named():
    feature = ObservationFeature(
        "bad", VariableSpec("ft", -1.0, 1.0, "x"), lambda obs, ctx: ctx["target_vs_fpm"]
    )
    env = toy_env_fn(toy_scenario(observation_features=(feature,)))()
    try:
        with pytest.raises(
            ValueError,
            match="bad reads 'target_vs_fpm', which the scenario does not declare in context_variables",
        ):
            scenario_fingerprint(env)
    finally:
        env.close()


def test_a_feature_that_fails_inside_the_canonical_box_is_named():
    feature = ObservationFeature(
        "root", VariableSpec("ft", 0.0, 100.0, "x"), lambda obs, ctx: math.sqrt(obs[V.ALTITUDE])
    )
    env = toy_env_fn(toy_scenario(observation_features=(feature,)))()
    try:
        with pytest.raises(
            ValueError, match=r"root cannot be fingerprinted on probe row \d+: "
        ) as raised:
            scenario_fingerprint(env)
        assert isinstance(raised.value.__cause__, ValueError)
    finally:
        env.close()


def test_the_probe_is_not_clipped_to_a_features_bounds():
    probe = toy_fingerprint()
    for declared in probe["features"]:
        values = probe["probe"]["values"][declared["name"]]
        if min(values) < declared["low"] or max(values) > declared["high"]:
            return
    pytest.fail("no probe value lies beyond its feature's declared bounds")


def test_a_reward_change_leaves_the_fingerprint_alone():
    """A policy file carries no reward, so retuning one must still resume and export."""
    assert _difference(toy_fingerprint(), toy_scenario(reward_fn=lambda t: (0.0, True))) is None


def test_a_gymnasium_env_has_no_fingerprint():
    env = gym.make("CartPole-v1")
    try:
        assert scenario_fingerprint(env) is None
    finally:
        env.close()
