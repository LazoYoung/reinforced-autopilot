"""Export: one file, named columns, and a check that catches a wrong file."""

from __future__ import annotations

import json
import os

import gymnasium as gym
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from aero_train import export
from aero_train.algo import Random
from aero_train.algo.dqn import DQN
from aero_train.config import DQNConfig
from aero_train.envs.features import scenario_of, trace_scenario
from aero_train.export import PolicyCheckError, check_policy_file, export_policy
from tests.flight_fakes import N_ACTIONS, N_FEATURES, toy_env_fn, toy_scenario

META = {"aero.config": json.dumps({"seed": 0}), "aero.frames": "0", "aero.scenario": "toy_hold"}


def _dqn() -> DQN:
    return DQN((N_FEATURES,), N_ACTIONS, DQNConfig(hidden_sizes=(16, 16)), seed=0)


def _check(path, policy):
    """Fly `path` against `policy` in a fresh toy env."""
    env = toy_env_fn()()
    try:
        return check_policy_file(path, policy, env, trace_scenario(scenario_of(env)), seed=0)
    finally:
        env.close()


@pytest.fixture
def exported(tmp_path):
    algo = _dqn()
    path = tmp_path / "policy.onnx"
    report = export_policy(path, algo, toy_env_fn(), seed=0, metadata=META)
    return algo, path, report


def test_export_writes_exactly_one_file(exported, tmp_path):
    _, _, report = exported
    assert sorted(p.name for p in tmp_path.iterdir()) == ["policy.onnx"]
    assert report.steps > 0 and report.flips == 0


def test_the_file_takes_and_gives_named_float_columns(exported):
    _, path, _ = exported
    model = onnx.load(str(path))
    assert [c.name for c in model.graph.input] == [
        "altitude", "heading", "roll", "pitch", "target_altitude_ft", "target_heading_deg",
        "prev.elevator", "prev.aileron", "prev.throttle",
    ]
    assert [c.name for c in model.graph.output] == ["elevator", "aileron", "throttle", "value"]
    for column in (*model.graph.input, *model.graph.output):
        tensor = column.type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT, column.name
        assert [d.dim_param for d in tensor.shape.dim] == ["batch"], column.name


def test_each_column_documents_itself_and_the_file_carries_its_metadata(exported):
    _, path, _ = exported
    model = onnx.load(str(path))
    docs = {c.name: c.doc_string for c in (*model.graph.input, *model.graph.output)}
    assert docs["target_altitude_ft"] == "ft, 4800 to 5200: altitude to hold"
    assert docs["prev.throttle"].startswith("norm, 0 to 1: 0 = idle, 1 = full throttle;")
    assert docs["value"].startswith("the policy's estimate")
    meta = {p.key: p.value for p in model.metadata_props}
    assert sorted(meta) == sorted(
        ["aero.format", "aero.agent_hz", "aero.aircraft", "aero.inputs", "aero.outputs", *META]
    )
    assert meta["aero.format"] == "1"
    assert meta["aero.agent_hz"] == "10.0"
    assert meta["aero.aircraft"] == "c172p"
    assert {key: meta[key] for key in META} == META


def test_the_metadata_tells_a_host_each_columns_role_unit_and_range(exported):
    """What a mode control panel needs: which inputs are targets, and their dials' range."""
    _, path, _ = exported
    model = onnx.load(str(path))
    meta = {p.key: p.value for p in model.metadata_props}
    inputs, outputs = json.loads(meta["aero.inputs"]), json.loads(meta["aero.outputs"])
    assert list(inputs) == [c.name for c in model.graph.input]
    assert list(outputs) == [c.name for c in model.graph.output]
    assert inputs["target_altitude_ft"] == {
        "role": "target", "unit": "ft", "low": 4800.0, "high": 5200.0,
        "description": "altitude to hold",
    }
    assert inputs["altitude"]["role"] == "reading"
    assert (inputs["altitude"]["low"], inputs["altitude"]["high"]) == (-1500.0, 85000.0)
    assert inputs["prev.throttle"] | {"description": ""} == {
        "role": "prev", "unit": "norm", "low": 0.0, "high": 1.0, "description": "",
    }
    assert outputs["elevator"]["role"] == "command"
    assert (outputs["elevator"]["low"], outputs["elevator"]["high"]) == (-1.0, 1.0)
    assert set(outputs["value"]) == {"role", "description"}
    assert outputs["value"]["role"] == "value"


def test_the_file_runs_on_any_batch_size(exported):
    _, path, _ = exported
    session = ort.InferenceSession(str(path))
    feed = {c.name: np.zeros(3, np.float32) for c in session.get_inputs()}
    assert [out.shape for out in session.run(None, feed)] == [(3,)] * 4


def test_readings_bind_by_name(exported):
    """Swapping two readings' values between their names must change the answer."""
    _, path, _ = exported
    session = ort.InferenceSession(str(path))
    feed = {c.name: np.zeros(1, np.float32) for c in session.get_inputs()}
    feed |= {
        "altitude": np.array([5100.0], np.float32),
        "heading": np.array([10.0], np.float32),
        "target_altitude_ft": np.array([5000.0], np.float32),
        "target_heading_deg": np.array([20.0], np.float32),
    }
    swapped = feed | {"altitude": feed["heading"], "heading": feed["altitude"]}
    (value,) = session.run(["value"], feed)
    (value_swapped,) = session.run(["value"], swapped)
    assert value[0] != pytest.approx(value_swapped[0])


def _rename(model, a: str, b: str, columns: str) -> None:
    """Swap two graph inputs' or outputs' names, rewiring every node that uses them."""
    swap = {a: b, b: a}
    for node in model.graph.node:
        wires = node.input if columns == "input" else node.output
        wires[:] = [swap.get(w, w) for w in wires]
    for port in model.graph.input if columns == "input" else model.graph.output:
        port.name = swap.get(port.name, port.name)


@pytest.mark.parametrize(
    "a, b, columns, match",
    [
        ("altitude", "heading", "input", r"step 0: value"),
        ("prev.elevator", "prev.aileron", "input", r"step 0: .*no action reaches"),
        ("elevator", "aileron", "output", r"step 0: .*no action reaches"),
    ],
    ids=["readings", "prev", "actions"],
)
def test_the_check_rejects_a_file_with_two_names_swapped(exported, tmp_path, a, b, columns, match):
    algo, path, _ = exported
    model = onnx.load(str(path))
    _rename(model, a, b, columns)
    bad = tmp_path / "swapped.onnx"
    onnx.save(model, str(bad))
    with pytest.raises(PolicyCheckError, match=match):
        _check(bad, algo.policy_module())


class _FlipOn(torch.nn.Module):
    """The policy, except that on the listed calls it picks the next action over."""

    def __init__(self, policy, calls):
        super().__init__()
        self.policy, self.calls, self.count = policy, set(calls), 0

    def forward(self, obs):
        action, value = self.policy(obs)
        if self.count in self.calls:
            action = (action + 1) % N_ACTIONS
        self.count += 1
        return action, value


def test_one_near_tie_flip_is_tolerated_and_counted(exported):
    algo, path, _ = exported
    assert _check(path, _FlipOn(algo.policy_module(), {3})).flips == 1


def test_flips_beyond_one_percent_fail_the_check(exported):
    """Three flips exceed max(1, steps // 100) for any episode of 200 steps or fewer."""
    algo, path, _ = exported
    with pytest.raises(PolicyCheckError, match=r"3 of \d+ steps"):
        _check(path, _FlipOn(algo.policy_module(), {3, 5, 7}))


def test_the_check_stops_at_its_step_cap(tmp_path, monkeypatch):
    """The cap, not the scenario, ends this episode: nothing terminates it."""
    monkeypatch.setattr(export, "CHECK_MAX_STEPS", 25)
    scenario = toy_scenario(max_episode_steps=300, reward_fn=lambda t: (1.0, False))
    report = export_policy(
        tmp_path / "p.onnx", _dqn(), toy_env_fn(scenario), seed=0, metadata=META
    )
    assert report.steps == 25


def test_a_failed_check_keeps_an_existing_file_and_leaves_no_temp(tmp_path, monkeypatch):
    path = tmp_path / "policy.onnx"
    path.write_bytes(b"the previous export")

    def fail(*args, **kwargs):
        raise PolicyCheckError("step 0: value expected 1, got 2")

    monkeypatch.setattr(export, "check_policy_file", fail)
    with pytest.raises(PolicyCheckError):
        export_policy(path, _dqn(), toy_env_fn(), seed=0, metadata=META)
    assert path.read_bytes() == b"the previous export"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["policy.onnx"]


def test_a_failed_rename_leaves_no_temp(tmp_path):
    """A rename failure, not just a failed check, must still clean up the temp file."""
    path = tmp_path / "policy.onnx"
    path.mkdir()
    with pytest.raises(OSError):
        export_policy(path, _dqn(), toy_env_fn(), seed=0, metadata=META)
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_an_algorithm_without_a_policy_is_refused_before_any_env_is_built(tmp_path):
    def no_env():
        raise AssertionError("no env should be built")

    with pytest.raises(NotImplementedError, match="Random cannot be exported"):
        export_policy(tmp_path / "p.onnx", Random(N_ACTIONS), no_env, seed=0, metadata=META)


def test_a_policy_without_the_stepping_wrapper_is_refused(tmp_path):
    with pytest.raises(ValueError, match="DeltaAction"):
        export_policy(
            tmp_path / "p.onnx", _dqn(), lambda: gym.make("CartPole-v1"), seed=0, metadata=META
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("key", ["aero.config", "aero.frames", "aero.scenario"])
def test_export_needs_what_only_the_checkpoint_knows(tmp_path, key):
    def no_env():
        raise AssertionError("no env should be built")

    metadata = {k: v for k, v in META.items() if k != key}
    with pytest.raises(ValueError, match=key):
        export_policy(tmp_path / "p.onnx", _dqn(), no_env, seed=0, metadata=metadata)


@pytest.mark.parametrize("key", ["aero.format", "aero.agent_hz", "aero.inputs"])
def test_export_refuses_metadata_it_writes_itself(tmp_path, key):
    def no_env():
        raise AssertionError("no env should be built")

    with pytest.raises(ValueError, match=key):
        export_policy(tmp_path / "p.onnx", _dqn(), no_env, seed=0, metadata=META | {key: "x"})


def test_missing_onnx_packages_name_the_extra(tmp_path, monkeypatch):
    real = export.importlib.util.find_spec
    monkeypatch.setattr(
        export.importlib.util,
        "find_spec",
        lambda name, *args: None if name == "onnxscript" else real(name, *args),
    )
    with pytest.raises(ModuleNotFoundError, match=r"onnxscript.*aero-train\[export\]"):
        export_policy(tmp_path / "p.onnx", _dqn(), toy_env_fn(), seed=0, metadata=META)


def test_a_target_that_never_varied_reaches_the_host_as_a_one_value_range(tmp_path):
    """Zero jitter declares low == high; a host must see exactly that, not a guess."""
    from aero_gym.scenario import ObservationFeature
    from aero_gym.variables import FlightVariable as V, VariableSpec

    base = toy_scenario()
    airspeed_error = ObservationFeature(
        "airspeed_error_kts",
        VariableSpec("kt", -20.0, 20.0, "airspeed minus target"),
        lambda obs, ctx: obs[V.AIRSPEED] - ctx["target_airspeed_kts"],
    )
    scenario = toy_scenario(observation_features=base.observation_features + (airspeed_error,))
    algo = DQN((N_FEATURES + 1,), N_ACTIONS, DQNConfig(hidden_sizes=(16, 16)), seed=0)
    path = tmp_path / "p.onnx"
    export_policy(path, algo, toy_env_fn(scenario), seed=0, metadata=META)
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    target = json.loads(meta["aero.inputs"])["target_airspeed_kts"]
    assert (target["role"], target["low"], target["high"]) == ("target", 100.0, 100.0)



def test_the_check_turns_off_onnxruntime_telemetry(exported, monkeypatch):
    """Its upload thread can abort the process at exit, after the file is written."""
    _, path, _ = exported
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    _check(path, _dqn().policy_module())
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"


def test_the_check_keeps_a_telemetry_setting_the_caller_chose(exported, monkeypatch):
    _, path, _ = exported
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    _check(path, _dqn().policy_module())
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "0"
