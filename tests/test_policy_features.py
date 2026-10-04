"""Every configured scenario's features survive export: traced, run by
onnxruntime, and compared with the float path the env computes them on."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml

from aero_gym.registry import build_scenario
from aero_gym.variables import OBSERVATION_SPECS
from aero_train.config import load_config
from aero_train.envs.features import trace_scenario
from aero_train.tensors import column_shapes, example_columns

CONFIGS = sorted(
    path
    for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")
    if yaml.safe_load(path.read_text())["env"].get("scenario")
)
SAMPLES = 4000


class _Unclipped(torch.nn.Module):
    """The traced features before the clip, which would hide most of their range."""

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, *inputs):
        return self.features.values(*inputs)


def test_there_are_configs_to_check():
    assert CONFIGS


@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.stem)
def test_a_configs_features_run_in_onnx_as_they_do_in_the_env(config, tmp_path):
    cfg = load_config(config)
    scenario = build_scenario(cfg.env.scenario, cfg.env.params)
    traced = trace_scenario(scenario)
    names = [p.name for p in traced.inputs]
    path = tmp_path / "features.onnx"
    with warnings.catch_warnings():
        # Every input shares one dim name ("batch") on purpose; this warning
        # is about that, not about anything wrong with the graph.
        warnings.filterwarnings(
            "ignore", message=r"# The axis name: batch will not be used.*", category=UserWarning
        )
        torch.onnx.export(
            _Unclipped(traced.features).eval(),
            example_columns(len(names)),
            str(path),
            dynamo=True,
            input_names=names,
            dynamic_shapes=column_shapes(len(names)),
            external_data=False,
            verbose=False,
        )
    rng = np.random.default_rng(0)
    columns = {
        var.value: rng.uniform(OBSERVATION_SPECS[var].low, OBSERVATION_SPECS[var].high, SAMPLES)
        for var in traced.readings
    }
    columns |= {
        c.name: rng.uniform(c.spec.low, c.spec.high, SAMPLES)
        for c in scenario.context_variables
        if c.name in traced.targets
    }
    columns = {name: values.astype(np.float32) for name, values in columns.items()}
    (got,) = ort.InferenceSession(str(path)).run(None, columns)
    want = [
        [
            feature.fn(
                {var: float(columns[var.value][i]) for var in traced.readings},
                {name: float(columns[name][i]) for name in traced.targets},
            )
            for feature in scenario.observation_features
        ]
        for i in range(SAMPLES)
    ]
    # float32 against float64: near 85000 ft one float32 step is 0.008 ft.
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-2)
