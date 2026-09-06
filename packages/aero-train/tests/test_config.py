from __future__ import annotations

import textwrap

import pytest

from aero_train.config import (
    DQNConfig,
    TrainConfig,
    apply_overrides,
    config_from_dict,
    config_to_dict,
    flatten_config,
    load_config,
)

MINIMAL = {
    "run_name": "t",
    "env": {"id": "CartPole-v1"},
    "algo": {"type": "dqn"},
}


def test_defaults_fill_in():
    cfg = config_from_dict(MINIMAL)
    assert cfg.run_name == "t"
    assert cfg.seed == 0
    assert isinstance(cfg.algo, DQNConfig)
    assert cfg.rollout.engine == "local"


def test_hidden_sizes_coerced_from_yaml_list_to_tuple():
    cfg = config_from_dict({**MINIMAL, "algo": {"type": "dqn", "hidden_sizes": [32, 16]}})
    assert cfg.algo.hidden_sizes == (32, 16)


def test_hidden_sizes_rejects_non_integer_values():
    # A float would silently truncate (3.7 -> 3), giving a network with different
    # layer widths than the config asked for; that must fail loudly instead.
    with pytest.raises(ValueError, match="hidden_sizes"):
        config_from_dict({**MINIMAL, "algo": {"type": "dqn", "hidden_sizes": [3.7, 2.2]}})


def test_hidden_sizes_rejects_bools():
    # bool is a subclass of int in Python, so this must be excluded explicitly.
    with pytest.raises(ValueError, match="hidden_sizes"):
        config_from_dict({**MINIMAL, "algo": {"type": "dqn", "hidden_sizes": [True]}})


def test_unknown_key_raises_type_error():
    with pytest.raises(TypeError, match="lr_rate"):
        config_from_dict({**MINIMAL, "algo": {"type": "dqn", "lr_rate": 0.1}})


def test_unknown_algo_type_raises_value_error():
    with pytest.raises(ValueError, match="ppo"):
        config_from_dict({**MINIMAL, "algo": {"type": "ppo"}})


def test_frames_per_iter_must_divide_the_topology():
    with pytest.raises(ValueError, match="frames_per_iter"):
        config_from_dict({**MINIMAL, "rollout": {"envs_per_worker": 8, "frames_per_iter": 100}})


def test_total_steps_below_warmup_raises():
    with pytest.raises(ValueError, match="warmup"):
        config_from_dict({**MINIMAL, "total_steps": 100, "algo": {"type": "dqn", "warmup_frames": 1000}})


def test_local_engine_rejects_multiple_workers():
    with pytest.raises(ValueError, match="n_workers"):
        config_from_dict({**MINIMAL, "rollout": {"engine": "local", "n_workers": 4}})


def test_overrides_coerce_scalar_types():
    data = apply_overrides(
        MINIMAL, ["rollout.n_workers=4", "algo.lr=0.001", "tracking.enabled=false"]
    )
    assert data["rollout"]["n_workers"] == 4
    assert data["algo"]["lr"] == pytest.approx(0.001)
    assert data["tracking"]["enabled"] is False


def test_overrides_do_not_mutate_the_input():
    data = dict(MINIMAL)
    apply_overrides(data, ["seed=7"])
    assert "seed" not in data


def test_override_cannot_bypass_validation():
    # applied to the raw dict *before* the dataclass is built, so __post_init__ still runs
    with pytest.raises(ValueError, match="frames_per_iter"):
        config_from_dict(apply_overrides(MINIMAL, ["rollout.frames_per_iter=100"]))


def test_malformed_override_raises():
    with pytest.raises(ValueError, match="key.path=value"):
        apply_overrides(MINIMAL, ["rollout.n_workers"])


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        textwrap.dedent(
            """
            run_name: yaml-run
            seed: 3
            env:
              id: CartPole-v1
            algo:
              type: dqn
              hidden_sizes: [64, 64]
            rollout:
              envs_per_worker: 4
              frames_per_iter: 64
            """
        )
    )
    cfg = load_config(path)
    assert cfg.seed == 3
    assert cfg.algo.hidden_sizes == (64, 64)
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_flatten_config_produces_dotted_string_params():
    flat = flatten_config(config_from_dict(MINIMAL))
    assert flat["rollout.n_workers"] == "1"
    assert flat["algo.type"] == "dqn"
    assert flat["env.id"] == "CartPole-v1"
