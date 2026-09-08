from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"

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


def test_warmup_above_buffer_size_raises():
    # warmup_frames > buffer_size means len(buffer) can never reach warmup_frames
    # -- the ring buffer caps at buffer_size -- so update() would return None
    # forever and the run would silently never take a gradient step.
    with pytest.raises(ValueError, match="warmup_frames cannot exceed"):
        config_from_dict(
            {**MINIMAL, "algo": {"type": "dqn", "buffer_size": 100, "warmup_frames": 200}}
        )


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


@pytest.mark.parametrize("spelling", ["1e-3", "5e-4", "1E-3", "1e3", "-1e-3"])
def test_overrides_coerce_bare_exponent_notation_to_float(spelling):
    # PyYAML is YAML 1.1, whose float grammar requires a decimal point before
    # the exponent, so yaml.safe_load(spelling) alone returns a str here. This
    # is exactly the spelling anyone types for a learning-rate sweep --
    # DQNConfig.lr defaults to 5e-4 -- so apply_overrides must coerce it itself.
    data = apply_overrides(MINIMAL, [f"algo.lr={spelling}"])
    assert isinstance(data["algo"]["lr"], float)
    assert data["algo"]["lr"] == pytest.approx(float(spelling))


@pytest.mark.parametrize("spelling", ["1.0e-3", "0.001"])
def test_overrides_still_accept_yaml_native_float_spellings(spelling):
    # These already parse as float under YAML 1.1; guard against a fix that
    # only special-cases the exponent gap and breaks the spellings that worked.
    data = apply_overrides(MINIMAL, [f"algo.lr={spelling}"])
    assert data["algo"]["lr"] == pytest.approx(0.001)


def test_overrides_leave_non_numeric_strings_as_strings():
    data = apply_overrides(MINIMAL, ["run_name=wings-level"])
    assert data["run_name"] == "wings-level"


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


def test_env_needs_exactly_one_of_id_and_scenario():
    from aero_train.config import EnvConfig

    with pytest.raises(ValueError, match="exactly one"):
        EnvConfig()
    with pytest.raises(ValueError, match="exactly one"):
        EnvConfig(id="CartPole-v1", scenario="steady_flight")


def test_scenario_only_settings_are_rejected_on_the_gymnasium_branch():
    from aero_train.config import EnvConfig

    # match=rf"env\.{name}" rather than the shared "aero_gym scenario" suffix
    # every branch of this rejection shares -- otherwise a message that names
    # the wrong offending field would still satisfy the assertion.
    for name, kwargs in (
        ("render_mode", {"render_mode": "flightgear"}),
        ("scenario_kwargs", {"scenario_kwargs": {"aircraft": "737"}}),
        ("delta", {"delta": 0.2}),
    ):
        with pytest.raises(ValueError, match=rf"env\.{name}"):
            EnvConfig(id="CartPole-v1", **kwargs)


def test_the_episode_limit_has_exactly_one_home():
    from aero_train.config import EnvConfig

    with pytest.raises(ValueError, match="env.max_episode_steps"):
        EnvConfig(scenario="steady_flight", scenario_kwargs={"max_episode_steps": 10})


def test_render_mode_and_delta_are_validated():
    from aero_train.config import EnvConfig

    with pytest.raises(ValueError, match="flightgear"):
        EnvConfig(scenario="steady_flight", render_mode="opengl")
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        EnvConfig(scenario="steady_flight", delta=1.5)


def test_a_scenario_env_round_trips_through_a_config_dict():
    from aero_train.config import config_from_dict, config_to_dict

    # render_mode and delta are given non-default values on purpose: a round
    # trip that silently dropped either back to its default would still pass
    # an equality check seeded with the defaults, the same blind spot
    # test_yaml_round_trip exists to close for the rest of TrainConfig.
    cfg = config_from_dict(
        {
            "run_name": "flight",
            "env": {
                "scenario": "steady_flight",
                "scenario_kwargs": {"aircraft": "737"},
                "render_mode": "flightgear",
                "delta": 0.05,
            },
            "algo": {"type": "dqn"},
            # render_mode "flightgear" needs exactly one environment in total;
            # the default rollout (envs_per_worker=8) would fail that check.
            "rollout": {"n_workers": 1, "envs_per_worker": 1},
        }
    )
    assert cfg.env.scenario == "steady_flight"
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_flightgear_needs_exactly_one_environment_in_total():
    """Not `n_workers > 1`: one worker with the default 8 envs per worker would
    still launch eight viewers, all fighting for one screen."""
    from aero_train.config import config_from_dict

    base = {
        "run_name": "fg",
        "env": {"scenario": "steady_flight", "render_mode": "flightgear"},
        "algo": {"type": "dqn"},
    }
    with pytest.raises(ValueError, match="one environment"):
        config_from_dict({**base, "rollout": {"n_workers": 1, "envs_per_worker": 8}})
    cfg = config_from_dict({**base, "rollout": {"n_workers": 1, "envs_per_worker": 1}})
    assert cfg.env.render_mode == "flightgear"


def test_the_three_steady_flight_configs_fly_the_identical_task():
    """dqn/random/constant_steady_flight.yaml exist to be compared against each
    other; a baseline flown on a different env or a different rollout topology
    would not be a baseline. Parsing all three is not enough to show that --
    this compares the actual env: and rollout: dataclasses field by field, so a
    config drifting out of sync fails here instead of silently producing an
    uncomparable curve in MLflow."""
    dqn = load_config(CONFIGS_DIR / "dqn_steady_flight.yaml")
    random_ = load_config(CONFIGS_DIR / "random_steady_flight.yaml")
    constant = load_config(CONFIGS_DIR / "constant_steady_flight.yaml")

    for other in (random_, constant):
        assert dataclasses.asdict(dqn.env) == dataclasses.asdict(other.env)
        assert dataclasses.asdict(dqn.rollout) == dataclasses.asdict(other.rollout)

    # And the configs must actually differ on the one thing they're supposed
    # to: which algorithm flies. Otherwise the equality checks above could
    # pass trivially against three copies of the same file.
    assert dqn.algo.type == "dqn"
    assert random_.algo.type == "random"
    assert constant.algo.type == "constant"
