"""Registration, discovery, and the two ways discovery can go wrong."""

from __future__ import annotations

import importlib.metadata
import sys
import textwrap
from dataclasses import dataclass

import pytest

from aero_gym import registry
from aero_gym.scenario import InitialConditions, Scenario, fixed_setup
from conftest import constant_reward

IC = InitialConditions(
    latitude_deg=0.0, longitude_deg=0.0, altitude_ft=5000.0, heading_deg=0.0
)


@dataclass(frozen=True)
class _Params:
    altitude_ft: float = 5000.0
    aircraft: str = "c172p"


@pytest.fixture
def clean(monkeypatch):
    """An empty registry with discovery un-run, restored afterwards."""
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DISCOVERED", None)
    # Every `clean` test also stubs _entry_point_roots; skip that here and
    # discover() would import the real sandbox and strand it in sys.modules.
    monkeypatch.setattr(registry, "_entry_point_roots", lambda: ())
    return registry


def _write_package(
    tmp_path, monkeypatch, files: dict[str, str], root="probe", stub_roots=True
):
    """Put a real importable package on sys.path, with real module files.

    Real files rather than stubs because the failure being tested is an import
    failure, and only an actual import produces the traceback discovery keeps.
    `stub_roots=False` leaves `_entry_point_roots` alone, for a test that
    exercises the real one against a crafted `importlib.metadata` entry point.
    """
    package = tmp_path / root
    (package / "scenarios").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name, body in files.items():
        (package / "scenarios" / name).write_text(textwrap.dedent(body))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in list(sys.modules):
        if name == root or name.startswith(root + "."):
            del sys.modules[name]
    if stub_roots:
        monkeypatch.setattr(registry, "_entry_point_roots", lambda: (f"{root}.scenarios",))


GOOD = """
    from dataclasses import dataclass

    from aero_gym.registry import scenario
    from aero_gym.scenario import InitialConditions, Scenario, fixed_setup


    @dataclass(frozen=True)
    class Params:
        altitude_ft: float = 5000.0


    @scenario("good", params=Params)
    def build(params):
        return Scenario(
            name="good",
            setup_fn=fixed_setup(InitialConditions(0.0, 0.0, params.altitude_ft, 0.0)),
            reward_fn=lambda t: (0.0, False),
        )
"""

BROKEN = "this is not valid python ((\n"


def test_a_registered_scenario_can_be_looked_up(clean):
    @clean.scenario("hold")
    def build():
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    found = clean.lookup("hold")
    assert found.name == "hold"
    assert found.build is build
    assert found.params_type is None


def test_a_duplicate_name_raises_and_names_both_registrations(clean):
    """Last-wins would make the shadowed scenario invisible, with no error."""

    @clean.scenario("hold")
    def first():
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    with pytest.raises(ValueError, match=r"first.*second|second.*first"):

        @clean.scenario("hold")
        def second():
            return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)


def test_an_unknown_name_lists_what_did_register(clean):
    @clean.scenario("hold")
    def build():
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    with pytest.raises(ValueError, match=r"unknown scenario 'climb'.*'hold'"):
        clean.lookup("climb")


def test_an_empty_registry_blames_the_missing_install_not_the_name(clean):
    """A missing `uv sync` and a typo are different problems, and reporting the
    first as "unknown scenario" sends the reader hunting in the wrong file."""
    with pytest.raises(ValueError, match="uv sync"):
        clean.lookup("hold")


def test_a_broken_module_does_not_stop_a_good_one_registering(
    clean, tmp_path, monkeypatch
):
    """`broken.py` sorts before `good.py`, which is exactly how a naive
    discovery loop loses every scenario to one unrelated syntax error."""
    _write_package(tmp_path, monkeypatch, {"broken.py": BROKEN, "good.py": GOOD})
    report = clean.discover()
    assert [name for name, _ in report.failures] == ["probe.scenarios.broken"]
    assert "probe.scenarios.good" in report.modules
    assert [r.name for r in clean.registered()] == ["good"]


def test_a_broken_module_warns_when_the_wanted_scenario_was_found_anyway(
    clean, tmp_path, monkeypatch
):
    _write_package(tmp_path, monkeypatch, {"broken.py": BROKEN, "good.py": GOOD})
    with pytest.warns(RuntimeWarning, match="probe.scenarios.broken"):
        assert clean.lookup("good").name == "good"


def test_a_broken_module_is_reported_with_its_traceback_when_the_name_is_missing(
    clean, tmp_path, monkeypatch
):
    """The typo is very likely inside the module that failed to import, so the
    collected traceback belongs in the raise, not only in a warning."""
    _write_package(tmp_path, monkeypatch, {"broken.py": BROKEN, "good.py": GOOD})
    with pytest.raises(ValueError) as excinfo:
        clean.lookup("climb")
    assert "probe.scenarios.broken" in str(excinfo.value)
    assert "SyntaxError" in str(excinfo.value)


def test_discovery_walks_nested_subpackages(clean, tmp_path, monkeypatch):
    """pkgutil.iter_modules reaches one level only; walk_packages is required."""
    package = tmp_path / "probe"
    (package / "scenarios" / "deep").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "scenarios" / "__init__.py").write_text("")
    (package / "scenarios" / "deep" / "__init__.py").write_text("")
    (package / "scenarios" / "deep" / "nested.py").write_text(textwrap.dedent(GOOD))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in list(sys.modules):
        if name == "probe" or name.startswith("probe."):
            del sys.modules[name]
    monkeypatch.setattr(registry, "_entry_point_roots", lambda: ("probe.scenarios",))
    clean.discover()
    assert [r.name for r in clean.registered()] == ["good"]


def test_a_pkg_colon_attr_entry_point_value_still_discovers_its_scenarios(
    tmp_path, monkeypatch
):
    """`pkg.scenarios:attr` is a legal entry-point value and a plausible
    mistype of `pkg.scenarios`, since the README hands readers a snippet to
    copy; only the module half is importable, so `_entry_point_roots` must
    use `.module`, not the whole `.value`. Not the `clean` fixture: it stubs
    out `_entry_point_roots`, and this test exercises the real one.
    """
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DISCOVERED", None)
    _write_package(tmp_path, monkeypatch, {"good.py": GOOD}, stub_roots=False)
    point = importlib.metadata.EntryPoint(
        name="probe", value="probe.scenarios:build", group=registry.ENTRY_POINT_GROUP
    )
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda group=None: (point,)
    )
    registry.discover()
    assert [r.name for r in registry.registered()] == ["good"]


def test_discovery_runs_once_per_process(clean, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        registry, "_entry_point_roots", lambda: calls.append(1) or ()
    )
    clean.discover()
    clean.discover()
    assert len(calls) == 1
    clean.discover(force=True)
    assert len(calls) == 2


def test_params_are_built_from_a_mapping(clean):
    @clean.scenario("hold", params=_Params)
    def build(params):
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    built = clean.build_params(clean.lookup("hold"), {"aircraft": "737"})
    assert built == _Params(altitude_ft=5000.0, aircraft="737")


def test_an_unknown_param_names_the_field_and_lists_the_valid_ones(clean):
    """The property the old inspect.signature bind existed for: one failure at
    factory construction, not eight inside workers."""

    @clean.scenario("hold", params=_Params)
    def build(params):
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    with pytest.raises(ValueError, match=r"aircrafft.*altitude_ft.*aircraft"):
        clean.build_params(clean.lookup("hold"), {"aircrafft": "737"})


def test_a_scenario_with_no_params_rejects_params(clean):
    @clean.scenario("hold")
    def build():
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    with pytest.raises(ValueError, match="takes no parameters"):
        clean.build_params(clean.lookup("hold"), {"aircraft": "737"})


def test_build_scenario_passes_the_params_object_to_the_builder(clean):
    seen = []

    @clean.scenario("hold", params=_Params)
    def build(params):
        seen.append(params)
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    assert clean.build_scenario("hold", {"aircraft": "737"}).name == "hold"
    assert seen == [_Params(aircraft="737")]


def test_build_scenario_calls_a_no_params_builder_with_no_argument(clean):
    @clean.scenario("hold")
    def build():
        return Scenario(name="hold", setup_fn=fixed_setup(IC), reward_fn=constant_reward)

    assert clean.build_scenario("hold").name == "hold"
