"""The sandbox is the root project: installed, importable, and self-declaring."""

from __future__ import annotations

import importlib
import importlib.metadata


def test_the_sandbox_module_is_importable():
    assert importlib.import_module("sandbox").__name__ == "sandbox"


def test_the_scenarios_subpackage_is_a_package_discovery_can_walk():
    """walk_packages needs __path__; a plain module would silently scan nothing."""
    package = importlib.import_module("sandbox.scenarios")
    assert hasattr(package, "__path__")


def test_the_sandbox_declares_its_scenario_package_as_an_entry_point():
    """The entry point is what makes discovery need no config field at all.

    Asserted against the installed metadata, not the pyproject text, because
    the metadata is what importlib.metadata reads and it only exists after a
    `uv sync`.
    """
    points = list(importlib.metadata.entry_points(group="aero_gym.scenarios"))
    assert [(p.name, p.value) for p in points] == [("sandbox", "sandbox.scenarios")]
