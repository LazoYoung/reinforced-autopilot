"""Global Constraint #3: within aero-train, only aero_train.envs may import
aero_gym -- it is what makes the CartPole integration test the cheap, honest
proof that the learning stack is correct rather than something that could be
accidentally exercising flight-model code. Nothing enforced this; it held only
because envs/factory.py doesn't import aero_gym yet (that branch arrives in
Phase 2), so the first violation anywhere else would have passed every test.

Uses ast rather than a substring search on the source text, because
aero_train genuinely has real, load-bearing *comments and docstrings* that
say "aero_gym" (cli.py:73, tracking.py:122, envs/factory.py:18-20) without a
single one of them being an import -- a grep for the word would either miss
real violations hiding after a `# noqa`-style false negative or, worse here,
false-positive on every one of these and have to be special-cased into
uselessness. ast.parse also means no module needs to be importable (no
JSBSim, no socket, no subprocess, aero_gym need not even be installed) for
this test to run against aero_train's source.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGES_DIR = Path(__file__).resolve().parents[2]
AERO_TRAIN_SRC = PACKAGES_DIR / "aero-train" / "src" / "aero_train"
AERO_GYM_SRC = PACKAGES_DIR / "aero-gym" / "src" / "aero_gym"


def _module_name(path: Path, src_root: Path, package: str) -> str:
    """Dotted module name for a source file, e.g. envs/factory.py -> package.envs.factory."""
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([package, *parts])


def _top_level_imports(path: Path) -> set[str]:
    """Top-level package names this module imports absolutely.

    Relative imports (`from . import x`, level > 0) are same-package internal
    imports by construction and are skipped -- they can never be aero_gym
    reaching into aero_train or vice versa across the package boundary this
    test checks.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_only_aero_train_envs_imports_aero_gym():
    violations = {
        _module_name(path, AERO_TRAIN_SRC, "aero_train"): sorted(imports)
        for path in sorted(AERO_TRAIN_SRC.rglob("*.py"))
        for imports in [_top_level_imports(path)]
        if "aero_gym" in imports
        and _module_name(path, AERO_TRAIN_SRC, "aero_train") != "aero_train.envs"
        and not _module_name(path, AERO_TRAIN_SRC, "aero_train").startswith("aero_train.envs.")
    }
    assert not violations, (
        f"aero_gym imported outside aero_train.envs by: {violations}; "
        "Global Constraint #3 reserves that import for aero_train.envs alone"
    )


def test_aero_gym_never_imports_aero_train():
    # The dependency is one-directional: aero_gym is the standalone simulation
    # package aero_train depends on, never the other way around.
    violations = {
        str(path.relative_to(AERO_GYM_SRC)): sorted(imports)
        for path in sorted(AERO_GYM_SRC.rglob("*.py"))
        for imports in [_top_level_imports(path)]
        if "aero_train" in imports
    }
    assert not violations, f"aero_train imported by aero_gym module(s): {violations}"
