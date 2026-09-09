"""Global Constraint #3: within aero-train, only aero_train.envs may import
aero_gym, and even there the import must be deferred to function scope --
it is what makes the CartPole integration test the cheap, honest proof that
the learning stack is correct rather than something that could be
accidentally exercising flight-model code. Before this test existed, nothing
enforced either half of that: a violation anywhere would have passed every
test that came before it.

Uses ast rather than a substring search on the source text, because
aero_train genuinely has real, load-bearing *comments and docstrings* that
say "aero_gym" (cli.py:73, tracking.py:122, envs/factory.py:34-38) without a
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


def _import_names(node: ast.AST) -> set[str]:
    """Top-level package name(s) named by a single Import/ImportFrom node.

    Relative imports (`from . import x`, level > 0) are same-package internal
    imports by construction and are skipped -- they can never be aero_gym
    reaching into aero_train or vice versa across the package boundary this
    test checks.
    """
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return {node.module.split(".")[0]}
    return set()


def _top_level_imports(path: Path) -> set[str]:
    """Top-level package names this module imports absolutely, anywhere in
    the file -- module scope or nested inside a function. Used to catch a
    dependency-direction violation regardless of where in the file it hides.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        names |= _import_names(node)
    return names


def _module_level_imports(path: Path) -> set[str]:
    """Top-level package names imported directly in the module body -- i.e.
    paid merely by `import`-ing this module, before any of its functions run.

    Only looks at `tree.body` (the module's direct statements), not the full
    walk `_top_level_imports` does, so an import nested inside a function is
    correctly excluded here even though that same import is a real,
    load-bearing dependency the other test must still see.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        names |= _import_names(node)
    return names


def _function_level_imports(path: Path) -> set[str]:
    """Top-level package names imported anywhere inside a function body --
    i.e. deferred until that function actually runs.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for outer in ast.walk(tree):
        if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(outer):
                if node is not outer:
                    names |= _import_names(node)
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


def test_the_aero_gym_import_in_envs_is_deferred_to_function_scope():
    """The CartPole integration test is only an honest proof that the
    learning stack works with no flight model nearby if merely `import
    aero_train.envs` (which the trainer and CLI both do) can never load
    aero_gym -- so no module-level `import aero_gym` may exist anywhere
    under aero_train.envs, in any of its files, not just factory.py.
    envs/__init__.py re-exports make_env_fn and is imported well outside any
    scenario branch, so a module-level import hiding in scale.py or delta.py
    would defeat the deferral just as surely as one in factory.py.
    """
    violations = {
        _module_name(path, AERO_TRAIN_SRC, "aero_train"): sorted(imports)
        for path in sorted((AERO_TRAIN_SRC / "envs").rglob("*.py"))
        for imports in [_module_level_imports(path)]
        if "aero_gym" in imports
    }
    assert not violations, (
        f"aero_gym imported at module scope by: {violations}; it must be "
        "imported inside a function so the Gymnasium branch never pays for it"
    )
    # And the deferred import must actually still exist -- otherwise this
    # test would pass equally well if the import vanished from factory.py
    # entirely, which would be a different, worse bug than a module-level one.
    factory_path = AERO_TRAIN_SRC / "envs" / "factory.py"
    assert "aero_gym" in _function_level_imports(factory_path), (
        "expected a function-level `import aero_gym` in envs/factory.py; found none"
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
