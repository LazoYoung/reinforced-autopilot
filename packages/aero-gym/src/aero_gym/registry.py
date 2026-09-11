"""Scenario registry: a name, a builder, and how the two are found.

aero_gym owns the *concept* of a scenario and ships none. Tasks live in a
sandbox, or in any installed package that declares the same entry-point group,
and register themselves with the `@scenario` decorator. Nothing here ever
learns the name of a task or of the package holding one.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import pkgutil
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .scenario import Scenario

ENTRY_POINT_GROUP = "aero_gym.scenarios"


@dataclass(frozen=True)
class Registration:
    """One registered builder, and where it was declared.

    `module` and `qualname` are captured rather than derived on demand so a
    duplicate-name error can name both registering functions precisely.
    """

    name: str
    build: Callable[..., Scenario]
    params_type: type | None
    module: str
    qualname: str


@dataclass(frozen=True)
class DiscoveryReport:
    """What one discovery pass imported, and what it could not."""

    modules: tuple[str, ...]
    failures: tuple[tuple[str, str], ...]

    def failure_text(self) -> str:
        return "\n".join(f"[{name}]\n{tb}" for name, tb in self.failures)


_REGISTRY: dict[str, Registration] = {}
_DISCOVERED: DiscoveryReport | None = None


def scenario(name: str, *, params: type | None = None):
    """Register the decorated function as the builder for `name`.

    `params` names the scenario's frozen params dataclass, or None for a
    scenario with no knobs, in which case the builder takes no argument. It is
    passed explicitly rather than read from the annotation because
    `from __future__ import annotations` turns annotations into strings, and
    resolving them would fail in ways that report badly.
    """

    def decorate(fn: Callable[..., Scenario]) -> Callable[..., Scenario]:
        existing = _REGISTRY.get(name)
        if existing is not None:
            raise ValueError(
                f"scenario {name!r} is already registered by "
                f"{existing.module}.{existing.qualname}; "
                f"{fn.__module__}.{fn.__qualname__} would shadow it"
            )
        _REGISTRY[name] = Registration(
            name=name,
            build=fn,
            params_type=params,
            module=fn.__module__,
            qualname=fn.__qualname__,
        )
        return fn

    return decorate


def _entry_point_roots() -> tuple[str, ...]:
    """Package names to scan, one per entry point in the group."""
    points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    return tuple(point.value for point in points)


def _import_tree(
    root: str, modules: list[str], failures: list[tuple[str, str]]
) -> None:
    """Import `root` and every module under it, one try per module.

    A single unimportable file must not hide every other scenario, so failures
    are collected with their tracebacks instead of propagating.
    """
    try:
        package = importlib.import_module(root)
    except Exception:
        failures.append((root, traceback.format_exc()))
        return
    modules.append(root)
    if not hasattr(package, "__path__"):
        return

    def on_error(name: str) -> None:
        failures.append((name, traceback.format_exc()))

    # walk_packages, not iter_modules: iter_modules reaches one level only, so
    # a scenario in a subpackage would never be imported.
    for info in pkgutil.walk_packages(package.__path__, root + ".", onerror=on_error):
        try:
            importlib.import_module(info.name)
        except Exception:
            failures.append((info.name, traceback.format_exc()))
        else:
            modules.append(info.name)


def discover(force: bool = False) -> DiscoveryReport:
    """Import every scenario module named by the entry-point group.

    Idempotent: the walk runs once per process and the report is cached, since
    every worker process calls it and imports are cached anyway.
    """
    global _DISCOVERED
    if _DISCOVERED is not None and not force:
        return _DISCOVERED
    modules: list[str] = []
    failures: list[tuple[str, str]] = []
    for root in _entry_point_roots():
        _import_tree(root, modules, failures)
    _DISCOVERED = DiscoveryReport(tuple(modules), tuple(failures))
    return _DISCOVERED


def registered() -> tuple[Registration, ...]:
    """Everything registered so far. Does not trigger discovery."""
    return tuple(_REGISTRY.values())


def lookup(name: str) -> Registration:
    """Discover, then resolve `name` to its registration."""
    report = discover()
    found = _REGISTRY.get(name)
    if found is not None:
        if report.failures:
            warnings.warn(
                "some scenario modules failed to import and their scenarios are "
                f"unavailable: {[module for module, _ in report.failures]}\n"
                f"{report.failure_text()}",
                RuntimeWarning,
                stacklevel=2,
            )
        return found
    if not _REGISTRY:
        # A missing entry point and a typo are different problems, and calling
        # the first "unknown scenario" sends the reader hunting in the wrong file.
        raise ValueError(
            f"no scenarios were discovered from entry-point group "
            f"{ENTRY_POINT_GROUP!r}. Run `uv sync` so the sandbox is installed "
            f"and its entry point registered."
            + (f"\nimport failures:\n{report.failure_text()}" if report.failures else "")
        )
    raise ValueError(
        f"unknown scenario {name!r}; registered scenarios are {sorted(_REGISTRY)}, "
        f"discovered from {list(report.modules)}"
        + (
            f"\nsome modules failed to import, and the name may be inside one of "
            f"them:\n{report.failure_text()}"
            if report.failures
            else ""
        )
    )


def build_params(
    registration: Registration, values: Mapping[str, Any] | None = None
) -> Any | None:
    """Construct a scenario's params object, naming a field it does not have.

    Separate from build_scenario so a caller can bind the params once, eagerly,
    and defer only the Scenario construction into a per-environment callable.
    """
    values = dict(values or {})
    if registration.params_type is None:
        if values:
            raise ValueError(
                f"scenario {registration.name!r} takes no parameters, got "
                f"{sorted(values)}"
            )
        return None
    # Declaration order, not sorted: an error naming the fields should read
    # in the order the dataclass defines them, not alphabetically.
    valid = [f.name for f in dataclasses.fields(registration.params_type)]
    unknown = sorted(set(values) - set(valid))
    if unknown:
        raise ValueError(
            f"scenario {registration.name!r} has no parameter(s) {unknown}; "
            f"valid parameters are {valid}"
        )
    return registration.params_type(**values)


def build_scenario(
    name: str, params: Mapping[str, Any] | None = None
) -> Scenario:
    """Look up, bind the params, and build. The convenience form."""
    registration = lookup(name)
    bound = build_params(registration, params)
    return registration.build() if bound is None else registration.build(bound)
