"""Helpers shared by sandbox scenarios."""

from __future__ import annotations

from collections.abc import Mapping

from aero_gym.variables import ALL_ACTION_VARIABLES


def delta_fractions(
    delta: float | Mapping[str, float] | None,
) -> tuple[float, ...] | None:
    """Resolve a params delta to one fraction per action variable, in order.

    A number applies to every axis; a mapping must name each axis exactly once.
    """
    if delta is None:
        return None
    if not isinstance(delta, Mapping):
        return tuple(float(delta) for _ in ALL_ACTION_VARIABLES)
    names = {var.value for var in ALL_ACTION_VARIABLES}
    unknown, missing = sorted(set(delta) - names), sorted(names - set(delta))
    if unknown or missing:
        raise ValueError(
            f"delta must name every action axis: unknown {unknown}, missing {missing}"
        )
    return tuple(float(delta[var.value]) for var in ALL_ACTION_VARIABLES)
