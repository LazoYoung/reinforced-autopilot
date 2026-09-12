"""What the observations did outside their declared bounds, for the whole run.

`AeroGymEnv._to_array` clips every feature against its `Box`, which is the
safety net that makes hand-written bounds workable: without it one bad bound
would put the policy's input scaling somewhere it never trained. The clip is
silent by construction, so this module remembers what it caught and a caller
prints the summary once, at the end.

Recorded per (scenario name, feature name) rather than per environment. A tight
bound is a property of the scenario definition, and a run builds many
environments from one definition: one per vector sub-env, plus a fresh one for
every evaluation. Keying on names collapses those into a single line, and lets
the peak be the worst reading any of them saw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Saturation:
    """One feature's whole history of leaving its bounds, in its own unit."""

    scenario: str
    feature: str
    low: float
    high: float
    peak: float
    """The reading furthest outside the bounds, sign included."""
    overshoot: float
    """How far outside `peak` was. 0.0 when only NaNs were seen."""
    steps: int
    nans: int


_RECORD: dict[tuple[str, str], Saturation] = {}


def reset_saturation() -> None:
    """Forget every recorded excursion. For tests, and for a second run in one process."""
    _RECORD.clear()


def record_saturation(
    scenario: str, feature: str, low: float, high: float, value: float
) -> None:
    """Note one clipped reading. Cheap enough for the hot path, which only
    reaches here for a value already known to be out of bounds."""
    key = (scenario, feature)
    seen = _RECORD.get(key)
    if math.isnan(value):
        overshoot, peak = (0.0, math.nan) if seen is None else (seen.overshoot, seen.peak)
        nans = 1 if seen is None else seen.nans + 1
    else:
        overshoot = low - value if value < low else value - high
        nans = 0 if seen is None else seen.nans
        peak = value
        if seen is not None and seen.overshoot >= overshoot:
            overshoot, peak = seen.overshoot, seen.peak
    _RECORD[key] = Saturation(
        scenario=scenario,
        feature=feature,
        low=low,
        high=high,
        peak=peak,
        overshoot=overshoot,
        steps=1 if seen is None else seen.steps + 1,
        nans=nans,
    )


def saturation_report() -> tuple[Saturation, ...]:
    """Every feature that left its bounds this run, worst first.

    NaNs sort ahead of any overshoot: a bound that is merely tight costs
    resolution, while a NaN means the feature itself is broken.
    """
    return tuple(
        sorted(_RECORD.values(), key=lambda s: (-s.nans, -s.overshoot, s.feature))
    )


def format_saturation_report(report: tuple[Saturation, ...]) -> str:
    """The summary a caller prints at the end of a run. Empty when nothing saturated."""
    if not report:
        return ""
    lines = [
        "observation bounds exceeded; the agent saw the clipped value, not the peak:"
    ]
    for s in report:
        detail = (
            f"peaked at {s.peak:g} against [{s.low:g}, {s.high:g}], "
            f"{s.overshoot:g} over, on {s.steps} step(s)"
        )
        if s.nans:
            detail = f"{s.nans} NaN reading(s); " + (detail if s.steps > s.nans else "")
        lines.append(f"  {s.scenario}.{s.feature}: {detail.rstrip('; ')}")
    lines.append("  widen the bound in the scenario's params, or fix the feature.")
    return "\n".join(lines)
