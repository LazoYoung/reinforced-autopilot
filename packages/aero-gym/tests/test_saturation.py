"""The saturation record: what the agent's observations did outside their bounds."""

from __future__ import annotations

import pytest

from aero_gym.saturation import (
    format_saturation_report,
    record_saturation,
    reset_saturation,
    saturation_report,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_saturation()
    yield
    reset_saturation()


def test_nothing_recorded_is_an_empty_report_and_empty_text():
    """A clean run must print nothing at all, or the summary becomes noise of
    its own at the end of every run that was fine."""
    assert saturation_report() == ()
    assert format_saturation_report(saturation_report()) == ""


def test_the_peak_survives_a_later_smaller_excursion():
    """The point of reporting at the end rather than on first sight: the worst
    excursion is what tells the author where to put the bound."""
    for value in (34.0, 51.2, 40.0):
        record_saturation("steady_flight", "pitch_rate", -30.0, 30.0, value)
    (entry,) = saturation_report()
    assert entry.peak == 51.2
    assert entry.overshoot == pytest.approx(21.2)
    assert entry.steps == 3


def test_a_low_side_excursion_measures_its_own_overshoot():
    record_saturation("steady_flight", "roll", -75.0, 75.0, -81.1)
    (entry,) = saturation_report()
    assert entry.peak == -81.1
    assert entry.overshoot == pytest.approx(6.1)


def test_the_further_side_wins_when_a_feature_leaves_both_bounds():
    """Peak is the reading furthest outside, not the largest signed value."""
    record_saturation("steady_flight", "roll", -75.0, 75.0, 80.0)
    record_saturation("steady_flight", "roll", -75.0, 75.0, -120.0)
    (entry,) = saturation_report()
    assert entry.peak == -120.0
    assert entry.overshoot == pytest.approx(45.0)


def test_one_entry_per_scenario_and_feature_however_many_environments():
    """Eight vector sub-envs plus one per evaluation all report the same
    scenario, so keying on the pair is what collapses them into one line."""
    for _ in range(10):
        record_saturation("steady_flight", "pitch_rate", -30.0, 30.0, 44.0)
    record_saturation("steady_flight", "roll", -75.0, 75.0, 80.0)
    record_saturation("wings_level", "pitch_rate", -30.0, 30.0, 44.0)
    assert len(saturation_report()) == 3
    assert saturation_report()[0].steps == 10


def test_a_nan_is_counted_apart_from_an_overshoot():
    """NaN trips the same out-of-bounds comparison, since NaN never equals
    itself, but it is a broken feature rather than a tight bound."""
    record_saturation("steady_flight", "aoa", -20.0, 20.0, float("nan"))
    (entry,) = saturation_report()
    assert entry.nans == 1
    assert entry.overshoot == 0.0


def test_the_report_is_worst_first():
    record_saturation("steady_flight", "roll", -75.0, 75.0, 80.0)
    record_saturation("steady_flight", "pitch_rate", -30.0, 30.0, 60.0)
    assert [e.feature for e in saturation_report()] == ["pitch_rate", "roll"]


def test_the_text_names_the_feature_the_peak_and_the_bound():
    """The three numbers an author needs to decide what to widen it to."""
    record_saturation("steady_flight", "pitch_rate", -30.0, 30.0, 51.2)
    text = format_saturation_report(saturation_report())
    assert "pitch_rate" in text
    assert "51.2" in text
    assert "30" in text
    assert "clipped" in text
