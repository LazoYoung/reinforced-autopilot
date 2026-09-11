"""ConsoleTracker: the only thing that makes a long run legible while it runs.

Every test here drives a fake clock rather than sleeping. Throttling is the
whole point of the class, and a test that slept for real would either take
5 seconds or prove nothing.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from aero_train.config import DQNConfig, EnvConfig, TrackingConfig, TrainConfig
from aero_train.tracking import (
    CompositeTracker,
    ConsoleTracker,
    MLflowTracker,
    NullTracker,
    Tracker,
    build_tracker,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make_cfg(**tracking):
    return TrainConfig(
        run_name="tracked-run",
        env=EnvConfig(scenario="steady_flight"),
        algo=DQNConfig(),
        seed=5,
        total_steps=10000,
        tracking=TrackingConfig(**tracking),
    )


def make_tracker(**tracking):
    stream, clock = io.StringIO(), FakeClock()
    tracker = ConsoleTracker(make_cfg(**tracking), stream=stream, clock=clock)
    return tracker, stream, clock


def iteration(frames, iteration_n=1, **extra):
    return {
        "progress/frames": float(frames),
        "progress/iteration": float(iteration_n),
        "throughput/fps": 1000.0,
        **extra,
    }


def test_console_tracker_satisfies_the_abc():
    tracker, _, _ = make_tracker()
    assert isinstance(tracker, Tracker)


def test_the_header_names_the_run_and_its_budget():
    """Printed once, so a scrollback (or a piped log) identifies which run it
    belongs to without going back to the YAML."""
    tracker, stream, _ = make_tracker()
    tracker.log_params(make_cfg())
    header = stream.getvalue()
    assert "tracked-run" in header
    assert "dqn" in header
    assert "steady_flight" in header
    assert "10000" in header      # total_steps


def test_iteration_lines_are_throttled_to_the_interval():
    """The reason this class exists rather than a bare print(): a 300k-frame
    run is ~1170 iterations, and an unthrottled line per iteration scrolls
    faster than it can be read."""
    tracker, stream, clock = make_tracker(console_interval_s=5.0)
    for i, t in enumerate([0.0, 1.0, 2.0, 3.0, 6.0]):
        clock.t = t
        tracker.log_metrics(iteration(frames=256 * (i + 1), iteration_n=i + 1), step=i)

    lines = stream.getvalue().splitlines()
    # t=0 (the first is never withheld) and t=6 (>= 5.0 since the last print).
    assert len(lines) == 2
    assert "256" in lines[0]
    assert "1280" in lines[1]


def test_the_first_iteration_prints_immediately():
    """A run that withheld its first line would look hung for `interval`
    seconds at exactly the moment the user is checking it started at all."""
    tracker, stream, _ = make_tracker(console_interval_s=600.0)
    tracker.log_metrics(iteration(frames=256), step=0)
    assert stream.getvalue().strip()


def test_eval_lines_are_never_throttled():
    """Evals are rare and are the reason to be watching. Suppressing one
    because an iteration line printed 2 seconds ago loses the only number in
    the stream that says whether the policy is any good."""
    tracker, stream, clock = make_tracker(console_interval_s=5.0)
    tracker.log_metrics(iteration(frames=256), step=0)        # prints at t=0
    clock.t = 0.5
    tracker.log_metrics(
        {
            "eval/return_mean": 58.3,
            "eval/length_mean": 91.0,
            "eval/return_per_step": 0.641,
            "eval/terminated_frac": 1.0,
        },
        step=1,
    )
    out = stream.getvalue()
    assert "58.3" in out
    assert "0.641" in out


def test_close_flushes_an_iteration_that_the_throttle_withheld():
    """interval has no reason to land on a run's final iteration, so without
    this the last thing printed is a stale line and the finishing frame count
    never appears at all."""
    tracker, stream, clock = make_tracker(console_interval_s=5.0)
    tracker.log_metrics(iteration(frames=256, iteration_n=1), step=0)
    clock.t = 1.0
    tracker.log_metrics(iteration(frames=512, iteration_n=2), step=1)   # withheld
    assert "512" not in stream.getvalue()

    tracker.close()
    # The iteration line specifically: close() also prints a footer carrying
    # the same frame count, which would pass with the flush deleted.
    iter_lines = [ln for ln in stream.getvalue().splitlines() if ln.startswith("iter")]
    assert len(iter_lines) == 2
    assert "512" in iter_lines[1]


def test_close_does_not_repeat_an_iteration_it_already_printed():
    tracker, stream, _ = make_tracker(console_interval_s=5.0)
    tracker.log_metrics(iteration(frames=256, iteration_n=1), step=0)
    tracker.close()
    assert stream.getvalue().count("iter") == 1


def test_a_line_shows_algorithm_metrics_when_the_algorithm_reports_them():
    tracker, stream, _ = make_tracker()
    tracker.log_metrics(iteration(frames=256, loss=0.0412, epsilon=0.94), step=0)
    line = stream.getvalue()
    assert "0.0412" in line
    assert "0.94" in line


def test_a_line_omits_algorithm_metrics_when_there_are_none():
    """The two baselines never take a gradient step, so Algorithm.update()
    returns None and no loss/epsilon key ever reaches the tracker. Formatting
    must degrade rather than KeyError: these are the runs DQN is measured
    against, so they cannot be the ones that crash."""
    tracker, stream, _ = make_tracker()
    tracker.log_metrics(iteration(frames=256), step=0)
    line = stream.getvalue()
    assert "loss" not in line
    assert "eps" not in line
    assert "256" in line


def test_a_line_omits_episode_stats_when_no_episode_finished():
    """Early iterations of a long-episode scenario complete no episode at all;
    _episode_metrics then reports only episode/count == 0."""
    tracker, stream, _ = make_tracker()
    tracker.log_metrics(iteration(frames=256, **{"episode/count": 0.0}), step=0)
    assert "ret" not in stream.getvalue()


def test_a_line_shows_episode_stats_when_episodes_finished():
    tracker, stream, _ = make_tracker()
    tracker.log_metrics(
        iteration(
            frames=256,
            **{
                "episode/count": 3.0,
                "episode/return_mean": 53.9,
                "episode/length_mean": 86.0,
                "episode/terminated_frac": 1.0,
            },
        ),
        step=0,
    )
    line = stream.getvalue()
    assert "53.9" in line
    assert "86" in line


def test_log_artifact_names_the_file_it_saved():
    tracker, stream, _ = make_tracker()
    tracker.log_artifact(Path("runs/x/checkpoints/best.pt"))
    assert "best.pt" in stream.getvalue()


def test_console_interval_must_not_be_negative():
    with pytest.raises(ValueError, match="console_interval_s"):
        TrackingConfig(console_interval_s=-1.0)


class _Recorder(Tracker):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def log_params(self, cfg) -> None:
        self.calls.append("params")

    def log_metrics(self, metrics, step) -> None:
        self.calls.append("metrics")

    def log_artifact(self, path) -> None:
        self.calls.append("artifact")

    def close(self) -> None:
        self.calls.append("close")


def test_composite_fans_every_call_out_to_every_tracker():
    a, b = _Recorder(), _Recorder()
    composite = CompositeTracker([a, b])
    composite.log_params(make_cfg())
    composite.log_metrics({"loss": 1.0}, step=1)
    composite.log_artifact(Path("x.pt"))
    composite.close()
    assert a.calls == ["params", "metrics", "artifact", "close"]
    assert b.calls == a.calls


def test_composite_closes_the_rest_even_when_one_close_raises():
    """close() runs in Trainer.run's finally block. A tracking backend failing
    to shut down must not strand the others, or MLflow's run stays open
    and the console footer would never print."""

    class _Exploding(_Recorder):
        def close(self) -> None:
            super().close()
            raise RuntimeError("backend went away")

    a, b = _Exploding(), _Recorder()
    with pytest.raises(RuntimeError, match="backend went away"):
        CompositeTracker([a, b]).close()
    assert b.calls == ["close"]


def test_build_tracker_keeps_the_console_when_mlflow_is_disabled():
    """tracking.enabled governs MLflow, not the terminal. A run with tracking
    off is exactly the run whose only output is stdout."""
    tracker = build_tracker(make_cfg(enabled=False))
    assert isinstance(tracker, ConsoleTracker)
    assert not isinstance(tracker, NullTracker)


def test_build_tracker_pairs_the_console_with_mlflow_when_enabled(tmp_path):
    cfg = make_cfg(enabled=True, experiment="unit", tracking_uri=tmp_path.as_uri())
    tracker = build_tracker(cfg)
    try:
        assert isinstance(tracker, CompositeTracker)
        kinds = [type(t) for t in tracker.trackers]
        assert ConsoleTracker in kinds
        assert MLflowTracker in kinds
    finally:
        tracker.close()


def test_build_tracker_returns_the_null_tracker_when_the_console_is_off_too():
    """Both channels disabled is the only way back to a fully silent run,
    what a benchmark harness or a test wants."""
    assert isinstance(build_tracker(make_cfg(enabled=False, console=False)), NullTracker)


@pytest.mark.parametrize(
    "seconds, expected",
    [(0.4, "0.4s"), (59.9, "59.9s"), (60.0, "1m 00s"), (2112.0, "35m 12s")],
)
def test_the_footer_reports_a_readable_duration(seconds, expected):
    """A 35-minute run reported as "0.6 hours", or a 12-second smoke test as
    "0.0 min", tells the reader nothing."""
    tracker, stream, clock = make_tracker()
    clock.t = seconds
    tracker.close()
    assert expected in stream.getvalue()
