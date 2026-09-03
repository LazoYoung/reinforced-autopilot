from __future__ import annotations

import itertools
import os
import plistlib
import socket
import threading
import types

import pytest

from aerogym.simulators.jsbsim import flightgear
from aerogym.simulators.jsbsim.flightgear import FlightGearViewer, find_fgfs


BUNDLE = "/Applications/FlightGear.app"


class _FakeProcess:
    """Stands in for a running fgfs subprocess."""

    returncode = None
    pid = 4321

    def poll(self):
        return None


def _fake_telnet_server(ready_event: threading.Event, replies: list[bytes]):
    """One-shot TCP server imitating FlightGear's telnet property server."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        ready_event.set()
        for reply in replies:
            conn, _ = server.accept()
            with conn:
                conn.recv(1024)
                conn.sendall(reply)
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    ready_event.wait(5.0)
    return port


def test_wait_until_ready_returns_true_once_scenery_loaded():
    port = _fake_telnet_server(
        threading.Event(),
        [
            b"/sim/sceneryloaded = 'false' (bool)\r\n/> ",
            b"/sim/sceneryloaded = 'true' (bool)\r\n/> ",
        ],
    )
    viewer = FlightGearViewer(telnet_port=port)
    viewer._process = _FakeProcess()
    assert viewer.wait_until_ready(timeout=10.0, poll_interval=0.05)


def test_wait_until_ready_times_out_with_warning():
    viewer = FlightGearViewer(telnet_port=1)  # nothing listens on port 1
    viewer._process = _FakeProcess()
    with pytest.warns(UserWarning, match="Timed out"):
        assert not viewer.wait_until_ready(timeout=0.2, poll_interval=0.05)


def test_wait_until_ready_without_process_is_a_noop():
    viewer = FlightGearViewer()
    assert not viewer.wait_until_ready(timeout=0.1)


def _make_bundle(root, name: str, executable: str = "FlightGear") -> str:
    """Create a minimal macOS FlightGear.app layout under ``root``.

    Mirrors the shipped 2024.x bundle: the binary is named after the app and
    declared in Info.plist, and no base package lives inside the bundle.
    """
    bundle = root / name
    macos = bundle / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    (macos / executable).write_text("")
    with open(bundle / "Contents" / "Info.plist", "wb") as info:
        plistlib.dump({"CFBundleExecutable": executable}, info)
    return str(bundle)


def _bundled(bundle: str, executable: str = "FlightGear") -> str:
    return os.path.join(bundle, "Contents", "MacOS", executable)


@pytest.fixture
def no_ambient_fgfs(monkeypatch):
    """Neutralize the override, PATH and Windows lookups so only globs matter."""
    monkeypatch.delenv("AEROGYM_FGFS", raising=False)
    monkeypatch.setattr(flightgear.shutil, "which", lambda _: None)
    monkeypatch.setattr(flightgear, "_WINDOWS_GLOBS", ())


def test_find_fgfs_locates_macos_app_bundle(tmp_path, monkeypatch, no_ambient_fgfs):
    bundle = _make_bundle(tmp_path, "FlightGear.app")
    monkeypatch.setattr(flightgear, "_MACOS_APP_GLOBS", (f"{tmp_path}/FlightGear*.app",))
    assert find_fgfs() == _bundled(bundle)


def test_find_fgfs_locates_a_bundle_with_the_legacy_fgfs_name(
    tmp_path, monkeypatch, no_ambient_fgfs
):
    bundle = _make_bundle(tmp_path, "FlightGear.app", executable="fgfs")
    monkeypatch.setattr(flightgear, "_MACOS_APP_GLOBS", (f"{tmp_path}/FlightGear*.app",))
    assert find_fgfs() == _bundled(bundle, "fgfs")


def test_find_fgfs_prefers_the_newest_bundle(tmp_path, monkeypatch, no_ambient_fgfs):
    _make_bundle(tmp_path, "FlightGear-2020.3.19.app")
    newest = _make_bundle(tmp_path, "FlightGear-2024.1.1.app")
    monkeypatch.setattr(flightgear, "_MACOS_APP_GLOBS", (f"{tmp_path}/FlightGear*.app",))
    assert find_fgfs() == _bundled(newest)


def test_find_fgfs_skips_a_bundle_without_an_executable(tmp_path, monkeypatch, no_ambient_fgfs):
    (tmp_path / "FlightGear-2024.app").mkdir()  # empty bundle, e.g. a failed install
    usable = _make_bundle(tmp_path, "FlightGear-2020.app")
    monkeypatch.setattr(flightgear, "_MACOS_APP_GLOBS", (f"{tmp_path}/FlightGear*.app",))
    assert find_fgfs() == _bundled(usable)


def test_find_fgfs_expands_a_home_relative_glob(tmp_path, monkeypatch, no_ambient_fgfs):
    monkeypatch.setenv("HOME", str(tmp_path))
    apps = tmp_path / "Applications"
    apps.mkdir()
    bundle = _make_bundle(apps, "FlightGear.app")
    monkeypatch.setattr(flightgear, "_MACOS_APP_GLOBS", ("~/Applications/FlightGear*.app",))
    assert find_fgfs() == _bundled(bundle)


def test_override_may_name_an_app_bundle(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, "FlightGear.app")
    monkeypatch.setenv("AEROGYM_FGFS", bundle)
    assert find_fgfs() == _bundled(bundle)


def test_override_naming_a_plain_executable_is_used_verbatim(monkeypatch):
    monkeypatch.setenv("AEROGYM_FGFS", "/opt/flightgear/bin/fgfs")
    assert find_fgfs() == "/opt/flightgear/bin/fgfs"


def _captured_launch(monkeypatch, executable: str):
    """Launch a viewer with Popen stubbed out; returns the call kwargs/args."""
    calls = {}

    def fake_popen(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(flightgear.subprocess, "Popen", fake_popen)
    assert FlightGearViewer(executable=executable).launch()
    return calls


def test_launch_does_not_override_cwd_off_windows(monkeypatch):
    monkeypatch.setattr(flightgear.os, "name", "posix")
    calls = _captured_launch(monkeypatch, "/usr/bin/fgfs")
    assert calls["kwargs"]["cwd"] is None


def test_launch_runs_from_the_bin_dir_on_windows(monkeypatch):
    monkeypatch.setattr(flightgear.os, "name", "nt")
    calls = _captured_launch(monkeypatch, "/opt/fg/bin/fgfs")
    assert calls["kwargs"]["cwd"] == "/opt/fg/bin"


def _launch_capturing_threads(monkeypatch, executable: str, platform: str):
    """Launch a viewer on a pretend platform, capturing threads it starts."""
    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._call = (target, args)

        def start(self):
            started.append(self._call)

    monkeypatch.setattr(flightgear.sys, "platform", platform)
    monkeypatch.setattr(flightgear.threading, "Thread", FakeThread)
    _captured_launch(monkeypatch, executable)
    return started


def test_launch_activates_the_bundle_on_macos(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, "FlightGear.app")
    started = _launch_capturing_threads(monkeypatch, _bundled(bundle), "darwin")
    assert started == [(flightgear._activate_bundle, (bundle, _FakeProcess.pid))]


def test_launch_does_not_activate_off_macos(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, "FlightGear.app")
    assert _launch_capturing_threads(monkeypatch, _bundled(bundle), "linux") == []


def test_launch_does_not_activate_a_non_bundled_executable(monkeypatch):
    assert _launch_capturing_threads(monkeypatch, "/usr/bin/fgfs", "darwin") == []


def _stub_activation(monkeypatch, registered, frontmost):
    """Record the ``open`` calls _activate_bundle makes against stubbed lookups."""
    opens = []
    monkeypatch.setattr(flightgear, "_lsappinfo_name", lambda _: next(registered))
    monkeypatch.setattr(flightgear, "_frontmost_app_name", lambda: next(frontmost))
    monkeypatch.setattr(flightgear.subprocess, "run", lambda cmd, **kw: opens.append(cmd))
    monkeypatch.setattr(flightgear.time, "sleep", lambda _: None)
    return opens


def test_activate_bundle_waits_for_launch_services_to_know_the_process(monkeypatch):
    """``open -a`` before registration starts a second copy, so never send it."""
    opens = _stub_activation(
        monkeypatch,
        registered=iter(["", "", "FlightGear", "FlightGear"]),
        frontmost=iter(["Terminal", "FlightGear"]),
    )
    flightgear._activate_bundle(BUNDLE, pid=4321, attempts=4, interval=0)
    assert opens == [["/usr/bin/open", "-a", BUNDLE]]


def test_activate_bundle_stops_once_the_app_is_frontmost(monkeypatch):
    """Don't keep stealing focus after the window has come up."""
    opens = _stub_activation(
        monkeypatch,
        registered=itertools.repeat("FlightGear"),
        frontmost=iter(["Terminal", "FlightGear"]),
    )
    flightgear._activate_bundle(BUNDLE, pid=4321, attempts=5, interval=0)
    assert opens == [["/usr/bin/open", "-a", BUNDLE]]


def test_activate_bundle_gives_up_after_its_attempts(monkeypatch):
    opens = _stub_activation(
        monkeypatch,
        registered=itertools.repeat("FlightGear"),
        frontmost=itertools.repeat("Terminal"),
    )
    flightgear._activate_bundle(BUNDLE, pid=4321, attempts=3, interval=0)
    assert len(opens) == 3


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('"LSDisplayName"="FlightGear"\n', "FlightGear"),
        ('"LSDisplayName"=[ NULL ] \n', ""),  # process LaunchServices doesn't know
        ("", ""),
    ],
)
def test_lsappinfo_name_parsing(monkeypatch, output, expected):
    monkeypatch.setattr(
        flightgear.subprocess,
        "run",
        lambda cmd, **kw: types.SimpleNamespace(stdout=output),
    )
    assert flightgear._lsappinfo_name("4321") == expected
