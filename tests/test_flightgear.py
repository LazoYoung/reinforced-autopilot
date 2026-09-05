from __future__ import annotations

import itertools
import os
import plistlib
import socket
import threading
import time
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


class _FakePropertyServer:
    """Stands in for FlightGear's telnet property server in ``data`` mode."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.sets: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        buf = b""
        with conn:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    self._handle(conn, line.decode().strip())

    def _handle(self, conn, line: str):
        verb, _, rest = line.partition(" ")
        if verb == "get":
            with self._lock:
                value = self.values.get(rest.strip(), "")
            conn.sendall(f"{value}\n".encode())
        elif verb == "set":
            prop, _, value = rest.partition(" ")
            with self._lock:
                self.values[prop] = value
                self.sets.append((prop, value))

    def recorded(self) -> dict[str, str]:
        with self._lock:
            return dict(self.sets)

    def close(self):
        self._stop.set()
        self._sock.close()


@pytest.fixture
def property_server():
    server = _FakePropertyServer()
    yield server
    server.close()


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_get_property_reads_a_value(property_server):
    property_server.values["/sim/aircraft"] = "c172p"
    viewer = FlightGearViewer(telnet_port=property_server.port)
    assert viewer.get_property("/sim/aircraft") == "c172p"


def test_get_property_returns_none_when_nothing_is_listening():
    assert FlightGearViewer(telnet_port=1).get_property("/sim/aircraft") is None


def test_property_sync_writes_what_the_source_returns(property_server):
    viewer = FlightGearViewer(telnet_port=property_server.port)
    viewer._process = _FakeProcess()
    viewer.start_property_sync(lambda: {"/controls/flight/aileron": 0.25}, rate_hz=50.0)
    try:
        assert _wait_for(lambda: "/controls/flight/aileron" in property_server.recorded())
        assert property_server.recorded()["/controls/flight/aileron"] == "0.25"
    finally:
        viewer.stop_property_sync()


def test_property_sync_copies_mirrored_values(property_server):
    property_server.values["/environment/pressure-inhg"] = "29.42"
    viewer = FlightGearViewer(telnet_port=property_server.port)
    viewer._process = _FakeProcess()
    viewer.start_property_sync(
        dict,
        mirrors=(("/systems/static-pressure-inhg-selected", "/environment/pressure-inhg"),),
        rate_hz=50.0,
    )
    try:
        assert _wait_for(
            lambda: property_server.recorded().get("/systems/static-pressure-inhg-selected")
            == "29.42"
        )
    finally:
        viewer.stop_property_sync()


def test_property_sync_does_not_start_without_a_running_flightgear(property_server):
    viewer = FlightGearViewer(telnet_port=property_server.port)
    viewer.start_property_sync(lambda: {"/controls/flight/aileron": 1.0})
    assert viewer._sync_thread is None


def test_property_sync_warns_when_flightgear_cannot_be_reached():
    viewer = FlightGearViewer(telnet_port=1)
    viewer._process = _FakeProcess()
    with pytest.warns(UserWarning, match="property server"):
        viewer._sync_properties(dict, (), rate_hz=50.0, initial={})


def test_property_sync_writes_the_initial_state_once(property_server):
    viewer = FlightGearViewer(telnet_port=property_server.port)
    viewer._process = _FakeProcess()
    viewer.start_property_sync(
        dict, rate_hz=50.0, initial={"/controls/switches/master-bat": "true"}
    )
    try:
        assert _wait_for(
            lambda: property_server.recorded().get("/controls/switches/master-bat") == "true"
        )
        time.sleep(0.2)
        written = [p for p, _ in property_server.sets if p.endswith("master-bat")]
        assert len(written) == 1
    finally:
        viewer.stop_property_sync()


def test_stop_property_sync_is_idempotent(property_server):
    viewer = FlightGearViewer(telnet_port=property_server.port)
    viewer._process = _FakeProcess()
    viewer.start_property_sync(dict, rate_hz=50.0)
    viewer.stop_property_sync()
    viewer.stop_property_sync()
    assert viewer._sync_thread is None


def test_launch_asks_for_a_property_server_fast_enough_to_drive_a_cockpit(monkeypatch):
    """--telnet=<port> would run it at 5 Hz, far too slow to animate controls."""
    calls = _captured_launch(monkeypatch, "/usr/bin/fgfs")
    assert "--props=socket,bi,60,localhost,5401,tcp" in calls["args"]
    assert not [arg for arg in calls["args"] if arg.startswith("--telnet")]


def test_get_reads_one_value_per_reply_when_flightgear_batches_them():
    """A single recv can carry several replies; each get must take just one."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.recv(1024)  # the "data" handshake
            time.sleep(0.05)
            conn.recv(1024)  # both gets, pipelined by the client
            conn.sendall(b"25.4\n25.9\n")
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    client = flightgear._PropertyClient(port)
    assert client.get("/a") == "25.4"
    assert client.get("/b") == "25.9"
    client.close()


def test_get_returns_none_when_the_connection_drops():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _ = server.accept()
        conn.recv(1024)
        conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    client = flightgear._PropertyClient(port)
    assert client.get("/sim/aircraft") is None
    client.close()


def _make_addon_catalog(root, catalog: str, aircraft: str) -> str:
    """Create the layout FlightGear's launcher installs add-on aircraft into."""
    path = os.path.join(str(root), "Aircraft", catalog, "Aircraft", aircraft)
    os.makedirs(path)
    open(os.path.join(path, f"{aircraft}-set.xml"), "w").close()
    return os.path.dirname(path)


def test_addon_aircraft_catalogs_under_fg_home_are_found(tmp_path, monkeypatch):
    """fgfs only searches its own data dir; the launcher's add-ons live elsewhere."""
    expected = _make_addon_catalog(tmp_path, "org.flightgear.fgaddon.stable_2024", "737-300")
    monkeypatch.setattr(flightgear, "_fg_home", lambda: str(tmp_path))
    assert flightgear.find_aircraft_paths() == [expected]


def test_several_catalogs_are_all_found(tmp_path, monkeypatch):
    a = _make_addon_catalog(tmp_path, "org.flightgear.fgaddon.stable_2024", "737-300")
    b = _make_addon_catalog(tmp_path, "org.example.catalog", "a320")
    monkeypatch.setattr(flightgear, "_fg_home", lambda: str(tmp_path))
    assert sorted(flightgear.find_aircraft_paths()) == sorted([a, b])


def test_no_fg_home_means_no_aircraft_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(flightgear, "_fg_home", lambda: str(tmp_path / "nothing-here"))
    assert flightgear.find_aircraft_paths() == []


def test_launch_points_flightgear_at_every_addon_catalog(monkeypatch):
    """Without this, --aircraft=737-300 finds nothing: fgfs cannot see add-ons."""
    monkeypatch.setattr(flightgear, "find_aircraft_paths", lambda: ["/catalog/Aircraft"])
    calls = _captured_launch(monkeypatch, "/usr/bin/fgfs")
    assert "--fg-aircraft=/catalog/Aircraft" in calls["args"]


def test_launch_asks_for_the_aircraft_the_viewer_was_given(monkeypatch):
    calls = {}

    def fake_popen(args, **kwargs):
        calls["args"] = args
        return _FakeProcess()

    monkeypatch.setattr(flightgear.subprocess, "Popen", fake_popen)
    assert FlightGearViewer(executable="/usr/bin/fgfs", aircraft="737-300").launch()
    assert "--aircraft=737-300" in calls["args"]
