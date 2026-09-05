from __future__ import annotations

import glob
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable, Sequence

_WINDOWS_GLOBS = (
    r"C:\Program Files\FlightGear*\bin\fgfs.exe",
    r"C:\Program Files (x86)\FlightGear*\bin\fgfs.exe",
)

_MACOS_APP_GLOBS = (
    "/Applications/FlightGear*.app",
    "~/Applications/FlightGear*.app",
)

_BUNDLE_EXECUTABLE_NAMES = ("fgfs", "FlightGear")

_PROPERTY_SERVER_HZ = 60


def _bundle_info(path: str) -> dict:
    """Read a macOS ``.app``'s Info.plist, or an empty dict if there isn't one."""
    try:
        with open(os.path.join(path, "Contents", "Info.plist"), "rb") as info:
            return plistlib.load(info)
    except (OSError, plistlib.InvalidFileException):
        return {}


def _bundle_executable(path: str) -> str | None:
    """Map a macOS ``.app`` bundle to the FlightGear binary inside it."""
    declared = _bundle_info(path).get("CFBundleExecutable")
    names = ([declared] if declared else []) + list(_BUNDLE_EXECUTABLE_NAMES)
    for name in names:
        candidate = os.path.join(path, "Contents", "MacOS", name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _containing_bundle(executable: str) -> str | None:
    """The ``.app`` an executable lives in, if it lives in one."""
    bundle = os.path.dirname(os.path.dirname(os.path.dirname(executable)))
    return bundle if bundle.endswith(".app") else None


def _lsappinfo_name(specifier: str) -> str:
    """The display name LaunchServices has for an app, or "" if it has none."""
    try:
        out = subprocess.run(
            ["/usr/bin/lsappinfo", "info", "-only", "name", specifier],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    name = out.partition("=")[2].strip()
    return name.strip('"') if name.startswith('"') else ""


def _frontmost_app_name() -> str:
    """The display name of the frontmost macOS app, or "" if it can't be read."""
    try:
        front = subprocess.run(
            ["/usr/bin/lsappinfo", "front"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return _lsappinfo_name(front) if front else ""


def _activate_bundle(bundle: str, pid: int, attempts: int = 10, interval: float = 3.0) -> None:
    """Bring a macOS ``.app`` to the front."""
    name = _bundle_info(bundle).get("CFBundleName") or os.path.basename(bundle)[: -len(".app")]
    for _ in range(attempts):
        time.sleep(interval)
        if not _lsappinfo_name(str(pid)):
            continue
        if _frontmost_app_name() == name:
            return
        try:
            subprocess.run(
                ["/usr/bin/open", "-a", bundle],
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return


def find_fgfs() -> str | None:
    """Locate the FlightGear executable.

    Checks the ``AEROGYM_FGFS`` environment variable, then ``PATH``, then the
    default Windows and macOS install locations (newest version wins). The
    environment variable may name a macOS ``.app`` bundle as well as a plain
    executable, since the bundle is what a user actually installs.
    """
    override = os.environ.get("AEROGYM_FGFS")
    if override:
        return _bundle_executable(override) or override
    on_path = shutil.which("fgfs")
    if on_path:
        return on_path
    for pattern in _WINDOWS_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    for pattern in _MACOS_APP_GLOBS:
        for bundle in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True):
            executable = _bundle_executable(bundle)
            if executable:
                return executable
    return None


class _PropertyClient:
    """A connection to FlightGear's property server, in ``data`` mode: replies
    are bare values rather than the interactive prompt's decorated output."""

    def __init__(self, port: int, timeout: float = 5.0) -> None:
        self._timeout = timeout
        self._buffer = b""
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self._send("data")
        self._drain()

    def _send(self, command: str) -> None:
        self._sock.sendall(command.encode() + b"\r\n")

    def _drain(self) -> None:
        """Discard queued output, so it can't be read as the next reply."""
        self._sock.setblocking(False)
        try:
            while self._sock.recv(4096):
                pass
        except OSError:
            pass
        self._sock.settimeout(self._timeout)

    def set(self, prop: str, value: object) -> None:
        self._send(f"set {prop} {value}")

    def get(self, prop: str, timeout: float = 1.0) -> str | None:
        """The property's value, or None if FlightGear didn't answer in time.

        FlightGear batches replies, so take one line and keep the remainder.
        """
        self._send(f"get {prop}")
        self._sock.settimeout(timeout)
        while b"\n" not in self._buffer:
            try:
                chunk = self._sock.recv(4096)
            except OSError:
                chunk = b""
            if not chunk:
                # Drop a partial line rather than answer a later request with it.
                self._buffer = b""
                return None
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode(errors="replace").strip() or None

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class FlightGearViewer:
    """A FlightGear subprocess configured as a native-FDM listener."""

    def __init__(
        self,
        port: int = 5550,
        rate_hz: int = 60,
        aircraft: str | None = None,
        executable: str | None = None,
        extra_args: Sequence[str] = (),
        telnet_port: int = 5401,
        ready_timeout: float = 180.0,
    ) -> None:
        self.port = port
        self.rate_hz = rate_hz
        self.aircraft = aircraft
        self.executable = executable
        self.extra_args = tuple(extra_args)
        self.telnet_port = telnet_port
        self.ready_timeout = ready_timeout
        self._process: subprocess.Popen | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_stop = threading.Event()

    def launch(
        self,
        lat_deg: float | None = None,
        lon_deg: float | None = None,
        heading_deg: float | None = None,
    ) -> bool:
        """Start FlightGear; returns False (with a warning) if not found."""
        if self._process is not None and self._process.poll() is None:
            return True
        exe = self.executable or find_fgfs()
        if exe is None:
            warnings.warn(
                "FlightGear not found! Running in headless mode.",
                stacklevel=2,
            )
            return False
        args = [
            exe,
            "--fdm=null",
            f"--native-fdm=socket,in,{self.rate_hz},,{self.port},udp",
            f"--props=socket,bi,{_PROPERTY_SERVER_HZ},localhost,{self.telnet_port},tcp",
            "--disable-ai-traffic",
            "--timeofday=noon",
        ]
        if lat_deg is not None:
            args.append(f"--lat={lat_deg}")
        if lon_deg is not None:
            args.append(f"--lon={lon_deg}")
        if heading_deg is not None:
            args.append(f"--heading={heading_deg}")
        if self.aircraft:
            args.append(f"--aircraft={self.aircraft}")
        args.extend(self.extra_args)
        # Windows builds resolve their data directory relative to the working
        # directory, so run from the bin dir there; elsewhere the install
        # locates itself and the caller's cwd is the less surprising choice.
        cwd = (os.path.dirname(exe) or None) if os.name == "nt" else None
        self._process = subprocess.Popen(args, cwd=cwd)
        bundle = _containing_bundle(exe) if sys.platform == "darwin" else None
        if bundle is not None:
            # Off the hot path: FlightGear needs a window before it will load
            # scenery, and nothing else will give it one. See _activate_bundle.
            threading.Thread(
                target=_activate_bundle, args=(bundle, self._process.pid), daemon=True
            ).start()
        return True

    def wait_until_ready(
        self, timeout: float | None = None, poll_interval: float = 2.0
    ) -> bool:
        """Block until FlightGear has finished loading its scenery.

        Polls ``/sim/sceneryloaded`` over FlightGear's telnet property server,
        which only answers once the simulator's main loop is up. Returns False
        (with a warning) on timeout or if FlightGear exited, so callers can
        continue headless rather than hang forever.
        """
        if self._process is None:
            return False
        deadline = time.monotonic() + (self.ready_timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                warnings.warn(
                    f"FlightGear exited during startup (code {self._process.returncode}); "
                    "Running in headless mode.",
                    stacklevel=2,
                )
                return False
            try:
                with socket.create_connection(("127.0.0.1", self.telnet_port), timeout=2.0) as sock:
                    sock.settimeout(2.0)
                    sock.sendall(b"get /sim/sceneryloaded\r\n")
                    if b"true" in sock.recv(1024):
                        return True
            except OSError:
                pass
            time.sleep(poll_interval)
        warnings.warn(
            "Timed out waiting for FlightGear to finish loading; "
            "the episode will start regardless.",
            stacklevel=2,
        )
        return False

    def get_property(self, name: str) -> str | None:
        """Read one property from the running FlightGear, or None if it can't."""
        try:
            client = _PropertyClient(self.telnet_port)
        except OSError:
            return None
        try:
            return client.get(name)
        finally:
            client.close()

    def start_property_sync(
        self,
        source: Callable[[], dict[str, object]],
        mirrors: Sequence[tuple[str, str]] = (),
        rate_hz: float = 20.0,
        initial: dict[str, object] | None = None,
    ) -> None:
        """Keep feeding FlightGear the state its own flight model would supply.

        ``initial`` is written once on connecting, for state the user should stay
        free to change afterwards; ``source`` is called each tick; ``mirrors``
        names ``(destination, origin)`` pairs to copy within FlightGear's own
        tree. Runs on a private thread, so the simulation never blocks.
        """
        if self._process is None or self._sync_thread is not None:
            return
        self._sync_stop.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_properties,
            args=(source, tuple(mirrors), rate_hz, initial or {}),
            daemon=True,
        )
        self._sync_thread.start()

    def stop_property_sync(self) -> None:
        if self._sync_thread is None:
            return
        self._sync_stop.set()
        self._sync_thread.join(timeout=5.0)
        self._sync_thread = None

    def _sync_properties(
        self,
        source: Callable[[], dict[str, object]],
        mirrors: tuple[tuple[str, str], ...],
        rate_hz: float,
        initial: dict[str, object],
    ) -> None:
        try:
            client = _PropertyClient(self.telnet_port)
        except OSError:
            warnings.warn(
                "Could not reach FlightGear's property server; its cockpit "
                "controls and instruments will not track the simulation.",
                stacklevel=2,
            )
            return
        interval = 1.0 / rate_hz
        try:
            for prop, value in initial.items():
                client.set(prop, value)
            while not self._sync_stop.wait(interval):
                for prop, value in source().items():
                    client.set(prop, value)
                for destination, origin in mirrors:
                    value = client.get(origin)
                    if value is not None:
                        client.set(destination, value)
        except OSError:
            pass  # FlightGear went away; the viewer is optional either way
        finally:
            client.close()

    def close(self) -> None:
        self.stop_property_sync()
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
