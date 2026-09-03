"""Launch FlightGear as an external-FDM viewer for a running simulation.

FlightGear is started with its own flight dynamics disabled (``--fdm=null``)
and listens for native-FDM UDP packets that JSBSim pushes, so it acts purely
as a visualizer that tracks the simulated aircraft.
"""

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
from collections.abc import Sequence

_WINDOWS_GLOBS = (
    r"C:\Program Files\FlightGear*\bin\fgfs.exe",
    r"C:\Program Files (x86)\FlightGear*\bin\fgfs.exe",
)

# macOS ships FlightGear as an application bundle, which is never on PATH.
_MACOS_APP_GLOBS = (
    "/Applications/FlightGear*.app",
    "~/Applications/FlightGear*.app",
)
# Release bundles name the binary after the app ("FlightGear"); source builds
# and older bundles keep the traditional "fgfs".
_BUNDLE_EXECUTABLE_NAMES = ("fgfs", "FlightGear")


def _bundle_info(path: str) -> dict:
    """Read a macOS ``.app``'s Info.plist, or an empty dict if there isn't one."""
    try:
        with open(os.path.join(path, "Contents", "Info.plist"), "rb") as info:
            return plistlib.load(info)
    except (OSError, plistlib.InvalidFileException):
        return {}


def _bundle_executable(path: str) -> str | None:
    """Map a macOS ``.app`` bundle to the FlightGear binary inside it.

    Prefers the name the bundle declares for itself, since neither hard-coded
    name has held across releases.
    """
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

    def launch(
        self,
        lat_deg: float | None = None,
        lon_deg: float | None = None,
        heading_deg: float | None = None,
    ) -> bool:
        """Start FlightGear; returns False (with a warning) if not found.

        Passing the scenario's initial position lets FlightGear load the right
        scenery immediately instead of booting at its default airport.
        """
        if self._process is not None and self._process.poll() is None:
            return True
        exe = self.executable or find_fgfs()
        if exe is None:
            warnings.warn(
                "FlightGear executable not found; running without visualization. "
                "Install FlightGear and/or set the AEROGYM_FGFS environment "
                "variable to the full path of fgfs(.exe), or of FlightGear.app "
                "on macOS.",
                stacklevel=2,
            )
            return False
        args = [
            exe,
            "--fdm=null",
            f"--native-fdm=socket,in,{self.rate_hz},,{self.port},udp",
            f"--telnet={self.telnet_port}",
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
                    "continuing without visualization.",
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

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
