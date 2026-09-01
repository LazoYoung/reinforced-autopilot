"""Launch FlightGear as an external-FDM viewer for a running simulation.

FlightGear is started with its own flight dynamics disabled (``--fdm=null``)
and listens for native-FDM UDP packets that JSBSim pushes, so it acts purely
as a visualizer that tracks the simulated aircraft.
"""

from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import time
import warnings
from collections.abc import Sequence

_WINDOWS_GLOBS = (
    r"C:\Program Files\FlightGear*\bin\fgfs.exe",
    r"C:\Program Files (x86)\FlightGear*\bin\fgfs.exe",
)


def find_fgfs() -> str | None:
    """Locate the FlightGear executable.

    Checks the ``AEROGYM_FGFS`` environment variable, then ``PATH``, then the
    default Windows install locations (newest version wins).
    """
    override = os.environ.get("AEROGYM_FGFS")
    if override:
        return override
    on_path = shutil.which("fgfs")
    if on_path:
        return on_path
    for pattern in _WINDOWS_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
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
                "variable to the full path of fgfs(.exe).",
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
        # cwd=bin dir so the FlightGear installation finds its own data on Windows.
        self._process = subprocess.Popen(args, cwd=os.path.dirname(exe) or None)
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
