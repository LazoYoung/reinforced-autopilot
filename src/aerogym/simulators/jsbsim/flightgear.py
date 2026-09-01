"""Launch FlightGear as an external-FDM viewer for a running simulation.

FlightGear is started with its own flight dynamics disabled (``--fdm=null``)
and listens for native-FDM UDP packets that JSBSim pushes, so it acts purely
as a visualizer that tracks the simulated aircraft.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
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
    ) -> None:
        self.port = port
        self.rate_hz = rate_hz
        self.aircraft = aircraft
        self.executable = executable
        self.extra_args = tuple(extra_args)
        self._process: subprocess.Popen | None = None

    def launch(self) -> bool:
        """Start FlightGear; returns False (with a warning) if not found."""
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
            "--disable-ai-traffic",
            "--timeofday=noon",
        ]
        if self.aircraft:
            args.append(f"--aircraft={self.aircraft}")
        args.extend(self.extra_args)
        # cwd=bin dir so the FlightGear installation finds its own data on Windows.
        self._process = subprocess.Popen(args, cwd=os.path.dirname(exe) or None)
        return True

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
