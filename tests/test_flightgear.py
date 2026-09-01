from __future__ import annotations

import socket
import threading

import pytest

from aerogym.simulators.jsbsim.flightgear import FlightGearViewer


class _FakeProcess:
    """Stands in for a running fgfs subprocess."""

    returncode = None

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
