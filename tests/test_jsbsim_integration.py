"""Integration tests against the real JSBSim FDM (skipped if not installed)."""

from __future__ import annotations

import numpy as np
import pytest

jsbsim = pytest.importorskip("jsbsim")

from aerogym import AeroGymEnv
from aerogym.scenarios import make_steady_flight_scenario
from aerogym.simulators.jsbsim import JSBSimSimulator
from aerogym.variables import OBSERVATION_SPECS, FlightVariable as V


@pytest.fixture(scope="module")
def env():
    scenario = make_steady_flight_scenario(max_episode_steps=None)
    env = AeroGymEnv(scenario, simulator=JSBSimSimulator())
    yield env
    env.close()


def test_reset_honours_initial_conditions(env):
    _, info = env.reset(seed=0)
    obs = info["obs"]
    assert obs[V.ALTITUDE] == pytest.approx(5000.0, abs=7.0)
    assert obs[V.HEADING] == pytest.approx(270.0, abs=1.0)
    assert abs(obs[V.ROLL]) < 1.0


def test_trimmed_flight_stays_in_envelope(env):
    env.reset(seed=0)
    action = np.array([0.05, 0.0, 0.0, 0.7], np.float32)  # elevator, aileron, rudder, throttle
    terminated = False
    for _ in range(100):  # 10 seconds at 10 Hz
        obs_arr, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs_arr)
        assert np.isfinite(reward)
        if terminated:
            break
    obs = info["obs"]
    assert not terminated, f"left the flight envelope: {obs}"
    assert obs[V.ALTITUDE] > 4000.0


def test_controls_have_canonical_sign(env):
    """Pitch-up elevator command must raise pitch rate, and aileron-right must roll right."""
    env.reset(seed=0)
    up = np.array([0.5, 0.0, 0.0, 0.7], np.float32)
    for _ in range(5):
        _, _, _, _, info = env.step(up)
    assert info["obs"][V.PITCH_RATE] > 0.0

    env.reset(seed=0)
    right = np.array([0.05, 0.5, 0.0, 0.7], np.float32)
    for _ in range(5):
        _, _, _, _, info = env.step(right)
    assert info["obs"][V.ROLL] > 0.0


def test_observations_cover_all_variables(env):
    _, info = env.reset(seed=0)
    for var, value in info["obs"].items():
        spec = OBSERVATION_SPECS[var]
        assert spec.low <= value <= spec.high, f"{var} out of declared bounds: {value}"


def test_flightgear_output_streams_udp_packets():
    """JSBSim must push native-FDM UDP packets to the configured port."""
    import socket

    from aerogym.scenarios import make_steady_flight_scenario

    port = 55511
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(5.0)
    sim = JSBSimSimulator()
    try:
        sim.enable_visualization(port=port, launch_flightgear=False)
        sim.reset(make_steady_flight_scenario())
        for _ in range(120):
            sim.run()
        packet, _ = sock.recvfrom(4096)
        assert len(packet) > 0
    finally:
        sim.close()
        sock.close()
