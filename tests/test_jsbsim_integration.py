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


class _StubViewer:
    """Records what the simulator asks of a FlightGear viewer, without one."""

    def __init__(self, aircraft=None):
        self.aircraft = aircraft
        self.launched = False
        self.synced = False

    def launch(self, **kwargs):
        self.launched = True
        return True

    def wait_until_ready(self, *args, **kwargs):
        return False  # skip the cockpit sync; nothing is listening

    def close(self):
        pass


def _reset_with_viewer(scenario, viewer):
    sim = JSBSimSimulator()
    sim._viewer = viewer
    sim.reset(scenario)
    return sim


def test_the_viewer_is_told_which_aircraft_the_scenario_flies():
    """FlightGear otherwise loads its default aircraft and renders a Cessna."""
    viewer = _StubViewer()
    sim = _reset_with_viewer(make_steady_flight_scenario(aircraft="737"), viewer)
    try:
        assert viewer.aircraft == "737-300"
    finally:
        sim.close()


def test_an_explicitly_requested_flightgear_aircraft_is_left_alone():
    viewer = _StubViewer(aircraft="737-300-yoke")
    sim = _reset_with_viewer(make_steady_flight_scenario(aircraft="737"), viewer)
    try:
        assert viewer.aircraft == "737-300-yoke"
    finally:
        sim.close()


def test_the_737_starts_clean_and_trimmed():
    """JSBSim's 737 initialises gear-down and untrimmed, which will not cruise."""
    sim = JSBSimSimulator()
    try:
        sim.reset(make_steady_flight_scenario(aircraft="737"))
        fdm = sim._fdm
        assert fdm["gear/gear-cmd-norm"] == 0.0
        assert fdm["fcs/throttle-cmd-norm[0]"] == pytest.approx(0.85)
        assert fdm["fcs/throttle-cmd-norm[1]"] == pytest.approx(0.85)
        assert fdm["fcs/pitch-trim-cmd-norm"] == pytest.approx(-0.2)
    finally:
        sim.close()


def test_c172p_still_gets_the_piston_controls_it_always_had():
    sim = JSBSimSimulator()
    try:
        sim.reset(make_steady_flight_scenario())
        assert sim._fdm["fcs/mixture-cmd-norm"] == 1.0
        assert sim._fdm["fcs/throttle-cmd-norm"] == pytest.approx(0.7)
    finally:
        sim.close()


def test_the_737_holds_its_cruise_without_leaving_the_envelope():
    scenario = make_steady_flight_scenario(aircraft="737", max_episode_steps=None)
    env = AeroGymEnv(scenario, simulator=JSBSimSimulator())
    try:
        _, info = env.reset(seed=0)
        assert info["obs"][V.ALTITUDE] == pytest.approx(30000.0, abs=50.0)
        action = np.array([0.0, 0.0, 0.0, 0.85], np.float32)
        for _ in range(100):  # 10 seconds at 10 Hz
            obs_arr, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs_arr)
            assert not terminated, f"left the flight envelope: {info['obs']}"
        assert info["obs"][V.ALTITUDE] > 29000.0
    finally:
        env.close()


def test_the_throttle_action_reaches_every_engine():
    """One lever in the action space commands the aircraft, not just engine 1.

    JSBSim's bare fcs/throttle-cmd-norm is engine 0 alone, so on a twin the
    second engine would keep whatever the startup left it at forever.
    """
    sim = JSBSimSimulator()
    try:
        sim.reset(make_steady_flight_scenario(aircraft="737"))
        sim.set_control(V.THROTTLE, 0.3)
        sim.run()
        assert sim._fdm["fcs/throttle-cmd-norm[0]"] == pytest.approx(0.3)
        assert sim._fdm["fcs/throttle-cmd-norm[1]"] == pytest.approx(0.3)
    finally:
        sim.close()


def test_a_single_engine_aircraft_still_takes_its_throttle():
    sim = JSBSimSimulator()
    try:
        sim.reset(make_steady_flight_scenario())
        sim.set_control(V.THROTTLE, 0.42)
        sim.run()
        assert sim._fdm["fcs/throttle-cmd-norm"] == pytest.approx(0.42)
    finally:
        sim.close()


def test_the_engine_count_follows_a_change_of_aircraft():
    """The same simulator can be reset onto a different model mid-life."""
    sim = JSBSimSimulator()
    try:
        sim.reset(make_steady_flight_scenario())
        sim.reset(make_steady_flight_scenario(aircraft="737"))
        sim.set_control(V.THROTTLE, 0.25)
        sim.run()
        assert sim._fdm["fcs/throttle-cmd-norm[1]"] == pytest.approx(0.25)
    finally:
        sim.close()


def test_both_engines_answer_the_agent_over_a_flight():
    """The cockpit showed engine 2 frozen because the simulation had it frozen."""
    scenario = make_steady_flight_scenario(aircraft="737", max_episode_steps=None)
    env = AeroGymEnv(scenario, simulator=JSBSimSimulator())
    try:
        env.reset(seed=0)
        for _ in range(20):
            env.step(np.array([0.0, 0.0, 0.0, 0.35], np.float32))
        fdm = env.simulator._fdm
        assert fdm["fcs/throttle-pos-norm[0]"] == pytest.approx(0.35, abs=0.01)
        assert fdm["fcs/throttle-pos-norm[1]"] == pytest.approx(0.35, abs=0.01)
    finally:
        env.close()
