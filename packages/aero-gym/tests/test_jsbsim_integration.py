"""Integration tests against the real JSBSim FDM (skipped if not installed)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jsbsim = pytest.importorskip("jsbsim")

from aero_gym import AeroGymEnv
from aero_gym.scenarios import make_steady_flight_scenario
from aero_gym.simulators.jsbsim import JSBSimSimulator
from aero_gym.variables import OBSERVATION_SPECS, FlightVariable as V


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


def test_trimmed_flight_stays_in_the_envelope(env):
    """Not a claim that a fixed action holds the band -- it does not, by design.
    What must hold is that the observation contract survives real flight: each
    feature's pre-clip value stays within the bound its Box entry declares.

    Not `env.observation_space.contains(obs_arr)` -- `_to_array` clips to
    those same bounds before returning (env.py:126-127), so containment holds
    by construction and could only ever catch a NaN, a shape mismatch, or a
    dtype problem, never a real excursion outside the declared range.
    """
    env.reset(seed=0)
    action = np.array([0.05, 0.0, 0.0, 0.7], np.float32)
    for _ in range(60):
        _, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        for feature in env.scenario.observation_features:
            value = feature.fn(info["obs"])
            assert feature.spec.low <= value <= feature.spec.high, (
                f"{feature.name}={value} outside [{feature.spec.low}, {feature.spec.high}]"
            )
        if terminated:
            break
    assert info["obs"][V.ALTITUDE] > 4000.0


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
    """All 15 canonical variables, not just the 12 steady_flight narrows to:
    query the simulator directly rather than iterating `info["obs"]`, which
    only carries the scenario's observed subset and would silently drop
    YAW, YAW_RATE and AOS from this check without the test's name noticing."""
    env.reset(seed=0)
    for var, spec in OBSERVATION_SPECS.items():
        value = env.simulator.observe(var)
        assert spec.low <= value <= spec.high, f"{var} out of declared bounds: {value}"


def test_flightgear_output_streams_udp_packets():
    """JSBSim must push native-FDM UDP packets to the configured port."""
    import socket

    from aero_gym.scenarios import make_steady_flight_scenario

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


def test_every_profile_publishes_its_control_positions():
    """A missing pos-norm property is an aircraft fact, so it is caught per
    aircraft -- and so is the surface's sign, because a left/right pair need not
    both follow the pilot convention and only the FDM can say which one does."""
    from aero_gym.simulators.jsbsim.aircraft import PROFILES
    from aero_gym.variables import ACTION_POSITION

    for model in PROFILES:
        sim = JSBSimSimulator()
        scenario = make_steady_flight_scenario(aircraft=model, max_episode_steps=None)
        scenario = dataclasses.replace(
            scenario, observation_variables=tuple(ACTION_POSITION.values())
        )
        try:
            sim.reset(scenario)
            for var in ACTION_POSITION.values():
                assert np.isfinite(sim.observe(var)), f"{model}: {var.name}"
            for action in (V.ELEVATOR, V.AILERON, V.RUDDER):
                sim.set_control(action, 0.5)
            for _ in range(20):  # let the actuators travel
                sim.run()
            for action in (V.ELEVATOR, V.AILERON, V.RUDDER):
                position = ACTION_POSITION[action]
                deflection = sim.observe(position)
                assert deflection > 0.0, f"{model}: {position.name} = {deflection}"
        finally:
            sim.close()


@pytest.mark.parametrize(
    "convert, expected_sign",
    [
        # Identity: the override hands the right surface's raw reading straight
        # through, and the 737 publishes that as the negation of its left one.
        # A negative answer is therefore proof the property name came from the
        # profile and not from OBSERVATION_MAP.
        (lambda x: x, -1.0),
        # The realistic escape hatch: an aircraft that exposes only its right
        # aileron supplies the negation itself, and the canonical roll-right
        # sign comes back out.
        (lambda x: -x, 1.0),
    ],
)
def test_a_profile_override_replaces_both_the_property_and_its_sign(
    monkeypatch, convert, expected_sign
):
    """observation_props is the escape hatch for an aircraft whose FCS exposes
    a different surface, so it has to carry a conversion, not just a name."""
    from aero_gym.simulators.jsbsim import aircraft, properties

    monkeypatch.setitem(
        aircraft.PROFILES,
        "737",
        dataclasses.replace(
            aircraft.PROFILES["737"],
            observation_props={
                V.AILERON_POS: properties.ObservationMapping(
                    "fcs/right-aileron-pos-norm", convert
                )
            },
        ),
    )
    sim = JSBSimSimulator()
    try:
        sim.reset(make_steady_flight_scenario(aircraft="737", max_episode_steps=None))
        sim.set_control(V.AILERON, 0.5)
        for _ in range(20):
            sim.run()
        assert sim.observe(V.AILERON_POS) == pytest.approx(expected_sign * 0.5, abs=0.01)
    finally:
        sim.close()


def test_an_unpublished_property_names_the_variable_the_property_and_the_model(
    monkeypatch,
):
    from aero_gym.simulator import SimulatorError
    from aero_gym.simulators.jsbsim import properties

    sim = JSBSimSimulator()
    scenario = make_steady_flight_scenario(max_episode_steps=None)
    monkeypatch.setitem(
        properties.OBSERVATION_MAP,
        V.AOS,
        properties.ObservationMapping(
            "fcs/no-such-property", properties.OBSERVATION_MAP[V.AOS].to_canonical
        ),
    )
    try:
        with pytest.raises(
            SimulatorError, match=r"AOS.*'fcs/no-such-property'.*'c172p'"
        ):
            sim.reset(dataclasses.replace(scenario, observation_variables=(V.AOS,)))
    finally:
        sim.close()


def test_the_first_observation_of_an_episode_is_not_stale():
    """fcs/*-pos-norm only updates inside run(), so without a settling step the
    throttle reads 0.0 while the profile just commanded 0.7."""
    from aero_gym.simulators.jsbsim.aircraft import profile_for
    from aero_gym.variables import FlightVariable as V

    scenario = make_steady_flight_scenario(max_episode_steps=None)
    env = AeroGymEnv(scenario, simulator=JSBSimSimulator())
    try:
        _, info = env.reset(seed=0)
        commanded = profile_for("c172p").startup["fcs/throttle-cmd-norm"]
        assert info["obs"][V.THROTTLE_POS] == pytest.approx(commanded, abs=1e-2)
    finally:
        env.close()


def test_initial_action_describes_the_aircrafts_startup_state():
    """Every axis, not just throttle: throttle is the one axis where command
    and position coincide (it has no trim term), so a throttle-only assertion
    passes just as happily on a seed read from the wrong property."""
    from aero_gym.simulators.jsbsim.aircraft import profile_for
    from aero_gym.simulators.jsbsim.properties import CONTROL_MAP

    scenario = make_steady_flight_scenario(max_episode_steps=None)
    env = AeroGymEnv(scenario, simulator=JSBSimSimulator())
    try:
        env.reset(seed=0)
        action = env.initial_action()
        assert action.shape == env.action_space.shape
        assert env.action_space.contains(action)
        startup = profile_for("c172p").startup
        for index, var in enumerate(scenario.action_variables):
            mapping = CONTROL_MAP[var]
            # Whatever the profile commanded at startup, in canonical sign;
            # a property the profile never writes is still at JSBSim's 0.0.
            expected = mapping.sign * startup.get(mapping.prop, 0.0)
            assert action[index] == pytest.approx(expected, abs=1e-2), var.name
        # ELEVATOR is the axis that discriminates: the profile leaves its
        # command at 0.0 while trimming to a ~0.2 elevator *position*.
        elevator_index = scenario.action_variables.index(V.ELEVATOR)
        assert action[elevator_index] == pytest.approx(0.0, abs=1e-2)
    finally:
        env.close()


def test_elevator_position_follows_the_commanded_sign(env):
    """Canonical +elevator is pitch-up, and the observed position must agree --
    asserted as a movement from the trimmed baseline, not against a constant."""
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.7], np.float32))
    baseline = info["obs"][V.ELEVATOR_POS]
    env.reset(seed=0)
    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0.5, 0.0, 0.0, 0.7], np.float32))
    assert info["obs"][V.ELEVATOR_POS] > baseline


def test_get_control_round_trips_every_axis(env):
    """get_control is the documented inverse of set_control.

    All four axes, because elevator and rudder are the ones whose canonical
    sign is the negation of JSBSim's: a get_control that forgot the sign would
    still pass on aileron and throttle alone.
    """
    env.reset(seed=0)
    sim = env.simulator
    for var, value in (
        (V.ELEVATOR, 0.35),
        (V.ELEVATOR, -0.42),
        (V.AILERON, 0.25),
        (V.RUDDER, 0.15),
        (V.RUDDER, -0.6),
        (V.THROTTLE, 0.8),
    ):
        sim.set_control(var, value)
        assert sim.get_control(var) == pytest.approx(value, abs=1e-6), var.name


def test_initial_action_seeds_the_elevator_command_not_its_position():
    """The c172p trims with fcs/pitch-trim-cmd-norm = 0.2, so its elevator
    *position* after reset is ~0.2 while its elevator *command* is 0.0. Seeding
    DeltaAction's accumulator from the position feeds that trim back in as a
    command and JSBSim adds trim a second time.
    """
    scenario = make_steady_flight_scenario(max_episode_steps=None)
    env = AeroGymEnv(scenario, simulator=JSBSimSimulator())
    try:
        env.reset(seed=0)
        elevator_index = scenario.action_variables.index(V.ELEVATOR)
        position = env.simulator.observe(V.ELEVATOR_POS)
        # Guard the premise: if the profile ever stops trimming, this test
        # would pass for the wrong reason.
        assert abs(position) > 0.1, "the c172p is expected to reset out of trim-neutral"
        assert env.initial_action()[elevator_index] == pytest.approx(0.0, abs=1e-6)
    finally:
        env.close()
