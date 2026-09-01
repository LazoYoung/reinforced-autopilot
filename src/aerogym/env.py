"""Gymnasium environment generic over any :class:`~aerogym.simulator.Simulator`."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

from .scenario import Scenario, Transition
from .simulator import Simulator
from .variables import ACTION_SPECS, OBSERVATION_SPECS, FlightVariable


class AeroGymEnv(gym.Env):
    """Fixed-wing flight control environment.

    The action and observation spaces are built from the scenario's variable
    selection; reward and termination are delegated to the scenario's reward
    function. Each agent step advances the simulator by
    ``1 / (dt * agent_hz)`` physics steps.
    """

    metadata = {"render_modes": ["flightgear"]}

    def __init__(
        self,
        scenario: Scenario,
        simulator: Simulator | None = None,
        render_mode: str | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        if simulator is None:
            from .simulators.jsbsim import JSBSimSimulator

            simulator = JSBSimSimulator()
        self.scenario = scenario
        self.simulator = simulator
        self.render_mode = render_mode

        self._steps_per_action = max(1, round(1.0 / (simulator.dt * scenario.agent_hz)))
        self._elapsed_steps = 0
        self._prev_obs: dict[FlightVariable, float] | None = None
        self._visualization_started = False
        self._next_frame_deadline: float | None = None

        act_low = np.array([ACTION_SPECS[v].low for v in scenario.action_variables], np.float32)
        act_high = np.array([ACTION_SPECS[v].high for v in scenario.action_variables], np.float32)
        obs_low = np.array([OBSERVATION_SPECS[v].low for v in scenario.observation_variables], np.float32)
        obs_high = np.array([OBSERVATION_SPECS[v].high for v in scenario.observation_variables], np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.render_mode == "flightgear" and not self._visualization_started:
            self.simulator.enable_visualization()
            self._visualization_started = True
        self.simulator.reset(self.scenario)
        self._elapsed_steps = 0
        self._next_frame_deadline = None
        obs_dict = self._observe()
        self._prev_obs = obs_dict
        return self._to_array(obs_dict), {"obs": obs_dict}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._prev_obs is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_dict = dict(zip(self.scenario.action_variables, (float(a) for a in action)))
        for var, value in action_dict.items():
            self.simulator.set_control(var, value)
        for _ in range(self._steps_per_action):
            self.simulator.run()
        if self.render_mode == "flightgear":
            self._pace_real_time()

        obs_dict = self._observe()
        transition = Transition(prev_obs=self._prev_obs, obs=obs_dict, action=action_dict)
        reward, terminated = self.scenario.reward_fn(transition)
        self._prev_obs = obs_dict
        self._elapsed_steps += 1
        truncated = (
            self.scenario.max_episode_steps is not None
            and self._elapsed_steps >= self.scenario.max_episode_steps
        )
        return self._to_array(obs_dict), float(reward), bool(terminated), truncated, {"obs": obs_dict}

    def close(self) -> None:
        self.simulator.close()

    def _observe(self) -> dict[FlightVariable, float]:
        return {v: self.simulator.observe(v) for v in self.scenario.observation_variables}

    def _to_array(self, obs: dict[FlightVariable, float]) -> np.ndarray:
        arr = np.array([obs[v] for v in self.scenario.observation_variables], np.float32)
        return np.clip(arr, self.observation_space.low, self.observation_space.high)

    def _pace_real_time(self) -> None:
        """Sleep so simulated time tracks wall-clock time for the viewer."""
        frame = self._steps_per_action * self.simulator.dt
        now = time.monotonic()
        if self._next_frame_deadline is None:
            self._next_frame_deadline = now + frame
            return
        delay = self._next_frame_deadline - now
        if delay > 0:
            time.sleep(delay)
            self._next_frame_deadline += frame
        else:
            self._next_frame_deadline = now + frame
