# AeroGym — Design

Date: 2026-09-01
Status: Implemented (initial scaffold)

## Goal

A Gymnasium environment for RL flight control experiments, backed by JSBSim
today but simulator-agnostic by construction, with live FlightGear
visualization. RL algorithms are explicitly out of scope for this phase; the
deliverable is a verified simulator interaction loop.

## Architecture

Three layers, each importable and testable on its own:

1. **Core abstractions** (`aerogym.variables`, `aerogym.scenario`,
   `aerogym.simulator`) — no dependency on any concrete simulator.
   - `FlightVariable`: canonical registry of the supported action variables
     (elevator, aileron, rudder, throttle) and observation variables
     (heading, altitude, vertical speed, pitch, roll, yaw, pitch/roll/yaw
     rates, AoA, AoS), each with canonical units and bounds.
     Canonical units: degrees for angles, deg/s for rates, metres for
     altitude, m/s for vertical speed, normalized [-1, 1] controls
     ([0, 1] throttle). Control sign convention is the pilot convention:
     +elevator = pitch up, +aileron = roll right, +rudder = yaw right.
   - `Scenario`: frozen dataclass selecting the subset of action/observation
     variables, the reward function, the initial condition (lat, lon,
     altitude, heading, pitch, roll — plus initial airspeed, without which
     the aircraft would start in free fall), the aircraft model
     (default `c172p`), agent interaction rate, and episode step limit.
   - `Simulator`: ABC every backend implements — `reset(scenario)`,
     `set_control(var, value)`, `run()` (advance one physics step, `dt`),
     `observe(var)` in canonical units, `enable_visualization()`, `close()`.
2. **Gymnasium env** (`aerogym.env.AeroGymEnv`) — generic over any
   `Simulator`. Builds `Box` spaces from the scenario's variable selection,
   runs N physics steps per agent step (physics 120 Hz, agent rate from the
   scenario), delegates reward/termination to the scenario's reward
   function, truncates at the step limit, and paces to wall-clock time when
   rendering so FlightGear shows real-time motion.
3. **JSBSim backend** (`aerogym.simulators.jsbsim`) — maps canonical
   variables to JSBSim properties with unit/sign conversions
   (`properties.py`), drives `FGFDMExec` (`simulator.py`), and streams
   native-FDM packets to FlightGear over UDP via a generated JSBSim output
   directive, optionally launching `fgfs` with `--fdm=null
   --native-fdm=socket,in,<rate>,,<port>,udp` (`flightgear.py`). The launch
   is deferred to the first `reset()`, which passes the scenario's initial
   position (`--lat/--lon/--heading`) and blocks until FlightGear reports
   `/sim/sceneryloaded` over its telnet server, so episodes only start once
   the viewer can actually show them; on timeout it warns and continues
   headless.

## Data flow

agent action (canonical) → env clips to bounds → simulator adapter converts
sign/units and writes `fcs/*-cmd-norm` → N × `fdm.run()` (JSBSim pushes UDP
frames to FlightGear as a side effect) → adapter reads properties, converts
to canonical units → env assembles the observation vector → scenario reward
fn receives the (prev_obs, obs, action) transition and returns
(reward, terminated).

## Reward functions

A scenario's reward function is any callable `Transition -> (float, bool)`.
`aerogym.scenarios.steady_flight` ships as the reference scenario: hold the
initial heading and altitude, with termination on gross attitude/altitude
envelope violations.

## Error handling

- Scenario validation at construction (unknown variables, empty selections,
  out-of-range agent rate) raises `ValueError` immediately.
- The JSBSim backend raises `SimulatorError` if the model fails to load or
  the FDM diverges (NaN state).
- FlightGear is best-effort: if `fgfs` cannot be found the env still runs
  headless and a warning tells the user how to point at the executable
  (`AEROGYM_FGFS`).

## Testing

- Unit tests for variables/scenario validation and for `AeroGymEnv` against
  a scripted `FakeSimulator` (spaces, stepping, clipping, truncation).
- Integration tests against real JSBSim (skipped if not installed): reset
  honours initial conditions, a trimmed aircraft keeps flying, observations
  stay within declared bounds.

## Out of scope (this phase)

RL algorithms, additional simulator backends (X-Plane etc. slot in as new
`Simulator` implementations), observation normalization (use Gymnasium
wrappers), wind/turbulence configuration.
