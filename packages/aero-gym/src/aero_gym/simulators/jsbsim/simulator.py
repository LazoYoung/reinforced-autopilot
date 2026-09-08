"""JSBSim implementation of the :class:`~aero_gym.simulator.Simulator` interface."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from aero_gym.scenario import Scenario
from aero_gym.simulator import Simulator, SimulatorError
from aero_gym.variables import FlightVariable
from .aircraft import AircraftProfile, profile_for
from .cockpit import CockpitSync
from .flightgear import FlightGearViewer
from .properties import CONTROL_MAP, OBSERVATION_MAP, ObservationMapping

_OUTPUT_DIRECTIVE = (
    '<?xml version="1.0"?>\n'
    '<output name="{host}" type="FLIGHTGEAR" port="{port}" protocol="UDP" rate="{rate}"/>\n'
)


class JSBSimSimulator(Simulator):
    """Drives a JSBSim ``FGFDMExec`` at a fixed physics rate.

    The FDM instance is created lazily on the first :meth:`reset` and reused
    across episodes; the aircraft model is only reloaded when the scenario
    asks for a different one.
    """

    def __init__(self, dt: float = 1.0 / 120.0, root_dir: str | None = None) -> None:
        self._dt = dt
        self._root_dir = root_dir
        self._fdm = None
        self._aircraft: str | None = None
        self._engines = 1
        self._profile: AircraftProfile | None = None
        self._viewer: FlightGearViewer | None = None
        self._viewer_synced = False
        self._visualization: dict[str, object] | None = None
        self._directive_path: Path | None = None

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self, scenario: Scenario) -> None:
        ic = scenario.initial_conditions
        profile = profile_for(scenario.aircraft)
        # observe() resolves per-aircraft property overrides through it
        self._profile = profile
        if self._viewer is not None:
            if self._viewer.aircraft is None:
                # Otherwise FlightGear renders whatever it defaults to, which
                # has nothing to do with the aircraft JSBSim is flying.
                self._viewer.aircraft = profile.flightgear
            # Launch first so FlightGear boots while we set up the FDM.
            self._viewer.launch(
                lat_deg=ic.latitude_deg,
                lon_deg=ic.longitude_deg,
                heading_deg=ic.heading_deg,
            )
        fdm = self._ensure_fdm(scenario.aircraft)
        fdm["ic/lat-geod-deg"] = ic.latitude_deg
        fdm["ic/long-gc-deg"] = ic.longitude_deg
        fdm["ic/h-sl-ft"] = ic.altitude_ft
        fdm["ic/psi-true-deg"] = ic.heading_deg % 360.0
        fdm["ic/theta-deg"] = ic.pitch_deg
        fdm["ic/phi-deg"] = ic.roll_deg
        fdm["ic/vc-kts"] = ic.airspeed_kts
        fdm["ic/p-rad_sec"] = 0.0
        fdm["ic/q-rad_sec"] = 0.0
        fdm["ic/r-rad_sec"] = 0.0
        if not fdm.run_ic():
            raise SimulatorError("JSBSim failed to initialize the requested flight condition")
        # Start with the engine(s) running and the aircraft's own controls set,
        # so it is flyable from the first step.
        fdm["propulsion/set-running"] = -1
        for prop, value in profile.startup.items():
            fdm[prop] = value
        # JSBSim recomputes fcs/*-pos-norm only inside run(), so right now every
        # position property still reads 0 while the commands above say 0.7.
        # Settling one agent step's worth of physics costs 0.1 s of simulated
        # time -- a few feet of altitude at cruise -- and buys a first
        # observation that is true. Measured on the c172p: after 12 steps
        # fcs/throttle-pos-norm reads 0.7 against a commanded 0.7 (the profile
        # has no fcs/elevator-cmd-norm, only fcs/pitch-trim-cmd-norm = 0.2,
        # which fcs/elevator-pos-norm converges toward instead).
        for _ in range(max(1, round(1.0 / (self._dt * scenario.agent_hz)))):
            self.run()
        if self._viewer is not None and not self._viewer_synced:
            # Don't start the episode while FlightGear is still on its splash
            # screen — nothing would be visualized. Warns and proceeds headless
            # if FlightGear never becomes ready.
            if self._viewer.wait_until_ready():
                self._start_cockpit_sync(fdm)
            self._viewer_synced = True
        self._validate_observations(scenario)

    def _validate_observations(self, scenario: Scenario) -> None:
        """Read every observed variable once so a missing property fails here.

        Without it the failure surfaces mid-rollout, with nothing pointing at
        the aircraft profile as the cause.
        """
        for var in scenario.observation_variables:
            self.observe(var)

    def _start_cockpit_sync(self, fdm) -> None:
        """Supply the cockpit state FlightGear's disabled flight model can't.

        Keyed on the JSBSim model, checked against what FlightGear says it
        loaded, since the caller may have pointed it at a different aircraft.
        """
        assert self._viewer is not None
        sync = CockpitSync(
            fdm,
            self._aircraft,
            flightgear_aircraft=self._viewer.get_property("/sim/aircraft"),
        )
        self._viewer.start_property_sync(
            sync.properties, sync.mirrors, initial=sync.initial_properties()
        )

    def set_control(self, var: FlightVariable, value: float) -> None:
        mapping = CONTROL_MAP[var]
        fdm = self._require_fdm()
        value = mapping.sign * value
        if mapping.per_engine:
            for i in range(self._engines):
                fdm[f"{mapping.prop}[{i}]"] = value
        else:
            fdm[mapping.prop] = value

    def run(self) -> None:
        fdm = self._require_fdm()
        fdm.run()
        if math.isnan(fdm["position/h-sl-ft"]):
            raise SimulatorError("JSBSim state diverged (NaN altitude)")

    def _mapping(self, var: FlightVariable) -> ObservationMapping:
        if self._profile is not None and var in self._profile.observation_props:
            return self._profile.observation_props[var]
        return OBSERVATION_MAP[var]

    def observe(self, var: FlightVariable) -> float:
        mapping = self._mapping(var)
        try:
            value = float(self._require_fdm()[mapping.prop])
        except KeyError as exc:
            # JSBSim's own message names only the property. Naming the variable
            # and pointing at the profile is what turns this from a mystery
            # hundreds of steps into a rollout into a one-line fix.
            raise SimulatorError(
                f"{var.name} needs JSBSim property {mapping.prop!r}, which model "
                f"{self._aircraft!r} does not publish; add an observation_props "
                "override to its AircraftProfile"
            ) from exc
        return mapping.to_canonical(value)

    def enable_visualization(
        self,
        host: str = "localhost",
        port: int = 5550,
        rate_hz: int = 60,
        launch_flightgear: bool = True,
        flightgear_aircraft: str | None = None,
        **viewer_kwargs: object,
    ) -> None:
        """Stream native-FDM frames to FlightGear and optionally launch it.

        Takes effect when the FDM is (re)created, i.e. on the next
        :meth:`reset` if none exists yet, immediately otherwise.
        """
        self._visualization = {"host": host, "port": port, "rate": rate_hz}
        if self._fdm is not None:
            self._apply_output_directive()
        if launch_flightgear and self._viewer is None:
            # The actual launch is deferred to reset(), where the scenario's
            # initial position is known and readiness can be awaited.
            self._viewer = FlightGearViewer(
                port=port,
                rate_hz=rate_hz,
                aircraft=flightgear_aircraft,
                **viewer_kwargs,  # type: ignore[arg-type]
            )

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            self._viewer_synced = False
        self._fdm = None
        if self._directive_path is not None:
            self._directive_path.unlink(missing_ok=True)
            self._directive_path = None

    def _ensure_fdm(self, aircraft: str):
        if self._fdm is None or aircraft != self._aircraft:
            try:
                import jsbsim
            except ImportError as exc:
                raise SimulatorError(
                    "the jsbsim package is not installed (pip install jsbsim)"
                ) from exc
            fdm = jsbsim.FGFDMExec(self._root_dir)
            fdm.set_debug_level(0)
            fdm.set_dt(self._dt)
            if not fdm.load_model(aircraft):
                raise SimulatorError(f"JSBSim could not load aircraft model {aircraft!r}")
            self._fdm = fdm
            self._aircraft = aircraft
            self._engines = fdm.get_propulsion().get_num_engines()
            if self._visualization is not None:
                self._apply_output_directive()
        return self._fdm

    def _require_fdm(self):
        if self._fdm is None:
            raise SimulatorError("simulator not initialized; call reset() first")
        return self._fdm

    def _apply_output_directive(self) -> None:
        assert self._visualization is not None
        content = _OUTPUT_DIRECTIVE.format(**self._visualization)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", prefix="aero_gym_fg_", delete=False
        ) as fh:
            fh.write(content)
            self._directive_path = Path(fh.name)
        if not self._fdm.set_output_directive(str(self._directive_path)):
            raise SimulatorError("JSBSim rejected the FlightGear output directive")
