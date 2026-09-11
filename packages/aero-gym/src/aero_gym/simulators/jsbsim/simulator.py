"""JSBSim implementation of the :class:`~aero_gym.simulator.Simulator` interface."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from aero_gym.scenario import InitialConditions
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

    def reset(
        self,
        aircraft: str,
        initial_conditions: InitialConditions,
        settle_steps: int = 1,
    ) -> None:
        """Put the FDM at the scenario's initial condition, engines running.

        Episodes from the same seed are bit-identical, because the FDM is
        discarded and rebuilt each time: ``run_ic()`` restores the flight
        condition but not engine state (rpm, manifold pressure, thrust), which
        otherwise carries into the next episode. A rebuild costs ~1.1 ms.

        Visualization suspends that. ``CockpitSync`` binds to one FDM for its
        lifetime and the output directive is rewritten on every build, so with a
        viewer (or with output configured but no viewer) the instance is
        reused and only the controls are cleared. Reset determinism is therefore
        a function of ``render_mode``, and a *training* run configured with a
        viewer is not reproducible. Eval is unaffected: ``cli._train`` strips
        ``render_mode`` from the eval env factory.

        Returns only once a full agent step has been simulated, so the first
        observation reports settled positions rather than the zeros JSBSim holds
        until ``run()`` is called.

        Does not pre-read the observed variables. AeroGymEnv.reset calls
        _observe() immediately after, which reads every one of them and raises
        the same SimulatorError from observe(), so the failure still surfaces
        inside reset() rather than mid-rollout.
        """
        ic = initial_conditions
        profile = profile_for(aircraft)
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
        if self._viewer is None and self._visualization is None:
            # Discarded, not reset: see the determinism contract above.
            # `_visualization` too, since output can be configured without a viewer.
            self._fdm = None
        fdm = self._ensure_fdm(aircraft)
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
        # run_ic() leaves the FCS commands alone, so a reused FDM would start
        # episode N+1 holding episode N's stick. Via set_control to match its keys.
        for var in CONTROL_MAP:
            self.set_control(var, 0.0)
        # Flyable from the first step. After the clear above, so the profile wins.
        fdm["propulsion/set-running"] = -1
        for prop, value in profile.startup.items():
            fdm[prop] = value
        # JSBSim recomputes fcs/*-pos-norm only inside run(), so until one agent
        # step has settled every position property reads 0 against a live command.
        for _ in range(max(1, settle_steps)):
            self.run()
        if self._viewer is not None and not self._viewer_synced:
            # Nothing renders while FlightGear is still on its splash screen.
            # wait_until_ready() warns and proceeds headless if it never comes up.
            if self._viewer.wait_until_ready():
                self._start_cockpit_sync(fdm)
            self._viewer_synced = True

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

    def get_control(self, var: FlightVariable) -> float:
        """Read back what set_control last wrote, undoing its sign.

        Same property and same sign as set_control, including the per-engine
        indexing: engine 0 is read back because the agent commands every engine
        identically, so it is representative of what was written.
        """
        mapping = CONTROL_MAP[var]
        fdm = self._require_fdm()
        prop = f"{mapping.prop}[0]" if mapping.per_engine else mapping.prop
        return mapping.sign * float(fdm[prop])

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
            # JSBSim's own message names only the property; naming the variable
            # and the profile is what makes this a one-line fix mid-rollout.
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
        # close() unlinks only the newest path, so each re-apply must unlink its
        # own predecessor or strand it in the temp dir.
        if self._directive_path is not None:
            self._directive_path.unlink(missing_ok=True)
            self._directive_path = None
        content = _OUTPUT_DIRECTIVE.format(**self._visualization)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", prefix="aero_gym_fg_", delete=False
        ) as fh:
            fh.write(content)
            self._directive_path = Path(fh.name)
        if not self._fdm.set_output_directive(str(self._directive_path)):
            raise SimulatorError("JSBSim rejected the FlightGear output directive")
