"""JSBSim implementation of the :class:`~aerogym.simulator.Simulator` interface."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from aerogym.scenario import Scenario
from aerogym.simulator import Simulator, SimulatorError
from aerogym.variables import FlightVariable
from .cockpit import CockpitSync
from .flightgear import FlightGearViewer
from .properties import CONTROL_MAP, OBSERVATION_MAP

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
        self._viewer: FlightGearViewer | None = None
        self._viewer_synced = False
        self._visualization: dict[str, object] | None = None
        self._directive_path: Path | None = None

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self, scenario: Scenario) -> None:
        ic = scenario.initial_conditions
        if self._viewer is not None:
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
        # Start with the engine(s) running and sensible engine controls so the
        # aircraft is flyable from the first step.
        fdm["propulsion/set-running"] = -1
        fdm["fcs/mixture-cmd-norm"] = 1.0
        fdm["fcs/throttle-cmd-norm"] = 0.7
        if self._viewer is not None and not self._viewer_synced:
            # Don't start the episode while FlightGear is still on its splash
            # screen — nothing would be visualized. Warns and proceeds headless
            # if FlightGear never becomes ready.
            if self._viewer.wait_until_ready():
                self._start_cockpit_sync(fdm)
            self._viewer_synced = True

    def _start_cockpit_sync(self, fdm) -> None:
        """Supply the cockpit state FlightGear's disabled flight model can't.

        Keyed on the aircraft FlightGear actually loaded, which the caller may
        have overridden.
        """
        assert self._viewer is not None
        aircraft = self._viewer.get_property("/sim/aircraft") or self._aircraft
        sync = CockpitSync(fdm, aircraft)
        self._viewer.start_property_sync(
            sync.properties, sync.mirrors, initial=sync.initial_properties()
        )

    def set_control(self, var: FlightVariable, value: float) -> None:
        mapping = CONTROL_MAP[var]
        self._require_fdm()[mapping.prop] = mapping.sign * value

    def run(self) -> None:
        fdm = self._require_fdm()
        fdm.run()
        if math.isnan(fdm["position/h-sl-ft"]):
            raise SimulatorError("JSBSim state diverged (NaN altitude)")

    def observe(self, var: FlightVariable) -> float:
        mapping = OBSERVATION_MAP[var]
        return mapping.to_canonical(float(self._require_fdm()[mapping.prop]))

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
            "w", suffix=".xml", prefix="aerogym_fg_", delete=False
        ) as fh:
            fh.write(content)
            self._directive_path = Path(fh.name)
        if not self._fdm.set_output_directive(str(self._directive_path)):
            raise SimulatorError("JSBSim rejected the FlightGear output directive")
