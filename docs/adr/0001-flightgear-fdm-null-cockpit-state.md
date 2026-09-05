# 1. Drive the FlightGear cockpit over the property server under `--fdm=null`

- Status: accepted
- Date: 2026-09-05

## Context

aero-gym flies the aircraft in JSBSim and uses FlightGear only to look at it.
FlightGear is therefore launched with `--fdm=null` and a native-FDM socket
(`--native-fdm=socket,in,<rate>,,<port>,udp`), so JSBSim pushes state in and
FlightGear renders it without integrating anything of its own. This is the only
arrangement that keeps one authoritative flight model: if FlightGear ran its own
FDM, the picture and the RL observations would diverge, and the environment's
step timing would be at the mercy of the viewer's frame rate.

`--fdm=null` has a consequence that is easy to miss. It does not merely disable
the equations of motion — it disables *everything* the aircraft's own JSBSim
configuration computes. FlightGear then knows only what arrives in the
native-FDM packet: position, orientation, control-surface deflections, and a
small amount of engine state. The packet is a rendering payload, not a
simulation state dump.

Everything else the cockpit depends on is normally produced by the JSBSim
configuration shipped with the *FlightGear* aircraft, and that configuration
never runs. Observed on `c172p`:

- The yoke and rudder pedals are animated from a JSBSim `<summer>`, so they sit
  frozen while the agent flies.
- Every pressure instrument is fed by a JSBSim `<fcs_function>` static source,
  so the airspeed indicator, altimeter, and VSI read nonsense.
- Fuel tanks are never filled, so the cockpit announces "Out of fuel!".
- Mach is absent from the packet, so it stays at zero and the pitot model
  collapses.
- The aircraft boots cold and dark; with the master switch off, the whole
  electrical panel reads zero no matter how good the data behind it is.

None of this affects training — it is a viewer — but a cockpit that contradicts
the simulation driving it is worse than no cockpit, because it is read as truth.

The alternatives were to accept the broken cockpit, to widen the native-FDM
packet (its layout is fixed by FlightGear, so this is not ours to change), or to
let FlightGear run its own FDM (rejected above).

## Decision

Keep `--fdm=null` and supply the missing cockpit state directly, over
FlightGear's property server, from the JSBSim instance aero-gym is actually
flying. The property server is a side channel for state the packet has no field
for; the packet stays the transport for everything it does carry.

Two rules keep the two channels from fighting:

- Never write what the packet already carries. Airspeed, for instance, is
  rewritten by FlightGear every frame, so writing it only causes a tug of war.
- State the user should stay free to change (the master switches) is written
  once on connect, not on every tick.

## Consequences

The unstaged changes implement this:

- `flightgear.py` swaps `--telnet=<port>` for
  `--props=socket,bi,60,localhost,<port>,tcp`. Same protocol, but the `--telnet`
  shorthand pins the property server at 5 Hz, which caps a read-and-write cycle
  near 2.5 Hz and leaves the cockpit visibly behind the simulation.
- `flightgear.py` adds `_PropertyClient` (a `data`-mode connection that reads one
  reply per `get`, since FlightGear batches replies into a single packet) plus
  `start_property_sync`/`stop_property_sync`, which run the writes on a daemon
  thread so the simulation never blocks on the viewer. If the property server
  cannot be reached, it warns and the visualization degrades rather than failing.
- `cockpit.py` builds the writes: pilot-convention control positions, Mach,
  per-engine throttle, tank capacity and selection, plus per-aircraft extras
  keyed by the aircraft FlightGear actually loaded — `c172p`'s yoke and pedal
  animation, and a *mirror* that points its static source at the pressure
  FlightGear still computes for itself, so instruments and the pitot model
  measure against one reference.
- `simulator.py` starts the sync only once `wait_until_ready()` confirms
  FlightGear is up, keyed on `/sim/aircraft` read back from FlightGear rather
  than on the scenario's aircraft, which can be overridden.

Costs we accept:

- Cockpit support is per-aircraft. `COCKPIT_MODELS` and `COCKPIT_MIRRORS` in
  `cockpit.py` currently cover `c172p` only; another aircraft renders and flies
  correctly but may show a dead yoke or bad instruments until an entry is added.
- The mapping depends on FlightGear-side property paths and on how a given
  aircraft is wired internally, so it can break on a FlightGear or aircraft
  update. The tests cover the transport, not FlightGear's tree.
- Fuel quantity is worked around rather than fixed: FlightGear reads the
  packet's pounds as US gallons, so tanks declare 1 lb/gal to keep every derived
  unit self-consistent.
- One more TCP connection and a 20 Hz write thread per episode with the viewer
  on — negligible, and absent entirely when running headless.
