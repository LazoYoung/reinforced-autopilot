# The policy file

`autopilot export` turns a flight checkpoint into one ONNX file that flies
without this repository's code. This page is that file's contract, format 1.

```sh
uv run autopilot export runs/dqn-steady-flight/v4/best.pt   # writes best.onnx beside it
uv run autopilot export runs/dqn-steady-flight/v4/best.pt -o exports/hold.onnx
```

Before keeping the file, export flies one greedy episode with it beside the
checkpoint. A failed check leaves no policy file (an existing file is
untouched), though it may leave behind a directory it just created.

Export traces the scenario code in the current checkout: export a checkpoint
with the code it was trained on. The check compares the file with that code,
not with the code at training time, so a checkpoint exported after its
scenario's features changed can still pass with a different file.

## Inputs and outputs

Every input and output is a float32 tensor of shape `[batch]`, in the
[canonical units](../README.md#canonical-units) and the pilot sign convention.
Each input's and each action output's ONNX doc string gives its unit, range
and meaning.

| Name | Direction | Meaning |
|---|---|---|
| a flight variable: `altitude`, `heading`, ... | in | this step's reading |
| a target: `target_altitude_ft`, ... | in | the value to hold |
| `prev.<action>`: `prev.elevator`, ... | in | the command applied now |
| an action: `elevator`, ..., `throttle` | out | the command to apply next, clipped to its range |
| `value` | out | the policy's estimate of the state's worth, comparable only within one file |

A file takes only what its policy reads: a policy that ignores airspeed has no
`target_airspeed_kts` input. Bind every input and output by name; their order
is not part of the contract.

`prev.*` is the command the aircraft is flying with now, read back from the
simulator on every call. It is not the file's last output. The two agree while
the file flies, and the read-back is also right on engagement, after a pilot
override, and across a mode switch. Feed the control *command*, never the
surface position: on a trimmed aircraft the position includes trim, and
feeding it back adds that trim a second time.

## Metadata

| Key | Value |
|---|---|
| `aero.format` | `1` |
| `aero.agent_hz` | how often to call the file, in Hz |
| `aero.aircraft` | the JSBSim model the policy was trained on |
| `aero.scenario` | the scenario's name |
| `aero.config` | the full training config, as JSON |
| `aero.frames` | how many frames the checkpoint had trained on |

`aero.format` covers the naming rules, the `[batch]` float32 shapes, the
meaning of `prev.*`, and the clipped outputs. A change to any of them bumps it.

## Flying it

```python
import numpy as np
import onnxruntime as ort


def tick(session, readings, targets, applied):
    """One call of the policy file.

    readings: this step's canonical readings, by name ("altitude", ...).
    targets: the targets the pilot set, by name ("target_altitude_ft", ...).
    applied: the commands the aircraft is flying with now, read back from the
    simulator, by action name ("elevator", ...).
    Returns the commands to apply next, by action name.
    """
    values = readings | targets | {f"prev.{name}": v for name, v in applied.items()}
    feed = {i.name: np.array([values[i.name]], np.float32) for i in session.get_inputs()}
    names = [o.name for o in session.get_outputs()]
    result = dict(zip(names, session.run(None, feed)))
    return {name: float(column[0]) for name, column in result.items() if name != "value"}


session = ort.InferenceSession("best.onnx")
hz = float(session.get_modelmeta().custom_metadata_map["aero.agent_hz"])
# call tick every 1 / hz seconds, holding each command between calls
```

## Host guidance

- **Engage** by setting each target to its current value (altitude, heading,
  airspeed), as real autopilots do; a hold policy was trained with the target
  starting where the aircraft is. Nothing else needs seeding, since `prev.*`
  is read back.
- **Disengage** by no longer calling the file. It holds no state.
- **Modes are files.** One file serves one combination of targets and axes;
  switch modes by switching files. Each file's names say what it takes and
  what it drives, and the `prev.*` read-back keeps the switch bumpless.
- **Stay in range.** Clamp each target to the range its doc string gives. A
  hold policy holds; capturing a distant target needs a policy trained for it.
- **Call at `aero.agent_hz`**, holding each command between calls.
