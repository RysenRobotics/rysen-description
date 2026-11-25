# ApexHand MuJoCo Virtual Prototype

When no physical dexterous hand is available, MuJoCo stands in for the hardware.
The existing SDK APIs still send commands and receive feedback without
modification, and the SDK does not need to know whether it is connected to the
simulation or to a real hand.

The virtual prototype uses the MJCF models generated under
`description/apex_hand/mjcf/` directly. It does not depend on the `simulate/`
directory or an external Runtime package.

## Start

The current MuJoCo environment already includes `mujoco` and `numpy`. Run the
virtual prototype from its directory:

```bash
cd description/apex_hand/virtual_device
./start_simulation.sh right
```

The MuJoCo window opens by default. On a machine without a desktop:

```bash
./start_simulation.sh right --headless
```

Common options:

```bash
./start_simulation.sh left
./start_simulation.sh dual
```

## SDK connection

Keep the virtual prototype running, then start the SDK program as usual in a
second terminal. Only the device IP needs to match the selected scene:

| Scene | Device IP |
| --- | --- |
| `left` | `127.0.0.1` |
| `right` | `127.0.0.2` |
| `dual` | left `127.0.0.1`, right `127.0.0.2` |

For example:

```bash
cpp/build/bin/rysen_example --ip 127.0.0.2
python3 python/example.py --ip 127.0.0.2
```

The control port is TCP `5856` and the feedback port is TCP `5857`. The SDK and
the virtual prototype must run on the same computer in the same network
namespace.

## Functional coverage

- Position control: `MoveJoint`, `MoveJPositionFollow`, and `MoveJControlFollow`
- MIT impedance control: start, reference and gain updates, state query, and stop
- Feedback: joint states, motor current, estimated temperature, and fingertip force
- Tactile: pressure images, callbacks, and zero calibration
- Fault flags: `GetHardwareErrorCode` and callbacks
- Device addressing: `SetDeviceIPAddress`

Temperature and tangential tactile values come from simplified models and are
not calibrated against the real hand. They are intended for validating the SDK
path and UI, not as real-hand measurements.

## Model and mounting

The scene is assembled from:

- `../mjcf/apex_hand_<side>/apex_hand_<side>_tactile.xml`

The hand models in `description/` are fixed to the world by default so they can
be mounted as arm assets. At runtime the virtual prototype adds a `freejoint` to
the palm and holds it with a mocap body plus weld, preserving the same dynamic
palm structure used for hardware-equivalent simulation. The base is at
`z = 0.5`, with the fingers pointing along world `+z` (up). The ground plane is
reused from `../mjcf/examples/scene/ground.xml`.

Physical parameters live in `config/hand_physics.json`, including the server
settings and per-joint `kp`, `kv`, `armature`, and `damping`. Coupling
constraints keep the defaults from the generated description model.

## Files

- `mujoco_server.py`: SDK protocol, control/feedback, and the simulation loop
- `mujoco_model.py`: assembles the virtual prototype scene from the description MJCF
- `device_network.py`: TCP listeners, device addressing, and reset
- `motor_model.py`: motor transmission, current-to-torque mapping, and temperature estimate
- `tactile_feedback.py`: tactile point mapping and feedback frames
- `start_simulation.sh`: startup entry point
- `config/hand_physics.json`: local physics and server parameters
