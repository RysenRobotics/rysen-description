# Rysen Robotics Descriptions

Simulation assets for Rysen Robotics hardware: URDF and MJCF models, the tooling
that generates and inspects them, and runnable example scenes. Every model is
available with or without the 619-point tactile array. USD assets are coming
soon.

<p align="center">
  <img src="apex_hand/urdf/images/frames_and_tactiles.png" alt="Coordinate frames and tactile array layout" width="720">
</p>

| Robot | DoF | Tactile | Formats | Docs |
|---|---|---|---|---|
| Apex Hand | 21 | 619 points | URDF, MJCF, USD (soon) | [apex_hand/](apex_hand/README.md) |

## Repository Structure

```
description/
├── apex_hand/          Apex Hand assets
│   ├── urdf/
│   ├── mjcf/
│   └── usd/            coming soon
└── (future robots)
```

One directory per robot. Within a robot, one directory per format, and the
tooling and data that belong to a format live inside that format's directory
rather than beside it:

```
<robot>/
├── urdf/               source model + meshes + tactile data + tooling
├── mjcf/               generated + meshes + tooling + examples
└── usd/                coming soon, same shape as the others
```

A `usd/` directory is coming soon: USD assets will be generated from `urdf/` the
way `mjcf/` is today, and will carry their own `tools/` and `examples/`.

Per-robot details — which variant to load, mesh caveats, how the model is
generated — are in that robot's `urdf/` and `mjcf/` READMEs.

## Quick Start

### MuJoCo

Needs MuJoCo >= 3.10.

```bash
# without the tactile array
python3 -m mujoco.viewer --mjcf=apex_hand/mjcf/apex_hand_right/apex_hand_right.xml

# with the 619 tactile points
python3 -m mujoco.viewer --mjcf=apex_hand/mjcf/apex_hand_right/apex_hand_right_tactile.xml

cd apex_hand/mjcf
python3 examples/view_hand.py --side right              # no tactile
python3 examples/view_hand.py --side right --tactiles   # with tactile
python3 examples/grasp_cylinder.py --side right         # drive onto a cylinder, close, lift
```

Or load a model in Python — both variants load the same way:

```python
import mujoco

plain = mujoco.MjModel.from_xml_path(
    "apex_hand/mjcf/apex_hand_right/apex_hand_right.xml")           # no tactile

tactile = mujoco.MjModel.from_xml_path(
    "apex_hand/mjcf/apex_hand_right/apex_hand_right_tactile.xml")   # 619 tactile points
```

A recorded grasp — 3D view on the left, 619-point pressure map on the right:

![Right hand grasp](apex_hand/mjcf/examples/grasp_right.gif)

The left hand is the mirror image; `apex_hand/mjcf/examples/record_grasp.py --side left`
records it.

### URDF

Needs ROS 2 Humble on Ubuntu 22.04. The viewer shows the plain model by default;
`--tactile` switches to the variant carrying the 619 point frames.

```bash
python3 apex_hand/urdf/tools/rviz_viewer.py --side right            # no tactile
python3 apex_hand/urdf/tools/rviz_viewer.py --side right --tactile  # with tactile
```

## Contact

Questions, bug reports or feature requests: [support@rysenbot.com](mailto:support@rysenbot.com).

## License

See [LICENSE](LICENSE).
