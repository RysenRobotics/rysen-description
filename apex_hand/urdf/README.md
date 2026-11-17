# Apex Hand URDF

The source model for the Apex Hand. Everything under `../mjcf/` is generated
from here, so this is the directory to edit.

```
urdf/
├── apex_hand_left/
│   ├── apex_hand_left.urdf
│   ├── apex_hand_left_tactile.urdf
│   └── meshes/            48 STLs
├── apex_hand_right/       same structure
├── tactile/               sensor data + the tactile point generator
└── tools/                 rviz_viewer.py
```

## Variants

| File | Contents |
|---|---|
| `apex_hand_<side>.urdf` | Links, joints, meshes |
| `apex_hand_<side>_tactile.urdf` | Base model plus 619 point frames |

## Tactile Points

Each hand has **619 tactile points** across 16 pad links, one per sensor cell,
each attached as a fixed joint to its pad link. They are massless reference
frames, not collision geometry.

| Tactile pad link | Source sensors file | Points |
|---|---|---|
| `left_palm_tactile` | `palm_link_left_sensors.urdf` | 89 |
| `right_palm_tactile` | `palm_link_right_sensors.urdf` | 89 |
| `{side}_thumb_tactile0` | `f0_link2_sensors.urdf` | 48 |
| `{side}_thumb_tactile1` | `f0_link3_sensors.urdf` | 32 |
| `left_thumb_tactile2` | `f0_link4_sensors.urdf` | 58 |
| `right_thumb_tactile2` | `right_f0_link4_sensors.urdf` | 58 |
| `{side}_{index,middle,ring,pinky}_tactile0` | `f1_link1_sensors.urdf` | 36 |
| `{side}_{index,middle,ring,pinky}_tactile1` | `f1_link2_sensors.urdf` | 14 |
| `{side}_{index,middle,ring,pinky}_tactile2` | `f1_link3_sensors.urdf` | 48 |

Total per hand: 89 + 48 + 32 + 58 + 4×(36 + 14 + 48) = **619**. The four
fingers share one sensor layout; only the thumb's last link has separate left
and right files; `{side}` is `left` or `right`.

A point's frame follows the URDF convention: its **+z is the outward surface
normal**, so a reading can be projected along +z without inverting anything.
The stored position is only a reference near the skin: it lands 0.02–2.8 mm
below the surface, and the palm's 89 points sit about 4.4 mm down, outside the
solid shell. The MJCF generator re-projects each point onto the outermost skin
along +z and insets it 1.5 mm, which is why the spheres vanish under a drawn pad
(see [`../NOTES.md`](../NOTES.md)).

### `sensors/` — Raw Point Data

Exported from point selection on the CAD surfaces, one XML fragment per pad
link. Two quirks matter: the files have **no common root node** (several
`<sensor>` elements side by side, so a standard parser reports "junk after
document element"), and each starts with an **example** `<sensor>` inside a
comment. The scripts strip the `<?xml?>` declaration, wrap the rest in
`<root>`, and count only `sensor_0` .. `sensor_N-1` — do not count the
commented-out example.

The sensor files label their angles `Roll Pitch Yaw`, but the generator reads
them as yaw-pitch-roll and swaps roll with yaw. Both readings were checked
against the pad meshes and give the same point z axis: the only difference is a
free rotation about z, which cannot change anything for a spherical point.

### `remap.json` — 2D Array Layout

Maps a point number onto the (row, column) cell of the physical sensor array, so
readings can be indexed by array position instead of a flat point number.

- `palm.left` / `palm.right` — a 14×14 array each (the hands are mirror images)
- `thumb["2"|"3"|"4"]` — 6×8 / 6×6 / 8×9 arrays
- `finger["1"|"2"|"3"]` — 6×6 / 4×5 / 8×8 arrays, shared by the four fingers
- Each cell holds a point number; `-1` is an empty slot.

### Generators

`tactile/attach_tactile_points.py` reads the base URDF plus `sensors/` and
writes `*_tactile.urdf`. The generated block is delimited by
`<!-- BEGIN/END GENERATED TACTILE POINTS -->`, so a re-run replaces it as a
whole.

```bash
python3 urdf/tactile/attach_tactile_points.py --dry-run   # print stats only
python3 urdf/tactile/attach_tactile_points.py             # write both hands
```

| Option | Default | Description |
|---|---|---|
| `--side` | `both` | `left` / `right` / `both` |
| `--sensors-dir` | `urdf/tactile/sensors` | Sensors fragment directory |
| `--radius` | `0.0008` | Point sphere radius (m) |
| `--mass` | `0.000001` | Point sphere mass (kg) |
| `--rgba` | `1 0.05 0.05 1` | Point sphere colour |
| `--dry-run` | off | Print stats only, do not write |

`tactile/correct_tactile_normal.py` aligns each point's +z with the surface
normal at the nearest point on the host link's mesh (trimesh closest-point
query). The palm group's normals are flipped, so its z points inward; points
already aligned (dot > 1-1e-8) keep their values.

```bash
python3 urdf/tactile/correct_tactile_normal.py
```

**Warning**: it overwrites `*_tactile.urdf` in place with no backup.

Both scripts need `numpy` and `trimesh` (registered in
`.devcontainer/requirements.txt`).

## Mesh Caveats

**Paths are relative.** Each URDF references `meshes/<name>.STL` relative to
its own directory. Tools that resolve mesh URIs against a different base
directory — several RViz setups, some CAD importers — need that path rewritten
to an absolute one first.

**Not all meshes are watertight.** About half of the collision meshes have open
edges, and `*_palm_link.STL` additionally carries duplicated back-to-back
triangles. Rendering is unaffected and MuJoCo convexifies them fine. Tools that
require a closed manifold (boolean operations, some simulators, volume
computations) will either reject them or need a voxel-closing pass first — see
`../mjcf/tools/lib/convex_decomp.py` for how the generator handles it.

**The palm file holds two shells.** `*_palm_link.STL` contains both the outer
skin and an inner frame structure — they are complementary, not duplicates, and
together they are one link in the URDF. Splitting them apart loses geometry; do
not "clean up" by deleting one.

## RViz Viewer

`tools/rviz_viewer.py` is a single-file viewer for the URDFs in this package. It
has no dependency on the rest of this repository and needs no colcon build — ROS 2
is the only requirement.

### Requirements

- **Ubuntu 22.04 + ROS 2 Humble.** The script sources `/opt/ros/*/setup.bash`
  itself, so nothing has to be sourced beforehand.
- Python 3.9+ (the 3.10 shipped with Humble is fine; no third-party Python
  packages beyond ROS).
- A graphical display — RViz needs X11 or Wayland.

### Installation

Install the full desktop metapackage, or only the five packages this tool
actually uses:

```bash
# Option 1: everything
sudo apt install ros-humble-desktop

# Option 2: minimal set
sudo apt install \
    ros-humble-robot-state-publisher \   # TF from URDF + joint angles
    ros-humble-rviz2 \                   # the GUI
    ros-humble-ros2cli \                 # the `ros2` command
    ros-humble-rclpy \                   # used by the joint publisher subprocess
    ros-humble-sensor-msgs               # JointState message definitions

# Only needed for --slider
sudo apt install python3-tk
```

### Use

```bash
python3 apex_hand/urdf/tools/rviz_viewer.py                          # left hand, zero pose
python3 apex_hand/urdf/tools/rviz_viewer.py --side right --tactile   # right hand with tactile points
python3 apex_hand/urdf/tools/rviz_viewer.py --slider                 # add a joint slider window
```

| Option | Default | Description |
|---|---|---|
| `--side` | `left` | `left` or `right` |
| `--tactile` | off | load `apex_hand_<side>_tactile.urdf` instead of the plain model |
| `--slider` | off | open a joint slider window alongside RViz |

Closing the RViz window, closing the slider window, or pressing Ctrl+C exits;
the ROS processes and temporary files it started are cleaned up automatically.

## Editing

`urdf/` is hand-maintained; there is no generator producing it from CAD. If you
change joint limits, link inertias or mesh assignments, re-run the MJCF
generator afterwards so the two formats stay in sync:

```bash
cd apex_hand/mjcf && python3 tools/urdf_to_mjcf.py
```

## Contact

Questions, bug reports or feature requests: [support@rysenbot.com](mailto:support@rysenbot.com).
