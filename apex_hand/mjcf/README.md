# Apex Hand MJCF

MuJoCo models generated from [`../urdf/`](../urdf/README.md).

```
mjcf/
├── apex_hand_left/
│   ├── apex_hand_left.xml
│   ├── apex_hand_left_tactile.xml
│   ├── apex_hand_left_convex_decomp.xml
│   ├── apex_hand_left_convex_decomp_tactile.xml
│   └── meshes/            mirrored from ../urdf/, convex/ is the hull cache
├── apex_hand_right/       same structure
├── tools/                 urdf_to_mjcf.py, config.yaml
└── examples/              view_hand.py, grasp_cylinder.py, record_grasp.py,
                           lib/ (tactile helpers), scene/ (ground, cylinder)
```

**Generated files.** Everything named `*.xml` at the top of each side is
produced by `tools/urdf_to_mjcf.py`. Editing it is lost on the next run.

## Variants

Four per side, from two independent choices: how collision is represented, and
whether tactile spheres are emitted.

| File | Collision | Tactile | Geoms |
|---|---|---|---|
| `apex_hand_<side>.xml` | raw mesh | — | 76 |
| `apex_hand_<side>_tactile.xml` | raw mesh | 619 spheres | 695 |
| `apex_hand_<side>_convex_decomp.xml` | convex hulls | — | 553–557 |
| `apex_hand_<side>_convex_decomp_tactile.xml` | convex hulls | 619 spheres | 1172–1176 |

Which to use:

- **`_convex_decomp`** for anything performance-sensitive or held for many
  steps. MuJoCo convexifies mesh geoms anyway, but doing it ahead of time lets
  the hull budget be tuned per link and drops the geom count by a third.
- **plain mesh** when the contact shape matters more than speed. The palm is a
  U-shaped shell that a solid convex hull can only approximate, so the hull
  version over-covers it by about 1.4×.
- **`_tactile`** whenever you need per-point contact readings. Adds 619 spheres
  of radius 1 mm under the 16 pad links, plus translucent pad shells so the
  spheres stay visible.

Both choices are independent — all four combinations exist.

## Simulation Settings

| Parameter | Value | Why |
|---|---|---|
| `timestep` | 2 ms | Coarsest step that keeps the mimicked joints stable |
| Integrator | `implicitfast` | Resists the stability limit of stiff contacts |
| Cone | `elliptic` | Correct for rubber pads, unlike the default pyramid |

Two joints have no motor on the way back on the real hand — the thumb's distal
joint and the fingers' middle joint are returned by a preloaded torsion spring.
It is a joint `stiffness` whose `springref` sits below the open limit, derived per
joint so the torque runs 0.02 N·m at the stop up to 0.05 at full flexion. Those
actuators' `forcerange` starts at 0: the motor pulls only towards closing, so
nothing but the spring opens the joint.

A torque cap on the servo is not equivalent: its force fades as the joint nears
its target and its `kv` term keeps braking. The preload matters too — without it
the return ends on a torque that has fallen to nothing, and the last few degrees
crawl against friction. With the spring, and those joints plus their mimics on
their own lower `damping` (0.0015, against 0.05 elsewhere — the global value
stands for the motor's rotor loss, which is gone with the motor off), the return
takes 0.19 s on the thumb and 0.24 s on the fingers.

Contact parameters live in `tools/config.yaml` alongside everything else that
can be tuned; see that file for the reasoning behind each value.

## Examples

All three run from any working directory and need only MuJoCo.

For the SDK-facing virtual device that loads these models over TCP, see
[`../virtual_device/README.md`](../virtual_device/README.md).

### view_hand.py

Loads the hand on a ground plane and lets you drag the joints. The hand is fixed
at the palm, so only the fingers move — useful for checking joint ranges.

```bash
python3 examples/view_hand.py                                   # right hand, mesh collision
python3 examples/view_hand.py --side both --collision convex_decomp --tactiles
```

| Option | Default | Description |
|---|---|---|
| `--side` | `right` | `left`, `right` or `both` |
| `--collision` | `mesh` | `mesh` or `convex_decomp` |
| `--tactiles` | off | load the variant carrying the 619 tactile spheres |
| `--show-collision` | off | draw the collision geoms, hidden by default |

### grasp_cylinder.py

Drives the hand onto a cylinder, closes the fingers and lifts it. The hand sits
on a mocap body welded rigidly to it — the same rigging a teleoperation or
arm-integration setup would use.

```bash
python3 examples/grasp_cylinder.py --side right
```

| Option | Default | Description |
|---|---|---|
| `--side` | `right` | `left` or `right` |
| `--collision` | `mesh` | `mesh` or `convex_decomp` |
| `--tactiles` | off | load the tactile variant and draw the contact forces |
| `--show-collision` | off | draw the collision geoms, hidden by default |

### record_grasp.py

Renders that grasp to a GIF, with the tactile pressure maps in a column beside
the 3D view. Always uses mesh collision and the tactile variant.

```bash
MUJOCO_GL=osmesa python3 examples/record_grasp.py --fps 30
```

| Option | Default | Description |
|---|---|---|
| `--side` | `right` | `left` or `right` |
| `--fps` | `30` | playback rate; the trajectory is 9 s long |
| `--width` / `--height` | `640` / `480` | 3D panel size, the default offscreen framebuffer |
| `--stride` | `1` | capture every Nth frame, for a quick preview |
| `--out` | `examples/grasp_<side>.gif` | output path |

Needs an OpenGL backend: on a machine with a GPU it works as is, in a container
without one set `MUJOCO_GL=osmesa`.

### examples/lib/

`mj_tactiles.py` turns contacts into per-point forces and pressure maps
(`RobotTactiles`); `tactile_viewer.py` draws them as a hand-shaped panel. Both
are imported by the scripts above rather than run on their own. The readings are
relative signals — which points are loaded and how the load moves — not
calibrated pressures, and they do not match the USD model in absolute terms.
Readings are clipped to the array's reporting range (0.1–20 N, 10–200 kPa), the
same limits the USD side uses; see
[`../README.md`](../README.md#reporting-range).

## Regenerating

```bash
python3 tools/urdf_to_mjcf.py
```

Writes all eight variants for both sides. Takes a few minutes the first time,
because each mesh is voxel-closed and hulled; later runs reuse the cache in
`<side>/meshes/convex/`.

**The cache is keyed by the mesh's bytes plus the `convex` values** — hull
budget, voxel pitch, resolution. Change one and the affected hulls rebuild on
the next run; nothing has to be deleted by hand. Only a change to the
decomposition code itself needs the `CACHE_STAMP` name in
`tools/lib/convex_decomp.py` bumped.

## Notes

**`meshdir` resolution.** The models set `meshdir="."`, which MuJoCo resolves
against the *process working directory*, not the model's own directory. Loading
with `MjModel.from_xml_path` is unaffected (it uses the file's directory), but
code that builds a scene in memory with `MjSpec.from_string` and `<include>`
must chdir to this directory first. The example scripts do exactly that, which
is why they run from anywhere.

**Tactile sphere placement.** Each sphere centre sits 1.5 mm inside the pad
skin, measured per point along its own +z so the clearance is constant. Slightly
more than the 1 mm radius, so a rendered pad shell hides the spheres entirely.
Points whose forward ray misses the mesh — they sit outside a thin wall — are
found by backing off along −z and casting forward again.

**Collision is disabled selectively.** Overlapping bodies are listed in
`<exclude>`, generated by compiling a model without excludes and collecting
every pair that interpenetrates. Parent/child pairs are kept in that list:
MuJoCo usually filters them, but a body nested under a fixed joint still
collides with its parent.

## Contact

Questions, bug reports or feature requests: [support@rysenbot.com](mailto:support@rysenbot.com).
