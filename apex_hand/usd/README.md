# Apex Hand USD

USD assets for Isaac Sim, generated from [`../urdf/`](../urdf/README.md). The
counterpart of [`../mjcf/`](../mjcf/README.md): same joints, same tactile array,
PhysX instead of MuJoCo.

Verified against **Isaac Sim 6.2.0** and **Isaac Lab 3.0.0**
(`/opt/isaacsim/venv/bin/python`), RTX 4070.

```
usd/
├── apex_hand_left/
│   ├── apex_hand_left[_mount][_tactile].usd
│   └── manifest.json
├── apex_hand_right/       same structure
├── tools/                 urdf_to_usd.py, config.yaml, lib/
└── examples/              view_hand.py, grasp_cylinder.py, record_grasp.py,
                           lib/ (hand, tactiles, tactile window), grasp_right.gif
```

**Generated files.** Everything named `*.usd` at the top of each side is
produced by `tools/urdf_to_usd.py`. Editing it is lost on the next run;
`manifest.json` is written by the same run.

## Variants

Four per side, from two independent choices: whether the 619 tactile bodies are
emitted, and whether the root carries a six-joint mount. Only the mount is a
suffix — a bare file is the plain hand.

| File | Bodies | Joints | Tactile | Mount |
|---|---|---|---|---|
| `apex_hand_<side>.usd` | 22 | 21 | — | — |
| `apex_hand_<side>_tactile.usd` | 641 | 21 | 619 spheres | — |
| `apex_hand_<side>_mount.usd` | 28 | 27 | — | 6-DOF |
| `apex_hand_<side>_mount_tactile.usd` | 647 | 27 | 619 spheres | 6-DOF |

Counts are the articulation's, so the `_mount` variants add six links to the 22
URDF bodies, and `_tactile` adds one rigid body per sensing element.

`_tactile` also draws translucent pad shells so the spheres stay visible.

The **mount** is three prismatic joints and three revolute ones, chained through
massless links between the world and the palm, with position drives. It exists so
a scene can move the hand without teleporting it: a written root pose discards
velocity and drags the colliders through whatever they touch, whereas an actuated
mount is pushed back by the contact it just made. The bare files omit it, which is
what an arm integration or a teleoperation rig wants — the hand is then a
floating articulation and the caller supplies its own root joint.

`hand_cfg(..., mount=False)` selects a bare variant; every example uses the
`_mount` one.

Each file is self-contained: the 38 source meshes are authored once under
`/apex_hand_<side>/meshes` and referenced internally, so a variant can be
dragged into Isaac Sim or handed off on its own.

`manifest.json` carries what the examples need at runtime and would be awkward
to dig out of USD: joint names, limits, the mimic relations, the mount joint
order, and every tactile point's position and normal. One file serves all four
variants, since they differ only by the mount chain and the tactile bodies; the
`variants` map says which file is which, and `mount_joints` is only meaningful
for the `_mount` ones.

## Simulation Settings

| Parameter | Value | Why |
|---|---|---|
| `timestep` | 1/120 s | Coarsest step that keeps the 619 tactile bodies stable |
| Collision | `convexDecomposition` (links), `convexHull` (`_tactile` shells) | Thin open CAD shells have no reliable SDF interior; the shells are too thin to voxelise |
| Contact model | compliant springs on all three materials | Allows the pad to give ~1 mm so an object reaches the buried spheres |

`tools/config.yaml` holds every value that can be tuned — collision cooking,
contact offsets, friction, actuator gains, tactile geometry — with the reasoning
next to each. `simulation.dt` is only written into the manifest.

## Examples

All need the Isaac Sim environment's Python, which brings `isaaclab` and `pxr`;
the `SimulationApp` is started by the scripts themselves. Run them from any
working directory.

### view_hand.py

Opens the hand in the viewport with a slider per finger joint. The palm is
held by the mount and only the fingers move — useful for checking joint ranges.
The mount's six joints get no sliders: the loop re-drives them to a fixed pose
every frame, so a slider there could not take effect. With `--side both` the two
hands stand 0.26 m apart, wrists 0.10 m up.

The URDF frame runs the fingers along +z out of the wrist, but the body frame
PhysX simulates is that frame turned 180° about X (its centre of mass resolves
68 mm below the link origin and every finger joint resolves to −z). An identity
rotation therefore hangs the fingers straight down, so the viewer applies
`REST_QUAT`, a half turn about X, to stand them up.

```bash
python3 examples/view_hand.py                       # right hand, viewport
python3 examples/view_hand.py --side both --tactiles
python3 examples/view_hand.py --headless            # sweep, no window
```

| Option | Default | Description |
|---|---|---|
| `--side` | `right` | `left`, `right` or `both` |
| `--tactiles` | off | load the variant carrying the 619 tactile bodies |
| `--show-collision` | off | draw the collision meshes instead of the visual ones |
| `--headless` | off | sweep every joint instead of opening a viewport |
| `--viz` | `kit` | launcher option; `none` keeps it console-only |

### grasp_cylinder.py

Drives the hand onto a cylinder, closes the fingers and lifts it. The hand's
root is driven through a six-joint mount, so a contact pushes back instead of
being teleported through.

```bash
python3 examples/grasp_cylinder.py --tactiles       # arrows + pressure window
python3 examples/grasp_cylinder.py --viz none --tactiles   # console only
```

| Option | Default | Description |
|---|---|---|
| `--side` | `right` | `left` or `right` |
| `--tactiles` | off | load the tactile variant and read the array every step |
| `--show-tactile` | off | force the pressure window without a viewport |
| `--viz` | `kit` | `none` keeps it console-only |

### record_grasp.py

Renders that grasp to a GIF, with the tactile pressure maps in a column beside
the 3D view. Always uses the tactile variant.

```bash
python3 examples/record_grasp.py
python3 examples/record_grasp.py --fps 30 --width 960 --height 720
```

| Option | Default | Description |
|---|---|---|
| `--side` | `right` | `left` or `right` |
| `--fps` | `20` | playback rate; the trajectory is 9 s long |
| `--width` / `--height` | `640` / `480` | 3D panel size |
| `--out` | `examples/grasp_<side>.gif` | output path |

`examples/grasp_right.gif` is the checked-in result.

### examples/lib/

`hand.py` builds the articulation config and writes the mount targets;
`grasp_trajectory.py` plays the recorded sequence; `tactiles.py` turns the
contact sensor into per-point forces and pressure maps (`RobotTactiles`);
`tactile_viewer.py` draws them into the viewport; `tactile_window.py` drives the
pressure window from a child process. All are imported by the scripts above
rather than run on their own.

## Tactile Readout

`tactiles.py` turns the `ContactSensor` into a sensor reading:

1. project each sphere body's net contact force onto that sphere's outward
   normal (its local +z, rotated into world), so shear does not leak in;
2. clip to the array's range — 0.1–20 N, with anything below the floor reading
   zero;
3. divide by the point's share of the pad area — the square of the distance to
   its nearest neighbour, a Voronoi cell — and clip the result to 10–200 kPa;
4. scatter into the 2D array from [`../urdf/tactile/remap.json`](../urdf/tactile/remap.json),
   so a reading can be indexed by (row, column) like the hardware.

Both the position and the normal come from the sphere's own rigid body, not
from the pad link: `body_pos_w` is a centre-of-mass frame, and rebuilding the
point from it was off by tens of millimetres.

Closing is position playback, the same as the MJCF model: every digit curls to a
fixed fraction of its range and the compliant pads stop it at the object. The
readout is for observation only — nothing in the control loop depends on it.

## Visualisation

Two views, as in the MuJoCo demo.

**Red arrows** in the viewport, one per loaded point, along the outward normal.
Their length follows √force over the array's range — 28 mm at the 0.1 N floor,
400 mm at the 20 N ceiling — so a weak point still shows up next to a strong
one; see the reporting range in [`../README.md`](../README.md#reporting-range).
Kit's debug-draw has lines but no arrow primitive, so each arrow is a shaft plus
four short edges forming a tip; the extension is enabled by the viewer at
start-up.

**The pressure window** is the MuJoCo demo's `TactilePanel`, reused unchanged:
pads tiled in the outline of the hand — palm at the bottom, four fingers as
columns, thumb alongside — one cell per sensing element, blue at rest and at
10 kPa, red at 200 kPa. It cannot be created inside Kit: this virtualenv ships
matplotlib with no GUI backend (Agg only), and importing Qt there crashes the
simulator. So it runs in its own process (`tactile_window.py` spawns
`tactile_panel_host.py`), fed over localhost. The host needs a Python whose
matplotlib has Tk or Qt; it tries `$TACTILE_WINDOW_PYTHON`, then
`/usr/bin/python3`, then `python3`, and degrades to a one-line warning if none
qualifies.

`record_grasp.py` draws both into the GIF instead: the arrows are projected into
the camera and drawn onto the frame (debug-draw is a viewport layer and never
reaches a sensor render), with a depth buffer hiding the shafts that are behind
the hand.

## Regenerating

```bash
python3 tools/urdf_to_usd.py                 # both hands, all four variants
python3 tools/urdf_to_usd.py --side right --variants mesh
python3 tools/urdf_to_usd.py --variants mount_tactile,mesh_tactile
```

Change `tools/lib/` or `config.yaml` rather than the generated files.

## Notes

**Why convex decomposition.** PhysX rejects triangle meshes on dynamic bodies,
and an SDF baked from these thin, open CAD shells has no reliable inside: a
cylinder pressed into a finger resolved against the wrong side and passed
through. Cooking parameters were measured per link — 32 hulls for the phalanges,
64 for the U-shaped palm, whose cavity is over-filled at 32.

**Bodies are merged.** The URDF's 48 links collapse to 22, because the 16 tactile
pads and the 5 `_tip` / `_pad` frames hang off a phalanx by fixed joints. The
`_tip` and `_pad` links carry only a locator sphere, so they become Xform
markers, matching the sites the MJCF model emits.

**Tactile spheres** are one rigid body each, fixed to the pad, with their own
contact material — PhysX reports a contact force per rigid body and nowhere
finer. Radius 1 mm, centre 1.5 mm under the skin (the top of the pad 0.5 mm
under) so the shell, not the array, is what an object touches first. A sphere is
symmetric, so its rigid-body frame has no centre-of-mass offset.

**Mimic joints** are a target copy in the control layer: PhysX has no mimic
constraint and the URDF importer drops the relation. The metadata is kept as
`customData["urdf:mimicJoint"]` on the joint prim.

**Self-collision is off.** Several neighbouring links interpenetrate at the zero
pose, and the 619 overlapping spheres would push each other.

**Angular units.** USD stores joint limits and velocities in degrees, and
`physics:lowerLimit` / `physics:upperLimit` are what PhysX reads —
`UsdPhysics.LimitAPI` is ignored on these joints.

**The grasp is offset from the world origin.** The six mount joints start at
zero, so on the first solver pass the whole hand sits at `(0,0,0)`. PhysX
resolves that overlap before any example code can move the mount, and an object
spawned at the origin is fired away at 6 m/s. `SCENE_XY` in
`grasp_trajectory.py` shifts the cylinder, the palm poses and the camera
together; the hand's rest pose stays untouched.

## Known Limitations

1. **Contact magnitudes are over-estimated during a grasp.** The array reads the
   force correctly but a position-controlled grip buries the pads in the
   cylinder by millimetres, so the reported pressure follows penetration. Which
   points are loaded is right; a force-controlled close would fix the rest. The
   absolute values are not comparable with the MJCF model either: different
   solver, different step, different penetration.
2. **Contact reports do not work in this build.** `PhysxContactReportAPI` plus
   `subscribe_contact_report_events` yields nothing in Isaac Sim 6.2, hence the
   per-point bodies and `ContactSensor`.
3. **No self-collision.** Fingers can pass through each other. Turn it on in
   `config.yaml` only after checking the rest pose.
4. **The `_pad` links do not collide.** They are coordinate frames; the
   `_tactile` shells are the contact surface.
5. **The left hand** is generated and loads with the same numbers, but the grasp
   trajectory is tuned for the right hand; it mirrors automatically (the asset is
   a mirror image, so the world-side pose is not).

## Contact

Questions, bug reports or feature requests: [support@rysenbot.com](mailto:support@rysenbot.com).
