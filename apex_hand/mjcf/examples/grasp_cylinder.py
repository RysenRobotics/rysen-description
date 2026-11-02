#!/usr/bin/env python3
"""Drive the hand onto a cylinder, close the fingers around it, and lift it.

Usage:
    python3 examples/grasp_cylinder.py                       # right, mesh
    python3 examples/grasp_cylinder.py --side left
    python3 examples/grasp_cylinder.py --collision convex_decomp
    python3 examples/grasp_cylinder.py --show-collision

The hand is mounted on a mocap body and welded to it, so moving the mocap body
moves the hand rigidly -- the same rigging a teleoperation or arm-integration
layer would use. The cylinder is a free body on the floor.

Sequence: approach with the thumb already swung open, close the fingers around
the cylinder, then lift it. The palm is the mount and the hand comes in from +x
with the palm -- not the back of the hand -- turned towards the cylinder.

With ``--tactiles`` the contact forces are read from the 619 tactile spheres and
shown two ways: red arrows in the viewer, and a separate window drawing each pad's
pressure map in kPa (blue at rest, red at 200 kPa).

Dependencies: mujoco; matplotlib for the pressure window.
"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
MJCF_DIR = EXAMPLES_DIR.parent

import sys
sys.path.insert(0, str(EXAMPLES_DIR / "lib"))
from mj_tactiles import RobotTactiles  # noqa: E402

# Collision variant -> filename suffix produced by tools/urdf_to_mjcf.py.
COLLISION_VARIANTS = {"mesh": "", "convex_decomp": "_convex_decomp"}
# Geom groups assigned by the generator: group 3 is the rigid collision, group 4
# the tactile pads' collision. Neither is drawn unless asked for.
COLLISION_GROUPS = (3, 4)
# Rendering rate. Physics still advances at the model's 2 ms timestep; this only
# decides how many steps are batched into one displayed frame.
FRAME_HZ = 60.0

# Hand geometry, measured on the asset with mj_forward (palm frame = identity):
#   index tip    = [0.008, 0.034, 0.232]     the fingers reach along +z
#   index_j0     0.9 of its range moves the tip +y   they curl towards +y
#   knuckles     at +x, fingers curl back    the palm normal is local -x
#   thumb pad    = [0.013, 0.155, 0.058]     the thumb sits on the +y side
CYLINDER_Z = 0.12            # cylinder centre height, matches cylinder.xml
CYLINDER_RADIUS = 0.042

# Where the hand waits before moving in. The hand comes in from -x, so it parks on
# that side and travels straight to the cylinder without sweeping through it.
PARK = (0.30, -0.12, 0.25)

# --- Pose 1: palm contact ---------------------------------------------------
# Both hand models share the same local frame: the knuckles march along +x and
# the fingers curl back from there, so the palm normal is local -x; the thumb
# sits on +/-y; the fingers extend along +z.
#
# The hand comes in from +x with the palm turned towards the cylinder:
#   local +x (palm back)   -> world -x : the hand approaches from +x
#   local -x (palm normal) -> world +x : the palm faces the cylinder
#   local +y (thumb side)  -> world +z : thumb up
#   local +z (fingers)     -> world +y : fingers lie sideways
CONTACT_QUAT = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
# Distance from the palm's origin to the surface of its collision shell along the
# local -x axis (measured from the collision meshes: they span x -0.029..+0.031,
# so the palm face is 0.029 from the origin).
PALM_SHELL = 0.029
# Small clearance so the palm rests against the cylinder rather than inside it.
# Kept at zero: the approach is supposed to end in contact, and the palm's
# position servo settles onto the wall instead of stopping short of it.
PALM_STANDOFF = 0.0
CONTACT_X = PALM_SHELL + PALM_STANDOFF + CYLINDER_RADIUS
CONTACT_Z = CYLINDER_Z
# The fingers reach along +y and curl back towards -y, so the palm is placed at
# -y of the cylinder axis for the curl to sweep across it.
CONTACT_Y = -0.12

# --- Pose 2: fingers wrapped around the cylinder ----------------------------
# The thumb's base joint swings out to 90 degrees first, which brings the thumb
# round to the far side of the cylinder instead of lying beside the fingers.
# Only then does everything curl in: the fingers wrap one side and the thumb the
# other, so the two squeeze rather than shoving the cylinder one way.
THUMB_OPEN_DEG = 90.0
# Closing is deliberately partial: curling to the limit drives the fingertips
# into each other and lifts the proximal phalanges off the cylinder. One shared
# fraction keeps the thumb and the fingers at the same bending angle, so neither
# side squeezes harder than the other.
CLOSE_FRACTION = 0.65
# The thumb finishes closing at this fraction of the closing phase, so it leads
# the fingers slightly instead of moving in lockstep with them.
THUMB_LEAD = 0.75

# --- Recorded trajectory ----------------------------------------------------
# The grasp as it was tuned: a list of phases, each held for a duration and then
# interpolated into the next one. Every phase sets the mocap position and the
# joint targets it should reach, so playing the script back is just interpolating
# down this table -- there is no per-step logic left to reason about.
#
#   t        seconds this phase runs, interpolating from the previous row
#   assume   named pose the row starts from (only meaningful for the first row)
#   pos      mocap position the hand should be at when the phase ends
#   close    fraction of each joint's range to curl to; None keeps the current
#            targets, and the thumb's own joints lead slightly (THUMB_LEAD)
MOVE_TIME = 2.0
CLOSE_TIME = 2.5
LIFT_TIME = 2.0
SETTLE_TIME = 0.5
HOLD_TIME = 1.5
# How far the grasp is lifted, to prove the cylinder is actually held.
LIFT_DELTA = 0.12

TRAJECTORY = (
    # Travel in from the park position, swinging the thumb open on the way.
    {"t": MOVE_TIME, "pos": (CONTACT_X, CONTACT_Y, CONTACT_Z), "open": True},
    # Hold there while the palm settles onto the cylinder.
    {"t": SETTLE_TIME, "pos": (CONTACT_X, CONTACT_Y, CONTACT_Z)},
    # Curl the fingers and thumb in around it.
    {"t": CLOSE_TIME, "pos": (CONTACT_X, CONTACT_Y, CONTACT_Z),
     "close": CLOSE_FRACTION},
    # Let the grip settle before lifting, so nothing creeps while it rises.
    {"t": SETTLE_TIME, "pos": (CONTACT_X, CONTACT_Y, CONTACT_Z)},
    # Lift, carrying the cylinder.
    {"t": LIFT_TIME, "pos": (CONTACT_X, CONTACT_Y, CONTACT_Z + LIFT_DELTA)},
    # Hold the result on screen.
    {"t": HOLD_TIME, "pos": (CONTACT_X, CONTACT_Y, CONTACT_Z + LIFT_DELTA)},
)


def variant_suffix(collision: str, tactiles: bool) -> str:
    """Filename suffix for a collision / tactile combination."""
    return COLLISION_VARIANTS[collision] + ("_tactile" if tactiles else "")


def build_model(side: str, collision: str, tactiles: bool = False) -> mujoco.MjModel:
    """Assemble the scene, then rig the hand to a mocap body.

    Built in two steps because the hand is emitted without a joint at its root --
    an asset that bakes in a free joint cannot be mounted on an arm. Here the
    root is given a mocap body plus a weld, which is the runtime rig the asset
    was designed for.
    """
    suffix = variant_suffix(collision, tactiles)
    hand = MJCF_DIR / f"apex_hand_{side}" / f"apex_hand_{side}{suffix}.xml"
    if not hand.exists():
        raise SystemExit(f"missing hand model: {hand}")

    # <include> resolves against the including file's directory; the hand models
    # live one level up, under apex_hand_<side>/, together with their meshes.
    hand_key = f"apex_hand_{side}/apex_hand_{side}{suffix}.xml"
    files = {
        "../scene/ground.xml": (EXAMPLES_DIR / "scene" / "ground.xml").read_bytes(),
        "scene/cylinder.xml": (EXAMPLES_DIR / "scene" / "cylinder.xml").read_bytes(),
        hand_key: (MJCF_DIR / hand_key).read_bytes(),
    }
    lines = ["<mujoco>"]
    lines += [f'<include file="{key}"/>' for key in files]
    lines.append("</mujoco>")

    spec = mujoco.MjSpec.from_string("\n".join(lines), include=files)

    root = spec.body(f"{side}_palm_link")
    if root is None:
        raise SystemExit(f"hand model has no '{side}_palm_link'")

    # The asset ships without a joint at its root, which is what makes it usable
    # on an arm. For this rig the root has to be jointed, otherwise MuJoCo treats
    # it as a fixed body whose pose comes from body_pos and the weld below has
    # nothing to drive. A free joint plus the weld is the standard mocap rig.
    root.pos = list(PARK)
    root.quat = list(CONTACT_QUAT)
    root.add_freejoint()

    # Mocap body at the grasp start, weld holding the palm to it. MuJoCo has no
    # dedicated weld helper in the Python API, so the equality is added by type.
    mocap = spec.worldbody.add_body()
    mocap.name = f"{side}_hand_mocap"
    mocap.mocap = True
    mocap.pos = list(PARK)
    mocap.quat = list(CONTACT_QUAT)
    weld = spec.add_equality(
        name=f"{side}_hand_weld",
        type=mujoco.mjtEq.mjEQ_WELD,
        name1=mocap.name,
        name2=root.name,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        active=True,
    )
    # The default softness lets the hand lag the mocap by centimetres while it is
    # being dragged. These are near-rigid values: with a 2 ms timestep the weld
    # error stays at float precision, so the hand follows its target.
    weld.solref = np.array([0.002, 1.0])
    weld.solimp = np.array([0.999, 0.9999, 0.00001, 0.5, 2.0])

    # ``meshdir="."`` is resolved against the process working directory, not
    # against where the model lives, so compile from mjcf/ and the script runs
    # from anywhere.
    previous = Path.cwd()
    os.chdir(MJCF_DIR)
    try:
        return spec.compile()
    finally:
        os.chdir(previous)



def _thumb_mask(model):
    """Boolean mask selecting the thumb's actuators."""
    mask = np.zeros(model.nu, dtype=bool)
    for index in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        if "thumb" in name:
            mask[index] = True
    return mask


def _open_targets(model, current):
    """Targets with the thumb's base joint swung out to THUMB_OPEN_DEG.

    Only j0 moves: it is the joint that carries the thumb round the cylinder.
    """
    targets = np.asarray(current, dtype=float).copy()
    for index in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        if "thumb" not in name or not name.endswith("_j0"):
            continue
        joint = model.actuator_trnid[index, 0]
        low, high = model.jnt_range[joint]
        targets[index] = float(np.clip(np.deg2rad(THUMB_OPEN_DEG), low, high))
    return targets


def _close_targets(model, current, fraction):
    """Targets that close the hand around the cylinder.

    j0 stays put on both thumb and fingers -- the knuckle at the palm would swing
    the whole digit outwards and push the cylinder away. Everything else curls to
    ``fraction`` of its range, and the mimicked joints follow by equality.
    """
    targets = np.asarray(current, dtype=float).copy()
    for index in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        if name.endswith("_j0") or ("thumb" in name and name.endswith("_j2")):
            continue
        joint = model.actuator_trnid[index, 0]
        low, high = model.jnt_range[joint]
        targets[index] = low + fraction * (high - low)
    return targets



def run(model, data):
    """Play TRAJECTORY back, one simulation step at a time.

    Yields once per step; the caller steps the model in between. Each phase
    interpolates position and joint targets from where the previous one ended, so
    the recorded grasp plays back exactly as it was tuned.
    """
    position = np.asarray(data.mocap_pos[0], dtype=float).copy()
    ctrl = np.asarray(data.ctrl, dtype=float).copy()
    for phase in TRAJECTORY:
        start_pos, start_ctrl = position.copy(), ctrl.copy()
        end_pos = np.asarray(phase["pos"], dtype=float)
        if phase.get("open"):
            end_ctrl = _open_targets(model, start_ctrl)
        elif "close" in phase:
            end_ctrl = _close_targets(model, start_ctrl, phase["close"])
        else:
            end_ctrl = start_ctrl
        for target_pos, target_ctrl in _phase_steps(
                model, position, end_pos, ctrl, end_ctrl, phase["t"]):
            data.mocap_pos[0] = target_pos
            data.ctrl[:] = target_ctrl
            yield
        position, ctrl = end_pos, end_ctrl


def _phase_steps(model, start_pos, end_pos, start_ctrl, end_ctrl, duration):
    """Yield (mocap_pos, ctrl) pairs along one phase."""
    thumb = _thumb_mask(model)
    for alpha in _ramp(duration, model.opt.timestep):
        # The thumb finishes at THUMB_LEAD of the phase, so it meets the cylinder
        # slightly before the fingers close in from the other side.
        progress = np.where(thumb, np.minimum(alpha / THUMB_LEAD, 1.0), alpha)
        yield (start_pos + alpha * (end_pos - start_pos),
               start_ctrl + progress * (end_ctrl - start_ctrl))

def _ramp(duration, timestep):
    """Yield a smooth 0..1 fraction over ``duration`` seconds."""
    steps = max(1, int(duration / timestep))
    for step in range(steps + 1):
        t = step / steps
        yield t * t * (3.0 - 2.0 * t)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--collision", choices=tuple(COLLISION_VARIANTS), default="mesh",
        help="collision geometry to load (default: mesh)",
    )
    parser.add_argument(
        "--tactiles", action="store_true",
        help="load the variant carrying the 619 tactile spheres and show their "
             "contact forces",
    )
    parser.add_argument(
        "--show-collision", action="store_true",
        help="also draw the collision geoms, hidden by default",
    )
    args = parser.parse_args()

    model = build_model(args.side, args.collision, args.tactiles)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"{'side':10s}: {args.side}")
    print(f"{'collision':10s}: {args.collision}")
    print(f"{'tactiles':10s}: {'on' if args.tactiles else 'off'}")
    print(f"nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
          f"nu={model.nu} nmocap={model.nmocap} neq={model.neq}")

    # Built after assembly: it collects the tactile spheres by name, so it has to
    # see the final model.
    tactiles = RobotTactiles(model) if args.tactiles else None
    panel = None
    if tactiles is not None:
        from tactile_viewer import TactilePanel
        panel = TactilePanel(args.side)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for group in COLLISION_GROUPS:
            viewer.opt.geomgroup[group] = int(args.show_collision)

        # Render at FRAME_HZ while stepping physics at the model's own 2 ms. One
        # sleep per physics step would overrun 2 ms every time -- step plus sync
        # plus scheduler granularity -- and the sequence would fall further
        # behind real time as it went; batching fixes the drift.
        steps_per_frame = max(1, round((1.0 / FRAME_HZ) / model.opt.timestep))
        script = run(model, data)
        done = False
        next_frame = time.perf_counter()
        while viewer.is_running():
            for _ in range(steps_per_frame):
                if not done:
                    try:
                        next(script)
                    except StopIteration:
                        done = True
                mujoco.mj_step(model, data)
            if tactiles is not None:
                # Read the contact forces after the physics step and turn them
                # into per-point readings the way the sensor would.
                tactiles.get_tactile_force(data)
            viewer.sync()
            if tactiles is not None:
                tactiles.visualize_tactile_force(viewer)
                panel.update(tactiles)
            next_frame += steps_per_frame * model.opt.timestep
            remaining = next_frame - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_frame = time.perf_counter()


if __name__ == "__main__":
    main()
