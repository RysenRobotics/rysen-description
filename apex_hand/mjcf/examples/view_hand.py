#!/usr/bin/env python3
"""Load the hand on a ground plane in the MuJoCo viewer and drag the fingers.

Usage:
    python3 examples/view_hand.py                                       # right, mesh
    python3 examples/view_hand.py --side left
    python3 examples/view_hand.py --side both
    python3 examples/view_hand.py --collision convex_decomp
    python3 examples/view_hand.py --tactiles
    python3 examples/view_hand.py --collision convex_decomp --tactiles --show-collision

Each hand is fixed to the world at the palm; only the finger joints move.
Double-click a joint in the viewer to drag it.

Dependencies: mujoco (``pip install mujoco``).
"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

EXAMPLES_DIR = Path(__file__).resolve().parent
MJCF_DIR = EXAMPLES_DIR.parent

# Palm height above the ground, chosen so the fingers clear the floor.
MOUNT_HEIGHT = 0.25
# Half the gap between the two palms in the two-hand scene.
BOTH_HAND_SPREAD = 0.12

# Collision variant -> filename suffix produced by tools/urdf_to_mjcf.py.
COLLISION_VARIANTS = {"mesh": "", "convex_decomp": "_convex_decomp"}
# Geom groups assigned by the generator: group 3 is the rigid collision, group 4
# the tactile pads' collision. Neither is drawn unless asked for.
COLLISION_GROUPS = (3, 4)
# Physics steps batched into one displayed frame. The model runs at 2 ms, which
# does not divide a 60 Hz frame, so the choice is expressed in steps rather than
# in Hz: 8 steps is 62.5 Hz, the closest achievable to 60.
STEPS_PER_FRAME = 8


def compile_spec(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile a spec built from memory, from the directory its meshes live in.

    ``<compiler meshdir=".">`` is resolved against the *process* working
    directory, not against the directory of the model or against the key an
    ``<include>`` was registered under. Compiling a ``from_string`` spec from
    ``examples/`` therefore looks for ``apex_hand_right/meshes/...`` under
    ``examples/`` and fails with "Error opening file". Chdir to ``mjcf/`` for
    the duration of the compile, so the script runs from anywhere.
    """
    previous = Path.cwd()
    os.chdir(MJCF_DIR)
    try:
        return spec.compile()
    finally:
        os.chdir(previous)


def build_model(side: str, collision: str, tactiles: bool) -> mujoco.MjModel:
    """Place the requested hands on the ground plane and pose them."""
    sides = ("left", "right") if side == "both" else (side,)
    suffix = COLLISION_VARIANTS[collision] + ("_tactile" if tactiles else "")

    # The scene is assembled here rather than kept as a dozen near-identical
    # files. <include> is used instead of MjSpec.attach() because attach drops the
    # child's <option>: the hand's timestep, integrator and cone would silently
    # be replaced by the ground's.
    #
    # The text lives in memory, so <include> is resolved from a mapping of the
    # paths the scene names onto the file contents. The key is the path as the
    # scene spells it; the value is where that file actually is.
    includes = {"../scene/ground.xml": EXAMPLES_DIR / "scene" / "ground.xml"}
    for name in sides:
        key = f"apex_hand_{name}/apex_hand_{name}{suffix}.xml"
        includes[key] = MJCF_DIR / key
    files = {}
    for key, path in includes.items():
        if not path.exists():
            raise SystemExit(f"missing model: {path}")
        files[key] = path.read_bytes()

    lines = ["<mujoco>"]
    lines += [f'<include file="{key}"/>' for key in includes]
    lines.append("</mujoco>")

    spec = mujoco.MjSpec.from_string("\n".join(lines), include=files)
    for name in sides:
        body = spec.body(f"{name}_palm_link")
        if body is None:
            raise SystemExit(f"hand model has no '{name}_palm_link'")
        y = BOTH_HAND_SPREAD if name == "right" else -BOTH_HAND_SPREAD
        body.pos = [0.0, y if len(sides) > 1 else 0.0, MOUNT_HEIGHT]
    return compile_spec(spec)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument(
        "--collision", choices=tuple(COLLISION_VARIANTS), default="mesh",
        help="collision geometry to load (default: mesh)",
    )
    parser.add_argument(
        "--tactiles", action="store_true",
        help="load the variant carrying the 619 tactile spheres, off by default",
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
    print(
        f"nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
        f"nsite={model.nsite} nu={model.nu}"
    )
    print(f"contacts at start: {data.ncon}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for group in COLLISION_GROUPS:
            viewer.opt.geomgroup[group] = int(args.show_collision)

        # Sleep once per frame, not per physics step: sleeping after every 2 ms
        # step would also include the loop's own overhead, so the simulation
        # falls further behind real time on every step. Batching keeps it in step.
        next_frame = time.perf_counter()
        while viewer.is_running():
            for _ in range(STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
            viewer.sync()
            next_frame += STEPS_PER_FRAME * model.opt.timestep
            remaining = next_frame - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_frame = time.perf_counter()


if __name__ == "__main__":
    main()
