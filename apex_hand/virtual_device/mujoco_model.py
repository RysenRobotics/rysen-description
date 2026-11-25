"""Assemble the description MJCF into the scene used by the virtual device."""

import os
from pathlib import Path

import mujoco

VIRTUAL_DEVICE_DIR = Path(__file__).resolve().parent
MJCF_DIR = VIRTUAL_DEVICE_DIR.parent / "mjcf"
EXAMPLES_DIR = MJCF_DIR / "examples"

MOUNT_HEIGHT = 0.5
BOTH_HAND_SPREAD = 0.2
# 手掌局部 +z 指向手指展开方向；单位四元数让手指朝世界 +z（上）。
PALM_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)


def build_model(side):
    """Build a hand scene with a free palm welded to a mocap body.

    The hand assets start fixed to the world because they are meant to be
    mounted on an arm. The SDK virtual device needs a dynamic palm so it can
    use the same mocap/weld rig as the hardware-equivalent scene.
    """
    if side not in ("left", "right", "dual"):
        raise ValueError(f"unsupported side: {side}")

    sides = ("left", "right") if side == "dual" else (side,)

    includes = {
        "../scene/ground.xml": EXAMPLES_DIR / "scene" / "ground.xml",
    }
    for name in sides:
        key = f"apex_hand_{name}/apex_hand_{name}_tactile.xml"
        includes[key] = MJCF_DIR / key

    files = {}
    for key, path in includes.items():
        if not path.exists():
            raise FileNotFoundError(f"missing MJCF asset: {path}")
        files[key] = path.read_bytes()

    lines = ["<mujoco>"]
    lines.extend(f'<include file="{key}"/>' for key in includes)
    lines.append("</mujoco>")

    spec = mujoco.MjSpec.from_string("\n".join(lines), include=files)
    for name in sides:
        root = spec.body(f"{name}_palm_link")
        if root is None:
            raise ValueError(f"hand model has no '{name}_palm_link'")

        y = BOTH_HAND_SPREAD if name == "left" else -BOTH_HAND_SPREAD
        position = [0.0, y if len(sides) > 1 else 0.0, MOUNT_HEIGHT]
        quaternion = list(PALM_QUATERNION_WXYZ)

        root.pos = position
        root.quat = quaternion
        freejoint = root.add_freejoint()
        freejoint.name = f"{name}_palm_freejoint"

        mocap = spec.worldbody.add_body()
        mocap.name = f"{name}_palm_mocap"
        mocap.mocap = True
        mocap.pos = position
        mocap.quat = quaternion

        spec.add_equality(
            name=f"{name}_palm_weld",
            type=mujoco.mjtEq.mjEQ_WELD,
            name1=mocap.name,
            name2=root.name,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            active=True,
            solref=[0.002, 1.0],
            solimp=[0.999, 0.9999, 0.00001, 0.5, 2.0],
        )

    # meshdir="." is resolved against the process working directory, not the
    # model file, so compile from mjcf/ and keep the caller's cwd untouched.
    previous = Path.cwd()
    os.chdir(MJCF_DIR)
    try:
        return spec.compile()
    finally:
        os.chdir(previous)
