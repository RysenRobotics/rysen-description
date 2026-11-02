"""Render individual MJCF tags.

Pure text out, plain data in: nothing here walks the tree or touches the files
system, which keeps the assembly logic in ``mjcf_builder`` readable.
"""

from pathlib import Path

import mujoco
import numpy as np

import mjcf_config as cfg


def fmt(value):
    text = f"{float(value):.10g}"
    return "0" if text in ("-0", "0") else text


def vec(values):
    return " ".join(fmt(v) for v in values)


def rpy_to_quat(rpy):
    """URDF rpy (rad) -> MuJoCo wxyz quaternion.

    This URDF stores its rotations in **zyx** order, i.e. the three values are
    applied about z, then y, then x. MuJoCo's euler order string names the axis
    of each component positionally, so the values must be reversed before being
    handed over as ``"zyx"``.

    Reading the same data as ``"xyz"`` happens to agree for the finger joints
    (their rpy has a single non-zero component) but folds the right thumb and
    the whole left hand back into the palm.
    """
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray(rpy, dtype=float)[::-1], "zyx")
    return vec(quat)


def mass_tag(link_name, link, indent):
    """Emit <inertial>, or nothing for a frame link or a missing inertia."""
    if link["inertia"] is None or link_name.endswith(("_tip", "_pad")):
        return ""
    ixx, iyy, izz, ixy, ixz, iyz = link["inertia"]
    if ixy or ixz or iyz:
        inertia = f'fullinertia="{vec((ixx, iyy, izz, ixy, ixz, iyz))}"'
    else:
        inertia = f'diaginertia="{vec((ixx, iyy, izz))}"'
    return (
        f'{"  " * indent}<inertial pos="{vec(link["com"])}" '
        f'mass="{fmt(link["mass"])}" {inertia}/>\n'
    )


def joint_tags(joint, indent):
    """Emit the <body> opener plus its hinge joint and actuator.

    Returns ``(body, joint, actuator)``; ``joint`` and ``actuator`` are empty
    for a fixed joint, and ``actuator`` is also empty for a mimicked joint.
    """
    pad = "  " * indent
    joint_tag = ""
    actuator_tag = ""
    if joint["type"] != "fixed":
        limit = joint["limit"]
        lower, upper = float(limit.get("lower", 0.0)), float(limit.get("upper", 0.0))
        effort = float(limit.get("effort", 1.0))
        low_effort = cfg.spring_return_limit(joint["name"], effort)
        joint_tag = (
            f'{pad}  <joint name="{joint["name"]}" type="hinge" axis="{vec(joint["axis"])}" '
            f'range="{fmt(lower)} {fmt(upper)}" armature="{fmt(cfg.JOINT_ARMATURE)}" '
            f'damping="{fmt(cfg.JOINT_DAMPING)}" frictionloss="{fmt(cfg.JOINT_FRICTIONLOSS)}"/>\n'
        )
        # A mimicked joint is driven through its equality, not by its own
        # actuator: giving both ends an actuator makes the two position servos
        # fight each other across the coupling, so the joint pair lags and the
        # equality settles at neither target.
        if joint["mimic"] is None:
            actuator_tag = (
                f'    <position name="act_{joint["name"]}" joint="{joint["name"]}" '
                f'kp="{fmt(cfg.DEFAULT_KP)}" kv="{fmt(cfg.DEFAULT_KV)}" '
                f'ctrllimited="true" ctrlrange="{fmt(lower)} {fmt(upper)}" '
                f'forcelimited="true" forcerange="-{fmt(low_effort)} {fmt(effort)}"/>\n'
            )
    body_tag = (
        f'{pad}<body name="{joint["child"]}" pos="{vec(joint["xyz"])}" '
        f'quat="{rpy_to_quat(joint["rpy"])}">\n'
    )
    return body_tag, joint_tag, actuator_tag


def site_tag(joint, indent):
    """Emit a fixed frame link as a <site> on its parent body."""
    pad = "  " * indent
    return (
        f'{pad}<site name="{joint["child"]}" pos="{vec(joint["xyz"])}" '
        f'quat="{rpy_to_quat(joint["rpy"])}" size="{fmt(cfg.SITE_SIZE)}" '
        f'group="{cfg.SITE_GROUP}"/>\n'
    )


def equality_tags(joints):
    """Emit <joint> equalities for every URDF <mimic>.

    URDF mimic is ``q_child = multiplier * q_parent + offset``; MJCF records the
    same relation as ``q_joint1 = a0 + a1*q_joint2 + ...``, so the constant term
    goes to a0 and the multiplier to a1.

    MuJoCo has no binary "hard constraint" switch; equality is always a soft
    constraint and ``solref``/``solimp`` decide how soft. The values below drive
    the residual down to float precision, which is why they are stiff rather
    than the defaults: a soft coupling lets the mimicked joint lag its driver by
    ~1e-3 rad, visible as the fingertip trailing the linkage.
    """
    tags = []
    for joint in joints:
        mimic = joint["mimic"]
        if mimic is None:
            continue
        multiplier = float(mimic.get("multiplier", 1.0))
        offset = float(mimic.get("offset", 0.0))
        tags.append(
            f'    <joint joint1="{joint["name"]}" joint2="{mimic["joint"]}" '
            f'polycoef="{fmt(offset)} {fmt(multiplier)} 0 0 0" '
            f'solref="{cfg.COUPLING_SOLREF}" solimp="{cfg.COUPLING_SOLIMP}"/>\n'
        )
    return tags


def default_section(robot_name, with_tactile):
    """Emit the geom classes referenced by the visual and collision geoms.

    contype/conaffinity are spelled out in every class rather than set once on
    the top-level <default>: a value there is inherited by all three classes,
    so setting the visual pair at top level silently disables collision too.
    Stating them per class keeps each one self-contained.

    The tactile-only classes (translucent visual shell, contact spheres) are
    declared only for the tactile variants, so the plain ones carry no unused
    class.
    """
    hard = " ".join(f'{k}="{v}"' for k, v in cfg.HARD_LINK_CONTACT.items())
    soft = " ".join(f'{k}="{v}"' for k, v in cfg.SOFT_TACTILE_CONTACT.items())
    point = " ".join(f'{k}="{v}"' for k, v in cfg.TACTILE_POINT_CONTACT.items())
    tactile_classes = (
        f'    <default class="{cfg.class_name(robot_name, cfg.SOFT_VISUAL_CLASS)}">\n'
        f'      <geom contype="0" conaffinity="0" rgba="{cfg.SOFT_VISUAL_RGBA}"/>\n'
        "    </default>\n"
        f'    <default class="{cfg.class_name(robot_name, cfg.TACTILE_CLASS)}">\n'
        f'      <geom condim="3" rgba="{cfg.TACTILE_POINT_RGBA}" {point}/>\n'
        "    </default>\n"
    ) if with_tactile else ""
    return (
        "  <default>\n"
        f'    <default class="{cfg.class_name(robot_name, cfg.VISUAL_CLASS)}">\n'
        '      <geom contype="0" conaffinity="0"/>\n'
        "    </default>\n"
        f'    <default class="{cfg.class_name(robot_name, cfg.HARD_LINK_CLASS)}">\n'
        f'      <geom condim="3" {hard}/>\n'
        "    </default>\n"
        f'    <default class="{cfg.class_name(robot_name, cfg.SOFT_TACTILE_CLASS)}">\n'
        f'      <geom condim="3" {soft}/>\n'
        "    </default>\n"
        + tactile_classes
        + "  </default>\n"
    )


def mesh_asset(name, path):
    return f'    <mesh name="{name}" file="{path}"/>\n'


def mesh_geom(name, group, class_suffix, indent=3, robot_name=""):
    pad = "  " * indent
    return (
        f'{pad}<geom name="{name}" type="mesh" mesh="{name}" '
        f'group="{group}" class="{cfg.class_name(robot_name, class_suffix)}"/>\n'
    )


def tactile_geoms(robot_name, points, radius):
    """Emit one sphere geom per tactile point.

    Each geom carries its point frame, whose local z axis is the outward contact
    normal (see ``tactile_points``). A sphere is symmetric, so the orientation is
    only there for downstream code: the force normal can be read off the geom's
    z axis without special-casing the palm.

    The centre is already placed under the pad surface, deeper than the radius,
    so the whole sphere stays inside the skin. MuJoCo filters contacts within a
    single body, so the spheres do not collide with the pad they sit in.
    """
    collision_class = cfg.class_name(robot_name, cfg.TACTILE_CLASS)
    out = ""
    for name, position, rpy in points:
        out += (
            f'      <geom name="{name}" type="sphere" size="{fmt(radius)}" '
            f'pos="{vec(position)}" quat="{rpy_to_quat(rpy)}" '
            f'group="{cfg.TACTILE_GROUP}" class="{collision_class}"/>\n'
        )
    return out


def mesh_file_name(path):
    """Basename of an asset as referenced from the output directory."""
    return Path(path).name
