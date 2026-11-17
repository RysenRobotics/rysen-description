"""Tunable constants shared by the MJCF writer.

Everything here is a value the generator has no URDF source for: timestep,
actuator gains, geom groups and contact classes. Kept in one place so a change
is obvious and nothing has to be threaded through the writer by hand.

The physical and geometric values live in ``tools/config.yaml`` and are loaded
over the defaults below. The defaults are kept working, so the generator runs
with no config file; YAML only has to list the keys it wants to change.
Structure -- variant names, geom groups, class names, which links are complex --
stays here on purpose: those are contracts with the writer and the generated
XML, not knobs.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# variant -> (filename suffix, collision style, emit tactile points).
# ``tactile`` in the name means the 619 tactile spheres are emitted under each
# pad body; the collision style is chosen independently, so all four
# combinations exist.
VARIANTS = {
    "mesh": ("", "mesh", False),
    "convex": ("_convex_decomp", "convex", False),
    "mesh_tactile": ("_tactile", "mesh", True),
    "convex_tactile": ("_convex_decomp_tactile", "convex", True),
}

# Defaults for every value that can be tuned from config.yaml. The rationale for
# each lives beside it in that file; these are only the fallbacks that keep the
# generator running when no config is present.
TIMESTEP = 2e-3
JOINT_ARMATURE = 1e-4
JOINT_DAMPING = 0.05
JOINT_FRICTIONLOSS = 1e-3
DEFAULT_KP = 20.0
DEFAULT_KV = 0.17
# Geometry and decomposition values, mirrored by tools/config.yaml. They have to
# exist here too: the geometry helpers read them off this module, and the
# generator is supposed to run with no config file present.
DEFAULT_RADIUS = 0.001
SURFACE_INSET = 0.0015
INSIDE_STEP = 0.001
FACES_PER_HULL = 160
MIN_HULLS = 12
MAX_HULLS = 128
COMPLEX_MAX_HULLS = 128
SOLID_PITCH = 0.0008
VOLUME_ERROR_PERCENT = 0.25
RESOLUTION = 100000
# Structural: which joints are on a return spring. The torque and damping are
# tunable.
SPRING_RETURN_JOINTS = ("thumb_j3", "index_j2", "middle_j2", "ring_j2", "pinky_j2")
SPRING_RETURN_TORQUE = 0.05
SPRING_RETURN_TORQUE_OPEN = 0.02
SPRING_RETURN_DAMPING = 0.0015


def is_spring_returned(joint_name):
    """True for a joint whose opening direction is driven by a spring only.

    The name is matched without the hand prefix, so ``right_thumb_j3`` and
    ``left_index_j2`` both hit the rule.
    """
    return joint_name.split("_", 1)[-1] in SPRING_RETURN_JOINTS


def spring_stiffness(lower, upper):
    """Torsional stiffness spanning ``SPRING_RETURN_TORQUE_OPEN`` to
    ``SPRING_RETURN_TORQUE`` over the joint's travel.

    Derived per joint rather than fixed, so every digit develops the same torque
    at its own limits even though the thumb's travel differs from the fingers'.
    """
    travel = float(upper) - float(lower)
    if travel <= 0:
        return 0.0
    return (SPRING_RETURN_TORQUE - SPRING_RETURN_TORQUE_OPEN) / travel


def spring_reference(lower, stiffness):
    """Angle at which the spring is unstretched, given its preload.

    Below the open limit, so the spring still pushes at the stop: the joint
    rests against it instead of on a torque that has faded to zero.
    """
    if stiffness <= 0:
        return float(lower)
    return float(lower) - SPRING_RETURN_TORQUE_OPEN / stiffness


COUPLING_SOLREF = "0.002 1"
COUPLING_SOLIMP = "0.999 0.9999 1e-05"

# Tip and pad links only define reference frames: they are emitted as <site>
# on their parent body instead of massless <body>, so their placeholder mass
# and inertia never reach the model.
#
# Sites get their own group. The viewer toggles sites through opt.sitegroup,
# which is independent of opt.geomgroup, so sharing a number with a geom group
# does not make them hide together -- it only makes the two lists inconsistent.
SITE_GROUP = 5
SITE_SIZE = 0.003

# Geom groups, so the viewer can toggle shells and tactile pads independently.
# MuJoCo shows/hides by group (0-5); assigning a group does not affect collision.
VISUAL_GROUP_LINK = 1
VISUAL_GROUP_TACTILE = 2
COLLISION_GROUP_LINK = 3
COLLISION_GROUP_TACTILE = 4
# Tactile spheres live in group 1 as well: they are a fine-grained view of the
# shell, and group 5 is the last usable one, kept free.
TACTILE_GROUP = 1
TACTILE_CLASS = "tactile_point"

# Contact classes. MuJoCo has no name-based default matching, so every geom
# names its class explicitly. The tables themselves are tunable from
# config.yaml under `contact`; see there for what each value does.
VISUAL_CLASS = "visual"
HARD_LINK_CLASS = "hard_link"
SOFT_TACTILE_CLASS = "soft_tactile"
SOFT_VISUAL_CLASS = "soft_visual"
SOFT_VISUAL_RGBA = "0.6 0.75 0.85 0.35"
HARD_LINK_CONTACT = dict(
    contype="1",
    conaffinity="1",
    friction="1.0 0.005 0.0001",
    solref="0.02 1",
    solimp="0.9 0.95 0.001 0.5 2",
)
SOFT_TACTILE_CONTACT = dict(
    contype="1",
    conaffinity="1",
    friction="1.6 0.01 0.001",
    solref="0.04 1",
    solimp="0.7 0.9 0.005 0.5 2",
)
# Tactile spheres reuse the rigid contact parameters for now: they sense through
# contact, so they need to be collidable, and a stub of rubber-like softness on
# an 1 mm sphere would only make the reading mushy. The red tint is purely for
# viewing -- the spheres are buried in the pad otherwise.
TACTILE_POINT_CONTACT = dict(HARD_LINK_CONTACT)
TACTILE_POINT_RGBA = "1 0 0 1"


def class_name(robot_name, suffix):
    """Prefix a geom class with the model name.

    MuJoCo rejects duplicate default class names across includes, so a scene
    that loads both hands cannot use the same class names twice.
    """
    return f"{robot_name}_{suffix}"


def model_name(robot_name, variant):
    return robot_name if variant == "mesh" else f"{robot_name}_{variant}"


# config.yaml key -> module constant. Only tunable values appear here; anything
# structural is absent on purpose, so a stray line in the config is reported as
# an unknown key instead of being silently ignored.
_TUNABLE = {
    ("simulation", "timestep"): "TIMESTEP",
    ("joint", "armature"): "JOINT_ARMATURE",
    ("joint", "damping"): "JOINT_DAMPING",
    ("joint", "frictionloss"): "JOINT_FRICTIONLOSS",
    ("actuator", "kp"): "DEFAULT_KP",
    ("actuator", "kv"): "DEFAULT_KV",
    ("spring_return", "torque_at_limit"): "SPRING_RETURN_TORQUE",
    ("spring_return", "torque_at_open"): "SPRING_RETURN_TORQUE_OPEN",
    ("spring_return", "damping"): "SPRING_RETURN_DAMPING",
    ("coupling", "solref"): "COUPLING_SOLREF",
    ("coupling", "solimp"): "COUPLING_SOLIMP",
    ("visual", "soft_visual_rgba"): "SOFT_VISUAL_RGBA",
    ("visual", "tactile_point_rgba"): "TACTILE_POINT_RGBA",
    ("tactile", "radius"): "DEFAULT_RADIUS",
    ("tactile", "surface_inset"): "SURFACE_INSET",
    ("tactile", "inside_step"): "INSIDE_STEP",
    ("convex", "faces_per_hull"): "FACES_PER_HULL",
    ("convex", "min_hulls"): "MIN_HULLS",
    ("convex", "max_hulls"): "MAX_HULLS",
    ("convex", "palm_max_hulls"): "COMPLEX_MAX_HULLS",
    ("convex", "solid_pitch"): "SOLID_PITCH",
    ("convex", "volume_error_percent"): "VOLUME_ERROR_PERCENT",
    ("convex", "resolution"): "RESOLUTION",
}
# Contact tables are merged rather than replaced, so a config that only
# overrides friction keeps the other four keys.
_CONTACT_TABLES = {
    "hard_link": "HARD_LINK_CONTACT",
    "soft_tactile": "SOFT_TACTILE_CONTACT",
}


def _apply_config(path=CONFIG_PATH):
    """Overlay ``config.yaml`` onto the defaults defined above.

    A missing file is fine -- the defaults are complete and the generator must
    run without one. Unknown keys are an error: they are almost always a typo,
    and ignoring one would leave the generator using a value its author thought
    they had changed.
    """
    if not path.exists():
        return
    document = yaml.safe_load(path.read_text()) or {}

    for section, values in document.items():
        if not isinstance(values, dict):
            raise ValueError(f"{path.name}: '{section}' must be a mapping")
        for key, value in values.items():
            if section == "contact":
                continue
            constant = _TUNABLE.get((section, key))
            if constant is None:
                raise ValueError(f"{path.name}: unknown key '{section}.{key}'")
            globals()[constant] = value

    for class_suffix, values in (document.get("contact") or {}).items():
        table = _CONTACT_TABLES.get(class_suffix)
        if table is None:
            raise ValueError(f"{path.name}: unknown contact class "
                             f"'contact.{class_suffix}'")
        unknown = set(values) - set(globals()[table])
        if unknown:
            raise ValueError(f"{path.name}: unknown key(s) "
                             f"{sorted(unknown)} in 'contact.{class_suffix}'")
        globals()[table].update(values)


_apply_config()
# The tactile spheres reuse the rigid table, so re-point it after the overlay
# and let the config override it separately if it ever needs to differ.
TACTILE_POINT_CONTACT.update(HARD_LINK_CONTACT)
