"""Tunable constants shared by the USD writer.

Everything here is a value the generator has no URDF source for. The physical
and geometric values live in ``tools/config.yaml`` and are loaded over the
defaults below, so a key may be omitted and the built-in default applies.

Structure -- variant names, the prim layout, which links are reference frames --
stays here on purpose: those are contracts with the writer and the generated
USD, not knobs.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Links that only define a reference frame. They carry no mesh and are authored
# as plain Xform markers on their parent body, never as rigid bodies.
FRAME_LINK_SUFFIXES = ("_tip", "_pad")
# Link name fragment that marks a tactile pad (the shell the point array sits
# on) as opposed to a tactile point.
# ``*_tactile*`` are the rubber shells drawn on the phalanges. Not to be
# confused with ``*_pad`` / ``*_tip``: those are reference frames only, with no
# rigid body and no collider, and exist for kinematics.
TACTILE_TAG = "_tactile"
POINT_TAG = "_point_"

# variant -> (filename suffix, emit tactile bodies, add the six-joint mount).
# Four variants per hand, from two independent choices: whether the 619 tactile
# bodies are emitted, and whether the root gets a driveable mount. All four
# carry the same collision geometry; how it is cooked depends on the link (see
# ``approximation_for``).
#
# Only the mount is a suffix: an unadorned file is the bare hand, the thing you
# attach to a flange or a teleoperation rig. `_mount` marks the self-holding
# variant a standalone scene needs.
VARIANTS = {
    "mesh": ("", False, False),
    "mesh_tactile": ("_tactile", True, False),
    "mesh_mount": ("_mount", False, True),
    "mesh_mount_tactile": ("_mount_tactile", True, True),
}

# ``physics:approximation`` token for the rigid links. Resolved from config.yaml
# at load time and patched in by ``_apply_config``.
MESH_STYLE_APPROXIMATION = "convexDecomposition"
# The tactile shells are 0.3 mm open surfaces. Convex decomposition needs a
# closed manifold to voxelise, so these cook as a single convex hull; the rigid
# links are decomposed into several closed hulls instead.
TACTILE_APPROXIMATION = "convexHull"

# A standalone scene has to hold the hand somewhere, and writing the root pose
# every step teleports it -- the colliders jump through whatever they are
# touching and the solver has no velocity to work with. The ``_mount`` variants
# therefore give the root a six-joint mount (three prismatic, three revolute)
# that the examples drive with position targets, like every other joint. The
# bare variants omit it, since an arm or teleoperation rig supplies its own
# root joint and a second chain would be in the way.
MOUNT_CHAIN = (
    ("tx", "prismatic", "X"),
    ("ty", "prismatic", "Y"),
    ("tz", "prismatic", "Z"),
    ("rx", "revolute", "X"),
    ("ry", "revolute", "Y"),
    ("rz", "revolute", "Z"),
)
MOUNT_SCOPE = "mount"
MOUNT_TRAVEL = 2.0
# Generous by design: a 180 deg limit sits exactly on the rotation a wrap-around
# grasp asks for, and PhysX parks the joint against it instead of tracking.
MOUNT_SWING_DEG = 720.0
# Links this light make the articulation's mass matrix ill conditioned: at
# 1e-6 kg the solver diverged to NaN, and at 1e-3 kg the six stages could not
# track a moving target. They are ballast, not structure, but they still have to
# be heavy enough for the solver to treat the chain as a chain.
MOUNT_BASE_MASS = 1e-1
MOUNT_LINK_MASS = 5e-2
# Stiff enough that the mount holds the hand where the script asks, soft enough
# that a contact can still push it aside instead of being overrun.
MOUNT_LINEAR_KP = 1e4
MOUNT_LINEAR_KV = 1e2
MOUNT_ANGULAR_KP = 1e3
MOUNT_ANGULAR_KV = 1e1

# Defaults for every value that can be tuned from config.yaml.
TIMESTEP = 2e-3
JOINT_ARMATURE = 1e-4
JOINT_FRICTION = 1e-3
DEFAULT_KP = 20.0
DEFAULT_KV = 0.17
CONTACT_OFFSET = 1e-3
REST_OFFSET = 0.0
# SDF cooking, used only when approximation is explicitly switched back to
# "sdf": how far the field is fattened, how thick the accurate band is, and the
# per-axis voxel count (256 puts a 50 mm phalanx at 0.2 mm).
SDF_MARGIN = 5e-4
SDF_NARROW_BAND = 2e-3
SDF_RESOLUTION = 256
# Convex decomposition cooking. Measured on the cooked hulls (see config.yaml):
# 32 hulls is the knee for the phalanges, 5% error beats 10% slightly, and
# shrinkWrap does most of the work by projecting hull vertices onto the source
# surface. The palm is ~4x the faces of a phalanx and has a U-shaped cavity: at
# 32 hulls its worst over-coverage grows from 8.1 mm to 12.7 mm, so it gets a
# separate, larger budget.
CONVEX_MAX_HULLS = 32
PALM_MAX_HULLS = 64
CONVEX_ERROR_PERCENTAGE = 5.0
CONVEX_SHRINK_WRAP = True
SELF_COLLISION = False
SOLVER_POSITION_ITERATIONS = 8
SOLVER_VELOCITY_ITERATIONS = 1
# Both are compliant spring-damper contacts, so every hand contact is soft.
# k = E*A/L and c = 2*zeta*sqrt(k*m_eff); see config.yaml for the estimates.
# Hard plastic is ~20x stiffer than the silicone shells.
HARD_LINK_MATERIAL = {
    "static_friction": 1.2,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
    "compliant_contact_stiffness": 2.0e5,
    "compliant_contact_damping": 9.0e1,
}
SOFT_TACTILE_MATERIAL = {
    "static_friction": 1.8,
    "dynamic_friction": 1.6,
    "restitution": 0.0,
    "compliant_contact_stiffness": 1.0e4,
    "compliant_contact_damping": 2.5e1,
}
# The tactile spheres are separate rigid bodies buried in the shells. They are
# stiffer than the silicone around them (a semi-rigid sensing element, E ~
# 30 MPa) but far softer than the plastic links. k = E*A/L with A = pi*r^2 and
# L ~ 2r; c = 2*zeta*sqrt(k*m_eff) with m_eff ~ 1e-3 kg. Giving them their own
# material lets the array compress differently from the pad surface it reports.
TACTILE_POINT_MATERIAL = {
    "static_friction": 1.8,
    "dynamic_friction": 1.6,
    "restitution": 0.0,
    "compliant_contact_stiffness": 5.0e4,
    "compliant_contact_damping": 1.2e1,
}
SHELL_RGB = (0.612, 0.659, 0.671)
# The tactile variant draws the pad shells translucent so the array buried under
# them shows through. Tinted like the MJCF model's soft_visual, which exists for
# the same reason.
TACTILE_SHELL_RGBA = (0.6, 0.75, 0.85, 0.35)
# The array itself is drawn opaque red, so through the translucent pad it reads
# as a grid of points. Same tint as the MJCF model's tactile_point.
TACTILE_POINT_RGB = (1.0, 0.0, 0.0)
TACTILE_RADIUS = 1e-3
TACTILE_SURFACE_INSET = 1e-3
TACTILE_INSIDE_STEP = 1e-3
TACTILE_STRIDE = 1
TACTILE_MASS = 1e-5

# Primitives that carry the same name as the MJCF tags they replace, so a reader
# moving between the two formats has one vocabulary.
JOINTS_SCOPE = "Physics"
# Reference targets for the visual and collision placements. Kept inside the
# default prim so the geometry survives a reference of this file as a whole.
GEOMETRY_SCOPE = "meshes"
VISUAL_SCOPE = "visual"
COLLISION_SCOPE = "collision"
FRAMES_SCOPE = "frames"
TACTILE_SCOPE = "tactile"
MATERIALS_SCOPE = "Materials"
SHAPE_PRIM = "shape"

MATERIAL_HARD_LINK = "hard_link"
MATERIAL_SOFT_TACTILE = "soft_tactile"
# Render-only: carries no physics, the spheres keep the soft contact material.
MATERIAL_TACTILE_POINT = "tactile_point"


def variant_suffix(variant):
    return VARIANTS[variant][0]


def material_name(link_name):
    """Physics material a link's collision geometry binds to.

    Tactile pads are rubber, everything else is rigid plastic/metal. Pads are
    the ``*_tactile*`` links; the palm and the phalanges are not.
    """
    return MATERIAL_SOFT_TACTILE if TACTILE_TAG in link_name else MATERIAL_HARD_LINK


def is_tactile_shell(name):
    return TACTILE_TAG in name and POINT_TAG not in name


def approximation_for(link_name):
    """How PhysX should cook one link's collision mesh."""
    return TACTILE_APPROXIMATION if is_tactile_shell(link_name) else MESH_STYLE_APPROXIMATION


def max_convex_hulls_for(link_name):
    """Hull budget for one link; the palm needs more than a phalanx."""
    return PALM_MAX_HULLS if link_name.endswith("_palm_link") else CONVEX_MAX_HULLS


# config.yaml key -> module constant. Only tunable values appear here; anything
# structural is absent on purpose, so a stray line is reported as an unknown key
# instead of being silently ignored.
_TUNABLE = {
    ("simulation", "dt"): "TIMESTEP",
    ("joint", "armature"): "JOINT_ARMATURE",
    ("joint", "friction"): "JOINT_FRICTION",
    ("actuator", "kp"): "DEFAULT_KP",
    ("actuator", "kv"): "DEFAULT_KV",
    ("collision", "convex_max_hulls"): "CONVEX_MAX_HULLS",
    ("collision", "palm_max_hulls"): "PALM_MAX_HULLS",
    ("collision", "convex_error_percentage"): "CONVEX_ERROR_PERCENTAGE",
    ("collision", "convex_shrink_wrap"): "CONVEX_SHRINK_WRAP",
    ("collision", "contact_offset"): "CONTACT_OFFSET",
    ("collision", "rest_offset"): "REST_OFFSET",
    ("collision", "self_collision"): "SELF_COLLISION",
    ("collision", "solver_position_iterations"): "SOLVER_POSITION_ITERATIONS",
    ("collision", "solver_velocity_iterations"): "SOLVER_VELOCITY_ITERATIONS",
    ("visual", "shell_rgb"): "SHELL_RGB",
    ("visual", "tactile_rgba"): "TACTILE_SHELL_RGBA",
    ("visual", "point_rgb"): "TACTILE_POINT_RGB",
    ("tactile", "radius"): "TACTILE_RADIUS",
    ("tactile", "surface_inset"): "TACTILE_SURFACE_INSET",
    ("tactile", "inside_step"): "TACTILE_INSIDE_STEP",
    ("tactile", "stride"): "TACTILE_STRIDE",
    ("tactile", "mass"): "TACTILE_MASS",
}
# Contact tables are merged rather than replaced, so a config that only
# overrides friction keeps the other keys.
_MATERIAL_TABLES = {
    MATERIAL_HARD_LINK: "HARD_LINK_MATERIAL",
    MATERIAL_SOFT_TACTILE: "SOFT_TACTILE_MATERIAL",
    MATERIAL_TACTILE_POINT: "TACTILE_POINT_MATERIAL",
}


def _apply_config(path=CONFIG_PATH):
    """Overlay ``config.yaml`` onto the defaults defined above.

    A missing file is fine -- the defaults are complete. Unknown keys are an
    error: they are almost always a typo, and ignoring one would leave the
    generator using a value its author thought they had changed.
    """
    global MESH_STYLE_APPROXIMATION

    if not path.exists():
        return
    document = yaml.safe_load(path.read_text()) or {}

    for section, values in document.items():
        if not isinstance(values, dict):
            raise ValueError(f"{path.name}: '{section}' must be a mapping")
        for key, value in values.items():
            if section == "material":
                continue
            if section == "collision" and key == "approximation":
                MESH_STYLE_APPROXIMATION = str(value)
                continue
            constant = _TUNABLE.get((section, key))
            if constant is None:
                raise ValueError(f"{path.name}: unknown key '{section}.{key}'")
            globals()[constant] = value

    for name, values in (document.get("material") or {}).items():
        table = _MATERIAL_TABLES.get(name)
        if table is None:
            raise ValueError(f"{path.name}: unknown material '{name}'")
        unknown = set(values) - set(globals()[table])
        if unknown:
            raise ValueError(f"{path.name}: unknown key(s) {sorted(unknown)} in 'material.{name}'")
        globals()[table].update(values)


_apply_config()
