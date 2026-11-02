"""Assemble one MJCF document from a parsed URDF.

The writer walks the link tree once, collecting assets, body text and actuator
tags, then renders the document twice: the first pass carries no exclusions so
the overlapping body pairs can be measured, the second one includes them.
"""

from pathlib import Path

import collision_filter
import convex_decomp
import mjcf_config as cfg
import mjcf_tags as tags
import tactile_points as tactile_points_loader
from urdf_loader import build_tree, is_frame_link, is_tactile_link, load_urdf

# Geoms are emitted one level deeper than the body that owns them; the body
# contents are re-indented afterwards so the caller controls the nesting.
GEOM_INDENT = 3


class Document:
    """Mutable accumulators for one document being rendered."""

    def __init__(self, urdf_path, variant, output_dir):
        self.urdf_path = urdf_path
        self.variant = variant
        self.output_dir = output_dir
        _suffix, self.collision_style, self.wants_tactile = cfg.VARIANTS[variant]
        self.robot_name, self.links, self.joints = load_urdf(urdf_path)
        self.assets = []
        self.world = []
        self.actuators = []

    def source_mesh(self, link_name):
        """Absolute path of the collision STL for a link, if it has one."""
        collision = self.links[link_name]["collision"]
        return self.urdf_path.parent / "meshes" / Path(collision).name if collision else None


def _link_geoms(doc, link_name):
    """Visual geoms, collision geoms and tactile spheres for one link."""
    link = doc.links[link_name]
    tactile = is_tactile_link(link_name)
    visual_group = cfg.VISUAL_GROUP_TACTILE if tactile else cfg.VISUAL_GROUP_LINK
    collision_group = (
        cfg.COLLISION_GROUP_TACTILE if tactile else cfg.COLLISION_GROUP_LINK)
    contact = cfg.SOFT_TACTILE_CLASS if tactile else cfg.HARD_LINK_CLASS
    # A tactile shell is drawn translucent so the contact spheres inside it show
    # through, but only in the tactile variants -- which are the ones that carry
    # the translucent class.
    shell = cfg.SOFT_VISUAL_CLASS if tactile and doc.wants_tactile else cfg.VISUAL_CLASS

    out = ""
    if link["visual"]:
        name = f"{link_name}_visual"
        doc.assets.append(tags.mesh_asset(
            name, f"meshes/{tags.mesh_file_name(link['visual'])}"))
        out += tags.mesh_geom(name, visual_group, shell,
                              GEOM_INDENT, doc.robot_name)

    if link["collision"]:
        # The tactile pads keep their original mesh in every variant. They are
        # thin curved plates whose hulls would balloon into the surrounding
        # geometry, and the pad surface is exactly what contact sensing depends
        # on -- decomposing it would move the contact patch away from the pad.
        if doc.collision_style == "mesh" or tactile:
            name = f"{link_name}_collision"
            doc.assets.append(tags.mesh_asset(
                name, f"meshes/{tags.mesh_file_name(link['collision'])}"))
            out += tags.mesh_geom(name, collision_group, contact,
                                  GEOM_INDENT, doc.robot_name)
        else:
            # convex: one geom per hull, meshes cached under meshes/convex/.
            hulls = convex_decomp.ensure_hulls(
                doc.source_mesh(link_name), output_dir=doc.output_dir)
            for index, hull in enumerate(hulls):
                name = f"{link_name}_collision_{index}"
                doc.assets.append(tags.mesh_asset(
                    name, hull.relative_to(doc.output_dir).as_posix()))
                out += tags.mesh_geom(name, collision_group, contact,
                                      GEOM_INDENT, doc.robot_name)
    return out


def _reindent(body_text, from_indent, to_indent):
    """Shift every line of a rendered block from one indent level to another."""
    trim = "  " * from_indent
    pad = "  " * to_indent
    lines = [line[len(trim):] if line.startswith(trim) else line
             for line in body_text.splitlines()]
    return "".join(pad + line + "\n" for line in lines)


def _body_contents(doc, link_name, indent, tactile_points, tactile_radius):
    """Inertial tag plus geoms, indented for a body's contents."""
    geoms = _link_geoms(doc, link_name)
    if doc.wants_tactile and tactile_points and link_name in tactile_points:
        geoms += tags.tactile_geoms(
            doc.robot_name, tactile_points[link_name], tactile_radius)
    geoms = _reindent(geoms, GEOM_INDENT, indent)
    return tags.mass_tag(link_name, doc.links[link_name], indent) + geoms


def _load_tactile(doc):
    """Return ``({body: points}, radius)`` for the tactile variant."""
    if not doc.wants_tactile:
        return {}, 0.0
    tactile_urdf = doc.urdf_path.with_name(f"{doc.urdf_path.stem}_tactile.urdf")
    if not tactile_urdf.exists():
        raise SystemExit(f"missing tactile URDF: {tactile_urdf}")
    return (tactile_points_loader.load_points(tactile_urdf),
            tactile_points_loader.DEFAULT_RADIUS)


def build(urdf_path, variant, output_dir):
    """Render the MJCF text for one collision ``variant``."""
    if variant not in cfg.VARIANTS:
        raise ValueError(f"unknown variant {variant}")

    doc = Document(urdf_path, variant, output_dir)
    root_name, children = build_tree(doc.links, doc.joints)
    tactile_points, tactile_radius = _load_tactile(doc)

    def walk(joint, indent):
        if is_frame_link(joint["child"]):
            if children.get(joint["child"]):
                raise ValueError(f"frame link {joint['child']} has children")
            doc.world.append(tags.site_tag(joint, indent))
            return
        body_tag, joint_tag, actuator_tag = tags.joint_tags(joint, indent)
        doc.world.append(body_tag + joint_tag)
        doc.actuators.append(actuator_tag)
        link_name = joint["child"]
        doc.world.append(_body_contents(
            doc, link_name, indent + 1, tactile_points, tactile_radius))
        for child_joint in children.get(link_name, []):
            walk(child_joint, indent + 1)
        doc.world.append(f'{"  " * indent}</body>\n')

    # No freejoint: the root body is fixed to whatever body includes this model.
    # Posing the hand is the mounting layer's business -- a mocap+weld pair for a
    # draggable hand, or a flange attach when the hand sits on an arm. Baking a
    # freejoint in here would make the model unusable on a robot arm, because a
    # free joint is only legal on a top-level body.
    doc.world.append(f'    <body name="{root_name}">\n')
    doc.world.append(_body_contents(
        doc, root_name, 3, tactile_points, tactile_radius))
    for joint in children.get(root_name, []):
        walk(joint, 3)
    doc.world.append("    </body>\n")

    equalities = tags.equality_tags(doc.joints)

    def assemble(excludes):
        # meshdir pins the mesh base to this directory. Without it MuJoCo
        # resolves asset paths against the process cwd, so the model only loaded
        # when the caller happened to sit one level above (or below) the XML.
        # An absolute meshdir would break as soon as the tree is relocated,
        # hence the relative one; MuJoCo resolves it against the file itself.
        return (
            f'<mujoco model="{cfg.model_name(doc.robot_name, variant)}">\n'
            '  <compiler angle="radian" meshdir="."/>\n'
            f'  <option timestep="{tags.fmt(cfg.TIMESTEP)}" '
            'integrator="implicitfast" cone="elliptic"/>\n'
            + tags.default_section(doc.robot_name, doc.wants_tactile)
            + "  <asset>\n" + "".join(doc.assets) + "  </asset>\n"
            "  <worldbody>\n" + "".join(doc.world) + "  </worldbody>\n"
            + excludes
            + "  <equality>\n" + "".join(equalities) + "  </equality>\n"
            "  <actuator>\n" + "".join(doc.actuators) + "  </actuator>\n"
            "</mujoco>\n"
        )

    # Mesh paths in the generated text are relative to the output directory
    # (meshes/... and meshes/convex/...), so that is where the throwaway model
    # has to be resolved from.
    measured = collision_filter.exclude_section(assemble(""), output_dir)
    return assemble(measured)
