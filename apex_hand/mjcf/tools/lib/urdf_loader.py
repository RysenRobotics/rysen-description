"""Parse the Apex hand URDF into plain dictionaries.

Kept separate from the MJCF writer so the geometry modules can consume the same
link/joint data without importing anything MuJoCo-specific.
"""

import xml.etree.ElementTree as ET

# Tip and pad links only define reference frames: they become <site> on their
# parent body instead of massless bodies.
FRAME_LINK_SUFFIXES = ("_tip", "_pad")


def floats(text, default):
    return list(default) if text is None else [float(v) for v in text.split()]


def origin_xyz_rpy(node):
    origin = node.find("origin") if node is not None else None
    if origin is None:
        return [0.0] * 3, [0.0] * 3
    return floats(origin.get("xyz"), (0, 0, 0)), floats(origin.get("rpy"), (0, 0, 0))


def mesh_file(tag):
    mesh = tag.find("geometry/mesh") if tag is not None else None
    return mesh.get("filename") if mesh is not None else None


def is_frame_link(name):
    return name.endswith(FRAME_LINK_SUFFIXES)


def is_tactile_link(name):
    return "tactile" in name


def load_urdf(path):
    """Return ``(robot_name, links, joints)`` for a URDF file."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        inertia = inertial.find("inertia") if inertial is not None else None
        com, _ = origin_xyz_rpy(inertial)
        links[link.get("name")] = {
            "mass": float(mass.get("value")) if mass is not None else 0.0,
            "com": com,
            "inertia": tuple(float(inertia.get(k))
                             for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
            if inertia is not None else None,
            "visual": mesh_file(link.find("visual")),
            "collision": mesh_file(link.find("collision")),
        }

    joints = []
    for joint in root.findall("joint"):
        xyz, rpy = origin_xyz_rpy(joint)
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": floats(axis.get("xyz") if axis is not None else None, (0, 0, 1)),
            "limit": dict(limit.attrib) if limit is not None else {},
            "mimic": dict(mimic.attrib) if mimic is not None else None,
        })
    return root.get("name"), links, joints


def build_tree(links, joints):
    """Return the root link and a parent-link -> child-joints mapping."""
    children, child_links = {}, set()
    for joint in joints:
        if joint["type"] not in ("fixed", "revolute", "continuous"):
            raise ValueError(f"unsupported joint type {joint['type']} on {joint['name']}")
        children.setdefault(joint["parent"], []).append(joint)
        child_links.add(joint["child"])
    roots = [name for name in links if name not in child_links]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, found {roots}")
    return roots[0], children
