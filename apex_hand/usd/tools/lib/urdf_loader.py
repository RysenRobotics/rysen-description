"""Parse the Apex hand URDF into bodies, joints and reference frames.

The URDF is a full CAD tree: 48 links, but only 22 of them move. The other 26
are tactile pads and reference frames hanging off a phalanx by fixed joints.
PhysX (like every other engine) would rather not simulate 26 massless bodies, so
this module flattens the fixed subtrees the same way the URDF importer's
"merge fixed joints" pass does: their geometry, mass and inertia move onto the
nearest ancestor that a revolute joint drives.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from transforms import compose, rpy_to_matrix, transform_matrix


def _floats(text, default):
    return list(default) if text is None else [float(v) for v in text.split()]


def _origin(node):
    origin = node.find("origin") if node is not None else None
    if origin is None:
        return [0.0] * 3, [0.0] * 3
    return _floats(origin.get("xyz"), (0, 0, 0)), _floats(origin.get("rpy"), (0, 0, 0))


def _mesh_file(tag):
    mesh = tag.find("geometry/mesh") if tag is not None else None
    return mesh.get("filename") if mesh is not None else None


def load_urdf(path):
    """Return ``(robot_name, links, joints)`` for a URDF file."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        inertia = inertial.find("inertia") if inertial is not None else None
        com, com_rpy = _origin(inertial)
        links[link.get("name")] = {
            "mass": float(mass.get("value")) if mass is not None else 0.0,
            "com": com,
            "com_rpy": com_rpy,
            "inertia": tuple(float(inertia.get(k))
                             for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
            if inertia is not None else None,
            "visual": _mesh_file(link.find("visual")),
            "collision": _mesh_file(link.find("collision")),
        }

    joints = []
    for joint in root.findall("joint"):
        xyz, rpy = _origin(joint)
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
            "axis": _floats(axis.get("xyz") if axis is not None else None, (0, 0, 1)),
            "limit": dict(limit.attrib) if limit is not None else {},
            "mimic": dict(mimic.attrib) if mimic is not None else None,
        })
    return root.get("name"), links, joints


class _Accumulator:
    """Mass properties collected for one body.

    Everything is summed about the body origin first and shifted to the combined
    centre of mass at the end: the parallel-axis term is additive that way, and
    subtracting it once is exact where subtracting it per link is not.
    """

    def __init__(self):
        self.mass = 0.0
        self.first_moment = np.zeros(3)
        self.about_origin = np.zeros((3, 3))

    def add(self, link, transform):
        inertia = link["inertia"]
        mass = link["mass"]
        if inertia is None or mass <= 0.0:
            return
        com = (transform @ np.array([*link["com"], 1.0]))[:3]
        rotation = transform[:3, :3] @ rpy_to_matrix(link["com_rpy"])
        ixx, iyy, izz, ixy, ixz, iyz = inertia
        tensor = rotation @ np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]) @ rotation.T
        self.mass += mass
        self.first_moment += mass * com
        self.about_origin += tensor + mass * (
            np.dot(com, com) * np.eye(3) - np.outer(com, com)
        )

    def result(self):
        if self.mass <= 0.0:
            return 0.0, (0.0, 0.0, 0.0), None
        com = self.first_moment / self.mass
        tensor = self.about_origin - self.mass * (
            np.dot(com, com) * np.eye(3) - np.outer(com, com)
        )
        return self.mass, tuple(com), (tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                       tensor[0, 1], tensor[0, 2], tensor[1, 2])


def build_bodies(links, joints, frame_suffixes):
    """Flatten the URDF tree into ``(root_body, bodies, movable_joints, frames)``.

    ``bodies`` is a list of dicts with the merged mass properties and the mesh
    placements (each a ``(mesh_filename, 4x4 transform)`` pair in the body
    frame). ``frames`` lists the ``*_tip`` / ``*_pad`` markers with the body they
    belong to and their placement in that body's frame.
    """
    for joint in joints:
        if joint["type"] not in ("fixed", "revolute", "continuous"):
            raise ValueError(f"unsupported joint type {joint['type']} on {joint['name']}")

    children = {}
    child_links = set()
    for joint in joints:
        children.setdefault(joint["parent"], []).append(joint)
        child_links.add(joint["child"])
    roots = [name for name in links if name not in child_links]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, found {roots}")
    root = roots[0]

    # A link is a body if a movable joint drives it, or if it is the root.
    body_links = {root}
    for joint in joints:
        if joint["type"] != "fixed":
            body_links.add(joint["child"])
            body_links.add(joint["parent"])

    bodies = []
    frames = []
    # link name -> (owning body, placement of the link frame in that body)
    placements_by_link = {}

    def collect(body_name, link_name, parent_transform, accumulator, meshes, is_frame=False):
        link = links[link_name]
        placements_by_link[link_name] = (body_name, parent_transform)
        accumulator.add(link, parent_transform)
        if not is_frame:
            # Frame links carry a locator sphere (0.5-1 mm) as their visual and
            # collision mesh. It is a marker for the RViz view, not part of the
            # hand -- the MJCF model turns these links into sites too, so their
            # geometry is dropped here and they are authored as plain Xforms.
            for key in ("visual", "collision"):
                filename = link[key]
                if filename:
                    meshes.append((key, filename, parent_transform))
        for joint in children.get(link_name, []):
            child = joint["child"]
            transform = compose(parent_transform, transform_matrix(joint["xyz"], joint["rpy"]))
            if joint["type"] != "fixed" or child in body_links:
                continue  # a movable joint starts another body; it is walked separately
            frame = child.endswith(tuple(frame_suffixes))
            if frame:
                frames.append({"name": child, "body": body_name,
                               "transform": transform, "root": body_name == root})
            collect(body_name, child, transform, accumulator, meshes, frame)

    def make_body(link_name):
        accumulator = _Accumulator()
        meshes = []
        collect(link_name, link_name, np.eye(4), accumulator, meshes)
        mass, com, inertia = accumulator.result()
        return {
            "name": link_name,
            "mass": mass,
            "com": com,
            "inertia": inertia,
            # (kind, mesh filename, placement in body frame)
            "meshes": meshes,
        }

    bodies.append(make_body(root))
    for joint in joints:
        if joint["type"] == "fixed":
            continue
        bodies.append(make_body(joint["child"]))

    movable = [joint for joint in joints if joint["type"] != "fixed"]
    return root, bodies, movable, frames, placements_by_link


def mesh_path(urdf_path, filename):
    """Absolute path of a URDF mesh reference, which is relative to the URDF."""
    return Path(urdf_path).parent / filename


def mesh_prim_name(filename):
    """Prim name for a mesh file: the STL stem, which is the link name."""
    return Path(filename).stem
