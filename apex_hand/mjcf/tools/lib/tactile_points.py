"""Extract tactile contact points from the tactile URDF.

Each tactile point is a tiny sphere in ``*_tactile.urdf``: a link with a sphere
geometry, attached to its pad body by a fixed joint. The joint origin carries
both the point position and its frame; the frame's local z axis is the contact
normal, already pointing out of the surface.

The stored position is a point on, or within a hair of, the pad skin -- not the
sphere centre. MuJoCo wants the geom centre, so the sphere is pushed inward
along that z axis by ``SURFACE_INSET``. The inset exceeds the radius, so the
whole sphere stays buried under the skin and disappears from view when the pad
is drawn.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

import mjcf_config as cfg

# Sphere radius, surface inset and probe step come from config.yaml; see that
# file for why each has the value it does.
DEFAULT_RADIUS = cfg.DEFAULT_RADIUS
SURFACE_INSET = cfg.SURFACE_INSET
INSIDE_STEP = cfg.INSIDE_STEP
# Link name fragment that marks a tactile point rather than a pad.
POINT_TAG = "_point_"


def _rotation(rpy):
    """URDF rpy (zyx order, as in the rest of this package) -> 3x3 rotation."""
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray(rpy, dtype=float)[::-1], "zyx")
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def _collision_mesh(urdf_path, link_name):
    """Load the collision mesh of ``link_name`` in its own frame, or None.

    Meshes are declared relative to the URDF's directory, which is also where
    ``meshes/`` lives.
    """
    root = ET.parse(urdf_path).getroot()
    for link in root.findall("link"):
        if link.get("name") != link_name:
            continue
        filename = link.find("collision/geometry/mesh")
        if filename is None:
            return None
        path = Path(urdf_path).parent / filename.get("filename")
        if not path.exists():
            return None
        return trimesh.load(str(path), force="mesh")
    return None


def _outermost_hit(mesh, point, axis):
    """Distance along ``axis`` from ``point`` to the outermost skin crossing.

    The frame's +z points out of the pad, so the last face met going that way is
    the outermost skin. A point already under the skin reaches it; a point
    floating a fraction above the surface sees nothing there, and then the first
    crossing inward is the skin it belongs to.

    Distances are signed along ``axis``: a point floating above the surface has
    no crossing ahead of it, so the ray is started from just under the skin
    instead, and the resulting hit is reported in the point's own frame -- that
    way subtracting the inset lands the sphere under the same face either way.
    Returns 0.0 when the pad is never crossed.
    """
    hits, _, _ = mesh.ray.intersects_location(
        ray_origins=[point],
        ray_directions=[axis],
        multiple_hits=True,
    )
    outward = [float(np.dot(hit - point, axis)) for hit in hits]
    if outward:
        return max(outward)

    # Nothing ahead: the point sits above the skin. Step in past the surface it
    # belongs to and re-fire outward, then express the hit relative to the point.
    _, distance, _ = trimesh.proximity.closest_point(mesh, [point])
    origin = point - axis * (float(distance[0]) + INSIDE_STEP)
    hits, _, _ = mesh.ray.intersects_location(
        ray_origins=[origin],
        ray_directions=[axis],
        multiple_hits=True,
    )
    reaches = [float(np.dot(hit - point, axis)) for hit in hits]
    if reaches:
        return max(reaches)
    return float(distance[0])


def load_points(urdf_path, inset=SURFACE_INSET):
    """Return ``{body_name: [(point_name, position, rpy), ...]}``.

    ``position`` is the geom centre in the pad body's frame: the point moved
    ``inset`` inward from the outermost skin it sees along +z. ``rpy`` is the
    point frame, passed through untouched so the geom keeps the URDF orientation.

    Points are grouped by the pad body they belong to, which is where they get
    emitted and the only place MuJoCo could collide them against.
    """
    root = ET.parse(urdf_path).getroot()

    # child link -> (parent link, origin xyz, origin rpy)
    attachments = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()] \
            if origin is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()] \
            if origin is not None else [0.0, 0.0, 0.0]
        attachments[child.get("link")] = (parent.get("link"), xyz, rpy)

    meshes = {}
    grouped = {}
    for link in root.findall("link"):
        name = link.get("name") or ""
        if POINT_TAG not in name:
            continue
        if link.find("visual/geometry/sphere") is None:
            continue
        attachment = attachments.get(name)
        if attachment is None:
            continue
        body, xyz, rpy = attachment
        axis = _rotation(rpy)[:, 2]
        point = np.asarray(xyz, dtype=float)

        if body not in meshes:
            meshes[body] = _collision_mesh(urdf_path, body)
        offset = _outermost_hit(meshes[body], point, axis) if meshes[body] is not None else 0.0
        position = point + (offset - inset) * axis
        grouped.setdefault(body, []).append((name, position.tolist(), rpy))

    for points in grouped.values():
        points.sort(key=lambda item: item[0])
    return grouped
