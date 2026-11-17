"""Locate the 619 tactile spheres from the tactile URDF.

Each tactile point is a tiny sphere in ``*_tactile.urdf``: a link with a sphere
geometry, attached to its pad link by a fixed joint. The joint origin carries
both the point position and its frame; the frame's local z axis is the contact
normal, already pointing out of the surface.

The stored position is a point on, or a hair off, the pad skin -- not the sphere
centre. PhysX wants the geom centre, so the sphere is pushed inward along that z
axis by ``surface_inset``. See ``config.yaml`` for why the inset here is 1 mm
where the MJCF model uses 1.5 mm.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from transforms import rpy_to_matrix

POINT_TAG = "_point_"


def _collision_mesh(urdf_path, link_name):
    """Load the collision mesh of ``link_name`` in its own frame, or None."""
    root = ET.parse(urdf_path).getroot()
    for link in root.findall("link"):
        if link.get("name") != link_name:
            continue
        mesh = link.find("collision/geometry/mesh")
        if mesh is None:
            return None
        path = Path(urdf_path).parent / mesh.get("filename")
        if not path.exists():
            return None
        return trimesh.load(str(path), force="mesh")
    return None


def _outermost_hit(mesh, point, axis, inside_step):
    """Distance along ``axis`` from ``point`` to the outermost skin crossing.

    The frame's +z points out of the pad, so the last face met going that way is
    the outermost skin. A point already under the skin reaches it; a point
    floating a fraction above the surface sees nothing there, and then the first
    crossing inward is the skin it belongs to.

    Returns 0.0 when the pad is never crossed.
    """
    hits, _, _ = mesh.ray.intersects_location(
        ray_origins=[point], ray_directions=[axis], multiple_hits=True)
    outward = [float(np.dot(hit - point, axis)) for hit in hits]
    if outward:
        return max(outward)

    # Nothing ahead: the point sits above the skin. Step in past the surface it
    # belongs to and re-fire outward, then express the hit relative to the point.
    _, distance, _ = trimesh.proximity.closest_point(mesh, [point])
    origin = point - axis * (float(distance[0]) + inside_step)
    hits, _, _ = mesh.ray.intersects_location(
        ray_origins=[origin], ray_directions=[axis], multiple_hits=True)
    reaches = [float(np.dot(hit - point, axis)) for hit in hits]
    if reaches:
        return max(reaches)
    return float(distance[0])


def load_points(urdf_path, inset, inside_step, stride=1):
    """Return ``{pad_link: [(point_name, position, rpy), ...]}``.

    ``position`` is the sphere centre in the pad link frame: the point moved
    ``inset`` inward from the outermost skin it sees along +z. ``rpy`` is the
    point frame, passed through untouched so the sphere keeps the URDF
    orientation -- a sphere is symmetric, but downstream code reads the force
    direction off that frame rather than special-casing the palm.

    ``stride`` keeps every Nth point of each pad, which is the only knob that
    makes the array affordable: every point becomes a rigid body in the USD.
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
        pad, xyz, rpy = attachment
        axis = rpy_to_matrix(rpy)[:, 2]
        point = np.asarray(xyz, dtype=float)

        if pad not in meshes:
            meshes[pad] = _collision_mesh(urdf_path, pad)
        offset = _outermost_hit(meshes[pad], point, axis, inside_step) \
            if meshes[pad] is not None else 0.0
        position = point + (offset - inset) * axis
        grouped.setdefault(pad, []).append((name, position.tolist(), rpy))

    for pad, points in grouped.items():
        points.sort(key=lambda item: item[0])
        if stride > 1:
            # Keep the first of every ``stride`` points. The 2D remap table keeps
            # its full shape, so the dropped cells simply read as empty.
            points[:] = points[::stride]
    return grouped
