#!/usr/bin/env python3
# 读取 "../urdf/apex_hand_left/apex_hand_left_tactile.urdf" 和 "../urdf/apex_hand_right/apex_hand_right_tactile.urdf"
# 将每个tactile_point的坐标系的z方向，修正为所在link的mesh的最近点的法向量。

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

WORKSPACE = Path(__file__).resolve().parents[1]
URDF_DIR = WORKSPACE / "urdf"
URDF_PATHS = (
    URDF_DIR / "apex_hand_left/apex_hand_left_tactile.urdf",
    URDF_DIR / "apex_hand_right/apex_hand_right_tactile.urdf",
)

POINT_JOINT_RE = re.compile(
    r'<joint name="[^"]+_point_\d+_fix" type="fixed">\s*'
    r'<origin xyz="(?P<xyz>[^"]+)" rpy="(?P<rpy>[^"]+)"\s*/>\s*'
    r'<parent link="(?P<parent>[^"]+)"\s*/>'
)


def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R):
    return (
        np.arctan2(R[2, 1], R[2, 2]),
        np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])),
        np.arctan2(R[1, 0], R[0, 0]),
    )


def align_z(R, normal):
    """Minimal rotation of R so its z-axis points along the (unit) normal."""
    z = R[:, 2]
    axis = np.cross(z, normal)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        if np.dot(z, normal) > 0:
            return R
        axis = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(z, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        axis /= s
        angle = np.arctan2(s, np.dot(z, normal))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return R + (np.sin(angle) * K + (1 - np.cos(angle)) * K @ K) @ R


def resolve_mesh_path(filename, hand_dir):
    """Locate a mesh referenced by a URDF, whatever URI style the URDF uses.

    The URI is interpreted literally against the URDF's own directory, then
    urdf/, then the package root; only if none matches is the file looked up by
    name in the hand's meshes/ directory (which copes with a stale middle layer
    such as "models/<hand>/meshes" in a package:// URI).
    """
    relative = re.sub(r"^package://[^/]+/", "", filename)
    path = Path(relative)
    if path.is_absolute():
        return path
    for root in (hand_dir, URDF_DIR, WORKSPACE):
        candidate = root / path
        if candidate.exists():
            return candidate
    candidate = hand_dir / "meshes" / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Mesh not found: {filename}")


def link_mesh(root, link_name, hand_dir):
    for link in root.findall("link"):
        if link.get("name") == link_name:
            filename = link.find("visual/geometry/mesh").get("filename")
            return trimesh.load(resolve_mesh_path(filename, hand_dir))
    raise KeyError(f"link not found: {link_name}")


def correct(path):
    text = path.read_bytes().decode("utf-8")
    root = ET.parse(path).getroot()
    hand_dir = path.parent
    meshes = {}

    def replace(match):
        xyz = np.array([float(v) for v in match.group("xyz").split()])
        R = rpy_to_matrix([float(v) for v in match.group("rpy").split()])
        parent = match.group("parent")
        if parent not in meshes:
            meshes[parent] = link_mesh(root, parent, hand_dir)
        _, _, face = trimesh.proximity.closest_point_naive(meshes[parent], [xyz])
        normal = meshes[parent].face_normals[face[0]]
        if "palm" in parent:  # palm上的z轴取反向
            normal = -normal
        if np.dot(R[:, 2], normal) > 1 - 1e-8:  # already aligned
            return match.group(0)
        R = align_z(R, normal)
        rpy = " ".join(f"{v:.6f}" for v in matrix_to_rpy(R))
        return match.group(0).replace(f'rpy="{match.group("rpy")}"', f'rpy="{rpy}"', 1)

    path.write_bytes(POINT_JOINT_RE.sub(replace, text).encode("utf-8"))


if __name__ == "__main__":
    for urdf in URDF_PATHS:
        correct(urdf)
