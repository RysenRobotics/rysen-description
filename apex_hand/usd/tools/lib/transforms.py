"""Rotation and transform helpers shared by the writer.

Quaternions are ``(w, x, y, z)`` -- USD's convention -- everywhere in this
module. The URDF rotations use the standard rollout ``R = Rz(yaw) Ry(pitch)
Rx(roll)``; note that the MJCF writer had to reverse the triple before handing
it to MuJoCo's euler helper to get this same result, which is a good way to
remember that the URDF itself stores the ordinary roll-pitch-yaw triple.
"""

import numpy as np


def rpy_to_matrix(rpy):
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_quat(matrix):
    """Rotation matrix -> ``(w, x, y, z)``.

    Shepperd's method: pick the largest diagonal term so the divisor never
    collapses, which matters for the 180-degree rotations the fingers carry.
    """
    m = np.asarray(matrix, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)


def quat_conjugate(quat):
    w, x, y, z = quat
    return np.array([w, -x, -y, -z])


def quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def axis_alignment_quat(axis):
    """Quaternion rotating ``axis`` onto +Z.

    URDF joints spin about an arbitrary unit axis; USD revolute joints only
    accept X, Y or Z, so the joint frame is pre-rotated to make the axis land on
    Z. The same rotation has to be undone on the child body's side, which is why
    the writer stores both ``localRot0`` and ``localRot1``.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("joint axis is zero-length")
    axis = axis / norm
    target = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(axis, target))
    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-9:
        # 180 degrees: any axis perpendicular to Z will do.
        return np.array([0.0, 1.0, 0.0, 0.0])
    cross = np.cross(axis, target)
    quat = np.concatenate([[1.0 + dot], cross])
    return quat / np.linalg.norm(quat)


def diagonalize_inertia(inertia):
    """URDF's six inertia terms -> USD's ``diagonalInertia`` + ``principalAxes``.

    ``inertia`` is ``(ixx, iyy, izz, ixy, ixz, iyz)`` in the link frame. USD wants
    the eigenvalues along with the rotation taking the principal axes onto the
    link frame, so the pair reproduces the same tensor: ``I = R diag(d) R^T``.
    """
    ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in inertia)
    tensor = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ])
    values, vectors = np.linalg.eigh(tensor)
    # eigh returns a right-handed basis only up to a sign; flip one axis if the
    # determinant came out negative so the result stays a rotation.
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 0] = -vectors[:, 0]
    return values, matrix_to_quat(vectors)


def transform_matrix(xyz, rpy):
    matrix = np.eye(4)
    matrix[:3, :3] = rpy_to_matrix(rpy)
    matrix[:3, 3] = np.asarray(xyz, dtype=float)
    return matrix


def compose(parent, child):
    return np.asarray(parent) @ np.asarray(child)
