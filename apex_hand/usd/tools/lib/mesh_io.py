"""Read STL meshes and author them as USD ``Mesh`` prims.

The STLs are triangle soup, so every triangle carries its own three vertices.
That is kept as-is: USD then derives one normal per face and the CAD parts stay
faceted, which is what they look like. Welding them would shrink the file by two
thirds but smooth every hard edge.

The 38 meshes of a hand are written once into the variant's own geometry scope
and referenced from both the visual and the collision placement, so each
placement carries no geometry of its own and the asset stays a single file.
"""

import numpy as np
import trimesh


def load_mesh(path):
    """Return ``{"points": (N,3) float32, "faces": (M,3) int32}`` for an STL.

    Winding is repaired: twenty of the thirty-eight collision STLs are not
    watertight and the palm also carries duplicated reversed triangles, while
    PhysX builds the signed distance field from the winding. A flipped patch
    turns the inside of the palm into the outside, which shows up as contacts
    that push with kilonewtons instead of newtons.
    """
    mesh = trimesh.load(str(path), force="mesh", process=False)
    mesh.fix_normals()
    points = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path}: expected triangles, got faces of shape {faces.shape}")
    return {"points": points, "faces": faces}


def mesh_bounds(data):
    points = data["points"]
    return points.min(axis=0), points.max(axis=0)


def write_mesh_prim(stage, prim_path, data):
    """Author one ``UsdGeom.Mesh`` with the given points and triangles."""
    from pxr import Gf, UsdGeom, Vt

    geom = UsdGeom.Mesh.Define(stage, prim_path)
    geom.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in data["points"]]))
    geom.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(data["faces"])))
    geom.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray([int(i) for i in data["faces"].reshape(-1)]))
    low, high = mesh_bounds(data)
    geom.CreateExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, low)),
                                               Gf.Vec3f(*map(float, high))]))
    geom.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    # The library copy is a reference target only; without this it would also
    # draw at the source pose on top of every placement.
    UsdGeom.Imageable(geom.GetPrim()).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    return geom.GetPrim()


def reference_mesh(stage, prim_path, mesh_prim_path, transform=None, purpose=None):
    """Place a geometry-library mesh at ``prim_path`` by internal reference.

    The library prim lives in this same stage, so the visual and collision
    placements of a mesh share one copy of the geometry and the file needs no
    companion asset.
    """
    from pxr import Gf, UsdGeom

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    prim = stage.OverridePrim(prim_path)
    prim.GetReferences().AddInternalReference(mesh_prim_path)
    if transform is not None:
        xform = UsdGeom.Xformable(prim)
        xform.MakeMatrixXform().Set(Gf.Matrix4d(*np.asarray(transform).T.reshape(-1).tolist()))
    if purpose is not None:
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(purpose)
    return prim


def matrix_to_gf(matrix):
    from pxr import Gf

    return Gf.Matrix4d(*np.asarray(matrix).T.reshape(-1).tolist())
