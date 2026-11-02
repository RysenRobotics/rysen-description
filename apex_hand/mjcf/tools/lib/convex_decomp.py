"""Convex-decompose a mesh into convex hulls, cached as STL files.

The "convex" collision variant replaces each link mesh with several convex
pieces, which MuJoCo can collide efficiently and exactly (a concave mesh has to
be approximated anyway).

Results are cached under ``meshes/convex/<mesh>/`` and only recomputed when the
source STL is newer than the cache, so repeat runs stay fast.
"""

from pathlib import Path

import trimesh

import mjcf_config as cfg

# Links treated as complex. Only the palm: a large, thin, curved shell whose
# over-coverage is driven by its U shape rather than its face count, so it keeps
# an explicit hull cap instead of the face-derived one. Structural, not a knob.
COMPLEX_LINK_SUFFIXES = ("_palm_link",)
# Written into a cache directory once its hulls are complete. Bump the name when
# the decomposition changes, so caches built by older code are rebuilt.
CACHE_STAMP = "v4.stamp"

# Tunables (resolution, hull budget, solid pitch) come from config.yaml; see
# that file for what each one does and the measurements behind the defaults.
RESOLUTION = cfg.RESOLUTION
VOLUME_ERROR_PERCENT = cfg.VOLUME_ERROR_PERCENT
FACES_PER_HULL = cfg.FACES_PER_HULL
MIN_HULLS = cfg.MIN_HULLS
MAX_HULLS = cfg.MAX_HULLS
COMPLEX_MAX_HULLS = cfg.COMPLEX_MAX_HULLS
SOLID_PITCH = cfg.SOLID_PITCH


def hull_budget(mesh_name, face_count):
    """Return the hull cap for a link.

    The palm keeps an explicit cap; everything else gets one hull per
    ``FACES_PER_HULL`` faces, so a 1.2k-face knuckle is not handed the same
    budget as a 6k-face proximal phalanx.
    """
    if mesh_name.endswith(COMPLEX_LINK_SUFFIXES):
        return COMPLEX_MAX_HULLS
    derived = round(face_count / FACES_PER_HULL)
    return max(MIN_HULLS, min(MAX_HULLS, derived))


def make_solid(mesh, pitch=SOLID_PITCH):
    """Rebuild ``mesh`` as a watertight solid.

    Most of these STLs are open shells, and the palm additionally carries
    duplicated back-to-back triangles that make it non-manifold. V-HACD only
    fills volume it can see through the surface, so on
    such a mesh it emits thin slivers hugging the visible faces and leaves most
    of the part uncovered -- the finger links came out at 69% of their convex
    volume and the palm at 34%, which is what made the hulls look wrong.

    Hole filling alone is not enough: the duplicate faces survive it, so the
    mesh never becomes watertight. Rebuilding through a filled voxel grid
    sidesteps both defects at once, and the pitch keeps the result accurate to
    well under a millimetre.
    """
    voxels = mesh.copy().voxelized(pitch=pitch).fill()
    solid = voxels.marching_cubes
    # marching_cubes works in voxel space; the voxelised grid carries the
    # transform back to the mesh's own coordinates.
    solid.apply_transform(voxels.transform)
    return solid


def decompose(mesh, max_hulls, resolution=RESOLUTION,
              volume_error=VOLUME_ERROR_PERCENT):
    """Return a list of convex ``trimesh`` solids covering ``mesh``.

    V-HACD can emit hulls that are only numerically convex, so every piece is
    re-hulled; MuJoCo rejects a mesh whose vertices are not in convex position.
    """
    parts = trimesh.decomposition.convex_decomposition(
        make_solid(mesh),
        resolution=resolution,
        minimumVolumePercentErrorAllowed=volume_error,
        maxConvexHulls=max_hulls,
    )
    hulls = []
    for part in parts:
        piece = part if isinstance(part, trimesh.Trimesh) else trimesh.Trimesh(**part)
        if len(piece.faces) == 0:
            continue
        hulls.append(piece.convex_hull)
    return hulls


def ensure_hulls(mesh_path, output_dir=None):
    """Return the cached hull files for ``mesh_path``, decomposing if needed.

    ``output_dir`` is the directory holding the generated MJCF. Hulls are written
    under ``<output_dir>/meshes/convex/<mesh>/`` so the MJCF can reference them
    as ``meshes/convex/...`` and keep working from any working directory. It
    defaults to the mesh's own directory for standalone use.

    The cache directory carries a version tag. Comparing source timestamps alone
    is not enough: changing the pitch or the hull budget leaves the source STL
    untouched, so a stale result would be reused after a code change.
    """
    mesh_path = Path(mesh_path)
    root = Path(output_dir) / "meshes" if output_dir else mesh_path.parent
    cache = root / "convex" / mesh_path.stem
    source_mtime = mesh_path.stat().st_mtime

    if cache.is_dir():
        cached = sorted(cache.glob("hull_*.stl"))
        if cached and all(p.stat().st_mtime >= source_mtime for p in cached):
            if (cache / CACHE_STAMP).exists():
                return cached

    mesh = trimesh.load(str(mesh_path), force="mesh")
    hulls = decompose(mesh, hull_budget(mesh_path.stem, len(mesh.faces)))
    if not hulls:
        raise RuntimeError(f"convex decomposition produced no hulls for {mesh_path}")

    cache.mkdir(parents=True, exist_ok=True)
    for stale in cache.glob("hull_*.stl"):
        stale.unlink()
    stamp = cache / CACHE_STAMP
    if stamp.exists():
        stamp.unlink()
    paths = []
    for index, hull in enumerate(hulls):
        path = cache / f"hull_{index:02d}.stl"
        hull.export(str(path))
        paths.append(path)
    # Written last, so an interrupted run leaves the cache unstamped and the
    # next run rebuilds it instead of trusting a partial set.
    stamp.touch()
    return paths
