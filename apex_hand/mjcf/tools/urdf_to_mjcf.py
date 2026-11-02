#!/usr/bin/env python3
"""Convert the Apex hand URDF into the MJCF collision variants.

Every run rewrites all variants for both hands so they can never drift apart.
They differ only in how a link's collision geometry is built:

    apex_hand_<side>.xml                         triangle mesh, as exported
    apex_hand_<side>_tactile.xml                 plus the 619 tactile spheres
    apex_hand_<side>_convex_decomp.xml           convex hulls from V-HACD
    apex_hand_<side>_convex_decomp_tactile.xml   both of the above

Visual geometry, joints, coupling, geom groups and contact classes are shared,
so the variants stay behaviourally identical apart from collision shape.

Usage:
    python3 mjcf/tools/urdf_to_mjcf.py
"""

import shutil
import sys
from pathlib import Path

import mujoco

# The helper modules live in lib/ so tools/ stays a flat list of entry points.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import mjcf_builder  # noqa: E402
import mjcf_config  # noqa: E402
import urdf_loader  # noqa: E402

MJCF_DIR = Path(__file__).resolve().parent.parent
URDF_DIR = MJCF_DIR.parent / "urdf"
SIDES = ("left", "right")


def referenced_meshes(urdf_path):
    """Mesh files the generated MJCF actually references, by file name.

    Frame links -- ``*_tip`` and ``*_pad`` -- carry no geometry in the output:
    the writer turns them into ``<site>`` markers on their parent body. Their
    STLs are 1 mm marker cylinders, so copying them would only ship dead weight.
    """
    _, links, _ = urdf_loader.load_urdf(urdf_path)
    names = set()
    for link_name, link in links.items():
        if urdf_loader.is_frame_link(link_name):
            continue
        for key in ("visual", "collision"):
            if link[key]:
                names.add(Path(link[key]).name)
    return names


def copy_meshes(urdf_path, output_dir):
    """Copy the meshes the generated MJCF references next to it.

    Deleting first keeps the copy honest: a mesh that is no longer referenced --
    dropped from the URDF, or only used by a frame link -- would otherwise
    linger here and be silently shipped. ``convex/`` is the hull cache and
    survives, since rebuilding it costs minutes.
    """
    source = urdf_path.parent / "meshes"
    target = output_dir / "meshes"
    if target.is_dir():
        for stale in target.iterdir():
            if stale.name == "convex":
                continue
            shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    target.mkdir(parents=True, exist_ok=True)
    for name in sorted(referenced_meshes(urdf_path)):
        shutil.copy2(source / name, target / name)


def main():
    for side in SIDES:
        name = f"apex_hand_{side}"
        urdf_path = URDF_DIR / name / f"{name}.urdf"
        output_dir = MJCF_DIR / name
        copy_meshes(urdf_path, output_dir)

        for variant, specs in mjcf_config.VARIANTS.items():
            suffix = specs[0]
            output = output_dir / f"{name}{suffix}.xml"
            output.write_text(
                mjcf_builder.build(urdf_path, variant, output_dir), encoding="utf-8")
            model = mujoco.MjModel.from_xml_path(str(output))
            print(
                f"{output.name:26s} nbody={model.nbody:3d} njnt={model.njnt:2d} "
                f"ngeom={model.ngeom:4d} nu={model.nu:2d} neq={model.neq} "
                f"nexclude={model.nexclude}"
            )


if __name__ == "__main__":
    main()
