#!/usr/bin/env python3
"""Generate the Isaac Sim USD assets for the Apex hand from ../urdf/.

Usage:
    python3 tools/urdf_to_usd.py                       # both sides, all four variants
    python3 tools/urdf_to_usd.py --side right
    python3 tools/urdf_to_usd.py --variants mesh       # plain model, no tactile
    python3 tools/urdf_to_usd.py --variants mount      # the self-holding variant

Writes, per side, into ../usd/apex_hand_<side>/:
    apex_hand_<side>[_mount][_tactile].usd   self-contained: geometry included
    manifest.json                            joints, bodies, tactile pads -- what
                                             the examples read

Why this is not a thin wrapper around ``UrdfConverter``: the conversion has to
inject 619 tactile rigid bodies and keep the visual and collision geometry
sharing one copy. The Isaac Sim URDF importer offers neither, so the stage is
authored directly with ``pxr`` + PhysxSchema -- but Isaac Sim's schema plugins
only load inside a running Kit app, hence the ``SimulationApp`` at the top.

Needs the Isaac Sim environment (``/opt/isaacsim/venv/bin/python``), trimesh and
rtree.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

APEX_DIR = HERE.parent.parent
URDF_DIR = APEX_DIR / "urdf"
USD_DIR = HERE.parent

VARIANT_ORDER = ("mesh", "mesh_tactile",
                 "mesh_mount", "mesh_mount_tactile")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument(
        "--variants", default="all",
        help="comma separated subset of: mesh, mesh_tactile, mesh_mount, "
             "mesh_mount_tactile, tactile, mount, all (default: all)")
    parser.add_argument(
        "--usd-dir", default=str(USD_DIR),
        help="output directory (default: ../usd)")
    return parser.parse_args()


ALIASES = {
    "tactile": "mesh_tactile",
    "mount": "mesh_mount",
    "mount_tactile": "mesh_mount_tactile",
}


def select_variants(text):
    if text == "all":
        return list(VARIANT_ORDER)
    names = [item.strip() for item in text.split(",") if item.strip()]
    names = [ALIASES.get(name, name) for name in names]
    unknown = [name for name in names if name not in VARIANT_ORDER]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}")
    return [name for name in VARIANT_ORDER if name in names]


def build_side(side, variants, usd_dir):
    """Generate every requested variant of one hand and write its manifest."""
    import usd_config as cfg
    import usd_builder

    robot = f"apex_hand_{side}"
    urdf_path = URDF_DIR / robot / f"{robot}.urdf"
    tactile_urdf_path = URDF_DIR / robot / f"{robot}_tactile.urdf"
    if not urdf_path.exists():
        raise SystemExit(f"missing URDF: {urdf_path}")

    asset_dir = Path(usd_dir) / robot
    asset_dir.mkdir(parents=True, exist_ok=True)

    # One manifest per hand, describing the hand itself. Where a variant differs
    # -- the mount chain, most of all -- the mounted one wins, because that is
    # what the examples load and its joint list is the superset. The bare files
    # simply have no mount joints to drive.
    manifest = None
    variants_usd = {}
    for variant in variants:
        suffix = cfg.variant_suffix(variant)
        path = asset_dir / f"{robot}{suffix}.usd"
        info = usd_builder.build_variant(
            path, urdf_path, tactile_urdf_path, variant)
        variants_usd[variant] = path.name
        if manifest is None or (info["has_mount"] and not manifest["has_mount"]):
            manifest = info
        elif info["tactile"] and not manifest["tactile"]:
            manifest["tactile"] = info["tactile"]
            manifest["tactile_bodies"] = info["tactile_bodies"]

    if manifest is None:
        return
    # Which variant this was copied from is an implementation detail; callers
    # ask `variants` for a file and read `mount_joints` to drive it.
    manifest.pop("has_mount", None)
    manifest["side"] = side
    manifest["timestep"] = float(cfg.TIMESTEP)
    manifest["variants"] = variants_usd
    manifest["asset_dir"] = asset_dir.name
    usd_builder.write_manifest(asset_dir, manifest)
    print(f"[{side}] manifest.json written")


def main():
    args = parse_args()
    variants = select_variants(args.variants)

    # Isaac Sim ships PhysxSchema as an extension plugin: importing it before the
    # app exists fails with "cannot import name 'PhysxSchema'".
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        sides = ("left", "right") if args.side == "both" else (args.side,)
        for side in sides:
            build_side(side, variants, args.usd_dir)
    finally:
        app.close()


if __name__ == "__main__":
    main()
