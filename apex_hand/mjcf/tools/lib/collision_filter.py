"""Derive <exclude> pairs by measuring which bodies overlap at the zero pose.

Each collision variant uses different geometry (mesh, primitive, convex hulls),
so the set of overlapping body pairs differs per variant and cannot be shared.
Rather than hand-maintain three tables, this module compiles a throwaway model
without any exclusions, steps it to the zero pose, and records every body pair
that reports a contact. Those are the pairs that overlap by construction -- the
finger base joints genuinely sit inside the palm shell, for instance.

Parent/child pairs are kept: MuJoCo usually filters those, but a body nested
under a fixed joint still collides with its parent, so they have to be listed
too.
"""

import os
from pathlib import Path

import mujoco


def measure_overlaps(xml_text, asset_dir):
    """Return sorted body-name pairs that penetrate at the zero pose.

    ``asset_dir`` is the directory the MJCF's relative mesh paths resolve
    against. MuJoCo resolves them against the process working directory when
    loading from a string, so the caller has to say where the files live.
    """
    previous = os.getcwd()
    os.chdir(asset_dir)
    try:
        model = mujoco.MjModel.from_xml_string(xml_text)
    finally:
        os.chdir(previous)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    name = lambda index: mujoco.mj_id2name(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, index)

    pairs = set()
    for contact in data.contact[: data.ncon]:
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        if body1 == body2:
            continue
        if contact.dist >= 0:
            continue
        # Parent/child pairs are kept. MuJoCo usually filters them, but a body
        # nested under a fixed joint still collides with its parent here (the
        # palm against the finger roots), so they must be excluded too.
        first, second = name(body1), name(body2)
        pairs.add((min(first, second), max(first, second)))
    return sorted(pairs)


def exclude_section(xml_text, asset_dir):
    """Build the <contact> section for ``xml_text``, or '' when nothing overlaps."""
    pairs = measure_overlaps(xml_text, Path(asset_dir))
    if not pairs:
        return ""
    body = "".join(
        f'    <exclude body1="{first}" body2="{second}"/>\n'
        for first, second in pairs
    )
    return "  <contact>\n" + body + "  </contact>\n"
