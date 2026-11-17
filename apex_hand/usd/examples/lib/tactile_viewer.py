"""Tactile arrows in the Kit viewport.

The pressure window is not here: it is the MuJoCo demo's ``TactilePanel``,
driven from a separate process by ``tactile_window.py`` because this virtualenv
has no GUI matplotlib backend.

What is left in this module is the part that only makes sense inside Kit: red
arrows drawn with the debug-draw overlay, one per loaded point, along the point
normal and with length proportional to the force (clamped at ``FORCE_MAX``).
They are only created when there is a GUI.
"""

import numpy as np

from tactiles import FORCE_MAX, FORCE_MIN

# Force-arrow appearance. Arrows follow the sensor's own range: they appear at
# FORCE_MIN and stop growing at FORCE_MAX, so an arrow can never imply a load the
# array could not report. The array spans 200:1, so the length is sqrt-scaled --
# drawn linearly, a 1 N point would be a 20 mm stub while a 13 N point reached
# 260 mm, and the weak points vanished behind the hand. With sqrt a full-scale
# point still gets ARROW_M_AT_MAX and a quarter-scale point gets half of it.
# Direction is unaffected: only the length is compressed.
ARROW_M_AT_MAX = 0.4       # 20 N -> a 400 mm arrow
ARROW_WIDTH_PX = 5
ARROW_COLOR = (1.0, 0.0, 0.0, 1.0)
ARROW_THRESHOLD = FORCE_MIN


def arrow_length(force):
    """Length in metres for a point reading ``force`` newtons.

    sqrt, not linear, for the reason above; clamped at full scale. Shared with
    ``record_grasp.py`` so the GIF and the viewport agree.
    """
    clamped = min(float(force), FORCE_MAX)
    if clamped <= 0.0:
        return 0.0
    return ARROW_M_AT_MAX * float(np.sqrt(clamped / FORCE_MAX))

# Kit's debug-draw has lines but no arrow primitive, so a short V at the tip
# stands in for MuJoCo's cone and makes the direction legible from a distance.
ARROW_HEAD_FRAC = 0.25     # head length as a fraction of the shaft
ARROW_HEAD_SPREAD = 0.4    # head half-width relative to its length


class TactileArrows:
    """Red per-point force arrows in the Kit viewport, replaced every frame."""

    def __init__(self):
        # Isaac Lab's minimal Kit experience knows the debug-draw extension but
        # does not enable it. Turn it on here rather than asking the user to pass
        # --experience or --kit-args; enabling is asynchronous, so pump a few
        # frames before importing its Python bindings.
        import omni.kit.app

        name = "isaacsim.util.debug_draw"
        app = omni.kit.app.get_app()
        manager = app.get_extension_manager()
        if not manager.is_extension_enabled(name):
            manager.set_extension_enabled(name, True)
            for _ in range(30):
                app.update()
        try:
            from isaacsim.util.debug_draw import _debug_draw
        except ImportError:
            self._draw = None
            return
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def update(self, tactiles):
        if self._draw is None:
            return
        starts, ends, colors, widths = [], [], [], []

        def segment(a, b):
            starts.append(tuple(float(v) for v in a))
            ends.append(tuple(float(v) for v in b))
            colors.append(ARROW_COLOR)
            widths.append(ARROW_WIDTH_PX)

        for pad, forces in tactiles.forces.items():
            points = tactiles.point_positions.get(pad)
            normals = tactiles.point_normals.get(pad)
            if points is None or normals is None:
                continue
            for index in np.nonzero(forces > ARROW_THRESHOLD)[0]:
                direction = np.asarray(normals[index], dtype=float)
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    continue
                # Along the outward normal, as the MuJoCo demo draws them.
                direction /= norm
                start = points[index]
                length = arrow_length(forces[index])
                tip = start + direction * length
                segment(start, tip)
                # Four edges back from the tip form a shallow pyramid, so the
                # head reads as a head from any camera angle rather than a flat V.
                head = length * ARROW_HEAD_FRAC
                base = tip - direction * head
                ref = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 \
                    else np.array([1.0, 0.0, 0.0])
                u = np.cross(direction, ref)
                u /= np.linalg.norm(u)
                v = np.cross(direction, u)
                for axis in (u, -u, v, -v):
                    segment(base + axis * head * ARROW_HEAD_SPREAD, tip)
        self._draw.clear_lines()
        if starts:
            self._draw.draw_lines(starts, ends, colors, widths)

    def close(self):
        if self._draw is not None:
            self._draw.clear_lines()
