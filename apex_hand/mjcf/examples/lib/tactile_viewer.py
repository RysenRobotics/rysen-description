#!/usr/bin/env python3
"""Show the tactile readings as a colour-coded pressure map in its own window.

Each pad is drawn with the same element layout as the physical sensor (see
urdf/tactile/remap.json), so a cell on screen is one sensing element. Colour runs from
blue (no load) through green and yellow to red at ``KPA_MAX``; anything above is
clamped to red.

The pads are tiled in the shape of the hand rather than in a table: the palm sits
at the bottom, the four fingers rise from it as columns (proximal pad at the
bottom, fingertip at the top), and the thumb lies alongside at the side it grows
from. That way a glance at the window reads like a glance at the hand.

Usage: build a ``TactilePanel(side)`` once, then call ``update(tactiles)`` per
frame after ``RobotTactiles.get_tactile_force`` has run.

Dependencies: numpy, matplotlib.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Pressure that maps to full red; above it everything saturates.
KPA_MAX = 200.0
CMAP = LinearSegmentedColormap.from_list(
    "tactile",
    ["#0d1b6b", "#1f77d0", "#2ec4b6", "#8fd14f", "#ffd400", "#ff8c00", "#d62728"])

# Side of one pad's box on screen, in axis units. The pads have different grids
# but are drawn the same size so the finger columns line up.
CELL = 1.0
# Clearance between pads, in cells.
GAP = 1.0
# The palm's grid (14x14) is the densest, so it is enlarged to keep its elements
# legible next to the fingers.
PALM_SCALE = 3.0
# How far below the finger bases the thumb sits, in axis units.
THUMB_DROP = 12.0
# Fingers in reading order, from the index to the pinky. The thumb is handled on
# its own because it is offset to the side instead of pointing straight up.
FINGERS = ("index", "middle", "ring", "pinky")
# Pads along one finger, from the proxmal segment to the fingertip.
PADS = ("tactile0", "tactile1", "tactile2")


def _split(body_name):
    """``right_index_tactile2`` -> ``("index", "tactile2")``; palm returns None."""
    parts = body_name.split("_")
    if len(parts) == 3 and parts[2].startswith("tactile") and parts[1] != "palm":
        return parts[1], parts[2]
    return None


class TactilePanel:
    """A matplotlib window drawing the pads in the outline of a hand."""

    def __init__(self, side="right"):
        self.side = side
        self.fig, self.ax = plt.subplots(figsize=(7.0, 8.0))
        self.fig.canvas.manager.set_window_title("Tactile pressure [kPa]")
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self._artists = {}
        self._placed = False
        bar = self.fig.colorbar(
            plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0.0, KPA_MAX)),
            ax=self.ax, fraction=0.03, pad=0.02)
        bar.set_label("kPa")
        plt.ion()
        plt.show(block=False)

    def update(self, tactiles):
        """Redraw from ``tactiles.tactile_maps_kpa`` (kPa, NaN = no element)."""
        maps = tactiles.tactile_maps_kpa
        if not self._placed:
            self._place(maps)
            self._placed = True
        for body_name, image in maps.items():
            artist = self._artists.get(body_name)
            if artist is not None:
                artist.set_data(image)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _place(self, maps):
        """Tile the pads the way they sit on the hand."""
        palm = None
        pads = {}
        for body_name in maps:
            split = _split(body_name)
            if split is None:
                palm = body_name
            else:
                pads.setdefault(split[0], {})[split[1]] = body_name

        # Each finger's three pads are drawn the same size and centred on a shared
        # axis, so a column reads as one finger rather than as three unrelated
        # rectangles -- the pads have different grids (6x6, 4x5, 8x8) and would
        # otherwise wander under their own widths.
        pad_side = max(
            max(maps[name].shape) for finger_pads in pads.values()
            for name in finger_pads.values()) * CELL
        pitch = pad_side + GAP * 3

        # Read the hand palm-on: the thumb is on the outer edge, and the fingers run
        # away from it -- index first, pinky furthest. So the fingers go left to
        # right for a left hand and right to left for a right hand, which is just
        # this order reversed when the thumb sits on the right.
        order = FINGERS if self.side == "left" else tuple(reversed(FINGERS))
        finger_x = {}
        x = 0.0
        for finger in order:
            if not pads.get(finger):
                continue
            finger_x[finger] = x
            x += pitch

        # The palm is the widest band and is blown up so its 14x14 grid is
        # readable at the same cell size as the fingers.
        palm_height = 0.0
        if palm is not None:
            image = maps[palm]
            side = max(image.shape) * CELL * PALM_SCALE
            palm_height = side + GAP * 4
            self._draw(palm, image, 0.0, -GAP * 2, size=side)

        for finger in FINGERS:
            finger_pads = pads.get(finger, {})
            if not finger_pads:
                continue
            y = palm_height
            for pad in PADS:
                body_name = finger_pads.get(pad)
                if body_name is None:
                    continue
                image = maps[body_name]
                self._draw(body_name, image, finger_x[finger], y, size=pad_side)
                y += pitch

        # Thumb on the outer side of the hand, dropped below the finger bases: it
        # lies lower on the hand than the fingers do. With the palm facing the
        # reader the thumb sits on the outside edge -- left of the fingers for a
        # left hand, right of them for a right hand.
        thumb_pads = pads.get("thumb", {})
        if thumb_pads:
            thumb_x = -pitch if self.side == "left" else x
            y = palm_height - THUMB_DROP
            for pad in PADS:
                body_name = thumb_pads.get(pad)
                if body_name is None:
                    continue
                image = maps[body_name]
                self._draw(body_name, image, thumb_x, y, size=pad_side)
                y += pitch

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.margins(0.04)

    def _draw(self, body_name, image, x, y, size):
        """Draw one pad occupying the square ``(x, y)`` to ``(x+size, y+size)``.

        ``extent`` is ``(left, right, bottom, top)`` and row 0 of a remap array
        is the pad's top edge, so the bottom edge of the box is ``y`` and the
        top is ``y + size`` -- with ``origin="upper"`` that lands row 0 at the
        top. The axis stays unflipped, so the layout coordinates mean what they
        say.

        ``size`` forces a square box regardless of the pad's own grid, which is
        what lines the three pads of a finger up on one axis.
        """
        artist = self.ax.imshow(
            image, cmap=CMAP, vmin=0.0, vmax=KPA_MAX, origin="upper",
            interpolation="nearest",
            extent=(x, x + size, y, y + size))
        self._artists[body_name] = artist
        self.ax.text(x + size / 2, y + size + CELL * 0.4,
                     body_name.replace("_tactile", "").replace("_", " "),
                     fontsize=6, ha="center", va="bottom")

    def close(self):
        plt.close(self.fig)
