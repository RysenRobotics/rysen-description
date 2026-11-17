"""The grasp sequence: approach, close, lift.

Position playback only, the same shape as the MuJoCo example: each phase
interpolates the mocap pose and the joint targets, and nothing here reads a
sensor. The pad hulls are compliant, so they stop the fingers where the cylinder
is; a fixed close fraction is enough and there is no per-step logic to reason
about.
"""

import numpy as np

# Hand geometry: the fingers reach along +z and curl towards +y, the palm normal
# is local -x, and the thumb sits on the +y side. Everything below is in the
# hand's own frame.
CYLINDER_RADIUS = 0.042
CYLINDER_HEIGHT = 0.24
CYLINDER_MASS = 0.30
CYLINDER_Z = 0.12

# The mount joints start at zero, so for the first solver pass the whole hand --
# all 38 colliders -- sits at the world origin. PhysX resolves that overlap
# before any Python code runs, and an object spawned at the origin is fired away
# at 6 m/s. Keep the grasp clear of the origin instead: cylinder, palm poses and
# camera all share this XY offset. Only XY moves; the cylinder still rests on z=0.
SCENE_XY = (0.50, 0.0)
CYLINDER_POS = (*SCENE_XY, CYLINDER_Z)
# The hand comes in from +x: park on that side so the approach never sweeps
# through the cylinder.
PARK = (0.30 + SCENE_XY[0], -0.12 + SCENE_XY[1], 0.25)

# local +x (palm back) -> world -x, local -x (palm normal) -> world +x,
# local +y (thumb side) -> world +z, local +z (fingers) -> world +y.
CONTACT_QUAT = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
# Palm origin to the outside of its collision shell along local -x.
PALM_SHELL = 0.029
# Kept at zero as in the MuJoCo model: the approach is meant to end in contact,
# and the palm's position servo settles onto the wall.
PALM_STANDOFF = 0.0
CONTACT_X = SCENE_XY[0] + PALM_SHELL + PALM_STANDOFF + CYLINDER_RADIUS
CONTACT_Z = CYLINDER_Z
# The fingers curl back towards -y, so the palm starts at -y of the axis.
CONTACT_Y = SCENE_XY[1] - 0.12
CONTACT_POS = (CONTACT_X, CONTACT_Y, CONTACT_Z)

# The thumb swings out to 90 deg on the way in so it clears the cylinder, and
# stays there while the fingers close: that is what carries its pads round to
# the far side, where they squeeze against the fingers.
THUMB_OPEN_DEG = 90.0
# How far every curling joint goes, as a fraction of its range. Closing to the
# limit drives the fingertips into each other and lifts the proximal phalanges
# off the cylinder, so this stays short of it and lets the pads settle on the
# object. Copied from the MuJoCo model, which is tuned against the same geometry.
CLOSE_FRACTION = 0.65
# The thumb finishes closing at this fraction of the close phase, so it arrives
# first and the fingers close in against it rather than shoving the cylinder.
THUMB_LEAD = 0.75

MOVE_TIME = 2.0
CLOSE_TIME = 2.5
LIFT_TIME = 2.0
SETTLE_TIME = 0.5
HOLD_TIME = 1.5
LIFT_DELTA = 0.12


def open_targets(limits, names):
    """Swing the thumb out; every other joint stays at zero."""
    targets = np.zeros(len(names), dtype=float)
    for index, name in enumerate(names):
        if "thumb" in name and name.endswith("_j0"):
            low, high = limits[index]
            targets[index] = float(np.clip(np.deg2rad(THUMB_OPEN_DEG), low, high))
    return targets


def closed_targets(limits, names, current):
    """Curl every digit to ``CLOSE_FRACTION`` of its range.

    j0 keeps the knuckle where it is: zero for the fingers, and the 90 deg swing
    for the thumb, which leaves the thumb pads on the far side of the cylinder.
    Curling j0 would push the digit out against the cylinder instead of around
    it. The same rule as the MuJoCo example's ``_close_targets``.
    """
    targets = np.array(current, dtype=float, copy=True)
    for index, name in enumerate(names):
        low, high = limits[index]
        if name.endswith("_j0") or ("thumb" in name and name.endswith("_j2")):
            continue
        targets[index] = low + CLOSE_FRACTION * (high - low)
    return targets


def ramp(duration, timestep):
    """Yield a smooth 0..1 fraction over ``duration`` s."""
    steps = max(1, int(round(duration / timestep)))
    for step in range(steps + 1):
        t = step / steps
        yield t * t * (3.0 - 2.0 * t)


class Grasp:
    """Plays the phases back one physics step at a time."""

    def __init__(self, limits, names, timestep):
        self.limits = np.asarray(limits, dtype=float)
        self.names = list(names)
        self.timestep = timestep
        self.position = np.asarray(PARK, dtype=float)
        self.targets = np.zeros(len(names), dtype=float)
        self.finished = False
        self._generator = self._run()

    def _phase(self, duration, position_end, targets_end):
        position_start = self.position.copy()
        targets_start = self.targets.copy()
        for alpha in ramp(duration, self.timestep):
            self.position = position_start + alpha * (position_end - position_start)
            self.targets = targets_start + alpha * (targets_end - targets_start)
            yield self.position, self.targets

    def _run(self):
        yield from self._phase(MOVE_TIME, np.asarray(CONTACT_POS, dtype=float),
                               open_targets(self.limits, self.names))
        # Hold at the contact pose before curling, as the MuJoCo sequence does:
        # the palm settles onto the cylinder first, and closing while the
        # approach is still arriving ends the grasp somewhere else.
        yield from self._phase(SETTLE_TIME, self.position.copy(), self.targets.copy())
        yield from self._close()
        yield from self._phase(SETTLE_TIME, self.position.copy(), self.targets.copy())
        yield from self._phase(LIFT_TIME,
                               self.position + np.array((0.0, 0.0, LIFT_DELTA)),
                               self.targets.copy())
        yield from self._phase(HOLD_TIME, self.position.copy(), self.targets.copy())

    def _close(self):
        """Ramp to the closed pose and hold; the pads stop the fingers."""
        start = self.targets.copy()
        goal = closed_targets(self.limits, self.names, start)
        thumb = np.array(["thumb" in name for name in self.names])
        for alpha in ramp(CLOSE_TIME, self.timestep):
            targets = start + alpha * (goal - start)
            ahead = min(alpha / THUMB_LEAD, 1.0)
            targets[thumb] = start[thumb] + ahead * (goal[thumb] - start[thumb])
            self.position = np.asarray(CONTACT_POS, dtype=float)
            self.targets = targets
            yield self.position, self.targets
        self.targets = goal

    def step(self):
        """Advance one step; returns False once the sequence is finished."""
        if self.finished:
            return False
        try:
            self.position, self.targets = next(self._generator)
        except StopIteration:
            self.finished = True
            return False
        return True
