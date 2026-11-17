"""Read the 619-point tactile array out of PhysX.

Each tactile sphere is its own rigid body (fixed to the pad it sits in), because
PhysX reports a contact force per rigid body and nowhere finer. Isaac Lab's
``ContactSensor`` gives exactly that: a ``(num_envs, num_bodies, 3)`` net contact
force per body, in world coordinates.

Turning that into what the sensor delivers is four steps, mirroring
``mjcf/examples/lib/mj_tactiles.py``:

* project the force on the point's own outward normal, so tangential shear does
  not leak into the reading;
* clip to the array's reporting range: below ``FORCE_MIN`` nothing is reported,
  above ``FORCE_MAX`` the point saturates;
* divide by the point's share of the pad area, which is approximated by the
  square of the distance to its nearest neighbour (a Voronoi cell), and clip
  that to the pressure range;
* scatter into the 2D array from ``urdf/tactile/remap.json``, so a reading can be
  indexed by (row, column) like the physical sensor.
"""

import json
import re
from pathlib import Path

import numpy as np


def _remap_path():
    return Path(__file__).resolve().parent.parent.parent.parent / "urdf" / "tactile" / "remap.json"


TACTILE_REMAP = json.loads(_remap_path().read_text(encoding="utf-8"))

# Sphere radius, matching tools/config.yaml ``tactile.radius``. The arrow starts
# at the sphere surface so it does not disappear inside the pad.
POINT_RADIUS = 1.0e-3

# The array reports over a fixed range, like the sensor it stands in for: below
# FORCE_MIN a point reads nothing, above FORCE_MAX it saturates, and the pressure
# derived from it has its own floor and ceiling. Clipping here instead of only in
# the colour scale keeps the console numbers, the arrows and the map consistent.
FORCE_MIN = 0.1            # N
FORCE_MAX = 20.0           # N
PRESSURE_MIN_KPA = 10.0
PRESSURE_MAX_KPA = 200.0


def _remap_key(pad_name):
    """Map a pad link name onto a ``remap.json`` key.

    ``right_index_tactile2`` -> ``("finger", "3")``; ``right_thumb_tactile0`` ->
    ``("thumb", "2")``; ``right_palm_tactile`` -> ``("palm", "right")``.

    The fingers' pads 0/1/2 are finger.1/2/3 and the thumb's are thumb.2/3/4
    (the thumb has no proximal pad), so the two do not share an offset.
    """
    match = re.fullmatch(r"(left|right)_palm_tactile", pad_name)
    if match:
        return "palm", match.group(1)
    match = re.fullmatch(r"(left|right)_(thumb|index|middle|ring|pinky)_tactile(\d+)", pad_name)
    if match is None:
        return None
    _, finger, pad = match.groups()
    if finger == "thumb":
        return "thumb", str(int(pad) + 2)
    return "finger", str(int(pad) + 1)


def _point_number(name):
    return int(name.rsplit("_", 1)[-1])


class RobotTactiles:
    """Per-point forces and pressure maps for one hand, from a ContactSensor."""

    def __init__(self, manifest, sensor, robot, side, verbose=True):
        self.side = side
        self.sensor = sensor
        self.tactile = manifest["tactile"]
        if not self.tactile:
            raise SystemExit(f"{side}: the loaded variant carries no tactile bodies")

        # ContactSensor orders its bodies by USD traversal, so index by name.
        self.body_names = [name.rsplit("/", 1)[-1] for name in sensor.body_names]
        order = {name: index for index, name in enumerate(self.body_names)}

        self.pads = {}
        missing = []
        for name, info in sorted(self.tactile.items()):
            if name not in order:
                missing.append(name)
                continue
            entry = self.pads.setdefault(info["pad"], {
                "names": [], "indices": [], "numbers": [],
                "positions": [], "map": self._remap_table(info["pad"]),
            })
            entry["names"].append(name)
            entry["indices"].append(order[name])
            entry["numbers"].append(_point_number(name))
            entry["positions"].append(info["pos"])
        if missing:
            raise SystemExit(
                f"{side}: {len(missing)} tactile bodies not found by the contact "
                f"sensor, e.g. {missing[:3]}")

        # Nearest-neighbour area per pad. All points of a pad share a frame, so
        # the pairwise distance is frame independent.
        for entry in self.pads.values():
            points = np.asarray(entry["positions"], dtype=float)
            entry["indices"] = np.asarray(entry["indices"], dtype=int)
            entry["numbers"] = np.asarray(entry["numbers"], dtype=int)
            distance = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            np.fill_diagonal(distance, np.inf)
            entry["areas"] = distance.min(axis=1) ** 2

        # Every tactile sphere is its own rigid body in the articulation. Its
        # world pose is authoritative for the 3D arrows: manifest "pos" is in the
        # USD link frame, while body_pos_w/body_quat_w are the rigid-body (COM)
        # frame, so rebuilding the point from the link pose was off by tens of
        # millimetres. Index the spheres directly instead.
        self.body_index = {name.rsplit("/", 1)[-1]: index
                           for index, name in enumerate(robot.body_names)}
        for entry in self.pads.values():
            absent = [name for name in entry["names"]
                      if name not in self.body_index]
            if absent:
                raise SystemExit(
                    f"{side}: {len(absent)} tactile bodies missing from the "
                    f"articulation, e.g. {absent[:3]}")
            entry["body_ids"] = np.asarray(
                [self.body_index[name] for name in entry["names"]], dtype=int)

        self.forces = {}
        self.pressure_kpa = {}
        # World-space sphere-surface points and outward normals, refreshed in
        # update() for the 3D force arrows. Keyed by pad, in the same order as
        # forces[pad]. Read from the sphere rigid bodies via body_ids, not
        # reconstructed from the link pose.
        self.point_positions = {}
        self.point_normals = {}
        self._vector_sum = np.zeros(3)
        if verbose:
            counts = ", ".join(f"{pad}:{len(entry['names'])}"
                               for pad, entry in sorted(self.pads.items()))
            print(f"[tactile] {side}: "
                  f"{sum(len(e['names']) for e in self.pads.values())} points on "
                  f"{len(self.pads)} pads")
            print(f"[tactile] {counts}")

    @staticmethod
    def _remap_table(pad):
        key = _remap_key(pad)
        if key is None:
            return None
        part, index = key
        table = (TACTILE_REMAP.get(part) or {}).get(index)
        return None if table is None else np.asarray(table)

    # ------------------------------------------------------------------ #
    def update(self, robot):
        """Read the sensor after a physics step and refresh forces and maps."""
        raw = self.sensor.data.net_forces_w
        if raw is None:
            raise SystemExit("contact sensor produced no data")
        force = raw.torch[0].cpu().numpy()
        body_pos = robot.data.body_pos_w.torch[0].cpu().numpy()
        body_quat = robot.data.body_quat_w.torch[0].cpu().numpy()
        self._vector_sum = np.zeros(3)

        for pad, entry in self.pads.items():
            ids = entry["body_ids"]
            selected = force[entry["indices"]]
            # The sphere's own +z is its outward normal, in world coordinates.
            # A sphere is perfectly symmetric, so its rigid-body frame has no
            # centre-of-mass offset and this is exact (unlike the link frame).
            # Third column of the (w,x,y,z) rotation matrix, vectorised over the
            # pad's points so a frame does not build hundreds of 3x3 matrices.
            w, x, y, z = body_quat[ids].T
            normals = np.stack([2 * (x * z + w * y),
                                2 * (y * z - w * x),
                                1 - 2 * (x * x + y * y)], axis=-1)
            # The force is applied at the sphere surface, not its centre.
            self.point_positions[pad] = body_pos[ids] + POINT_RADIUS * normals
            self.point_normals[pad] = normals
            # Absolute value: the contact force points into the hand, the normal
            # points out of it, and only the magnitude is meaningful.
            magnitude = np.abs(np.sum(selected * normals, axis=1))
            # Sensor response: nothing below the noise floor, full scale above.
            magnitude[magnitude < FORCE_MIN] = 0.0
            np.minimum(magnitude, FORCE_MAX, out=magnitude)
            self.forces[pad] = magnitude
            # Kept separately: the vector sum over the whole hand is the load the
            # sensor is really carrying, so it can be checked against gravity.
            self._vector_sum += selected.sum(axis=0)

        for pad, entry in self.pads.items():
            table = entry["map"]
            if table is None:
                continue
            pressure = self.forces[pad] / entry["areas"] / 1000.0  # kPa
            pressure[pressure < PRESSURE_MIN_KPA] = 0.0
            np.minimum(pressure, PRESSURE_MAX_KPA, out=pressure)
            image = np.full(table.shape, np.nan)
            # A cell holds a point number; subsampled or absent points leave NaN.
            for number, value in zip(entry["numbers"], pressure):
                image[table == number] = value
            self.pressure_kpa[pad] = image
        return self.forces

    # ------------------------------------------------------------------ #
    def total_force(self):
        """Sum of the per-point magnitudes; doubles when a pad is squeezed."""
        return float(sum(float(np.sum(forces)) for forces in self.forces.values()))

    def net_force(self):
        """Magnitude of the summed force vectors, i.e. the load actually carried."""
        return float(np.linalg.norm(self._vector_sum))

    def peak_kpa(self):
        if not self.pressure_kpa:
            return 0.0
        return float(np.nanmax([np.nanmax(image) if np.any(~np.isnan(image)) else 0.0
                                for image in self.pressure_kpa.values()]))

    def active_points(self, threshold=0.0):
        """Points with a reading; the forces are already clipped to their range,
        so anything below the noise floor is exactly zero."""
        return int(sum(int(np.sum(forces > threshold)) for forces in self.forces.values()))
