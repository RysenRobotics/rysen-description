"""Encode MuJoCo tactile contact forces in the ApexHand feedback format."""

from collections import defaultdict
import re
import struct

import mujoco
import numpy as np


FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
THUMB_TIP_ROW_ENDS = (2, 7, 14, 21, 30, 39, 48, 57)
FINGER_MIDDLE_CHANNELS = (12, 13, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 0, 1)
RAW_PA_PER_UNIT = 20.0
MAX_RAW_PRESSURE = 65534
SHEAR_ADC_BASE = 2048
SHEAR_FULL_SCALE_N = 5.0
SHEAR_SENSITIVITY = 0.05
TACTILE_SAMPLE_PERIOD_S = 0.01
TACTILE_RECORD = struct.Struct('<HH')


def tactile_channel(side, sensor, point):
    # Translate tactile point indices through the image coordinates in tactile_image_mapper.cpp.
    if sensor == 1:
        row_start = 0
        for row_end in THUMB_TIP_ROW_ENDS:
            if point <= row_end:
                return row_start + row_end - point
            row_start = row_end + 1
        raise ValueError(f'Invalid thumb tip taxel: {point}')
    if sensor == 6:
        return point + 1
    if 7 <= sensor <= 10:
        return FINGER_MIDDLE_CHANNELS[point]
    if sensor == 11:
        row, col = divmod(point, 8)
        return 8 + 8 * row + (7 - col if side == 'left' else col)
    if sensor >= 12:
        row, col = divmod(point, 6)
        return 6 * row + 5 - col
    return point


class TactileFeedback:
    def __init__(self, model, side):
        self.model = model
        self.taxels = {}
        self._next_sample_time = 0.0
        self._phase = 0
        self._payloads = (b'', b'')
        groups = defaultdict(list)
        pattern = re.compile(rf'{side}_(palm|thumb|index|middle|ring|pinky)_tactile([0-2]?)_point_(\d+)$')
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            match = pattern.fullmatch(name or '')
            if match is None:
                continue
            finger, segment, point = match.groups()
            sensor = 0 if finger == 'palm' else 1 + FINGERS.index(finger) + 5 * (2 - int(segment))
            groups[sensor].append((geom_id, tactile_channel(side, sensor, int(point))))

        for sensor, points in groups.items():
            geom_ids = [geom_id for geom_id, _ in points]
            # Match the tactile model's nearest-neighbour effective area.
            surface_points = model.geom_pos[geom_ids].copy()
            rotation = np.empty(9)
            for index, geom in enumerate(geom_ids):
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                surface_points[index] += model.geom_size[geom, 0] * rotation.reshape(3, 3)[:, 2]
            deltas = surface_points[:, None] - surface_points[None, :]
            squared_distance = np.sum(deltas ** 2, axis=-1)
            np.fill_diagonal(squared_distance, np.inf)
            areas = squared_distance.min(axis=1)
            if not np.all(np.isfinite(areas) & (areas > 0)):
                raise ValueError(f"传感区 {side}:{sensor} 的触点间距无效")
            for (geom_id, channel), area in zip(points, areas):
                self.taxels[geom_id] = (sensor, channel, area)

    def update(self, data):
        if not self.taxels or data.time < self._next_sample_time:
            return
        self._next_sample_time = data.time + TACTILE_SAMPLE_PERIOD_S
        pressure = dict.fromkeys(self.taxels, 0.0)
        shear = np.zeros((5, 2))
        wrench = np.empty(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.geom1 not in self.taxels and contact.geom2 not in self.taxels:
                continue
            mujoco.mj_contactForce(self.model, data, index, wrench)
            force_world = contact.frame.reshape(3, 3).T @ wrench[:3]
            for geom, sign in ((contact.geom1, -1), (contact.geom2, 1)):
                taxel = self.taxels.get(geom)
                if taxel is None:
                    continue
                sensor, _, area = taxel
                force_local = sign * (data.geom_xmat[geom].reshape(3, 3).T @ force_world)
                pressure[geom] += abs(force_local[2]) / area
                if 1 <= sensor <= 5:
                    shear[sensor - 1] += force_local[:2]

        shear_records = bytearray()
        for sensor, force in enumerate(shear, start=1):
            x, y = np.clip(force / SHEAR_FULL_SCALE_N, -1, 1)
            adc_span = 2 * SHEAR_ADC_BASE * SHEAR_SENSITIVITY
            values = (SHEAR_ADC_BASE + x * adc_span, SHEAR_ADC_BASE,
                      SHEAR_ADC_BASE - y * adc_span, SHEAR_ADC_BASE)
            for channel, value in enumerate(values):
                shear_records.extend(TACTILE_RECORD.pack(sensor | (channel << 4) | (1 << 14), round(value)))

        # 619 taxels do not fit in one 1500 byte payload. Alternate halves; always include shear.
        phases = [bytearray(shear_records), bytearray(shear_records)]
        for index, (geom, (sensor, channel, _)) in enumerate(self.taxels.items()):
            raw = min(MAX_RAW_PRESSURE, round(pressure[geom] / RAW_PA_PER_UNIT))
            phases[index % 2].extend(TACTILE_RECORD.pack(sensor | (channel << 4), raw))
        self._payloads = tuple(bytes(phase) for phase in phases)

    def next_payload(self):
        payload = self._payloads[self._phase]
        self._phase ^= 1
        return payload
