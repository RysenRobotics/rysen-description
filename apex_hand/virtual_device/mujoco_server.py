import argparse
from collections import deque
from contextlib import ExitStack, contextmanager
import sys
import threading
import time
import struct
import math
import json
from pathlib import Path
import mujoco

from device_network import DeviceNetwork
from motor_model import DEFAULT_VELOCITY_LIMIT_DEG_S, FingerDrive
from mujoco_model import PALM_QUATERNION_WXYZ, build_model
from tactile_feedback import FINGERS, TactileFeedback


ACTIVE_JOINT_SUFFIXES = (
    "thumb_j0", "thumb_j1", "thumb_j2", "thumb_j3",
    "index_j0", "index_j1", "index_j2",
    "middle_j0", "middle_j1", "middle_j2",
    "ring_j0", "ring_j1", "ring_j2",
    "pinky_j0", "pinky_j1", "pinky_j2",
)
PASSIVE_COUPLED_JOINT_SUFFIXES = (
    "thumb_j4", "index_j3", "middle_j3", "ring_j3", "pinky_j3",
)


@contextmanager
def passive_viewer(model, data):
    from mujoco import viewer as mujoco_viewer

    previous_threads = set(threading.enumerate())
    viewer = mujoco_viewer.launch_passive(model, data)
    # MuJoCo 3.10 的 close() 只请求关闭，GLFW 退出前需要等待渲染线程结束。
    render_target = getattr(mujoco_viewer, "_launch_internal", None)
    render_threads = [
        thread for thread in threading.enumerate()
        if thread not in previous_threads and render_target is not None
        and getattr(thread, "_target", None) is render_target
    ]
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in render_threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                print("MuJoCo 渲染线程未在 3 秒内退出。", file=sys.stderr, flush=True)


def _merged_physics_params(defaults, overrides, joint_suffix, full_joint_name):
    params = dict(defaults)
    params.update(overrides.get(joint_suffix, {}))
    params.update(overrides.get(full_joint_name, {}))
    return params


def _finite_float(params, field_name, owner, *, positive=False, nonnegative=False):
    if field_name not in params:
        raise ValueError(f"{owner} 缺少物理参数 {field_name}")
    value = float(params[field_name])
    if not math.isfinite(value):
        raise ValueError(f"{owner}.{field_name} 必须是有限数值")
    if positive and value <= 0.0:
        raise ValueError(f"{owner}.{field_name} 必须大于0")
    if nonnegative and value < 0.0:
        raise ValueError(f"{owner}.{field_name} 不能为负数")
    return value


def apply_physics_config(model, config):
    """将JSON中的逐关节动力学覆盖应用到已编译的MuJoCo模型。"""
    joint_defaults = config["joint_defaults"]
    actuator_defaults = config["actuator_defaults"]
    joint_overrides = config.get("joint_overrides", {})
    actuator_overrides = config.get("actuator_overrides", {})

    detected_sides = [
        side
        for side in ("left", "right")
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_thumb_j0"
        ) >= 0
    ]
    for side in detected_sides:
        for joint_suffix in (
            ACTIVE_JOINT_SUFFIXES + PASSIVE_COUPLED_JOINT_SUFFIXES
        ):
            full_joint_name = f"{side}_{joint_suffix}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, full_joint_name
            )
            if joint_id < 0:
                raise ValueError(f"物理配置对应的关节不存在: {full_joint_name}")
            group_name = (
                "active"
                if joint_suffix in ACTIVE_JOINT_SUFFIXES
                else "passive_coupled"
            )
            params = _merged_physics_params(
                joint_defaults[group_name],
                joint_overrides,
                joint_suffix,
                full_joint_name,
            )
            owner = f"joint:{full_joint_name}"
            dof_id = int(model.jnt_dofadr[joint_id])
            model.dof_armature[dof_id] = _finite_float(
                params, "armature", owner, nonnegative=True
            )
            model.dof_damping[dof_id] = _finite_float(
                params, "damping", owner, nonnegative=True
            )
            model.dof_frictionloss[dof_id] = _finite_float(
                params, "frictionloss", owner, nonnegative=True
            )

        for joint_suffix in ACTIVE_JOINT_SUFFIXES:
            full_joint_name = f"{side}_{joint_suffix}"
            actuator_name = f"act_{full_joint_name}"
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise ValueError(f"物理配置对应的执行器不存在: {actuator_name}")
            params = _merged_physics_params(
                actuator_defaults,
                actuator_overrides,
                joint_suffix,
                full_joint_name,
            )
            owner = f"actuator:{actuator_name}"
            kp = _finite_float(params, "kp", owner, positive=True)
            kv = _finite_float(params, "kv", owner, nonnegative=True)
            force_range = params.get("force_range")
            if not isinstance(force_range, list) or len(force_range) != 2:
                raise ValueError(f"{owner}.force_range 必须是两个数的数组")
            force_min = float(force_range[0])
            force_max = float(force_range[1])
            if (
                not math.isfinite(force_min)
                or not math.isfinite(force_max)
                or force_min >= force_max
            ):
                raise ValueError(f"{owner}.force_range 无效: {force_range}")

            # MuJoCo position actuator:
            # force = kp * ctrl - kp * qpos - kv * qvel.
            model.actuator_gainprm[actuator_id, 0] = kp
            model.actuator_biasprm[actuator_id, 1] = -kp
            model.actuator_biasprm[actuator_id, 2] = -kv
            model.actuator_forcerange[actuator_id] = [force_min, force_max]

    # armature等质量相关字段会参与MuJoCo的派生惯量和约束权重计算。
    # JSON覆盖发生在XML编译之后，必须显式重算常量，才能与将同一配置
    # 直接烘焙进XML后的模型保持一致。
    constant_data = mujoco.MjData(model)
    mujoco.mj_setConst(model, constant_data)


# ================= 协议常数定义 =================
MAGIC0 = 0xAA
MAGIC1 = 0x55
VERSION = 0x03
SDK_ID = 0x02
HAND_ID = 0xF0  # 灵巧手硬件系列的通用标识符，不可修改。

CMD_00_CONNECT = 0x00
CMD_0F_RESET = 0x0F
CMD_C0_IP_ADDRESS_SET = 0xC0
CMD_CA_CTRL = 0xCA
CMD_DD_FEEDBACK = 0xDD

CTRL_CMD_IDLE = 0x01
CTRL_CMD_RUN_POSITION = 0x07
CTRL_CMD_RUN_CURRENT = 0xCC

END0 = 0xDE
END1 = 0xAD

# ================= 数据帧解包/打包格式 =================
THUMB_CTRL_STRUCT = struct.Struct("<B4f4f4fBB")
FINGER_CTRL_STRUCT = struct.Struct("<B3f3f3fBB")
CONTROL_PAYLOAD_SIZE = THUMB_CTRL_STRUCT.size + FINGER_CTRL_STRUCT.size * 4

FINGER_JOINT_RANGES = (
    range(0, 4),
    range(4, 7),
    range(7, 10),
    range(10, 13),
    range(13, 16),
)

THUMB_FEEDBACK = struct.Struct("<BBBf4f4f5f4h4bQ")
FINGER_FEEDBACK = struct.Struct("<BBBf3f3f4f3h3bQ")

def compute_checksum(data_bytes):
    return sum(data_bytes) & 0xFFFF

def build_frame(cmd, payload):
    # 恢复使用全局的 HAND_ID
    header_tmp = struct.pack("< B B H B B B B B B H",
                             MAGIC0, MAGIC1, 0, VERSION, HAND_ID, SDK_ID, cmd, 0, 0, len(payload))
    data_to_sum = header_tmp[4:] + payload
    checksum = compute_checksum(data_to_sum)
    header = struct.pack("< B B H B B B B B B H",
                         MAGIC0, MAGIC1, checksum, VERSION, HAND_ID, SDK_ID, cmd, 0, 0, len(payload))
    end_bytes = struct.pack("< B B", END0, END1)
    return header + payload + end_bytes

def parse_control_frame(data):
    payload_data = data[12:-2]
    if len(payload_data) != CONTROL_PAYLOAD_SIZE:
        raise ValueError(
            f"控制帧 payload 长度错误: {len(payload_data)}，期望 {CONTROL_PAYLOAD_SIZE}"
        )

    commands = []
    offset = 0
    for codec, joint_count in ((THUMB_CTRL_STRUCT, 4),) + (
        (FINGER_CTRL_STRUCT, 3),
    ) * 4:
        values = codec.unpack_from(payload_data, offset)
        commands.append(
            {
                "cmd": values[0],
                "positions_deg": list(values[1 : 1 + joint_count]),
                "velocities_deg_s": list(
                    values[1 + joint_count : 1 + joint_count * 2]
                ),
                "max_motor_torque_pct": values[-2],
            }
        )
        offset += codec.size
    return commands


def pop_next_frame(buffer):
    """从 TCP 字节流取出一帧；校验行为与固件 AcpMsgCheck 保持一致。"""
    while True:
        magic_index = buffer.find(bytes((MAGIC0, MAGIC1)))
        if magic_index < 0:
            if buffer and buffer[-1] == MAGIC0:
                del buffer[:-1]
            else:
                buffer.clear()
            return None
        if magic_index > 0:
            del buffer[:magic_index]
        if len(buffer) < 14:
            return None

        payload_len = struct.unpack_from("<H", buffer, 10)[0]
        total_len = 12 + payload_len + 2
        if total_len > 4096:
            del buffer[0]
            continue
        if len(buffer) < total_len:
            return None

        frame = bytes(buffer[:total_len])
        expected_checksum = struct.unpack_from("<H", frame, 2)[0]
        checksum_ok = compute_checksum(frame[4:-2]) == expected_checksum
        end_ok = frame[-2:] == bytes((END0, END1))
        if not checksum_ok or not end_ok:
            del buffer[0]
            continue

        del buffer[:total_len]
        return frame


def receive_frames(client):
    buffer = bytearray()
    while (data := client.recv(4096)):
        buffer.extend(data)
        while (frame := pop_next_frame(buffer)) is not None:
            yield frame


class HandState:
    def __init__(self, side, is_single):
        self.side = side
        self.side_code = 0 if side == "left" else 1

        self.ip = "127.0.0.1" if side == "left" else "127.0.0.2"
        self.ctrl_port = 5856
        self.data_port = 5857

        y_offset = 0.0 if is_single else (0.2 if side == "left" else -0.2)
        self.mocap_pos = [0.0, y_offset, 0.5]
        self.mocap_quat = PALM_QUATERNION_WXYZ

        self.joint_addrs = []
        self.joint_dof_addrs = []
        self.passive_joint_addrs = []
        self.actuator_ids = []
        self.mocap_idx = -1
        self.palm_qpos_addr = -1
        self.palm_dof_addr = -1

        self.finger_modes = [CTRL_CMD_IDLE] * 5
        self.finger_torque_limits = [100] * 5
        self.finger_errors = [0] * 5
        self.drives = []
        self.control_client = None
        self.previous_command_targets = []
        self.command_targets = []
        self.firmware_targets = []
        self.interpolation_steps_remaining = []
        self.feedback_velocity_deg_s = []
        self.pending_position_commands = []
        self.feedback_position_history = []
        self.feedback_velocity_history = []


class VirtualHardwareServer:
    def __init__(self, side, config_path):
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = json.load(file)
        if config.get("schema_version") != 1:
            raise ValueError(f"不支持的物理参数配置版本: {config.get('schema_version')}")
        server_config = config["server"]
        self.command_deadband_rad = float(server_config["command_deadband_rad"])
        self.render_hz = float(server_config["render_hz"])
        self.max_catch_up_steps = int(server_config["max_catch_up_steps"])
        self.firmware_target_max_delta_rad = math.radians(
            float(server_config.get("firmware_target_max_delta_deg", 6.0))
        )
        self.firmware_target_interpolation_steps = int(
            server_config.get("firmware_target_interpolation_steps", 10)
        )
        self.feedback_velocity_filter_alpha = float(
            server_config.get("feedback_velocity_filter_alpha", 0.2)
        )
        self.apply_four_finger_firmware_filter = bool(
            server_config.get("apply_four_finger_firmware_filter", True)
        )
        self.apply_thumb_firmware_filter = bool(
            server_config.get("apply_thumb_firmware_filter", False)
        )
        self.thumb_command_transport_delay_ms = self._load_thumb_delay_config(
            server_config, "thumb_command_transport_delay_ms"
        )
        self.thumb_feedback_transport_delay_ms = self._load_thumb_delay_config(
            server_config, "thumb_feedback_transport_delay_ms"
        )
        if self.command_deadband_rad < 0:
            raise ValueError("command_deadband_rad 不能为负数")
        if self.render_hz <= 0:
            raise ValueError("render_hz 必须大于 0")
        if self.max_catch_up_steps < 1:
            raise ValueError("max_catch_up_steps 必须至少为 1")
        if self.firmware_target_max_delta_rad <= 0:
            raise ValueError("firmware_target_max_delta_deg 必须大于 0")
        if self.firmware_target_interpolation_steps < 1:
            raise ValueError("firmware_target_interpolation_steps 必须至少为 1")
        if not 0 < self.feedback_velocity_filter_alpha <= 1:
            raise ValueError("feedback_velocity_filter_alpha 必须位于 (0, 1]")

        self.model = build_model(side)
        apply_physics_config(self.model, config)
        self.data = mujoco.MjData(self.model)
        physics_period_ms = float(self.model.opt.timestep) * 1000.0
        if physics_period_ms <= 0.0:
            raise ValueError("MuJoCo timestep 必须大于0")
        self.thumb_command_transport_delay_steps = [
            int(round(delay_ms / physics_period_ms))
            for delay_ms in self.thumb_command_transport_delay_ms
        ]
        self.thumb_feedback_transport_delay_steps = [
            int(round(delay_ms / physics_period_ms))
            for delay_ms in self.thumb_feedback_transport_delay_ms
        ]
        self.simulation_step_index = 0
        self.lock = threading.Lock()

        self.hands = {}

        detected_sides = []
        for side in ["left", "right"]:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_thumb_j0") != -1:
                detected_sides.append(side)

        if not detected_sides:
            raise ValueError("[!] 警告: 未检测到左右手骨骼！")

        is_single = (len(detected_sides) == 1)

        for side in detected_sides:
            h = HandState(side, is_single)
            h.tactile = TactileFeedback(self.model, side)

            joint_names = [f"{side}_{suffix}" for suffix in ACTIVE_JOINT_SUFFIXES]

            missing_model_items = []
            for name in joint_names:
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if j_id == -1:
                    missing_model_items.append(f"joint:{name}")
                    continue
                h.joint_addrs.append(self.model.jnt_qposadr[j_id])
                h.joint_dof_addrs.append(self.model.jnt_dofadr[j_id])

                act_name = f"act_{name}"
                a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                if a_id == -1:
                    missing_model_items.append(f"actuator:{act_name}")
                    continue
                h.actuator_ids.append(a_id)
                self.data.ctrl[a_id] = 0

            if missing_model_items:
                missing_text = ", ".join(missing_model_items)
                raise ValueError(f"{side} 手模型不完整: {missing_text}")

            for finger, joints in enumerate(FINGER_JOINT_RANGES):
                passive_name = f"{side}_{PASSIVE_COUPLED_JOINT_SUFFIXES[finger]}"
                passive_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, passive_name)
                if passive_id < 0:
                    raise ValueError(f"手模型缺少被动关节: {passive_name}")
                h.passive_joint_addrs.append(int(self.model.jnt_qposadr[passive_id]))
                indices = slice(joints.start, joints.stop)
                h.drives.append(FingerDrive(
                    self.model, side, finger, h.actuator_ids[indices], h.joint_dof_addrs[indices],
                    self.model.jnt_dofadr[passive_id],
                ))

            mocap_name = f"{side}_palm_mocap"
            mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
            if mocap_id != -1:
                h.mocap_idx = self.model.body_mocapid[mocap_id]
            if h.mocap_idx < 0:
                raise ValueError(f"{side} 手缺少有效的 Mocap body: {mocap_name}")

            palm_freejoint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_palm_freejoint",
            )
            if palm_freejoint_id < 0:
                raise ValueError(f"{side} 手缺少动态手掌 freejoint")
            if (
                self.model.jnt_type[palm_freejoint_id]
                != mujoco.mjtJoint.mjJNT_FREE
            ):
                raise ValueError(f"{side}_palm_freejoint 不是free joint")
            h.palm_qpos_addr = int(
                self.model.jnt_qposadr[palm_freejoint_id]
            )
            h.palm_dof_addr = int(
                self.model.jnt_dofadr[palm_freejoint_id]
            )

            # mocap目标和动态手掌必须从同一位姿开始，否则首个物理步会由
            # stiff weld产生一次无意义的大冲击。
            self.data.mocap_pos[h.mocap_idx] = h.mocap_pos
            self.data.mocap_quat[h.mocap_idx] = h.mocap_quat
            self.data.qpos[h.palm_qpos_addr:h.palm_qpos_addr + 3] = (
                h.mocap_pos
            )
            self.data.qpos[
                h.palm_qpos_addr + 3:h.palm_qpos_addr + 7
            ] = h.mocap_quat
            self.data.qvel[h.palm_dof_addr:h.palm_dof_addr + 6] = 0.0

            initial_targets = [self.data.qpos[addr] for addr in h.joint_addrs]
            h.previous_command_targets = list(initial_targets)
            h.command_targets = list(initial_targets)
            h.firmware_targets = list(initial_targets)
            h.interpolation_steps_remaining = [0] * len(initial_targets)
            h.feedback_velocity_deg_s = [0.0] * len(initial_targets)
            h.pending_position_commands = [deque() for _ in initial_targets]
            feedback_delays = self.thumb_feedback_transport_delay_steps + [0] * 12
            h.feedback_position_history = [
                deque(
                    [initial_targets[index]] * (feedback_delays[index] + 1),
                    maxlen=feedback_delays[index] + 1,
                )
                for index in range(len(initial_targets))
            ]
            h.feedback_velocity_history = [
                deque(
                    [0.0] * (feedback_delays[index] + 1),
                    maxlen=feedback_delays[index] + 1,
                )
                for index in range(len(initial_targets))
            ]

            self.hands[side] = h
            print(f"[*] 成功加载 {side.upper()} 手模型 | IP: {h.ip} | 端口: {h.ctrl_port}/{h.data_port}")
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _load_thumb_delay_config(server_config, field_name):
        values = server_config.get(field_name, [0.0, 0.0, 0.0, 0.0])
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"{field_name} 必须是包含4个数的数组")
        delays = [float(value) for value in values]
        if not all(math.isfinite(value) and value >= 0.0 for value in delays):
            raise ValueError(f"{field_name} 必须全部是非负有限数值")
        return delays

    def _uses_firmware_target_filter(self, finger_index, source_module):
        if source_module != SDK_ID:
            return False
        if finger_index == 0:
            return self.apply_thumb_firmware_filter
        return self.apply_four_finger_firmware_filter

    def _target_interpolation_steps(self, finger_index):
        if finger_index == 0:
            enabled = self.apply_thumb_firmware_filter
        else:
            enabled = self.apply_four_finger_firmware_filter
        return self.firmware_target_interpolation_steps if enabled else 1

    def _apply_position_target_locked(
        self, hand_info, finger_index, joint_index, target_rad, source_module
    ):
        if self._uses_firmware_target_filter(finger_index, source_module):
            current_target = hand_info.firmware_targets[joint_index]
            target_rad = max(
                current_target - self.firmware_target_max_delta_rad,
                min(current_target + self.firmware_target_max_delta_rad, target_rad),
            )
            interpolation_steps = self.firmware_target_interpolation_steps
        else:
            interpolation_steps = 1

        if (
            abs(target_rad - hand_info.command_targets[joint_index])
            <= self.command_deadband_rad
        ):
            return False

        hand_info.previous_command_targets[joint_index] = (
            hand_info.command_targets[joint_index]
        )
        hand_info.command_targets[joint_index] = target_rad
        hand_info.interpolation_steps_remaining[joint_index] = interpolation_steps
        return True

    def _drain_delayed_commands_locked(self, hand_info):
        for joint_index, pending in enumerate(hand_info.pending_position_commands):
            while pending and pending[0][0] <= self.simulation_step_index:
                _, finger_index, target_rad, source_module = pending.popleft()
                if hand_info.finger_modes[finger_index] != CTRL_CMD_RUN_POSITION:
                    continue
                self._apply_position_target_locked(
                    hand_info,
                    finger_index,
                    joint_index,
                    target_rad,
                    source_module,
                )

    @staticmethod
    def _feedback_joint_state_locked(hand_info, joint_index):
        return (
            hand_info.feedback_position_history[joint_index][0],
            hand_info.feedback_velocity_history[joint_index][0],
        )

    def build_connect_ack_payload(self, hand_info):
        # 【核心修复】：利用序列号 (MAC地址) 区分左右手，防止 ROS 内部重叠！
        sn1 = 111 if hand_info.side == "left" else 444
        sn2 = 222 if hand_info.side == "left" else 555
        sn3 = 333 if hand_info.side == "left" else 666
        tactile_version = (1, 0, 0) if hand_info.tactile.taxels else (0, 0, 0)

        return struct.pack("< B B 3B 3B 3B 3B B 3I",
            0, 1, 1, 0, 0, 3, 2, 10, *tactile_version, *tactile_version,
            hand_info.side_code,
            sn1, sn2, sn3
        )

    def build_feedback_payload(self, hand_info):
        # 左手 side_code=0 -> bitfield = 0x0E | 0 = 0x0E (1110)
        # 右手 side_code=1 -> bitfield = 0x0E | 1 = 0x0F (1111)
        bitfield = 0x0E | hand_info.side_code
        payload = bytearray()
        with self.lock:
            for finger, joints in enumerate(FINGER_JOINT_RANGES):
                drive = hand_info.drives[finger]
                positions, velocities = zip(*(self._feedback_joint_state_locked(hand_info, j) for j in joints))
                codec = THUMB_FEEDBACK if finger == 0 else FINGER_FEEDBACK
                payload.extend(codec.pack(
                    bitfield, hand_info.finger_modes[finger], drive.torque_limit_pct,
                    min(drive.velocity_limit_deg_s),
                    *(math.degrees(position) for position in positions), *velocities,
                    *(1000.0 * drive.applied_torque_nm),
                    *(max(-32768, min(32767, round(value))) for value in drive.applied_current_ma),
                    *(max(-128, min(127, round(value))) for value in drive.temperature_c),
                    hand_info.finger_errors[finger],
                ))
            payload.extend(hand_info.tactile.next_payload())
        return bytes(payload)

    def set_simulated_fault(self, side, finger, mask):
        """Set a firmware fault flag for a simulated diagnostic scenario."""
        if side not in self.hands or finger not in FINGERS or not 0 <= mask < 2**64:
            raise ValueError("故障参数须指定场景中的手、手指和 64 位错误码")
        with self.lock:
            self.hands[side].finger_errors[FINGERS.index(finger)] = mask

    def apply_control_command(self, hand_info, finger_commands, source_module=SDK_ID, control_client=None):
        """接收 SDK 控制帧，并复刻已确认的四指固件目标预处理。"""
        if len(finger_commands) != len(FINGER_JOINT_RANGES):
            raise ValueError(
                f"手指控制块数量错误: {len(finger_commands)}，期望 {len(FINGER_JOINT_RANGES)}"
            )
        if any(not math.isfinite(value) for command in finger_commands
               for field in ("positions_deg", "velocities_deg_s") for value in command[field]):
            raise ValueError("控制帧包含无效位置、速度或电流目标")
        deg_to_rad = math.pi / 180.0
        updated_count = 0
        with self.lock:
            hand_info.control_client = control_client
            for finger_index, (command, joint_range) in enumerate(
                zip(finger_commands, FINGER_JOINT_RANGES)
            ):
                mode = int(command["cmd"])
                hand_info.finger_modes[finger_index] = mode
                drive = hand_info.drives[finger_index]
                hand_info.finger_torque_limits[finger_index] = min(100, command["max_motor_torque_pct"])
                if mode == CTRL_CMD_RUN_CURRENT:
                    drive.target_current_ma[:] = command["positions_deg"]
                elif mode == CTRL_CMD_RUN_POSITION:
                    drive.velocity_limit_deg_s[:] = [
                        abs(value) or DEFAULT_VELOCITY_LIMIT_DEG_S for value in command["velocities_deg_s"]
                    ]

                if mode != CTRL_CMD_RUN_POSITION:
                    for joint_index in joint_range:
                        current = self.data.qpos[hand_info.joint_addrs[joint_index]]
                        hand_info.pending_position_commands[joint_index].clear()
                        hand_info.previous_command_targets[joint_index] = current
                        hand_info.command_targets[joint_index] = current
                        hand_info.firmware_targets[joint_index] = current
                        hand_info.interpolation_steps_remaining[joint_index] = 0
                    continue

                for joint_index, target_position_deg in zip(
                    joint_range, command["positions_deg"]
                ):
                    target_rad = target_position_deg * deg_to_rad
                    delay_steps = (
                        self.thumb_command_transport_delay_steps[joint_index]
                        if finger_index == 0
                        else 0
                    )
                    if delay_steps > 0:
                        hand_info.pending_position_commands[joint_index].append(
                            (
                                self.simulation_step_index + delay_steps + 1,
                                finger_index,
                                target_rad,
                                source_module,
                            )
                        )
                        updated_count += 1
                    else:
                        updated_count += int(
                            self._apply_position_target_locked(
                                hand_info,
                                finger_index,
                                joint_index,
                                target_rad,
                                source_module,
                            )
                        )
        return updated_count

    def handle_control_stream(self, hand_info, client):
        try:
            for frame in receive_frames(client):
                if frame[7] == CMD_CA_CTRL:
                    self.apply_control_command(hand_info, parse_control_frame(frame), frame[5], client)
        finally:
            with self.lock:
                if hand_info.control_client is client:
                    hand_info.control_client = None
                    hand_info.finger_modes[:] = [CTRL_CMD_IDLE] * len(hand_info.drives)
                    for pending in hand_info.pending_position_commands:
                        pending.clear()

    def handle_data_stream(self, hand_info, client, network):
        for frame in receive_frames(client):
            command = frame[7]
            if command == CMD_00_CONNECT:
                client.sendall(build_frame(command, self.build_connect_ack_payload(hand_info)))
            elif command == CMD_DD_FEEDBACK:
                client.sendall(build_frame(command, self.build_feedback_payload(hand_info)))
            elif command == CMD_C0_IP_ADDRESS_SET:
                accepted = network.prepare_address(frame[12:-2])
                client.sendall(build_frame(command, b"\x00" if accepted else b"\x01"))
            elif command == CMD_0F_RESET:
                if len(frame[12:-2]) == 4:
                    delay_ms = struct.unpack_from("<I", frame, 12)[0]
                    client.sendall(build_frame(command, b"\x00"))
                    network.schedule_reset(delay_ms)
                else:
                    client.sendall(build_frame(command, b"\x01"))

    def reset_hand(self, hand):
        with self.lock:
            all_joint_addrs = hand.joint_addrs + hand.passive_joint_addrs
            self.data.qpos[all_joint_addrs] = self.model.qpos0[all_joint_addrs]
            self.data.qpos[hand.palm_qpos_addr:hand.palm_qpos_addr + 3] = hand.mocap_pos
            self.data.qpos[hand.palm_qpos_addr + 3:hand.palm_qpos_addr + 7] = hand.mocap_quat
            self.data.qvel[hand.palm_dof_addr:hand.palm_dof_addr + 6] = 0
            for drive in hand.drives:
                self.data.qvel[drive.dofs] = 0
                drive.configure_step(self.model, self.data, False, 100)
                drive.target_current_ma.fill(0)
                drive.applied_current_ma.fill(0)
                drive.applied_torque_nm.fill(0)
                drive.velocity_limit_deg_s.fill(DEFAULT_VELOCITY_LIMIT_DEG_S)
            initial_positions = self.data.qpos[hand.joint_addrs]
            self.data.ctrl[hand.actuator_ids] = initial_positions
            hand.finger_modes[:] = [CTRL_CMD_IDLE] * len(hand.drives)
            hand.finger_torque_limits[:] = [100] * len(hand.drives)
            hand.control_client = None
            for index, position in enumerate(initial_positions):
                hand.previous_command_targets[index] = position
                hand.command_targets[index] = position
                hand.firmware_targets[index] = position
                hand.pending_position_commands[index].clear()
                hand.interpolation_steps_remaining[index] = 0
                hand.feedback_velocity_deg_s[index] = 0.0
                position_history = hand.feedback_position_history[index]
                speed_history = hand.feedback_velocity_history[index]
                position_history.extend([float(position)] * position_history.maxlen)
                speed_history.extend([0.0] * speed_history.maxlen)
            mujoco.mj_forward(self.model, self.data)

    def step_simulation(self):
        """推进一个固定物理步；网络线程通过同一把锁访问 mjData。"""
        with self.lock:
            self.simulation_step_index += 1
            timestep = float(self.model.opt.timestep)

            for h in self.hands.values():
                self._drain_delayed_commands_locked(h)
                self.data.mocap_pos[h.mocap_idx] = h.mocap_pos
                self.data.mocap_quat[h.mocap_idx] = h.mocap_quat

                for finger_index, joint_range in enumerate(FINGER_JOINT_RANGES):
                    mode = h.finger_modes[finger_index]
                    drive = h.drives[finger_index]
                    drive.configure_step(
                        self.model, self.data, mode == CTRL_CMD_RUN_CURRENT,
                        h.finger_torque_limits[finger_index],
                    )
                    for joint_index in joint_range:
                        if mode == CTRL_CMD_RUN_POSITION:
                            remaining = h.interpolation_steps_remaining[joint_index]
                            if remaining > 0:
                                total_steps = self._target_interpolation_steps(
                                    finger_index
                                )
                                completed_steps = total_steps - remaining + 1
                                start = h.previous_command_targets[joint_index]
                                target = h.command_targets[joint_index]
                                h.firmware_targets[joint_index] = start + (
                                    target - start
                                ) * (completed_steps / total_steps)
                                h.interpolation_steps_remaining[joint_index] -= 1
                            actuator = h.actuator_ids[joint_index]
                            joint_offset = joint_index - joint_range.start
                            maximum_step = math.radians(drive.velocity_limit_deg_s[joint_offset]) * timestep
                            delta = h.firmware_targets[joint_index] - self.data.ctrl[actuator]
                            self.data.ctrl[actuator] += max(-maximum_step, min(maximum_step, delta))
                        else:
                            current = self.data.qpos[h.joint_addrs[joint_index]]
                            h.previous_command_targets[joint_index] = current
                            h.command_targets[joint_index] = current
                            h.firmware_targets[joint_index] = current
                            h.interpolation_steps_remaining[joint_index] = 0
                            self.data.ctrl[h.actuator_ids[joint_index]] = current

            mujoco.mj_step(self.model, self.data)

            rad_to_deg = 180.0 / math.pi
            alpha = self.feedback_velocity_filter_alpha
            for h in self.hands.values():
                h.tactile.update(self.data)
                for drive in h.drives:
                    drive.update_feedback(self.data, timestep)
                for joint_index, dof_addr in enumerate(h.joint_dof_addrs):
                    velocity_deg_s = self.data.qvel[dof_addr] * rad_to_deg
                    h.feedback_velocity_deg_s[joint_index] += alpha * (
                        velocity_deg_s - h.feedback_velocity_deg_s[joint_index]
                    )
                    position = self.data.qpos[h.joint_addrs[joint_index]]
                    h.feedback_position_history[joint_index].append(float(position))
                    h.feedback_velocity_history[joint_index].append(
                        h.feedback_velocity_deg_s[joint_index]
                    )

    def run_simulation(self, headless=False):
        with ExitStack() as resources:
            networks = []
            for hand in self.hands.values():
                try:
                    network = DeviceNetwork(
                        hand.ip, (hand.ctrl_port, hand.data_port),
                        lambda hand=hand: self.reset_hand(hand),
                    )
                except OSError as error:
                    raise OSError(f"无法监听 {hand.ip}:{hand.ctrl_port}/{hand.data_port}: {error}") from error
                resources.callback(network.close)
                networks.append((hand, network))

            viewer = None
            if not headless:
                viewer = resources.enter_context(passive_viewer(self.model, self.data))

            for hand, network in networks:
                network.start(
                    lambda client, hand=hand: self.handle_control_stream(hand, client),
                    lambda client, route, hand=hand: self.handle_data_stream(hand, client, route),
                )
            print("[*] 仿真服务就绪，可以连接 SDK", flush=True)
            self._run_loop(viewer)

    def _run_loop(self, viewer):
        physics_period = float(self.model.opt.timestep)
        render_period = 1.0 / self.render_hz
        next_physics_step = time.perf_counter()
        next_render = next_physics_step
        last_overrun_report = 0.0

        while viewer is None or viewer.is_running():
            now = time.perf_counter()
            step_count = 0
            while now >= next_physics_step and step_count < self.max_catch_up_steps:
                self.step_simulation()
                next_physics_step += physics_period
                step_count += 1
                now = time.perf_counter()

            if now >= next_physics_step and step_count == self.max_catch_up_steps:
                skipped_steps = int((now - next_physics_step) / physics_period) + 1
                next_physics_step = now + physics_period
                if now - last_overrun_report >= 1.0:
                    print(f"[!] 物理循环过载，丢弃 {skipped_steps} 个追赶步")
                    last_overrun_report = now

            sleep_until = next_physics_step
            if viewer is not None:
                if now >= next_render:
                    viewer.sync()
                    next_render = now + render_period
                sleep_until = min(sleep_until, next_render)

            sleep_seconds = sleep_until - time.perf_counter()
            if sleep_seconds > 0:
                time.sleep(min(sleep_seconds, physics_period))


def main():
    parser = argparse.ArgumentParser(description="ApexHand MuJoCo 虚拟样机")
    parser.add_argument(
        "--side",
        choices=("left", "right", "dual"),
        default="right",
        help="虚拟手场景，默认右手",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "hand_physics.json",
        help="本地物理参数 JSON",
    )
    parser.add_argument("--headless", action="store_true", help="无界面运行")
    args = parser.parse_args()

    try:
        VirtualHardwareServer(args.side, args.config).run_simulation(
            headless=args.headless
        )
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"启动或运行仿真失败: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
