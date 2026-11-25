"""MuJoCo 上的 SDK 1.6.0 电机传动和遥测模型。"""

import numpy as np


MAX_CURRENT_MA = 1500.0
TORQUE_NM_PER_MA = 0.4 / 1000.0
AMBIENT_C = 25.0
RATED_TEMPERATURE_RISE_C = 45.0
THERMAL_TIME_CONSTANT_S = 60.0
DEFAULT_VELOCITY_LIMIT_DEG_S = 500.0


def firmware_transmission(side, finger_index):
    """把固件 MotorId 顺序的电机力矩映射到 SDK 关节坐标。"""
    # 传动半径、左右手符号及电机顺序来自 SDK 的 joint_dynamics.cpp 和 apexhand_controller.cpp。
    if finger_index == 0:
        signs = np.array([-1, -1, 1, -1, -1] if side == 'left' else [1, 1, 1, 1, 1])
        matrix = [
            [-1, -1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -26.0 / 17.0, signs[3] * 8.5 / 3.6],
            [0, 0, 0, 6.0 / 3.6],
            [0, 0, 0, 2.55 / 3.6],
        ]
    else:
        signs = np.array([-1, -1, -1, -1] if side == 'left' else [1, -1, -1, -1])
        # 固件四指 M0/M1 的顺序与运动学模型相反。
        matrix = [
            [8.0 / 4.7, 8.0 / 4.7, 0],
            [-7.8 / 4.7, 10.0 / 4.7, 6.0 / 3.6],
            [0, 0, 6.4 / 3.6],
            [0, 0, 2.55 / 3.6],
        ]
    return signs[:, None] * np.asarray(matrix) * TORQUE_NM_PER_MA


class FingerDrive:
    def __init__(self, model, side, finger_index, actuator_ids, active_dofs, passive_dof):
        self.actuator_ids = np.asarray(actuator_ids, dtype=int)
        self.active_dofs = np.asarray(active_dofs, dtype=int)
        self.dofs = np.append(self.active_dofs, passive_dof)
        self.transmission = firmware_transmission(side, finger_index)
        self.inverse = np.linalg.inv(self.transmission[:-1])
        self.position_gain = model.actuator_gainprm[self.actuator_ids, :].copy()
        self.position_bias = model.actuator_biasprm[self.actuator_ids, :].copy()
        self.position_force_range = model.actuator_forcerange[self.actuator_ids].copy()
        self.current_mode = False
        self.torque_limit_pct = 100
        self.velocity_limit_deg_s = np.full(self.active_dofs.size, DEFAULT_VELOCITY_LIMIT_DEG_S)
        self.target_current_ma = np.zeros(self.active_dofs.size)
        self.applied_current_ma = np.zeros(self.active_dofs.size)
        self.applied_torque_nm = np.zeros(self.dofs.size)
        self.temperature_c = np.full(self.active_dofs.size, AMBIENT_C)

    def configure_step(self, model, data, current_mode, torque_limit_pct):
        if current_mode != self.current_mode:
            model.actuator_gainprm[self.actuator_ids] = 0 if current_mode else self.position_gain
            model.actuator_biasprm[self.actuator_ids] = 0 if current_mode else self.position_bias
            self.current_mode = current_mode
        if torque_limit_pct != self.torque_limit_pct:
            self.torque_limit_pct = torque_limit_pct
            model.actuator_forcerange[self.actuator_ids] = self.position_force_range * (torque_limit_pct / 100)
        if current_mode:
            current_limit = MAX_CURRENT_MA * (torque_limit_pct / 100)
            np.clip(self.target_current_ma, -current_limit, current_limit, out=self.applied_current_ma)
            self.applied_torque_nm[:] = self.transmission @ self.applied_current_ma
            data.qfrc_applied[self.dofs] = self.applied_torque_nm
        else:
            data.qfrc_applied[self.dofs] = 0

    def update_feedback(self, data, timestep):
        if not self.current_mode:
            self.applied_torque_nm[:-1] = data.qfrc_actuator[self.active_dofs]
            self.applied_current_ma[:] = self.inverse @ self.applied_torque_nm[:-1]
            self.applied_torque_nm[-1] = self.transmission[-1] @ self.applied_current_ma
        # 简化热模型用于给出随负载变化的模拟遥测，参数未经硬件标定。
        load = np.minimum(np.abs(self.applied_current_ma) / MAX_CURRENT_MA, 1.0)
        heating = RATED_TEMPERATURE_RISE_C * load ** 2
        self.temperature_c += (AMBIENT_C + heating - self.temperature_c) * (timestep / THERMAL_TIME_CONSTANT_S)
