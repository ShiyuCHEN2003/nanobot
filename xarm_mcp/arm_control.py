"""
右臂运动控制（Arm 类）。

用法：
    from xarm_mcp.arm_control import Arm
    arm = Arm("can2")           # 初始化并建立重力补偿模型
    arm.move_joints(q7_target)  # 自动重力补偿插值运动
    arm.gripper_open()
    arm.gripper_close()
    q = arm.read_q7()
"""

import os
import time

import numpy as np

from xarm_mcp.xarm_controller.gravity_comp import ArmDynamics
from xarm_mcp.xarm_controller.XArmcontroller import (
    XArmController,
    MITParam,
    ARM_7DOF_JOINT_LIMITS,
    ARM_8DOF_LIMITS,
    ARM_8DOF_SLAVE_IDS,
    ARM_8DOF_MASTER_IDS,
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
)

CANPORT = "can2"
_HERE   = os.path.dirname(os.path.abspath(__file__))
_URDF   = os.path.join(_HERE, "xarm_controller", "xarm_bimanual.urdf")

# ── 各关节 PD 增益 ──
# 由 pid_sweep_test.py 扫描得出（KP_scale=1.50, KD_scale=2.00，综合得分最优）
KP = [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0]
KD = [ 1.2,  1.2,  1.2,  1.2,  0.3,  0.2,  0.3]
JOINT_LIMITS = ARM_7DOF_JOINT_LIMITS

GRIPPER_KP = 50.0
GRIPPER_KD = 1.0


class Arm:
    """右臂控制封装：XArmController + 重力补偿，canport 在构造时绑定。"""

    def __init__(self, canport: str = CANPORT, use_gravity: bool = True,
                 urdf: str = _URDF, arm_prefix: str = "xarm_right_"):
        self.canport   = canport
        self._ctrl     = XArmController(
            {canport: ARM_8DOF_LIMITS},
            slave_ids=ARM_8DOF_SLAVE_IDS,
            master_ids=ARM_8DOF_MASTER_IDS,
        )
        self._ctrl.enable_all()
        time.sleep(0.1)
        self._ctrl.recv_all(canport, timeout=0.05)
        self._dynamics = ArmDynamics(urdf, arm_prefix) if use_gravity else None

    # ── 状态读取 ──────────────────────────────────────────────────────────────

    def read_q7(self) -> np.ndarray:
        """读取 7 个关节当前角度 (rad)。"""
        return np.array(self._ctrl.get_positions(self.canport)[:7])

    # ── 运动控制 ──────────────────────────────────────────────────────────────

    def move_joints(self, q7_target: np.ndarray,
                    duration: float = 3.0, n_steps: int = 300):
        """从当前位置线性插值运动到 q7_target (rad)，自动重力补偿。"""
        self._ctrl.recv_all(self.canport, timeout=0.05)
        all_pos = self._ctrl.get_positions(self.canport)
        q0  = np.array(all_pos[:7])
        q1  = self._clip_q7(q7_target)
        g_q = all_pos[7]

        dt = duration / n_steps
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q      = q0 + (q1 - q0) * alpha
            tau_ff = self._dynamics.get_gravity(q) if self._dynamics is not None else None
            self._ctrl.mit_control_all(self.canport, self._arm_params(q, g_q, tau_ff))
            time.sleep(dt)
            self._ctrl.recv_all(self.canport, timeout=0.005)

    def gripper_open(self):
        """张开夹爪。"""
        self._gripper_move(GRIPPER_OPEN, duration=1.0)

    def gripper_close(self):
        """闭合夹爪（轻力夹取）。"""
        self._gripper_move(GRIPPER_CLOSE, duration=1.0)

    # ── 私有辅助 ─────────────────────────────────────────────────────────────

    def _clip_q7(self, q7: np.ndarray) -> np.ndarray:
        q = q7.copy()
        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            q[i] = np.clip(q[i], lo, hi)
        return q

    def _arm_params(self, q7: np.ndarray, gripper_q: float,
                    tau_ff: np.ndarray = None) -> list:
        params = [
            MITParam(kp=KP[j], kd=KD[j], q=float(q7[j]),
                     tau=float(tau_ff[j]) if tau_ff is not None else 0.0)
            for j in range(7)
        ]
        params.append(MITParam(kp=GRIPPER_KP, kd=GRIPPER_KD, q=float(gripper_q)))
        return params

    def _gripper_move(self, target_q: float,
                      duration: float = 1.0, n_steps: int = 100):
        self._ctrl.recv_all(self.canport, timeout=0.05)
        g0 = self._ctrl.get_gripper_position(self.canport)
        dt = duration / n_steps
        for i in range(n_steps + 1):
            alpha = i / n_steps
            gq = g0 + (target_q - g0) * alpha
            self._ctrl.gripper_position_control(self.canport, gq)
            self._ctrl.recv_all(self.canport, timeout=0.005)
            time.sleep(dt)
