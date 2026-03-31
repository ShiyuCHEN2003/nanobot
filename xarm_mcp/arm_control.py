"""
右臂运动控制工具模块（基于 XArmController）。

提供：
  - build_arm(canport)          → ctrl (XArmController)
  - read_q7(ctrl, canport)      → np.ndarray (7,) 当前关节角 rad
  - move_joints(ctrl, q7_target, duration, canport)  插值运动到目标关节角
  - gripper_open(ctrl, canport)
  - gripper_close(ctrl, canport)
"""

import os
import sys
import time

import numpy as np

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

# ── 各关节 PD 增益 ──
# 由 pid_sweep_test.py 扫描得出（KP_scale=1.50, KD_scale=2.00，综合得分最优）
# 原始值：KP=[300,300,150,150,40,40,30], KD=[2.5,2.5,2.5,2.5,0.8,0.8,0.8]
# KP = [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0]
# KD = [1.2, 1.2, 1.2, 1.2, 0.3, 0.2, 0.3]
KP = [300,300,150,150,40,40,30]
KD = [2.5,2.5,2.5,2.5,0.8,0.8,0.8]
# ── 关节限位 (rad)，从 XArmcontroller 统一维护 ──
JOINT_LIMITS = ARM_7DOF_JOINT_LIMITS

# ── 夹爪参数 ──
GRIPPER_KP = 50.0
GRIPPER_KD = 1.0


def build_arm(canport: str = CANPORT) -> XArmController:
    """初始化机械臂，返回 XArmController。"""
    ctrl = XArmController(
        {canport: ARM_8DOF_LIMITS},
        slave_ids=ARM_8DOF_SLAVE_IDS,
        master_ids=ARM_8DOF_MASTER_IDS,
    )
    ctrl.enable_all()
    time.sleep(0.1)
    ctrl.recv_all(canport, timeout=0.05)
    return ctrl


def read_q7(ctrl: XArmController, canport: str = CANPORT) -> np.ndarray:
    """读取 7 个关节当前角度 (rad)。"""
    return np.array(ctrl.get_positions(canport)[:7])


def _clip_q7(q7: np.ndarray) -> np.ndarray:
    """将目标关节角夹到限位内。"""
    q = q7.copy()
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        q[i] = np.clip(q[i], lo, hi)
    return q


def _arm_params(q7: np.ndarray, gripper_q: float) -> list:
    """构建 8 个 MITParam：7 个臂关节 + 1 个夹爪保持。"""
    params = [
        MITParam(kp=KP[j], kd=KD[j], q=float(q7[j]))
        for j in range(7)
    ]
    params.append(MITParam(kp=GRIPPER_KP, kd=GRIPPER_KD, q=float(gripper_q)))
    return params


def move_joints(ctrl: XArmController, q7_target: np.ndarray,
                duration: float = 3.0, n_steps: int = 600,
                canport: str = CANPORT):
    """从当前位置线性插值运动到 q7_target (rad)。"""
    ctrl.recv_all(canport, timeout=0.05)
    all_pos = ctrl.get_positions(canport)
    q0  = np.array(all_pos[:7])
    q1  = _clip_q7(q7_target)
    g_q = all_pos[7]   # 保持夹爪当前位置

    dt = duration / n_steps
    for i in range(n_steps + 1):
        alpha = i / n_steps
        q = q0 + (q1 - q0) * alpha
        ctrl.mit_control_all(canport, _arm_params(q, g_q))
        
        time.sleep(dt)
        ctrl.recv_all(canport, timeout=0.005)

# def move_to_initial_position(
#     ctrl:    XArmController,
#     channel: str,
#     steps:   int = 220,
#     step_ms: int = 10,
# ) -> None:
#     """线性插值 + PD 控制，平滑移动到 INIT_POS，总时长约 2.2 s。"""
#     kp = [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0]
#     kd = [ 1.2,  1.2,  1.2,  1.2,  0.3,  0.2,  0.3]

#     ctrl.recv_all(channel=channel, timeout=0.05)
#     all_pos = ctrl.get_positions(channel)
#     n       = min(len(all_pos), len(INIT_POS))   # 只控制臂关节（忽略多余电机如夹爪）
#     start   = all_pos[:n]

#     fmt = lambda v: f"{v:.3f}"
#     print(f"[INFO] {channel} 移动至初始位置（{steps * step_ms / 1000:.1f} s）...")
#     print(f"[INFO]   起始: [{', '.join(map(fmt, start))}]")
#     print(f"[INFO]   目标: [{', '.join(map(fmt, INIT_POS))}]")

#     # 若总线上有额外电机（如夹爪），额外生成保持当前位置的参数
#     extra = [MITParam(kp=5.0, kd=0.5, q=all_pos[i]) for i in range(n, len(all_pos))]

#     for step in range(steps):
#         alpha  = (step + 1) / steps
#         pos    = [start[i] * (1 - alpha) + INIT_POS[i] * alpha for i in range(n)]
#         params = [MITParam(kp=kp[i], kd=kd[i], q=pos[i]) for i in range(n)] + extra
#         ctrl.mit_control_all(channel, params)
#         time.sleep(step_ms / 1000.0)
#         ctrl.recv_all(channel=channel, timeout=0.005)

#     print(f"[INFO] {channel} 已到达初始位置。")
    
def _gripper_move(ctrl: XArmController, target_q: float,
                  duration: float = 1.0, n_steps: int = 100,
                  canport: str = CANPORT):
    """插值运动夹爪到目标角度。"""
    ctrl.recv_all(canport, timeout=0.05)
    g0 = ctrl.get_gripper_position(canport)
    dt = duration / n_steps
    for i in range(n_steps + 1):
        alpha = i / n_steps
        gq = g0 + (target_q - g0) * alpha
        ctrl.gripper_position_control(canport, gq)
        ctrl.recv_all(canport, timeout=0.005)
        time.sleep(dt)


def gripper_open(ctrl: XArmController, canport: str = CANPORT):
    """张开夹爪。"""
    _gripper_move(ctrl, GRIPPER_OPEN, duration=1.0, canport=canport)


def gripper_close(ctrl: XArmController, canport: str = CANPORT):
    """闭合夹爪（轻力夹取）。"""
    _gripper_move(ctrl, GRIPPER_CLOSE, duration=1.0, canport=canport)
