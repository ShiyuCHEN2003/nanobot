"""
右臂运动控制工具模块。

提供：
  - build_arm(canport)          → (xarm, arm, gripper)
  - read_q7(arm)                → np.ndarray (7,) 当前关节角 rad
  - move_joints(xarm, arm, q7_target, duration)  插值运动到目标关节角
  - gripper_open(xarm, gripper)
  - gripper_close(xarm, gripper)
"""

import time
import numpy as np
import xarm_can as oa

# ── 电机配置 ──
MOTOR_TYPES = [
    oa.MotorType.DM8009, oa.MotorType.DM8009,
    oa.MotorType.DM4340, oa.MotorType.DM4340,
    oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
]
MASTER_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
SLAVE_IDS  = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]

# ── 各关节 PD 增益 ──
KP = [300.0, 300.0, 150.0, 150.0, 40.0, 40.0, 30.0]
KD = [  2.5,   2.5,   2.5,   2.5,  0.8,  0.8,  0.8]

# ── 关节限位 (rad) ──
JOINT_LIMITS = [
    (np.deg2rad(-80),  np.deg2rad(200)),
    (np.deg2rad(-10),  np.deg2rad(190)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(0),    np.deg2rad(140)),
    (np.deg2rad(-90),  np.deg2rad(90)),
    (np.deg2rad(-45),  np.deg2rad(45)),
    (np.deg2rad(-90),  np.deg2rad(90)),
]


def build_arm(canport: str = "can2"):
    """初始化机械臂，返回 (xarm, arm, gripper)。"""
    xarm = oa.XArm(canport, True)
    xarm.init_arm_motors(MOTOR_TYPES, MASTER_IDS, SLAVE_IDS)
    xarm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
    xarm.set_callback_mode_all(oa.CallbackMode.STATE)
    xarm.enable_all()
    time.sleep(0.1)
    xarm.refresh_all()
    xarm.recv_all()
    arm     = xarm.get_arm()
    gripper = xarm.get_gripper()
    return xarm, arm, gripper


def read_q7(arm) -> np.ndarray:
    """读取 7 个关节当前角度 (rad)。"""
    return np.array([m.get_position() for m in arm.get_motors()])


def _clip_q7(q7: np.ndarray) -> np.ndarray:
    """将目标关节角夹到限位内。"""
    q = q7.copy()
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        q[i] = np.clip(q[i], lo, hi)
    return q


def move_joints(xarm, arm, q7_target: np.ndarray,
                duration: float = 3.0, n_steps: int = 300):
    """
    从当前位置插值运动到 q7_target (rad)。

    Parameters
    ----------
    xarm       : XArm 对象
    arm        : ArmComponent 对象
    q7_target  : 目标关节角，shape (7,)，rad
    duration   : 运动时间，秒
    n_steps    : 插值步数
    """
    xarm.refresh_all()
    xarm.recv_all()
    q0 = read_q7(arm)
    q1 = _clip_q7(q7_target)

    dt = duration / n_steps
    for i in range(n_steps + 1):
        alpha = i / n_steps
        q = q0 + (q1 - q0) * alpha
        params = [oa.MITParam(KP[j], KD[j], float(q[j]), 0.0, 0.0)
                  for j in range(7)]
        arm.mit_control_all(params)
        xarm.recv_all()
        time.sleep(dt)


def hold(xarm, arm, duration: float = 0.5):
    """原地保持当前姿态。"""
    xarm.refresh_all()
    xarm.recv_all()
    q = read_q7(arm)
    t_end = time.time() + duration
    while time.time() < t_end:
        params = [oa.MITParam(KP[j], KD[j], float(q[j]), 0.0, 0.0)
                  for j in range(7)]
        arm.mit_control_all(params)
        xarm.recv_all()
        time.sleep(0.005)


GRIPPER_KP   = 10.0
GRIPPER_KD   = 0.9
GRIPPER_OPEN  = np.deg2rad(-50)  # 张开角度（机械限位约 -60°）
GRIPPER_CLOSE = np.deg2rad(0)    # 闭合角度


def _gripper_move(xarm, gripper, target_q: float, duration: float = 1.0, n_steps: int = 100):
    """插值运动夹爪到目标角度。"""
    xarm.refresh_all()
    xarm.recv_all()
    q0 = gripper.get_motors()[0].get_position()
    dt = duration / n_steps
    for i in range(n_steps + 1):
        alpha = i / n_steps
        q = q0 + (target_q - q0) * alpha
        gripper.mit_control_all([oa.MITParam(GRIPPER_KP, GRIPPER_KD, float(q), 0.0, 0.0)])
        xarm.recv_all()
        time.sleep(dt)


def gripper_open(xarm, gripper):
    """张开夹爪。"""
    _gripper_move(xarm, gripper, GRIPPER_OPEN, duration=1.0)


def gripper_close(xarm, gripper):
    """闭合夹爪（轻力夹取）。"""
    _gripper_move(xarm, gripper, GRIPPER_CLOSE, duration=1.0)
