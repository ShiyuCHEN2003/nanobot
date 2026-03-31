#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
双臂重力补偿 Python 实现
- CAN 通信：使用 XArmController（XArmcontroller.py）
- 动力学计算：使用 Pinocchio
- 补偿内容：重力 + 科氏力（可调增益）+ 摩擦（可调增益）

用法：
  python gravity_comp.py          # 双臂（默认）
  python gravity_comp.py right    # 仅右臂 (can0)
  python gravity_comp.py left     # 仅左臂 (can1)
"""

import math
import os
import signal
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from XArmcontroller import XArmController, ARM_7DOF_LIMITS, MITParam
except ImportError as e:
    print(f"[ERROR] 无法导入 XArmController: {e}", file=sys.stderr)
    sys.exit(1)

URDF_PATH = "./xarm_bimanual.urdf"
RIGHT_CAN = "can2"
LEFT_CAN  = "can1"

# INIT_POS = [0.0, 0.0, 0.0, 1.283, 0.0, 0.0, 0.0]
INIT_POS = [-0.049, 0.53, -0.3313, 1.272, 0.521, 0.05855, 0.605]


GRAVITY_SCALE  = 1.0   # 重力矩缩放因子，若手臂下垂可逐步增大（如 1.5, 2.0）
FRICTION_GAIN  = 0.3
CORIOLIS_GAIN  = 0.1

# ===== 动力学计算（Pinocchio）=====

class ArmDynamics:
    """利用 Pinocchio 计算单臂的重力矩和科氏力矩。"""
    def __init__(self, urdf_path: str, arm_prefix: str):
        full_model = pin.buildModelFromUrdf(urdf_path)
        arm_joint_names = {f"{arm_prefix}joint{i}" for i in range(1, 8)}

        joints_to_lock: List[int] = []
        for jid in range(1, full_model.njoints):
            if full_model.names[jid] not in arm_joint_names:
                joints_to_lock.append(jid)

        q_ref  = pin.neutral(full_model)
        result = pin.buildReducedModel(full_model, joints_to_lock, q_ref)
        self.model: pin.Model = result[0] if isinstance(result, tuple) else result
        self.data:  pin.Data  = self.model.createData()

        if self.model.nv != 7:
            raise RuntimeError(
                f"简化模型 DOF={self.model.nv}，期望 7。"
                f"请检查 arm_prefix='{arm_prefix}' 及 URDF 关节名称。")

        print(f"[INFO] {arm_prefix}dynamics: {self.model.nv} DOF，"
              f"关节顺序: {list(self.model.names[1:])}")

    def get_gravity(self, q: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    def get_coriolis(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        return (C @ dq).copy()

# ===== 摩擦力补偿 =====
def compute_friction(vel: float, Fc: float, k: float, Fv: float, Fo: float) -> float:
    return Fc * math.tanh(0.1 * k * vel) + Fv * vel + Fo

_Fc = [0.5,  0.5,  0.3,  0.3,  0.2,  0.2,  0.2]
_k  = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
_Fv = [0.1,  0.1,  0.1,  0.1,  0.05, 0.05, 0.05]
_Fo = [0.0] * 7

# 右臂 per-joint kd（与 C++ 一致，joint2 kd=3.0 提供关键稳定性）
# RIGHT_KD = [0.5, 3.0, 0.1, 0.1, 0.1, 0.2, 0.2]
RIGHT_KD = [0.5, 0.5, 0.1, 0.1, 0.1, 0.2, 0.2]
# ===== 移动到初始位置 =====

def move_to_initial_position(
    ctrl:    XArmController,
    channel: str,
    steps:   int = 220,
    step_ms: int = 10,
) -> None:
    """线性插值 + PD 控制，平滑移动到 INIT_POS，总时长约 2.2 s。"""
    kp = [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0]
    kd = [ 1.2,  1.2,  1.2,  1.2,  0.3,  0.2,  0.3]

    ctrl.recv_all(channel=channel, timeout=0.05)
    all_pos = ctrl.get_positions(channel)
    n       = min(len(all_pos), len(INIT_POS))   # 只控制臂关节（忽略多余电机如夹爪）
    start   = all_pos[:n]

    fmt = lambda v: f"{v:.3f}"
    print(f"[INFO] {channel} 移动至初始位置（{steps * step_ms / 1000:.1f} s）...")
    print(f"[INFO]   起始: [{', '.join(map(fmt, start))}]")
    print(f"[INFO]   目标: [{', '.join(map(fmt, INIT_POS))}]")

    # 若总线上有额外电机（如夹爪），额外生成保持当前位置的参数
    extra = [MITParam(kp=5.0, kd=0.5, q=all_pos[i]) for i in range(n, len(all_pos))]

    for step in range(steps):
        alpha  = (step + 1) / steps
        pos    = [start[i] * (1 - alpha) + INIT_POS[i] * alpha for i in range(n)]
        params = [MITParam(kp=kp[i], kd=kd[i], q=pos[i]) for i in range(n)] + extra
        ctrl.mit_control_all(channel, params)
        time.sleep(step_ms / 1000.0)
        ctrl.recv_all(channel=channel, timeout=0.005)

    print(f"[INFO] {channel} 已到达初始位置。")


# ===== 主程序 =====

def main() -> None:
    # 解析命令行：可选传入 right 或 left，默认 both
    mode = "both"
    if len(sys.argv) >= 2:
        mode = sys.argv[1].lower()
        if mode not in ("left", "right", "both"):
            print(f"用法: python gravity_comp.py [right|left|both]")
            sys.exit(1)

    print("=== Dual Arm Dynamics Compensation ===")
    print(f"Mode           : {mode}")
    print(f"Right CAN      : {RIGHT_CAN}  (can0)")
    print(f"Left  CAN      : {LEFT_CAN}  (can1)")
    print(f"Gravity scale  : {GRAVITY_SCALE}")
    print(f"Friction gain  : {FRICTION_GAIN}")
    print(f"Coriolis gain  : {CORIOLIS_GAIN}")
    print(f"URDF path      : {URDF_PATH}")
    print(f"Init position  : {INIT_POS}")

    if not os.path.exists(URDF_PATH):
        print(f"[ERROR] URDF 文件未找到: {URDF_PATH}", file=sys.stderr)
        sys.exit(1)

    # ---- 初始化动力学模型 ----
    left_dyn:  Optional[ArmDynamics] = None
    right_dyn: Optional[ArmDynamics] = None

    if mode in ("left", "both"):
        print("[INFO] 初始化左臂动力学...")
        left_dyn = ArmDynamics(URDF_PATH, "xarm_left_")

    if mode in ("right", "both"):
        print("[INFO] 初始化右臂动力学...")
        right_dyn = ArmDynamics(URDF_PATH, "xarm_right_")

    # ---- 初始化 CAN 控制器 ----
    bus_configs: Dict = {}
    if mode in ("left",  "both"):
        bus_configs[LEFT_CAN]  = ARM_7DOF_LIMITS
    if mode in ("right", "both"):
        bus_configs[RIGHT_CAN] = ARM_7DOF_LIMITS

    ctrl = XArmController(bus_configs)
    ctrl.enable_all()
    ctrl.recv_all(timeout=0.05)

    # ---- 移动到初始位置 ----
    if mode in ("left",  "both"):
        move_to_initial_position(ctrl, LEFT_CAN)
    if mode in ("right", "both"):
        move_to_initial_position(ctrl, RIGHT_CAN)

    # ---- 控制循环 ----
    keep_running = True

    def _sigint(sig, frame):
        nonlocal keep_running
        print("\nCtrl+C detected. Exiting...")
        keep_running = False

    signal.signal(signal.SIGINT, _sigint)

    print("=== Starting dynamics compensation ===")
    print("Compensating: Gravity + Coriolis + Friction")
    print("Press Ctrl+C to stop")

    start_time   = time.monotonic()
    last_display = start_time
    frame_count  = 0
    N = 7

    try:
        while keep_running:
            frame_count += 1
            now = time.monotonic()

            # ---- 左臂 ----
            if left_dyn is not None:
                q  = np.array(ctrl.get_positions(LEFT_CAN))
                dq = np.array(ctrl.get_velocities(LEFT_CAN))

                g_tau   = left_dyn.get_gravity(q)
                cor_tau = left_dyn.get_coriolis(q, dq)

                params = []
                for i in range(N):
                    fric  = compute_friction(dq[i], _Fc[i], _k[i], _Fv[i], _Fo[i])
                    total = g_tau[i] * GRAVITY_SCALE + cor_tau[i] * CORIOLIS_GAIN + fric * FRICTION_GAIN
                    params.append(MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=total))

                ctrl.mit_control_all(LEFT_CAN, params)
                ctrl.recv_all(channel=LEFT_CAN)   # 默认 10ms，等待全部 7 个电机回包

            # ---- 右臂 ----
            if right_dyn is not None:
                q  = np.array(ctrl.get_positions(RIGHT_CAN))
                dq = np.array(ctrl.get_velocities(RIGHT_CAN))
                # print(f"right q: {q}")
                g_tau   = right_dyn.get_gravity(q)
                cor_tau = right_dyn.get_coriolis(q, dq)

                params = []
                for i in range(N):
                    fric  = compute_friction(dq[i], _Fc[i], _k[i], _Fv[i], _Fo[i])
                    total = g_tau[i] * GRAVITY_SCALE + cor_tau[i] * CORIOLIS_GAIN + fric * FRICTION_GAIN
                    params.append(MITParam(kp=0.0, kd=RIGHT_KD[i],
                                           q=0.0, dq=0.0, tau=total))

                ctrl.mit_control_all(RIGHT_CAN, params)
                ctrl.recv_all(channel=RIGHT_CAN)  # 默认 10ms，等待全部 7 个电机回包

            # ---- 每秒打印频率和力矩 ----
            if now - last_display >= 1.0:
                elapsed = now - start_time
                hz = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Frequency: {hz:.1f} Hz")
                if left_dyn is not None:
                    q = np.array(ctrl.get_positions(LEFT_CAN))
                    g = left_dyn.get_gravity(q)
                    tau_meas = ctrl.get_torques_measured(LEFT_CAN)
                    # print(f"  left  pos:      {[f'{v:.3f}' for v in q.tolist()]}")
                    # print(f"  left  tau_cmd:  {[f'{v:.3f}' for v in g.tolist()]}")
                    # print(f"  left  tau_meas: {[f'{v:.3f}' for v in tau_meas]}")
                if right_dyn is not None:
                    q = np.array(ctrl.get_positions(RIGHT_CAN))
                    g = right_dyn.get_gravity(q)
                    # tau_meas = ctrl.get_torques_measured(RIGHT_CAN)
                    # print(f"  right pos:      {[f'{v:.3f}' for v in q.tolist()]}")
                    # print(f"  right tau_cmd:  {[f'{v:.3f}' for v in g.tolist()]}")
                    # print(f"  right tau_meas: {[f'{v:.3f}' for v in tau_meas]}")
                last_display = now

    finally:
        ctrl.disable_all()
        ctrl.recv_all(timeout=0.05)
        ctrl.close()
        print("=== Dynamics compensation stopped ===")


if __name__ == "__main__":
    main()
