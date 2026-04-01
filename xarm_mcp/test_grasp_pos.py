#!/usr/bin/env python3
"""直接用指定世界坐标测试抓取流程（跳过相机/YOLO）。"""

import os, sys, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from xarm_mcp.grasp_pipeline import build_ik
from xarm_mcp.arm_control import Arm

CANPORT     = "can2"
PRE_GRASP_Z = 0.10   # 预抓取比目标高 10cm

# 目标坐标 (cm → m)
target_cm = (31.1, -17.0, 26.6)
pos_world = np.array(target_cm) / 100.0

print(f"目标世界坐标: {target_cm} cm  →  {pos_world} m")

ik  = build_ik()
arm = Arm(CANPORT)
q7_now = arm.read_q7()

# 预抓取点
pre_grasp = pos_world.copy()
pre_grasp[2] += PRE_GRASP_Z

print(f"预抓取点: ({pre_grasp*100}) cm")

# IK
result_pre, q7_pre = ik(pre_grasp, q7_now)
print(f"[IK 预抓取] 误差: {result_pre.pos_errors.get('right',0):.1f}mm  "
      f"关节角(deg): {np.round(np.degrees(q7_pre),1).tolist()}")
if not result_pre.success:
    print(f"预抓取 IK 失败: {result_pre.message}")
    sys.exit(1)

result_grasp, q7_grasp = ik(pos_world, q7_pre)
print(f"[IK 抓取]   误差: {result_grasp.pos_errors.get('right',0):.1f}mm  "
      f"关节角(deg): {np.round(np.degrees(q7_grasp),1).tolist()}")
if not result_grasp.success:
    print(f"抓取 IK 失败: {result_grasp.message}")
    sys.exit(1)

print("\n张开夹爪...")
arm.gripper_open()

q7_j1only = np.zeros(7); q7_j1only[0] = q7_pre[0]
print(f"→ 移动到预抓取点（仅J1）: {np.round(np.degrees(q7_j1only),1).tolist()} deg")
arm.move_joints(q7_j1only)

print("→ 下降到抓取点")
arm.move_joints(q7_grasp)

print("→ 夹爪闭合")
arm.gripper_close()

arm.move_joints(q7_pre)
print("完成！")
