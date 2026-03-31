#!/usr/bin/env python3
"""直接用指定世界坐标测试抓取流程（跳过相机/YOLO）。"""

import os, sys, time, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from xarm_mcp.grasp_pipeline import build_ik
from xarm_mcp.arm_control import build_arm, read_q7, move_joints, gripper_open, gripper_close

CANPORT = "can2"
PRE_GRASP_Z = 0.10   # 预抓取比目标高 10cm
SETTLE_SECS = 1.0   # 各段运动间收敛等待时间

# 目标坐标 (cm → m)
target_cm = (31.1, -17.0, 26.6)
pos_world = np.array(target_cm) / 100.0

print(f"目标世界坐标: {target_cm} cm  →  {pos_world} m")

ik = build_ik()
ctrl = build_arm(CANPORT)
q7_now = read_q7(ctrl, CANPORT)

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
    ctrl.disable_all(); ctrl.close(); sys.exit(1)

result_grasp, q7_grasp = ik(pos_world, q7_pre)
print(f"[IK 抓取]   误差: {result_grasp.pos_errors.get('right',0):.1f}mm  "
      f"关节角(deg): {np.round(np.degrees(q7_grasp),1).tolist()}")
if not result_grasp.success:
    print(f"抓取 IK 失败: {result_grasp.message}")
    ctrl.disable_all(); ctrl.close(); sys.exit(1)

# print("\n张开夹爪...")
# gripper_open(ctrl, CANPORT)

print("→ 移动到预抓取点")
# move_joints(ctrl, q7_pre, canport=CANPORT)
# time.sleep(SETTLE_SECS)

print("→ 下降到抓取点")
move_joints(ctrl, q7_grasp, canport=CANPORT)
time.sleep(SETTLE_SECS)

# print("→ 夹爪闭合")
# gripper_close(ctrl, CANPORT)
# time.sleep(SETTLE_SECS)

# print("→ 抬起")
# move_joints(ctrl, q7_pre, canport=CANPORT)
# time.sleep(SETTLE_SECS)

print("完成！")
# ctrl.disable_all()
# ctrl.close()
