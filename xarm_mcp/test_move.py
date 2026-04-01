#!/usr/bin/env python3
"""测试：从零位运动到 [0, 20, 0, 60, 0, 0, 0] 度，带重力补偿。"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "xarm_controller"))

from xarm_mcp.arm_control import build_arm, move_joints
from xarm_mcp.xarm_controller.gravity_comp import ArmDynamics

CANPORT  = "can2"
URDF     = os.path.join(os.path.dirname(__file__), "xarm_controller", "xarm_bimanual.urdf")

TARGET_DEG = [0, 20, 0, 60, 0, 0, 0]
# TARGET_DEG = [0, 20, 0, 0, 0, 0, 0]
TARGET_RAD = np.deg2rad(TARGET_DEG)

def main():
    print("初始化动力学模型...")
    dyn = ArmDynamics(URDF, "xarm_right_")

    print("初始化机械臂...")
    ctrl = build_arm(CANPORT)
    print("就绪")

    print("归零...")
    move_joints(ctrl, np.zeros(7), canport=CANPORT, dynamics=dyn)
    print(f"运动到 {TARGET_DEG} deg ...")
    move_joints(ctrl, TARGET_RAD, canport=CANPORT, dynamics=dyn)
    print("完成")
    move_joints(ctrl, np.zeros(7), canport=CANPORT, dynamics=dyn)
    print(f"运动到 {TARGET_DEG} deg ...")
    move_joints(ctrl, TARGET_RAD, canport=CANPORT, dynamics=dyn)
    print("完成")
    # print("归零...")
    # move_joints(ctrl, np.zeros(7), duration=4.0, canport=CANPORT)
    # print(f"运动到 {TARGET_DEG} deg ...")
    # move_joints(ctrl, TARGET_RAD, duration=4.0, canport=CANPORT)
    # print("完成")
    # move_joints(ctrl, np.zeros(7), duration=4.0, canport=CANPORT)
    # print(f"运动到 {TARGET_DEG} deg ...")
    # move_joints(ctrl, TARGET_RAD, duration=4.0, canport=CANPORT)
    # print("完成")


if __name__ == "__main__":
    main()
