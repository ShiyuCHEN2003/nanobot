#!/usr/bin/env python3
"""测试：从零位运动到 [0, 20, 0, 60, 0, 0, 0] 度，带重力补偿。"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "xarm_controller"))

from xarm_mcp.arm_control import Arm

CANPORT    = "can2"
TARGET_DEG = [0, 20, 0, 60, 0, 0, 0]
TARGET_RAD = np.deg2rad(TARGET_DEG)

def main():
    print("初始化机械臂（含重力补偿模型）...")
    arm = Arm(CANPORT)
    print("就绪")

    print("归零...")
    arm.move_joints(np.zeros(7))
    print(f"运动到 {TARGET_DEG} deg ...")
    arm.move_joints(TARGET_RAD)
    print("完成")
    arm.move_joints(np.zeros(7))
    print(f"运动到 {TARGET_DEG} deg ...")
    arm.move_joints(TARGET_RAD)
    print("完成")


if __name__ == "__main__":
    main()
