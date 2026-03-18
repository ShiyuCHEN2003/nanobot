#!/usr/bin/env python3
# debug_compare.py  —— 实时对比 ROS 读到的 q7 和 FK 结果
import subprocess, json, time, numpy as np, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NANOBOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _NANOBOT)
from openarm_mcp.ik.xarm_ik_solver import XarmIKSolver

SCENE_XML = os.path.join(_HERE, "../xml/scene.xml")
solver = XarmIKSolver(xml_path=SCENE_XML, right_ee_name="openarm_right_hand_tcp")
base_qpos = solver.data.qpos.copy()

# 直接读一帧 joint_states
cmd = "source /opt/ros/humble/setup.bash && ros2 topic echo /openarm_leader/joint_states --once"
out = subprocess.check_output(["bash","-c",cmd], text=True)
print("ROS raw output:\n", out)

# 解析 position
import re
positions = re.findall(r"^- ([-\d.e+]+)", out, re.MULTILINE)
names     = re.findall(r"^- (arm_\w+|hand_\w+)", out, re.MULTILINE)
print(f"\njoint names : {names}")
print(f"positions   : {[round(float(p),4) for p in positions]}")

q7 = np.array([float(p) for p in positions[:7]])
print(f"\nq7 (前7个)  : {np.round(q7,4)}")

qpos = base_qpos.copy()
qpos[9:16] = q7
solver.reset(qpos)
pos, rot = solver.get_ee_pose("right")
print(f"FK EE pos   : {np.round(pos,4)}")

# 现在把机械臂摆到你认为的"大概零位"或某个已知姿态
# 对比 FK 输出的位置和你实际目测的末端位置
print("\n请目测机械臂末端位置（米），与 FK 输出对比")
print("如果差距 > 5cm，说明关节角定义不一致")
