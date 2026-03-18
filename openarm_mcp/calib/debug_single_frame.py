#!/usr/bin/env python3
# debug_single_frame.py
import numpy as np, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NANOBOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _NANOBOT)

from openarm_mcp.ik.xarm_ik_solver import XarmIKSolver

SCENE_XML = os.path.join(_HERE, "../xml/scene.xml")
RIGHT_ARM_QPOS_SLICE = slice(9, 16)

solver = XarmIKSolver(xml_path=SCENE_XML, right_ee_name="openarm_right_hand_tcp")
base_qpos = solver.data.qpos.copy()

# ── 用你第1帧的真实关节角 ──
# 从 ros2 topic echo /openarm_leader/joint_states 读取，只取前7个臂关节
q7 = np.array([
    -0.0532, -0.1997, -0.0872, 0.9146, -0.2138, -0.4122, 0.1410
])  # ← 替换成你采集第1帧时的实际值

qpos = base_qpos.copy()
qpos[RIGHT_ARM_QPOS_SLICE] = q7
solver.reset(qpos)
pos, rot = solver.get_ee_pose("right")

print(f"FK EE position : {np.round(pos, 4)}")
print(f"FK EE rpy (deg): {np.round(np.degrees(np.array([np.arctan2(rot[2,1],rot[2,2]), np.arcsin(-rot[2,0]), np.arctan2(rot[1,0],rot[0,0])])), 2)}")

# ── 第1帧的 PnP 结果（从你的输出贴入）──
# cam->board t=(+0.198,+0.049,+0.280)
t_cam_board = np.array([0.198, 0.049, 0.280])

print(f"\n第1帧 cam->board Z = {t_cam_board[2]:.3f} m")
print(f"第1帧 FK EE Z      = {pos[2]:.3f} m")
print(f"\n── qpos 全貌（检查 slice）──")
print(f"qpos 总长度: {len(base_qpos)}")
for i in range(0, len(base_qpos), 1):
    print(f"  qpos[{i}] = {base_qpos[i]:.4f}")
