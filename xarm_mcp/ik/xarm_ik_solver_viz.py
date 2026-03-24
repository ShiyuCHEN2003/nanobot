"""
带可视化的 IK 求解器演示脚本。

用法:
    python xarm_ik_solver_viz.py /path/to/scene.xml
    
    - 显示实时 IK 求解过程
    - 左右两个机械臂同时显示
    - 可交互操作
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from xarm_ik_solver import XarmIKSolver
import sys


def visualize_ik_solving(xml_path: str, target_offset=None):
    """
    实时可视化 IK 求解过程。
    
    Parameters
    ----------
    xml_path : str
        MuJoCo XML 场景路径。
    target_offset : dict, optional
        对左臂和右臂的目标位置偏移量，格式: {"left": [dx, dy, dz], "right": [dx, dy, dz]}
    """
    
    solver = XarmIKSolver(xml_path=xml_path)
    
    # 读取初始末端位姿
    pos_l, rot_l = solver.get_ee_pose("left")
    pos_r, rot_r = solver.get_ee_pose("right")
    
    print("=== 初始末端执行器位姿 ===")
    print(f"Left EE:  pos={np.round(pos_l, 4)}")
    print(f"Right EE: pos={np.round(pos_r, 4)}")
    
    # 应用目标偏移（可选）
    if target_offset:
        if "left" in target_offset:
            pos_l = pos_l + np.array(target_offset["left"])
        if "right" in target_offset:
            pos_r = pos_r + np.array(target_offset["right"])
        print("\n=== 应用偏移后的目标位置 ===")
        print(f"Left target:  {np.round(pos_l, 4)}")
        print(f"Right target: {np.round(pos_r, 4)}")
    
    targets = {"left": (pos_l, rot_l), "right": (pos_r, rot_r)}
    
    # 打开 MuJoCo 查看器
    with mujoco.viewer.launch_passive(solver.model, solver.data) as viewer:
        print("\n=== 开始 IK 求解可视化 ===")
        print("按 Ctrl+C 停止")
        
        # 迭代求解并实时更新可视化
        max_steps = 500
        tol = 1e-3
        q_prev = solver.data.qpos.copy()
        
        for i in range(max_steps):
            # 执行单步 IK 求解
            q = solver.solve_step(targets, update_state=True)
            
            # 更新 MuJoCo 数据用于可视化
            solver.data.qpos[:] = q
            solver.data.ctrl[:] = q
            mujoco.mj_forward(solver.model, solver.data)
            
            # 计算收敛情况
            delta = np.max(np.abs(q - q_prev))
            
            # 每 5 步打印一次进度
            if i % 5 == 0:
                pos_l_fk, _ = solver.get_ee_pose("left")
                pos_r_fk, _ = solver.get_ee_pose("right")
                print(f"iter {i:3d}  Δq={delta:.2e}  "
                      f"left_pos_err={np.linalg.norm(pos_l_fk - pos_l)*1000:.2f}mm  "
                      f"right_pos_err={np.linalg.norm(pos_r_fk - pos_r)*1000:.2f}mm")
            
            # 同步可视化
            viewer.sync()
            
            # 检查收敛
            if delta < tol:
                print(f"\n✓ 收敛! (iter={i}, Δq={delta:.2e})")
                break
            
            q_prev = q
        
        # 最终结果评估
        result = solver.solve_until_converged(targets, max_steps=200, tol=1e-4, verbose=False)
        print(f"\n=== 最终结果 ===\n{result}")
        
        # 等待用户关闭查看器
        print("\n(保持窗口打开，按 Ctrl+C 关闭)")
        while viewer.is_running():
            viewer.sync()


def main():
    if len(sys.argv) < 2:
        xml_path = "/home/vlai-minipc/nanobot/xarm_mcp/xml/scene.xml"
    else:
        xml_path = sys.argv[1]
    
    # 示例 1: 演示当前位姿（零误差验证）
    # print("=" * 60)
    # print("演示 1: 验证当前位姿（应该零误差收敛）")
    # print("=" * 60)
    # visualize_ik_solving(xml_path)
    
    # 示例 2: 演示位置偏移目标
    print("\n" + "=" * 60)
    print("演示 2: 向上平移 10cm 目标")
    print("=" * 60)
    visualize_ik_solving(xml_path, target_offset={"left": [0.1, 0, 0.1], "right": [0, 0, 0.1]})


if __name__ == "__main__":
    main()
