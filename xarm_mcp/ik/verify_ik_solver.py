"""
验证 XarmIKSolver 的 IK 求解精度。

运行:
    python verify_ik_solver.py /path/to/scene.xml
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, __file__.rsplit("/scripts", 1)[0])
from scripts.simulation.xarm_ik_solver import XarmIKSolver



def verify_case(solver, label, targets, max_steps=500, tol=1e-4):
    print(f"\n{'='*50}")
    print(f"Case: {label}")
    print(f"{'='*50}")

    solver.reset()
    result = solver.solve_until_converged(targets, max_steps=max_steps, tol=tol)

    for side, (pos_tgt, _) in targets.items():
        pos_fk, _ = solver.get_ee_pose(side)
        pos_err = result.pos_errors[side]
        rot_err = result.rot_errors[side]
        status = "✅" if pos_err < 5.0 and rot_err < 2.0 else "❌"
        print(f"  [{side}] {status}  pos_err={pos_err:.2f} mm  rot_err={rot_err:.2f} deg  iters={result.iterations}")
        print(f"         target pos : {np.round(pos_tgt, 4)}")
        print(f"         actual pos : {np.round(pos_fk, 4)}")

    return result.q


def main():
    import os
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    else:
        # 从脚本位置动态构建路径
        repo_root = __file__.rsplit("/scripts", 1)[0]
        xml_path = os.path.join(repo_root, "xarm/scene.xml")
    
    solver = XarmIKSolver(xml_path=xml_path)

    # ── Case 1: 零误差测试（以当前位姿为目标，误差应近似 0）──────────────
    pos_l0, rot_l0 = solver.get_ee_pose("left")
    pos_r0, rot_r0 = solver.get_ee_pose("right")
    verify_case(solver, "零误差基准（当前位姿 → IK → FK）",
                {"left": (pos_l0, rot_l0), "right": (pos_r0, rot_r0)})

    # ── Case 2: 只移动位置（保持姿态）────────────────────────────────────
    pos_l1 = pos_l0 + np.array([0.05, 0.0, 0.05])
    pos_r1 = pos_r0 + np.array([0.05, 0.0, 0.05])
    verify_case(solver, "位置偏移 +5cm X/Z（保持初始姿态）",
                {"left": (pos_l1, rot_l0), "right": (pos_r1, rot_r0)})

    # ── Case 3: 只移动左臂 ────────────────────────────────────────────────
    solver.reset()
    pos_l2, rot_l2 = solver.get_ee_pose("left")
    pos_l2 = pos_l2 + np.array([0.0, -0.05, 0])
    verify_case(solver, "仅左臂偏移",
                {"left": (pos_l2, rot_l2)})

    # ── Case 4: 旋转目标 ─────────────────────────────────────────────────
    solver.reset()
    pos_l3, rot_l3 = solver.get_ee_pose("left")
    pos_r3, rot_r3 = solver.get_ee_pose("right")
    rot_l3_new = R.from_euler("z", 30, degrees=True).as_matrix() @ rot_l3
    rot_r3_new = R.from_euler("z", -30, degrees=True).as_matrix() @ rot_r3
    verify_case(solver, "末端绕 Z 轴旋转 ±30°",
                {"left": (pos_l3, rot_l3_new), "right": (pos_r3, rot_r3_new)})

    # ── Case 5: 圆弧轨迹跟踪（左臂在 XZ 平面画圆）─────────────────────────
    print(f"\n{'='*50}")
    print("Case: 圆弧轨迹跟踪（左臂 XZ 平面，半径 5cm，16 个点）")
    print(f"{'='*50}")
    solver.reset()
    pos_c, rot_c = solver.get_ee_pose("left")
    angles = np.linspace(0, 2 * np.pi, 17)[:-1]
    traj_errors = []
    for ang in angles:
        target_pos = pos_c + np.array([0.05 * np.cos(ang), 0.0, 0.05+0.05 * np.sin(ang)])
        q = solver.solve_until_converged({"left": (target_pos, rot_c)}, max_steps=300, tol=1e-4)
        pos_fk, _ = solver.get_ee_pose("left")
        err = np.linalg.norm(pos_fk - target_pos) * 1000
        traj_errors.append(err)
        status = "✅" if err < 5.0 else "❌"
        print(f"  ang={np.degrees(ang):6.1f}°  {status}  err={err:.2f} mm  target={np.round(target_pos,3)}")
    print(f"  均值={np.mean(traj_errors):.2f} mm  最大={np.max(traj_errors):.2f} mm")

    # # ── Case 6: 双臂反向运动（对称测试）────────────────────────────────────
    # solver.reset()
    # pos_l5, rot_l5 = solver.get_ee_pose("left")
    # pos_r5, rot_r5 = solver.get_ee_pose("right")
    # verify_case(solver, "双臂反向运动（左+X，右-X，各 8cm）",
    #             {
    #                 "left":  (pos_l5 + np.array([0.08, 0.0, 0.0]), rot_l5),
    #                 "right": (pos_r5 + np.array([-0.08, 0.0, 0.0]), rot_r5),
    #             })

    # # ── Case 7: 大角度旋转（±90°）────────────────────────────────────────
    # solver.reset()
    # pos_l6, rot_l6 = solver.get_ee_pose("left")
    # pos_r6, rot_r6 = solver.get_ee_pose("right")
    # rot_l6_new = R.from_euler("x", 90, degrees=True).as_matrix() @ rot_l6
    # rot_r6_new = R.from_euler("x", -90, degrees=True).as_matrix() @ rot_r6
    # verify_case(solver, "大角度旋转（绕 X 轴 ±90°）",
    #             {"left": (pos_l6, rot_l6_new), "right": (pos_r6, rot_r6_new)},
    #             max_steps=1000)

    # ── Case 8: 连续增量步进（模拟实时控制）────────────────────────────────
    print(f"\n{'='*50}")
    print("Case: 连续增量步进（每步 1mm，共 50 步，模拟实时控制）")
    print(f"{'='*50}")
    solver.reset()
    pos_s, rot_s = solver.get_ee_pose("left")
    step_errors = []
    for i in range(50):
        pos_s = pos_s + np.array([0.001, 0.0, 0.0])  # 每步 +1mm X
        q = solver.solve_step({"left": (pos_s, rot_s)})  # 单步，不迭代到收敛
        pos_fk, _ = solver.get_ee_pose("left")
        err = np.linalg.norm(pos_fk - pos_s) * 1000
        step_errors.append(err)
    print(f"  50 步均值误差={np.mean(step_errors):.2f} mm  最大={np.max(step_errors):.2f} mm")
    status = "✅" if np.mean(step_errors) < 5.0 else "❌"
    print(f"  {status} 实时单步精度{'正常' if np.mean(step_errors) < 5.0 else '偏大，建议用 solve_until_converged'}")

    print(f"\n{'='*50}")
    print("判断标准: pos_err < 5 mm, rot_err < 2 deg 视为通过")


if __name__ == "__main__":
    main()
