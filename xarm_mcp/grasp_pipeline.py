#!/usr/bin/env python3
"""
抓取流水线：胸部相机感知 → IK → 运动 → 抓取

流程：
  1. 胸部相机 + YOLOv8 检测目标物体
  2. 深度 → 相机系 3D 坐标 → 世界系坐标
  3. IK 求解目标关节角（接近点比目标高 10cm）
  4. 机械臂运动到预抓取位置
  5. 下降 10cm 到抓取点
  6. 夹爪闭合

用法：
  cd nanobot/
  DISPLAY=:1 conda run -n lerobot python -m xarm_mcp.grasp_pipeline \\
      --canport can2 --target bottle

控制：
  [SPACE]  对当前检测结果执行抓取
  [q]      退出
"""

import os, sys, json, time, argparse
import numpy as np
import cv2
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass
import pyrealsense2 as rs
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from xarm_mcp.calib.detect_3d import load_calib, pixel_to_world, sample_depth
from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver
from xarm_mcp.arm_control import Arm

SCENE_XML            = os.path.join(_HERE, "xml/scene.xml")
YOLO_MODEL           = os.path.join(_ROOT, "yolov8n.pt")
SERIAL_CHEST         = "314422070707"
RIGHT_ARM_QPOS_SLICE = slice(9, 16)

CONF_THRESH   = 0.4
DEPTH_MARGIN  = 5
PRE_GRASP_Z   = 0.10   # 预抓取点比目标高 10cm
DESCEND_Z     = 0.10   # 下降距离 10cm


def build_ik():
    solver = XarmIKSolver(xml_path=SCENE_XML,
                          right_ee_name="xarm_right_hand_tcp",
                          orientation_cost=0.0)
    base_qpos = solver.data.qpos.copy()

    # 几个不同的初始姿态种子（手臂伸向不同方向）
    SEEDS = [
        np.array([0,  0,  0,  np.pi/2, 0, 0, 0]),       # 肘部弯90°
        np.array([0,  0.3,  0,  np.pi/2, 0, 0, 0]),
        np.array([0.3, 0,  0,  np.pi/2, 0, 0, 0]),
        np.array([-0.3, 0, 0,  np.pi/2, 0, 0, 0]),
        np.array([0,  0,  0,  np.deg2rad(70), 0, 0, 0]),
    ]

    def ik(target_pos, q7_hint, pos_tol_mm=15.0):
        """尝试多个初始种子，返回误差最小的解。误差 < pos_tol_mm 视为成功。"""
        seeds = [q7_hint] + SEEDS
        best_result, best_q7, best_err = None, None, float('inf')

        for seed in seeds:
            qpos = base_qpos.copy()
            qpos[RIGHT_ARM_QPOS_SLICE] = seed
            solver.reset(qpos)
            result = solver.solve_until_converged(
                {"right": (target_pos, None)},
                max_steps=1000, tol=1e-4,
                pos_tol_mm=pos_tol_mm,
            )
            q7 = solver.data.qpos[RIGHT_ARM_QPOS_SLICE].copy()
            if result.success:
                return result, q7
            err = result.pos_errors.get('right', float('inf'))  # 已经是 mm
            if err < best_err:
                best_err = err
                best_result = result
                best_q7 = q7.copy()

        # 误差在容忍范围内，强制标记成功
        if best_err < pos_tol_mm:
            best_result.success = True
            best_result.message = f"OK (pos_err={best_err:.1f}mm)"
        return best_result, best_q7

    return ik


def best_detection(results, model, target_label, depth_f, K, T_base_cam):
    """在 YOLO 结果中找置信度最高的目标，返回 (world_pos, label, conf)。"""
    best = None
    best_conf = 0.0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        if target_label and label != target_label:
            continue
        if conf < best_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        depth_m = sample_depth(depth_f, cx_px, cy_px, DEPTH_MARGIN)
        if depth_m <= 0:
            continue
        pos_world = pixel_to_world(cx_px, cy_px, depth_m, K, T_base_cam)
        best_conf = conf
        best = (pos_world, label, conf, cx_px, cy_px, x1, y1, x2, y2)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    parser.add_argument("--target",  default="",
                        help="YOLO 类别名，空字符串=任意目标")
    args = parser.parse_args()

    K, dist, T_base_cam = load_calib()
    print(f"标定加载完毕。相机世界坐标: {T_base_cam[:3,3].round(3)} m")

    model = YOLO(YOLO_MODEL)
    print(f"YOLO 加载完毕")

    print("加载 IK 解算器...")
    ik = build_ik()
    print("IK 就绪")

    print(f"初始化机械臂 (CAN: {args.canport})...")
    arm = Arm(args.canport)
    print("机械臂就绪")

    # 启动相机
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL_CHEST)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(20):
        pipeline.wait_for_frames()
    print("相机就绪")

    print("张开夹爪...")
    arm.gripper_open()
    print("夹爪就绪")

    WIN = "抓取流水线  [SPACE] 执行抓取  [q] 退出"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 960)

    last_detection = None

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f or not depth_f:
                continue

            color = np.asanyarray(color_f.get_data())
            disp  = color.copy()

            results  = model(color, conf=CONF_THRESH, verbose=False)[0]
            det = best_detection(results, model, args.target,
                                 depth_f, K, T_base_cam)

            if det is not None:
                pos_world, label, conf, cx, cy, x1, y1, x2, y2 = det
                last_detection = det

                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 220, 0), 2)
                cv2.circle(disp, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                txt = (f"{label} {conf:.2f} | "
                       f"({pos_world[0]*100:.1f},"
                       f"{pos_world[1]*100:.1f},"
                       f"{pos_world[2]*100:.1f}) cm")
                cv2.putText(disp, txt, (int(x1), max(int(y1)-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
                status = f"检测到: {label}  [SPACE] 执行抓取"
                color_s = (0, 220, 0)
            else:
                last_detection = None
                status = "未检测到目标"
                color_s = (0, 60, 220)

            cv2.rectangle(disp, (0, 0), (640, 40), (0, 0, 0), -1)
            cv2.putText(disp, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_s, 2)

            cv2.imshow(WIN, cv2.resize(disp, (1280, 960)))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' ') and last_detection is not None:
                pos_world = last_detection[0]
                label     = last_detection[1]
                print(f"\n[抓取] 目标 '{label}' 世界坐标: "
                      f"({pos_world[0]*100:.1f}, {pos_world[1]*100:.1f}, "
                      f"{pos_world[2]*100:.1f}) cm")

                # 当前关节角
                q7_now = arm.read_q7()

                # 预抓取点（目标上方 PRE_GRASP_Z）
                pre_grasp = pos_world.copy()
                pre_grasp[2] += PRE_GRASP_Z

                print(f"[抓取] 预抓取点: ({pre_grasp[0]*100:.1f}, "
                      f"{pre_grasp[1]*100:.1f}, {pre_grasp[2]*100:.1f}) cm")

                # IK 解算预抓取点
                result_pre, q7_pre = ik(pre_grasp, q7_now)
                print(f"[IK] 预抓取误差: {result_pre.pos_errors.get('right',0):.1f}mm  "
                      f"关节角(deg): {np.round(np.degrees(q7_pre),1).tolist()}")
                if not result_pre.success:
                    print(f"[抓取] 预抓取 IK 失败: {result_pre.message}")
                    continue

                # IK 解算抓取点
                result_grasp, q7_grasp = ik(pos_world, q7_pre)
                if not result_grasp.success:
                    print(f"[抓取] 抓取 IK 失败: {result_grasp.message}")
                    continue

                print(f"[抓取] IK 成功，开始运动...")

                # 1. 运动到预抓取点
                print("  → 移动到预抓取点")
                arm.move_joints(q7_pre)

                # 2. 下降到抓取点
                print("  → 下降到抓取点")
                arm.move_joints(q7_grasp)

                # 3. 夹爪闭合
                print("  → 夹爪闭合")
                arm.gripper_close()

                # 4. 抬起
                print("  → 抬起")
                arm.move_joints(q7_pre)

                print("[抓取] 完成！\n")

            elif key == ord(' ') and last_detection is None:
                print("  未检测到目标，无法执行抓取")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        arm._ctrl.disable_all()
        arm._ctrl.close()


if __name__ == "__main__":
    main()
