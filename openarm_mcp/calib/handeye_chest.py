#!/usr/bin/env python3
"""
Chest camera hand-eye calibration (eye-to-hand)

Usage:
  python handeye_chest.py --canport can2 --serial 314422070707

Steps:
  Attach ChArUco board to the robot arm end-effector
  Move the arm so the chest camera sees the board clearly
  Press [SPACE] to capture (board must be detected)
  Press [c] to compute after 15+ frames
  Press [q] to quit

Result: T_cam_base  (base origin in camera frame)
        T_base_cam  (camera origin in base frame)
"""

import sys, os, argparse, json, time, subprocess, threading
import numpy as np
import cv2
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass
from scipy.spatial.transform import Rotation as Rot
import pyrealsense2 as rs

# Add parent directory to path for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_NANOBOT = os.path.dirname(os.path.dirname(_HERE))  # Go up to nanobot/
if _NANOBOT not in sys.path:
    sys.path.insert(0, _NANOBOT)

from openarm_mcp.ik.xarm_ik_solver import XarmIKSolver

# ── Paths ──
SCENE_XML = os.path.join(_HERE, "../xml/scene.xml")
CALIB_IN  = os.path.join(_HERE, "calib_chest.json")

# ── Board parameters ──
COLS, ROWS = 7, 5
SQUARE_M   = 0.030
MARKER_M   = 0.022
DICT_ID    = cv2.aruco.DICT_4X4_50
MIN_FRAMES = 15

RIGHT_ARM_QPOS_SLICE = slice(9, 16)


class RosJointReader:
    """用 Python 3.10 子进程读 ROS2 joint states，通过 stdout JSON 行传给主进程。"""
    def __init__(self, topic: str, arm_side: str = "right"):
        self.latest_q7 = None
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ros_joint_helper.py")
        cmd = f"source /opt/ros/humble/setup.bash && /usr/bin/python3 {helper} {topic} {arm_side}"
        self._proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for line in self._proc.stdout:
            try:
                self.latest_q7 = np.array(json.loads(line.strip()), dtype=float)
            except Exception:
                pass

    def destroy(self):
        self._proc.terminate()
        self._proc.wait(timeout=2)


# ────────────────────────────────────────────
# FK via XarmIKSolver
# ────────────────────────────────────────────
def build_fk(scene_xml):
    solver = XarmIKSolver(
        xml_path=scene_xml,
        right_ee_name="openarm_right_hand_tcp",
    )
    base_qpos = solver.data.qpos.copy()

    def fk(q7: np.ndarray) -> np.ndarray:
        qpos = base_qpos.copy()
        qpos[RIGHT_ARM_QPOS_SLICE] = q7
        solver.reset(qpos)
        pos, rot = solver.get_ee_pose("right")
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3,  3] = pos
        return T

    return fk


# ────────────────────────────────────────────
# Board detection
# ────────────────────────────────────────────
def make_detector():
    dic      = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board    = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_M, MARKER_M, dic)
    par      = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dic, par)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    return dic, board, detector, charuco_detector


def detect(gray, dic, board, detector, charuco_detector, K, dist):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return False, None, None, None, None, None, None

    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return False, None, None, None, None, None, None

    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < 4:
        return False, None, None, None, None, None, None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        K,
        dist,
        iterationsCount=100,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return False, None, None, None, None, None, None

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    img  = img_pts.reshape(-1, 2)
    reproj_rmse = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
    board_dist = float(np.linalg.norm(tvec))

    return True, rvec, tvec, (corners, ids), (ch_corners, ch_ids), reproj_rmse, board_dist


def to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return T


def rvec_to_rpy_deg(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return Rot.from_matrix(R).as_euler('xyz', degrees=True)


def draw_frame_axes_safe(img, K, dist, rvec, tvec, axis_len=0.05):
    h, w = img.shape[:2]
    obj = np.float32([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ])
    pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    pts = pts.reshape(-1, 2)

    def in_view(p):
        return 0 <= p[0] < w and 0 <= p[1] < h

    o, x, y, z = pts
    if in_view(o) and in_view(x):
        cv2.line(img, tuple(np.int32(o)), tuple(np.int32(x)), (0, 0, 255), 2)
    if in_view(o) and in_view(y):
        cv2.line(img, tuple(np.int32(o)), tuple(np.int32(y)), (0, 255, 0), 2)
    if in_view(o) and in_view(z):
        cv2.line(img, tuple(np.int32(o)), tuple(np.int32(z)), (255, 0, 0), 2)


def inv_T(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3,  3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


def rot_angle_deg(Ra, Rb):
    r = Ra.T @ Rb
    c = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    parser.add_argument("--serial",  default="314422070707")
    parser.add_argument("--out",     default="calib_handeye_chest.json")
    parser.add_argument("--joint-topic", default="/openarm_leader/joint_states")
    parser.add_argument("--arm-side", default="right", choices=["right", "left"])
    args = parser.parse_args()

    with open(CALIB_IN) as f:
        c = json.load(f)
    K    = np.array(c["K"],   dtype=np.float64)
    dist = np.array(c["dist"], dtype=np.float64)

    print("Loading FK solver...")
    fk = build_fk(SCENE_XML)
    print("FK ready")

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    for _ in range(20):
        pipeline.wait_for_frames()
    print("Camera ready")

    dic, board, detector, charuco_detector = make_detector()
    WIN = "Chest Eye-to-Hand Calib  [SPACE] capture  [c] compute  [q] quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 960)

    joint_reader = RosJointReader(args.joint_topic, args.arm_side)
    print(f"ROS joint reader ready: topic={args.joint_topic}, arm_side={args.arm_side}")
    if "follower" in args.joint_topic.lower():
        print("[WARN] 当前使用 follower 话题，可能是目标关节而非实测关节，手眼标定易失真。")
        print("[WARN] 建议改为 /openarm_leader/joint_states 或对应硬件反馈话题。")

    # eye-to-hand: collect T_base_ee (FK) and T_cam_board (PnP)
    R_gripper2base, t_gripper2base = [], []
    R_target2cam,   t_target2cam   = [], []
    T_base_ee_list, T_cam_board_list = [], []
    count = 0

    print("\n=== Eye-to-Hand Calibration (Chest Camera) ===")
    print("Attach board to arm EE, move arm so chest camera sees board")
    print("[SPACE] capture  |  [c] compute  |  [q] quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color  = np.asanyarray(frames.get_color_frame().get_data())
            gray   = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            q7 = joint_reader.latest_q7

            ok, rvec, tvec, aruco_draw, charuco_draw, reproj_rmse, board_dist = detect(
                gray, dic, board, detector, charuco_detector, K, dist)

            disp = color.copy()
            if ok:
                cv2.aruco.drawDetectedMarkers(disp, aruco_draw[0], aruco_draw[1])
                cv2.aruco.drawDetectedCornersCharuco(disp, charuco_draw[0], charuco_draw[1])
                draw_frame_axes_safe(disp, K, dist, rvec, tvec, axis_len=0.05)

            cv2.rectangle(disp, (0, 0), (640, 130), (0, 0, 0), -1)
            q7_text = np.round(q7, 2) if q7 is not None else "waiting ROS joints..."
            cv2.putText(disp, f"Captured: {count}  q7={q7_text}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            hint_color = (0, 220, 0) if ok else (0, 60, 220)
            if ok:
                hint_text = f"Board detected  dist={board_dist:.3f}m  rmse={reproj_rmse:.2f}px"
            else:
                hint_text = "No board - adjust angle"
            cv2.putText(disp, hint_text, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hint_color, 2)
            if ok:
                t = tvec.ravel()
                rpy_cam_board = rvec_to_rpy_deg(rvec)
                cv2.putText(
                    disp,
                    f"cam->board t=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})m  rpy=({rpy_cam_board[0]:+.1f},{rpy_cam_board[1]:+.1f},{rpy_cam_board[2]:+.1f})deg",
                    (10, 94),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
            if q7 is None:
                cv2.putText(disp, "No JointState yet - check ROS topic", (10, 122),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)

            disp_big = cv2.resize(disp, (1280, 960))
            cv2.imshow(WIN, disp_big)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and ok:
                if q7 is None:
                    print("  [skip] no ROS JointState data yet")
                    continue
                if board_dist < 0.10 or board_dist > 1.20:
                    print(f"  [skip] invalid board_dist={board_dist:.3f}m")
                    continue
                if reproj_rmse > 2.5:
                    print(f"  [skip] high reproj error rmse={reproj_rmse:.2f}px")
                    continue

                T_base_ee   = fk(q7)
                T_cam_board = to_T(rvec, tvec)
                
                # Debug: print raw transformation matrices
                print(f"\n--- 帧{count+1} 原始矩阵 ---")
                print(f"T_base_ee:\n{T_base_ee.round(4)}")
                print(f"T_cam_board:\n{T_cam_board.round(4)}")
                
                T_ee_base = inv_T(T_base_ee)  # eye-to-hand 需要传 base2gripper（取逆）
                R_gripper2base.append(T_ee_base[:3, :3])
                t_gripper2base.append(T_ee_base[:3,  3].reshape(3, 1))
                R_target2cam.append(T_cam_board[:3, :3])
                t_target2cam.append(T_cam_board[:3,  3].reshape(3, 1))
                T_base_ee_list.append(T_base_ee)
                T_cam_board_list.append(T_cam_board)
                count += 1
                rpy = Rot.from_matrix(T_base_ee[:3,:3]).as_euler('xyz', degrees=True)
                rpy_cam_board = rvec_to_rpy_deg(rvec)
                t_cam_board = T_cam_board[:3, 3]
                print(f"  [{count:02d}] ee={T_base_ee[:3,3].round(3)}  "
                      f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f})deg  "
                      f"board_dist={board_dist:.3f}m  rmse={reproj_rmse:.2f}px")
                print(f"       cam->board t=({t_cam_board[0]:+.3f},{t_cam_board[1]:+.3f},{t_cam_board[2]:+.3f})m  "
                      f"rpy=({rpy_cam_board[0]:+.1f},{rpy_cam_board[1]:+.1f},{rpy_cam_board[2]:+.1f})deg")

            elif key == ord('c'):
                if count < MIN_FRAMES:
                    print(f"Not enough frames ({count}/{MIN_FRAMES}), keep capturing")
                else:
                    break

            elif key == ord('q'):
                print("Quit")
                return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        joint_reader.destroy()

    # ── Compute (eye-to-hand) ──
    # eye-to-hand 正确约定：
    #   第1组传 T_gripper2base（即 T_base_ee，FK直接输出，不需要取逆）
    #   第2组传 T_target2cam  （即 T_cam_board，PnP直接输出）
    # 结果 R_x, t_x 是 T_cam_base（相机相对于base的变换）
    from scipy.optimize import minimize

    # ── 方法1: calibrateHandEye (使用多种方法取最优) ──
    best_T_base_cam = None
    best_std = float('inf')

    for method_name, method in [
        ("TSAI",   cv2.CALIB_HAND_EYE_TSAI),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
        ("ANDREFF",cv2.CALIB_HAND_EYE_ANDREFF),
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]:
        try:
            # eye-to-hand 
            R_x, t_x = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,   # inv(T_base_ee) 的 R, t（eye-to-hand 需取逆）
                R_target2cam,   t_target2cam,      # T_cam_board 的 R, t，结果为 T_base_cam
                method=method,
            )
            T_base_cam_try = np.eye(4)
            T_base_cam_try[:3, :3] = R_x
            T_base_cam_try[:3,  3] = t_x.ravel()

            # 计算 consistency std
            pts = []
            for T_be, T_cb in zip(T_base_ee_list, T_cam_board_list):
                T_eb = inv_T(T_be) @ T_base_cam_try @ T_cb
                pts.append(T_eb[:3, 3])
            pts = np.array(pts)
            std_total = float(np.sum(np.std(pts, axis=0)) * 1000)
            print(f"  {method_name:12s}: std={std_total:.1f}mm  cam_pos={T_base_cam_try[:3,3].round(3)}")

            if std_total < best_std:
                best_std = std_total
                best_T_base_cam = T_base_cam_try
        except Exception as e:
            print(f"  {method_name:12s}: 失败 {e}")

    # ── 方法2: 直接优化（当 calibrateHandEye 结果差时自动启用）──
    print(f"\ncalibrateHandEye 最优 std={best_std:.1f}mm，尝试直接优化...")

    def cost(params):
        R_bc = Rot.from_rotvec(params[:3]).as_matrix()
        T_bc = np.eye(4)
        T_bc[:3, :3] = R_bc
        T_bc[:3,  3] = params[3:]
        pts = []
        for T_be, T_cb in zip(T_base_ee_list, T_cam_board_list):
            pts.append((inv_T(T_be) @ T_bc @ T_cb)[:3, 3])
        pts = np.array(pts)
        return float(np.sum(np.var(pts, axis=0)))

    # 用 calibrateHandEye 最优结果作为初始值
    r0 = Rot.from_matrix(best_T_base_cam[:3, :3]).as_rotvec()
    t0 = best_T_base_cam[:3, 3]
    res = minimize(cost, np.concatenate([r0, t0]), method='Nelder-Mead',
                   options={'maxiter': 200000, 'xatol': 1e-7, 'fatol': 1e-9})

    T_base_cam_opt = np.eye(4)
    T_base_cam_opt[:3, :3] = Rot.from_rotvec(res.x[:3]).as_matrix()
    T_base_cam_opt[:3,  3] = res.x[3:]

    pts_opt = []
    for T_be, T_cb in zip(T_base_ee_list, T_cam_board_list):
        pts_opt.append((inv_T(T_be) @ T_base_cam_opt @ T_cb)[:3, 3])
    pts_opt = np.array(pts_opt)
    std_opt = float(np.sum(np.std(pts_opt, axis=0)) * 1000)
    print(f"  直接优化:    std={std_opt:.1f}mm  cam_pos={T_base_cam_opt[:3,3].round(3)}")

    # 选择更好的结果
    if std_opt < best_std:
        T_base_cam = T_base_cam_opt
        method_used = "direct_optimization"
        final_std = std_opt
        print("  → 使用直接优化结果")
    else:
        T_base_cam = best_T_base_cam
        method_used = "calibrateHandEye"
        final_std = best_std
        print("  → 使用 calibrateHandEye 结果")

    T_cam_base = inv_T(T_base_cam)

    print(f"\nT_cam_base:\n{T_cam_base.round(4)}")
    print(f"\nT_base_cam:\n{T_base_cam.round(4)}")
    print(f"Camera position in base frame (m): {T_base_cam[:3,3].round(4)}")
    cam_z = T_base_cam[:3, :3] @ np.array([0, 0, 1])
    print(f"Camera optical axis in base frame: {cam_z.round(4)}")

    # ── Consistency check ──
    ee_board_Ts = []
    for T_base_ee, T_cam_board in zip(T_base_ee_list, T_cam_board_list):
        T_ee_board = inv_T(T_base_ee) @ T_base_cam @ T_cam_board
        ee_board_Ts.append(T_ee_board)

    if len(ee_board_Ts) >= 2:
        p = np.array([T[:3, 3] for T in ee_board_Ts])
        p_std_mm = np.std(p, axis=0) * 1000.0
        r_ref = ee_board_Ts[0][:3, :3]
        r_err = [rot_angle_deg(r_ref, T[:3, :3]) for T in ee_board_Ts[1:]]
        print("\n[Quality]")
        print(f"  EE->Board position std (mm): {np.round(p_std_mm, 2)}")
        print(f"  EE->Board mean t (m): {p.mean(axis=0).round(4)}  (板固定在EE上应为固定小值)")
        if len(r_err) > 0:
            print(f"  EE->Board rotation diff wrt frame0 (deg): mean={np.mean(r_err):.2f}, max={np.max(r_err):.2f}")

    with open(args.out, "w") as f:
        json.dump({
            "T_cam_base": T_cam_base.tolist(),
            "T_base_cam": T_base_cam.tolist(),
            "method": method_used,
            "n_frames": count,
            "consistency_std_mm": np.std(p, axis=0).tolist() if len(ee_board_Ts) >= 2 else None,
            # 保存原始数据供离线重分析
            "T_base_ee_list": [T.tolist() for T in T_base_ee_list],
            "T_cam_board_list": [T.tolist() for T in T_cam_board_list],
        }, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()