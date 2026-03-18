#!/usr/bin/env python3
"""
Wrist camera hand-eye calibration (eye-in-hand)

Usage:
  python handeye_calib.py --canport can2 --serial 352122271543

Steps:
  Fix ChArUco board on table, move arm so camera sees board
  Press [SPACE] to capture (board must be detected)
  Press [c] to compute after 15+ frames
  Press [q] to quit
"""

import sys, os, argparse, json, time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
cv2.setLogLevel(0)  # 屏蔽 OpenCV C++ 层警告（如 drawFrameAxes out-of-frame）
import pyrealsense2 as rs
import xarm_can as oa
import mujoco

# 支持直接执行脚本
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openarm_mcp.ik.xarm_ik_solver import XarmIKSolver

# ── Paths ──
SCENE_XML = os.path.join(_HERE, "../xml/scene.xml")
CALIB_IN  = os.path.join(_HERE, "calib_wrist.json")

# ── Board parameters ──
COLS, ROWS = 7, 5
SQUARE_M   = 0.030
MARKER_M   = 0.022
DICT_ID    = cv2.aruco.DICT_4X4_50
MIN_FRAMES = 15

# ── Motor config ──
MOTOR_TYPES = [
    oa.MotorType.DM8009, oa.MotorType.DM8009,
    oa.MotorType.DM4340, oa.MotorType.DM4340,
    oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
]
MASTER_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
SLAVE_IDS  = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]

# Right arm joints occupy qpos[9:16] in the dual-arm scene.xml
RIGHT_ARM_QPOS_SLICE = slice(9, 16)


# ────────────────────────────────────────────
# FK via XarmIKSolver (mujoco + mink)
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
        return False, None, None, None, None

    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return False, None, None, None, None

    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < 4:
        return False, None, None, None, None

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return False, None, None, None, None

    return True, rvec, tvec, (corners, ids), (ch_corners, ch_ids)


def to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return T


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    parser.add_argument("--serial",  default="352122271543")
    parser.add_argument("--out",     default="calib_handeye_wrist.json")
    args = parser.parse_args()

    # Camera intrinsics
    with open(CALIB_IN) as f:
        c = json.load(f)
    K    = np.array(c["K"],   dtype=np.float64)
    dist = np.array(c["dist"], dtype=np.float64)

    # FK
    print("Loading FK solver...")
    fk = build_fk(SCENE_XML)
    print("FK ready")

    # Camera
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
    WIN = "Hand-Eye Calibration  [SPACE] capture  [c] compute  [q] quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 960)

    # Arm (read-only)
    xarm = oa.XArm(args.canport, True)
    xarm.init_arm_motors(MOTOR_TYPES, MASTER_IDS, SLAVE_IDS)
    xarm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
    xarm.set_callback_mode_all(oa.CallbackMode.STATE)
    xarm.enable_all()
    time.sleep(0.1)
    xarm.recv_all()
    arm = xarm.get_arm()
    print("Arm ready")

    R_ee, t_ee, R_cam, t_cam = [], [], [], []
    count = 0

    print("\n=== Eye-in-Hand Calibration ===")
    print("Fix board on table, move arm so camera sees board")
    print("[SPACE] capture  |  [c] compute  |  [q] quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color  = np.asanyarray(frames.get_color_frame().get_data())
            gray   = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            xarm.refresh_all()
            xarm.recv_all()
            q7 = np.array([m.get_position() for m in arm.get_motors()])

            ok, rvec, tvec, aruco_draw, charuco_draw = detect(
                gray, dic, board, detector, charuco_detector, K, dist)

            disp = color.copy()
            if ok:
                cv2.aruco.drawDetectedMarkers(disp, aruco_draw[0], aruco_draw[1])
                cv2.aruco.drawDetectedCornersCharuco(disp, charuco_draw[0], charuco_draw[1])
                cv2.drawFrameAxes(disp, K, dist, rvec, tvec, 0.05)

            cv2.rectangle(disp, (0, 0), (640, 80), (0, 0, 0), -1)
            cv2.putText(disp, f"Captured: {count}  q7={np.round(q7,2)}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            hint_color = (0, 220, 0) if ok else (0, 60, 220)
            hint_text = "Board detected  [SPACE] capture" if ok else "No board - adjust angle"
            cv2.putText(disp, hint_text, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hint_color, 2)

            disp_big = cv2.resize(disp, (1280, 960))
            cv2.imshow(WIN, disp_big)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and ok:
                T_ee_mat  = fk(q7)
                T_cb_mat  = to_T(rvec, tvec)
                R_ee.append(T_ee_mat[:3, :3])
                t_ee.append(T_ee_mat[:3,  3].reshape(3, 1))
                R_cam.append(T_cb_mat[:3, :3])
                t_cam.append(T_cb_mat[:3,  3].reshape(3, 1))
                count += 1
                rpy = Rot.from_matrix(T_ee_mat[:3,:3]).as_euler('xyz', degrees=True)
                print(f"  [{count:02d}] ee={T_ee_mat[:3,3].round(3)}  "
                      f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f})deg  "
                      f"board_dist={np.linalg.norm(tvec):.3f}m")

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
        xarm.disable_all()

    # ── Compute ──
    print(f"\nComputing hand-eye transform using {count} frames (Tsai)...")
    R_x, t_x = cv2.calibrateHandEye(R_ee, t_ee, R_cam, t_cam,
                                     method=cv2.CALIB_HAND_EYE_TSAI)
    T_ee_cam = np.eye(4)
    T_ee_cam[:3, :3] = R_x
    T_ee_cam[:3,  3] = t_x.ravel()

    print(f"\nT_ee_cam:\n{T_ee_cam.round(4)}")
    print(f"Translation (m): {t_x.ravel().round(4)}")

    with open(args.out, "w") as f:
        json.dump({"T_ee_cam": T_ee_cam.tolist(),
                   "method": "TSAI", "n_frames": count}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
