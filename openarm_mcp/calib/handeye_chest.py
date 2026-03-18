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

import sys, os, argparse, json, time
import numpy as np
import cv2


def _quiet_cv2_logs():
    try:
        cv2.setLogLevel(0)
        return
    except AttributeError:
        pass

    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass


_quiet_cv2_logs()

from scipy.spatial.transform import Rotation as Rot
import pyrealsense2 as rs
import xarm_can as oa

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

# ── Motor config ──
MOTOR_TYPES = [
    oa.MotorType.DM8009, oa.MotorType.DM8009,
    oa.MotorType.DM4340, oa.MotorType.DM4340,
    oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
]
MASTER_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
SLAVE_IDS  = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]

RIGHT_ARM_QPOS_SLICE = slice(9, 16)


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

    xarm = oa.XArm(args.canport, True)
    xarm.init_arm_motors(MOTOR_TYPES, MASTER_IDS, SLAVE_IDS)
    xarm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
    xarm.set_callback_mode_all(oa.CallbackMode.STATE)
    xarm.enable_all()
    time.sleep(0.1)
    xarm.recv_all()
    arm = xarm.get_arm()
    print("Arm ready")

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

            xarm.refresh_all()
            xarm.recv_all()
            q7 = np.array([m.get_position() for m in arm.get_motors()])

            ok, rvec, tvec, aruco_draw, charuco_draw, reproj_rmse, board_dist = detect(
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
            if ok:
                hint_text = f"Board detected  dist={board_dist:.3f}m  rmse={reproj_rmse:.2f}px"
            else:
                hint_text = "No board - adjust angle"
            cv2.putText(disp, hint_text, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hint_color, 2)

            disp_big = cv2.resize(disp, (1280, 960))
            cv2.imshow(WIN, disp_big)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and ok:
                if board_dist < 0.10 or board_dist > 1.20:
                    print(f"  [skip] invalid board_dist={board_dist:.3f}m")
                    continue
                if reproj_rmse > 2.5:
                    print(f"  [skip] high reproj error rmse={reproj_rmse:.2f}px")
                    continue

                T_base_ee   = fk(q7)
                T_cam_board = to_T(rvec, tvec)
                R_gripper2base.append(T_base_ee[:3, :3])
                t_gripper2base.append(T_base_ee[:3,  3].reshape(3, 1))
                R_target2cam.append(T_cam_board[:3, :3])
                t_target2cam.append(T_cam_board[:3,  3].reshape(3, 1))
                T_base_ee_list.append(T_base_ee)
                T_cam_board_list.append(T_cam_board)
                count += 1
                rpy = Rot.from_matrix(T_base_ee[:3,:3]).as_euler('xyz', degrees=True)
                print(f"  [{count:02d}] ee={T_base_ee[:3,3].round(3)}  "
                      f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f})deg  "
                      f"board_dist={board_dist:.3f}m  rmse={reproj_rmse:.2f}px")

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

    # ── Compute (eye-to-hand: invert EE poses) ──
    R_base2gripper, t_base2gripper = [], []
    for R, t in zip(R_gripper2base, t_gripper2base):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t.ravel()
        Ti = inv_T(T)
        R_base2gripper.append(Ti[:3, :3])
        t_base2gripper.append(Ti[:3,  3].reshape(3, 1))

    print(f"\nComputing eye-to-hand transform using {count} frames (Tsai)...")
    R_x, t_x = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper,
        R_target2cam,   t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_cam_base = np.eye(4)
    T_cam_base[:3, :3] = R_x
    T_cam_base[:3,  3] = t_x.ravel()
    T_base_cam = inv_T(T_cam_base)

    print(f"\nT_cam_base:\n{T_cam_base.round(4)}")
    print(f"\nT_base_cam:\n{T_base_cam.round(4)}")
    print(f"Camera position in base frame (m): {T_base_cam[:3,3].round(4)}")

    # ── Consistency check: T_ee_board should be almost constant across captures ──
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
        if len(r_err) > 0:
            print(f"  EE->Board rotation diff wrt frame0 (deg): mean={np.mean(r_err):.2f}, max={np.max(r_err):.2f}")

    with open(args.out, "w") as f:
        json.dump({
            "T_cam_base": T_cam_base.tolist(),
            "T_base_cam": T_base_cam.tolist(),
            "method": "TSAI",
            "n_frames": count,
        }, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
