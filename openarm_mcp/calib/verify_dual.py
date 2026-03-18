#!/usr/bin/env python3
"""
Verify both wrist (eye-in-hand) and chest (eye-to-hand) calibrations simultaneously.

Fix board on table. Move arm to different poses.
Both cameras compute board origin in world frame independently.
If both calibrations are correct, results should match.

  Wrist:  T_world_board = T_world_ee @ T_ee_cam  @ T_cam_board_wrist
  Chest:  T_world_board = T_base_cam @ T_cam_board_chest

Usage:
  cd nanobot/
  DISPLAY=:1 python -m openarm_mcp.calib.verify_dual --canport can2
"""

import sys, os, argparse, json, time
import numpy as np
import cv2
cv2.setLogLevel(0)
import pyrealsense2 as rs
import xarm_can as oa
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from openarm_mcp.ik.xarm_ik_solver import XarmIKSolver

SCENE_XML          = os.path.join(_HERE, "../xml/scene.xml")
CALIB_WRIST        = os.path.join(_HERE, "calib_wrist.json")
CALIB_CHEST        = os.path.join(_HERE, "calib_chest.json")
CALIB_HANDEYE_WRIST = os.path.join(_HERE, "calib_handeye_wrist.json")
CALIB_HANDEYE_CHEST = os.path.join(_HERE, "calib_handeye_chest.json")

SERIAL_WRIST = "352122271543"
SERIAL_CHEST = "314422070707"

COLS, ROWS = 7, 5
SQUARE_M   = 0.030
MARKER_M   = 0.022
DICT_ID    = cv2.aruco.DICT_4X4_50

MOTOR_TYPES = [
    oa.MotorType.DM8009, oa.MotorType.DM8009,
    oa.MotorType.DM4340, oa.MotorType.DM4340,
    oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
]
MASTER_IDS = [0x01,0x02,0x03,0x04,0x05,0x06,0x07]
SLAVE_IDS  = [0x11,0x12,0x13,0x14,0x15,0x16,0x17]
RIGHT_ARM_QPOS_SLICE = slice(9, 16)


def build_fk():
    solver = XarmIKSolver(xml_path=SCENE_XML,
                          right_ee_name="openarm_right_hand_tcp")
    base_qpos = solver.data.qpos.copy()
    def fk(q7):
        qpos = base_qpos.copy()
        qpos[RIGHT_ARM_QPOS_SLICE] = q7
        solver.reset(qpos)
        pos, rot = solver.get_ee_pose("right")
        T = np.eye(4); T[:3,:3] = rot; T[:3,3] = pos
        return T
    return fk


def make_detector():
    dic = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_M, MARKER_M, dic)
    detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    charuco_detector = cv2.aruco.CharucoDetector(board)
    return dic, board, detector, charuco_detector


def detect_board(gray, board, detector, charuco_detector, K, dist):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None, None
    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None
    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < 6:
        return None, None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None, None
    Rm, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3] = Rm; T[:3,3] = tvec.ravel()
    return T, (rvec, tvec, corners, ids, ch_corners, ch_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    args = parser.parse_args()

    # Load calibrations
    with open(CALIB_WRIST) as f: cw = json.load(f)
    with open(CALIB_CHEST) as f: cc = json.load(f)
    with open(CALIB_HANDEYE_WRIST) as f: hw = json.load(f)
    with open(CALIB_HANDEYE_CHEST) as f: hc = json.load(f)

    K_wrist  = np.array(cw["K"],  dtype=np.float64)
    d_wrist  = np.array(cw["dist"], dtype=np.float64)
    K_chest  = np.array(cc["K"],  dtype=np.float64)
    d_chest  = np.array(cc["dist"], dtype=np.float64)
    T_ee_cam   = np.array(hw["T_ee_cam"])
    T_base_cam = np.array(hc["T_base_cam"])

    print("Loading FK solver...")
    fk = build_fk()
    print("FK ready")

    # Start both cameras
    ctx = rs.context()
    pipelines = {}
    for serial, name in [(SERIAL_WRIST, "wrist"), (SERIAL_CHEST, "chest")]:
        p = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        p.start(cfg)
        pipelines[name] = p
    for p in pipelines.values():
        for _ in range(20): p.wait_for_frames()
    print("Both cameras ready")

    dic, board, detector, charuco_detector = make_detector()
    WIN_W = "Wrist Camera"
    WIN_C = "Chest Camera"
    cv2.namedWindow(WIN_W, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN_W, 800, 600)
    cv2.namedWindow(WIN_C, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN_C, 800, 600)

    xarm = oa.XArm(args.canport, True)
    xarm.init_arm_motors(MOTOR_TYPES, MASTER_IDS, SLAVE_IDS)
    xarm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
    xarm.set_callback_mode_all(oa.CallbackMode.STATE)
    xarm.enable_all()
    time.sleep(0.1); xarm.recv_all()
    arm = xarm.get_arm()
    print("Arm ready")
    print("\n=== Dual Camera Verification ===")
    print("Fix board on table, move arm to different poses")
    print("[SPACE] capture  |  [v] result  |  [q] quit\n")

    samples = []  # list of (pos_wrist, pos_chest, diff_mm)

    try:
        while True:
            # Read both cameras
            fw = pipelines["wrist"].wait_for_frames()
            fc = pipelines["chest"].wait_for_frames()
            color_w = np.asanyarray(fw.get_color_frame().get_data())
            color_c = np.asanyarray(fc.get_color_frame().get_data())
            gray_w = cv2.cvtColor(color_w, cv2.COLOR_BGR2GRAY)
            gray_c = cv2.cvtColor(color_c, cv2.COLOR_BGR2GRAY)

            xarm.refresh_all(); xarm.recv_all()
            q7 = np.array([m.get_position() for m in arm.get_motors()])

            T_cam_board_w, dw = detect_board(gray_w, board, detector, charuco_detector, K_wrist, d_wrist)
            T_cam_board_c, dc = detect_board(gray_c, board, detector, charuco_detector, K_chest, d_chest)

            disp_w = color_w.copy()
            disp_c = color_c.copy()

            w_ok = T_cam_board_w is not None
            c_ok = T_cam_board_c is not None

            if w_ok:
                cv2.aruco.drawDetectedMarkers(disp_w, dw[2], dw[3])
                cv2.drawFrameAxes(disp_w, K_wrist, d_wrist, dw[0], dw[1], 0.05)
            if c_ok:
                cv2.aruco.drawDetectedMarkers(disp_c, dc[2], dc[3])
                cv2.drawFrameAxes(disp_c, K_chest, d_chest, dc[0], dc[1], 0.05)

            for disp, ok, name in [(disp_w, w_ok, "Wrist"), (disp_c, c_ok, "Chest")]:
                cv2.rectangle(disp, (0,0), (640,45), (0,0,0), -1)
                color_hint = (0,220,0) if ok else (0,60,220)
                cv2.putText(disp, f"{name}: {'Board detected' if ok else 'No board'}  samples={len(samples)}",
                            (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hint, 2)

            cv2.imshow(WIN_W, cv2.resize(disp_w, (800,600)))
            cv2.imshow(WIN_C, cv2.resize(disp_c, (800,600)))
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and w_ok and c_ok:
                T_world_ee    = fk(q7)

                # Wrist: T_world_board = T_world_ee @ T_ee_cam @ T_cam_board
                T_world_board_w = T_world_ee @ T_ee_cam @ T_cam_board_w
                p_w = T_world_board_w[:3, 3]

                # Chest: T_world_board = T_base_cam @ T_cam_board
                T_world_board_c = T_base_cam @ T_cam_board_c
                p_c = T_world_board_c[:3, 3]

                diff_mm = np.linalg.norm(p_w - p_c) * 1000
                samples.append((p_w, p_c, diff_mm))

                print(f"  [{len(samples):02d}]  wrist: {p_w.round(4)}  chest: {p_c.round(4)}  diff={diff_mm:.1f}mm")

            elif key == ord(' ') and not (w_ok and c_ok):
                missing = []
                if not w_ok: missing.append("wrist")
                if not c_ok: missing.append("chest")
                print(f"  Board not detected by: {', '.join(missing)}")

            elif key == ord('v'):
                _report(samples)

            elif key == ord('q'):
                break

    finally:
        for p in pipelines.values():
            p.stop()
        cv2.destroyAllWindows()
        xarm.disable_all()

    _report(samples)


def _report(samples):
    if not samples:
        print("No samples collected")
        return

    diffs = [s[2] for s in samples]
    pw_list = np.array([s[0] for s in samples])
    pc_list = np.array([s[1] for s in samples])

    print(f"\n=== Dual Camera Verification ({len(samples)} samples) ===")
    print(f"{'':4s}  {'Wrist board pos':30s}  {'Chest board pos':30s}  {'Diff':>8s}")
    for i, (pw, pc, d) in enumerate(samples):
        print(f"  [{i+1:02d}]  {str(pw.round(4)):30s}  {str(pc.round(4)):30s}  {d:6.1f}mm")

    # Wrist consistency
    pw_mean = pw_list.mean(axis=0)
    pw_maxerr = np.linalg.norm(pw_list - pw_mean, axis=1).max() * 1000
    # Chest consistency
    pc_mean = pc_list.mean(axis=0)
    pc_maxerr = np.linalg.norm(pc_list - pc_mean, axis=1).max() * 1000

    print(f"\nWrist  consistency: max_err={pw_maxerr:.1f}mm  mean={pw_mean.round(4)}")
    print(f"Chest  consistency: max_err={pc_maxerr:.1f}mm  mean={pc_mean.round(4)}")
    print(f"Cross-camera diff : mean={np.mean(diffs):.1f}mm  max={np.max(diffs):.1f}mm")
    print(f"\nVerdict:")
    print(f"  Wrist  calib: {'OK' if pw_maxerr < 15 else 'needs recalib'} (max_err={pw_maxerr:.1f}mm, threshold 15mm)")
    print(f"  Chest  calib: {'OK' if pc_maxerr < 15 else 'needs recalib'} (max_err={pc_maxerr:.1f}mm, threshold 15mm)")
    print(f"  Cross-camera: {'OK' if np.mean(diffs) < 30 else 'mismatch'} (mean diff={np.mean(diffs):.1f}mm, threshold 30mm)")


if __name__ == "__main__":
    main()
