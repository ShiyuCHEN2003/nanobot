#!/usr/bin/env python3
"""
Verify wrist camera hand-eye calibration (consistency method).

Fix the board on the table. Move arm to 5+ different poses.
For each pose compute:
  T_world_board = T_world_ee @ T_ee_cam @ T_cam_board

If calibration is correct, all poses should give the same T_world_board.
Report position std dev (mm) and rotation std dev (deg).

Usage (from nanobot/):
  DISPLAY=:1 python -m openarm_mcp.calib.verify_handeye --canport can2 --serial 352122271543
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

SCENE_XML     = os.path.join(_HERE, "../xml/scene.xml")
CALIB_WRIST   = os.path.join(_HERE, "calib_wrist.json")
CALIB_HANDEYE = os.path.join(_HERE, "calib_handeye_wrist.json")

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
        T = np.eye(4); T[:3, :3] = rot; T[:3, 3] = pos
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
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = tvec.ravel()
    return T, (rvec, tvec, corners, ids, ch_corners, ch_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    parser.add_argument("--serial",  default="352122271543")
    args = parser.parse_args()

    with open(CALIB_WRIST) as f:
        c = json.load(f)
    K    = np.array(c["K"],    dtype=np.float64)
    dist = np.array(c["dist"], dtype=np.float64)

    with open(CALIB_HANDEYE) as f:
        h = json.load(f)
    T_ee_cam = np.array(h["T_ee_cam"])

    print("Loading FK solver...")
    fk = build_fk()
    print("FK ready")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    for _ in range(20):
        pipeline.wait_for_frames()
    print("Camera ready")

    dic, board, detector, charuco_detector = make_detector()
    WIN = "Verify Hand-Eye  [SPACE] sample  [q] quit"
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
    print("\n=== Hand-Eye Verification (Consistency Method) ===")
    print("Fix board on table, move arm to different poses")
    print("[SPACE] capture  |  [v] result  |  [q] quit\n")

    board_poses = []  # list of T_world_board per sample

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color  = np.asanyarray(frames.get_color_frame().get_data())
            gray   = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            xarm.refresh_all(); xarm.recv_all()
            q7 = np.array([m.get_position() for m in arm.get_motors()])

            T_cam_board, draw_data = detect_board(
                gray, board, detector, charuco_detector, K, dist)

            disp = color.copy()
            board_ok = T_cam_board is not None
            if board_ok and draw_data is not None:
                rvec, tvec, corners, ids, ch_corners, ch_ids = draw_data
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)
                cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids)
                cv2.drawFrameAxes(disp, K, dist, rvec, tvec, 0.05)

            cv2.rectangle(disp, (0,0), (640,85), (0,0,0), -1)
            cv2.putText(disp, f"Samples: {len(board_poses)}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            hint = "Board detected - [SPACE] capture" if board_ok else "No board detected"
            cv2.putText(disp, hint, (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0,220,0) if board_ok else (0,60,220), 2)

            cv2.imshow(WIN, cv2.resize(disp, (1280, 960)))
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and board_ok:
                T_world_ee    = fk(q7)
                T_world_board = T_world_ee @ T_ee_cam @ T_cam_board
                board_poses.append(T_world_board)
                p = T_world_board[:3, 3]
                print(f"  [{len(board_poses):02d}] board_origin in world: {p.round(4)}")

            elif key == ord('v'):
                _report(board_poses)

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        xarm.disable_all()

    _report(board_poses)


def _report(board_poses):
    if len(board_poses) < 2:
        print("Need at least 2 samples")
        return
    positions = np.array([T[:3, 3] for T in board_poses])
    mean_pos  = positions.mean(axis=0)
    max_err   = np.linalg.norm(positions - mean_pos, axis=1).max() * 1000
    std_mm    = np.linalg.norm(positions.std(axis=0)) * 1000

    rots = [Rot.from_matrix(T[:3, :3]) for T in board_poses]
    mean_rot  = Rot.mean(rots)
    rot_errs  = [np.degrees((mean_rot.inv() * r).magnitude()) for r in rots]

    print(f"\n=== Verification Result ({len(board_poses)} samples) ===")
    print(f"Board position in world (m):")
    for i, T in enumerate(board_poses):
        print(f"  [{i+1:02d}] {T[:3,3].round(4)}")
    print(f"Mean pos : {mean_pos.round(4)}")
    print(f"Pos std  : {std_mm:.1f} mm   Max err: {max_err:.1f} mm")
    print(f"Rot max  : {max(rot_errs):.2f} deg")
    ok = max_err < 15.0 and max(rot_errs) < 3.0
    print(f"Verdict  : {'OK' if ok else 'Too large - recalibrate'} "
          f"(threshold: <15mm, <3deg)")


if __name__ == "__main__":
    main()
