#!/usr/bin/env python3
"""
Detect ChArUco board lower-left corner, solve IK, print target joint angles.

Usage:
  cd nanobot/
  DISPLAY=:1 python -m openarm_mcp.calib.reach_board_corner --canport can2 --serial 352122271543

Press [SPACE] to detect board and solve IK.
"""

import sys, os, argparse, json, time
import numpy as np
import cv2
cv2.setLogLevel(0)
import pyrealsense2 as rs
import xarm_can as oa

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


def build_solver():
    solver = XarmIKSolver(xml_path=SCENE_XML,
                          right_ee_name="openarm_right_hand_tcp",
                          orientation_cost=0.0)
    base_qpos = solver.data.qpos.copy()

    def fk(q7):
        qpos = base_qpos.copy()
        qpos[RIGHT_ARM_QPOS_SLICE] = q7
        solver.reset(qpos)
        pos, rot = solver.get_ee_pose("right")
        T = np.eye(4); T[:3, :3] = rot; T[:3, 3] = pos
        return T

    def ik(target_pos, target_rot, q7_init):
        qpos = base_qpos.copy()
        qpos[RIGHT_ARM_QPOS_SLICE] = q7_init
        solver.reset(qpos)
        result = solver.solve_until_converged(
            {"right": (target_pos, target_rot)},
            max_steps=500, tol=1e-4,
        )
        q7 = solver.data.qpos[RIGHT_ARM_QPOS_SLICE].copy()
        return result, q7

    return fk, ik


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

    print("Loading FK/IK solver...")
    fk, ik = build_solver()
    print("Solver ready")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    for _ in range(20):
        pipeline.wait_for_frames()
    print("Camera ready")

    dic, board, detector, charuco_detector = make_detector()
    WIN = "Reach Board Corner  [SPACE] detect+IK  [q] quit"
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
    print("\n[SPACE] detect board and solve IK for bottom-right corner\n")

    last_q_target = None

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

                # Project all 4 corners and label them
                board_corners_3d = np.array([
                    [0,                  0,                 0],   # top-left
                    [COLS * SQUARE_M,    0,                 0],   # top-right
                    [0,                  ROWS * SQUARE_M,   0],   # bottom-left
                    [COLS * SQUARE_M,    ROWS * SQUARE_M,   0],   # bottom-right (TARGET)
                ], dtype=np.float32)
                labels = ["TL", "TR", "BL", "BR(target)"]
                colors = [(200,200,200), (200,200,200), (200,200,200), (0,0,255)]
                pts_2d, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, K, dist)
                for pt, label, color_c in zip(pts_2d, labels, colors):
                    x, y = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(disp, (x, y), 10, color_c, -1)
                    cv2.putText(disp, label, (x+12, y+6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_c, 2)

            cv2.rectangle(disp, (0,0), (640,55), (0,0,0), -1)
            hint = "Board detected - [SPACE] solve IK" if board_ok else "No board"
            cv2.putText(disp, hint, (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,220,0) if board_ok else (0,60,220), 2)

            cv2.imshow(WIN, cv2.resize(disp, (1280, 960)))
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and board_ok:
                T_world_ee    = fk(q7)
                T_world_cam   = T_world_ee @ T_ee_cam
                T_world_board = T_world_cam @ T_cam_board

                # Bottom-right corner in board frame
                p_board = np.array([COLS * SQUARE_M, ROWS * SQUARE_M, 0.0, 1.0])
                corner_world = (T_world_board @ p_board)[:3]

                print(f"\nBoard bottom-right corner in world: {corner_world.round(4)}")
                print("Solving IK (position only)...")

                result, q7_target = ik(corner_world, None, q7)

                print(f"IK result: {result}")
                print(f"\nTarget joint angles (rad):")
                print(f"  {np.round(q7_target, 4).tolist()}")
                print(f"Target joint angles (deg):")
                print(f"  {np.round(np.degrees(q7_target), 2).tolist()}")

                last_q_target = q7_target

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        xarm.disable_all()


if __name__ == "__main__":
    main()
