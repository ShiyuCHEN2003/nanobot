#!/usr/bin/env python3
"""
相机内参标定脚本（ChArUco board）

用法:
  python intrinsic_calib.py --serial 352122271543 --name wrist
  python intrinsic_calib.py --serial 314422070707 --name chest

操作说明:
  - 把标定板放在相机前方，从不同角度/距离移动
  - 按 [空格] 采集当前帧（检测到角点才能采集）
  - 采集 20 张以上后按 [c] 开始计算
  - 按 [q] 退出

结果保存到 calib_<name>.npz
"""

import argparse
import json
import sys
import numpy as np
import cv2
import pyrealsense2 as rs

# ── 标定板参数（需与 gen_board.py 一致）──
COLS     = 7
ROWS     = 5
SQUARE_M = 0.030
MARKER_M = 0.022
DICT_ID  = cv2.aruco.DICT_4X4_50
MIN_FRAMES = 15   # 最少采集帧数


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_M, MARKER_M, dictionary)
    return board, dictionary


def start_camera(serial: str | None):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    # 暖机
    for _ in range(20):
        pipeline.wait_for_frames()
    return pipeline


def run(serial, name):
    board, dictionary = make_board()
    detector_params = cv2.aruco.DetectorParameters()
    pipeline = start_camera(serial)

    all_corners = []
    all_ids = []
    img_size = None
    count = 0

    print("=== 相机内参标定 ===")
    print("把标定板放在相机前，从各个角度缓慢移动")
    print("[空格] 采集  |  [c] 计算  |  [q] 退出")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = np.asanyarray(frames.get_color_frame().get_data())
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]  # (w, h)

            # 检测 ArUco
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)

            display = color.copy()
            charuco_ok = False

            if ids is not None and len(ids) >= 4:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board)
                if ret and ret >= 6:
                    cv2.aruco.drawDetectedCornersCharuco(display, ch_corners, ch_ids)
                    charuco_ok = True

            status = f"已采集: {count} 帧"
            color_status = (0, 200, 0) if charuco_ok else (0, 0, 200)
            hint = "检测到标定板 ✓ 按[空格]采集" if charuco_ok else "未检测到标定板，请调整角度"
            cv2.putText(display, status, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(display, hint,   (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
            cv2.imshow("Intrinsic Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and charuco_ok:
                all_corners.append(ch_corners)
                all_ids.append(ch_ids)
                count += 1
                print(f"  采集第 {count} 帧")

            elif key == ord('c'):
                if count < MIN_FRAMES:
                    print(f"帧数不足（{count}/{MIN_FRAMES}），继续采集")
                else:
                    break

            elif key == ord('q'):
                print("退出")
                return
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ── 计算内参 ──
    print(f"\n正在用 {count} 帧计算内参...")
    flags = (cv2.CALIB_RATIONAL_MODEL)
    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_size, None, None, flags=flags)

    print(f"\n重投影误差 RMS: {rms:.4f} px  {'✓ 良好' if rms < 1.0 else '⚠ 偏大，建议重新采集'}")
    print(f"\n相机矩阵 K:\n{K}")
    print(f"\n畸变系数 dist:\n{dist.ravel()}")

    out = f"calib_{name}.npz"
    np.savez(out, K=K, dist=dist, rms=rms, img_size=img_size)
    print(f"\n结果已保存到 {out}")

    # 同时保存一份可读的 JSON
    json_out = f"calib_{name}.json"
    with open(json_out, "w") as f:
        json.dump({
            "camera": name,
            "serial": serial,
            "rms": float(rms),
            "img_size": list(img_size),
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
        }, f, indent=2)
    print(f"JSON 格式已保存到 {json_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None, help="RealSense 序列号")
    parser.add_argument("--name",   default="camera", help="相机名称，用于命名输出文件")
    args = parser.parse_args()
    run(args.serial, args.name)


if __name__ == "__main__":
    main()
