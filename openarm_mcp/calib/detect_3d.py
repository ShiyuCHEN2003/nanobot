#!/usr/bin/env python3
"""
Perception chain verification (Step B):
  Chest camera detects objects via YOLOv8 → depth → 3D world coordinates.

Usage:
  cd nanobot/
  conda run -n lerobot python -m openarm_mcp.calib.detect_3d

Controls:
  [q] quit
"""

import os, sys, json, time
import numpy as np
import cv2
cv2.setLogLevel(0)
import pyrealsense2 as rs
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

CALIB_CHEST      = os.path.join(_HERE, "calib_chest.json")
CALIB_HANDEYE    = os.path.join(_HERE, "calib_handeye_chest.json")
YOLO_MODEL       = os.path.join(_ROOT, "yolov8n.pt")
SERIAL_CHEST     = "314422070707"

CONF_THRESH = 0.4   # YOLO confidence threshold
DEPTH_MARGIN = 5    # pixel radius to average depth


def load_calib():
    with open(CALIB_CHEST) as f:
        cc = json.load(f)
    with open(CALIB_HANDEYE) as f:
        hc = json.load(f)
    K          = np.array(cc["K"],         dtype=np.float64)
    dist       = np.array(cc["dist"],      dtype=np.float64)
    T_base_cam = np.array(hc["T_base_cam"], dtype=np.float64)
    return K, dist, T_base_cam


def pixel_to_world(u, v, depth_m, K, T_base_cam):
    """Deproject (u, v) + depth → world frame position (m)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) * depth_m / fx
    y_cam = (v - cy) * depth_m / fy
    z_cam = depth_m
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    p_world = T_base_cam @ p_cam
    return p_world[:3]


def sample_depth(depth_frame, u, v, margin):
    """Average valid depth in a square window around (u, v)."""
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    vals = []
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            pu, pv = int(u + dx), int(v + dy)
            if 0 <= pu < w and 0 <= pv < h:
                d = depth_frame.get_distance(pu, pv)
                if d > 0.05:   # ignore invalid / too-close readings
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0


def main():
    K, dist, T_base_cam = load_calib()
    print(f"T_base_cam loaded. Camera pos in world: {T_base_cam[:3,3].round(3)} m")

    model = YOLO(YOLO_MODEL)
    print(f"YOLO loaded: {YOLO_MODEL}")

    # RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL_CHEST)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(cfg)

    align = rs.align(rs.stream.color)

    # Warm-up
    for _ in range(20):
        pipeline.wait_for_frames()
    print("Camera ready")

    WIN = "Chest 3D Detection  [q] quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 960)

    try:
        while True:
            frames    = pipeline.wait_for_frames()
            aligned   = align.process(frames)
            color_f   = aligned.get_color_frame()
            depth_f   = aligned.get_depth_frame()
            if not color_f or not depth_f:
                continue

            color = np.asanyarray(color_f.get_data())
            disp  = color.copy()

            # YOLO detection
            results = model(color, conf=CONF_THRESH, verbose=False)[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]

                cx_px = (x1 + x2) / 2
                cy_px = (y1 + y2) / 2
                depth_m = sample_depth(depth_f, cx_px, cy_px, DEPTH_MARGIN)

                if depth_m > 0:
                    pos_world = pixel_to_world(cx_px, cy_px, depth_m, K, T_base_cam)
                else:
                    pos_world = None

                detections.append((label, conf, cx_px, cy_px, depth_m, pos_world,
                                   x1, y1, x2, y2))

                # Draw bounding box
                color_box = (0, 220, 0) if pos_world is not None else (0, 80, 200)
                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
                cv2.circle(disp, (int(cx_px), int(cy_px)), 5, (0, 0, 255), -1)

                if pos_world is not None:
                    txt = (f"{label} {conf:.2f} | "
                           f"d={depth_m*100:.1f}cm | "
                           f"world=({pos_world[0]*100:.1f},{pos_world[1]*100:.1f},"
                           f"{pos_world[2]*100:.1f})cm")
                else:
                    txt = f"{label} {conf:.2f} | no depth"

                cv2.putText(disp, txt, (int(x1), max(int(y1) - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

                # Console print
                if pos_world is not None:
                    print(f"  {label:15s} conf={conf:.2f}  "
                          f"depth={depth_m*100:.1f}cm  "
                          f"world=({pos_world[0]*100:.1f}, "
                          f"{pos_world[1]*100:.1f}, "
                          f"{pos_world[2]*100:.1f}) cm")

            if not detections:
                cv2.putText(disp, "No detections", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 60, 220), 2)

            cv2.imshow(WIN, cv2.resize(disp, (1280, 960)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
