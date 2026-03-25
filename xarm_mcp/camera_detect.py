#!/usr/bin/env python3
"""
RealSense wrist camera capture + YOLOv8 detection.
Called as subprocess by the MCP server.

Usage:
  python camera_detect.py capture --output /tmp/frame.jpg [--serial XXXXXX]
  python camera_detect.py detect --image /tmp/frame.jpg [--model yolov8n.pt] [--conf 0.25]
"""

import argparse
import json
import sys
import os


def capture(args):
    import pyrealsense2 as rs
    import numpy as np
    import cv2

    pipeline = rs.pipeline()
    config = rs.config()

    if args.serial:
        config.enable_device(args.serial)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    try:
        # 丢弃前几帧，等待曝光稳定
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print(json.dumps({"error": "no color frame"}))
            sys.exit(1)

        img = np.asanyarray(color_frame.get_data())
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        cv2.imwrite(args.output, img)
        print(json.dumps({"ok": True, "path": args.output,
                          "width": img.shape[1], "height": img.shape[0]}))
    finally:
        pipeline.stop()


def detect(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model(args.image, conf=args.conf, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],  # x1,y1,x2,y2
            })

    print(json.dumps({"detections": detections}))


def detect_world(args):
    import json as _json
    import numpy as np
    import pyrealsense2 as rs
    import cv2
    from ultralytics import YOLOWorld

    # ── 加载标定 ──
    calib_dir = args.calib_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")
    with open(os.path.join(calib_dir, "calib_chest.json")) as f:
        cc = _json.load(f)
    with open(os.path.join(calib_dir, "calib_handeye_chest.json")) as f:
        hc = _json.load(f)
    K          = np.array(cc["K"],         dtype=np.float64)
    T_base_cam = np.array(hc["T_base_cam"], dtype=np.float64)

    def _pixel_to_world(u, v, depth_m):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        p_cam = np.array([(u-cx)*depth_m/fx, (v-cy)*depth_m/fy, depth_m, 1.0])
        return (T_base_cam @ p_cam)[:3]

    def _sample_depth(depth_frame, u, v, margin=5):
        h, w = depth_frame.get_height(), depth_frame.get_width()
        vals = [depth_frame.get_distance(int(u+dx), int(v+dy))
                for dy in range(-margin, margin+1)
                for dx in range(-margin, margin+1)
                if 0 <= int(u+dx) < w and 0 <= int(v+dy) < h]
        vals = [d for d in vals if d > 0.05]
        return float(np.median(vals)) if vals else 0.0

    # ── 打开相机（color + depth 对齐）──
    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(10):
            pipeline.wait_for_frames(timeout_ms=10000)
        frames   = align.process(pipeline.wait_for_frames(timeout_ms=10000))
        color_f  = frames.get_color_frame()
        depth_f  = frames.get_depth_frame()
        if not color_f or not depth_f:
            print(_json.dumps({"error": "no frames"})); return
        color = np.asanyarray(color_f.get_data())
    finally:
        pipeline.stop()

    # 保存彩色图供查看
    cv2.imwrite("/tmp/nanobot_chest.jpg", color)

    # ── YOLO-World 推理 ──
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    model = YOLOWorld(args.model)
    model.set_classes(labels)
    results = model(color, conf=args.conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        u, v = (x1+x2)/2, (y1+y2)/2
        depth_m = _sample_depth(depth_f, u, v)
        world_xyz = _pixel_to_world(u, v, depth_m).tolist() if depth_m > 0 else None
        detections.append({
            "label":      labels[int(box.cls[0])] if int(box.cls[0]) < len(labels) else model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox":       [round(x, 1) for x in [x1, y1, x2, y2]],
            "depth_m":    round(depth_m, 3),
            "world_xyz_m": [round(v, 4) for v in world_xyz] if world_xyz else None,
        })

    print(_json.dumps({"detections": detections}))


def capture_world(args):
    """一次性采集 RGB + depth，保存到磁盘（相机只开一次）。"""
    import pyrealsense2 as rs
    import numpy as np
    import cv2
    import json as _json

    rgb_path   = args.output or "/tmp/nanobot_chest.jpg"
    depth_path = rgb_path.replace(".jpg", "_depth.npy")

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(10):
            pipeline.wait_for_frames(timeout_ms=10000)
        frames  = align.process(pipeline.wait_for_frames(timeout_ms=10000))
        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        if not color_f or not depth_f:
            print(_json.dumps({"error": "no frames"}))
            return
        color = np.asanyarray(color_f.get_data())
        depth = np.asanyarray(depth_f.get_data())   # uint16，单位 mm
    finally:
        pipeline.stop()

    cv2.imwrite(rgb_path, color)
    np.save(depth_path, depth)
    print(_json.dumps({"ok": True, "image": rgb_path, "depth": depth_path,
                       "width": color.shape[1], "height": color.shape[0]}))


def detect_world_offline(args):
    """用磁盘上的 RGB + depth.npy 运行 YOLO-World，无需开相机。"""
    import json as _json
    import numpy as np
    import cv2
    from ultralytics import YOLOWorld

    calib_dir = args.calib_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")
    with open(os.path.join(calib_dir, "calib_chest.json")) as f:
        cc = _json.load(f)
    with open(os.path.join(calib_dir, "calib_handeye_chest.json")) as f:
        hc = _json.load(f)
    K          = np.array(cc["K"],          dtype=np.float64)
    T_base_cam = np.array(hc["T_base_cam"], dtype=np.float64)

    def _pixel_to_world(u, v, depth_m):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p_cam = np.array([(u - cx) * depth_m / fx,
                          (v - cy) * depth_m / fy,
                          depth_m, 1.0])
        return (T_base_cam @ p_cam)[:3]

    def _sample_depth_array(u, v, margin=5):
        h, w = depth_raw.shape
        vals = [depth_raw[int(v + dy), int(u + dx)] / 1000.0
                for dy in range(-margin, margin + 1)
                for dx in range(-margin, margin + 1)
                if 0 <= int(u + dx) < w and 0 <= int(v + dy) < h]
        vals = [d for d in vals if d > 0.05]
        return float(np.median(vals)) if vals else 0.0

    color     = cv2.imread(args.image)
    depth_raw = np.load(args.depth)   # uint16 mm，shape (H, W)

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    model  = YOLOWorld(args.model)
    model.set_classes(labels)
    results = model(color, conf=args.conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        u, v    = (x1 + x2) / 2, (y1 + y2) / 2
        depth_m = _sample_depth_array(u, v)
        world_xyz = _pixel_to_world(u, v, depth_m).tolist() if depth_m > 0 else None
        lbl_idx   = int(box.cls[0])
        detections.append({
            "label":       labels[lbl_idx] if lbl_idx < len(labels) else "unknown",
            "confidence":  round(float(box.conf[0]), 3),
            "bbox":        [round(x, 1) for x in [x1, y1, x2, y2]],
            "depth_m":     round(depth_m, 3),
            "world_xyz_m": [round(c, 4) for c in world_xyz] if world_xyz else None,
        })
    print(_json.dumps({"detections": detections}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--output", default="/tmp/nanobot_wrist.jpg")
    p_cap.add_argument("--serial", default=None, help="RealSense serial number")

    p_det = sub.add_parser("detect")
    p_det.add_argument("--image", required=True)
    p_det.add_argument("--model", default="yolov8n.pt")
    p_det.add_argument("--conf", type=float, default=0.25)

    p_world = sub.add_parser("detect_world")
    p_world.add_argument("--serial", default=None)
    p_world.add_argument("--labels", required=True, help="逗号分隔的目标名称")
    p_world.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov8s-world.pt"))
    p_world.add_argument("--conf", type=float, default=0.1)
    p_world.add_argument("--calib-dir", default=None)

    p_cw = sub.add_parser("capture_world")
    p_cw.add_argument("--output", default="/tmp/nanobot_chest.jpg")
    p_cw.add_argument("--serial", default=None)

    _default_model_world = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov8s-world.pt")
    p_dwo = sub.add_parser("detect_world_offline")
    p_dwo.add_argument("--image",     required=True)
    p_dwo.add_argument("--depth",     required=True, help="depth .npy 文件路径")
    p_dwo.add_argument("--labels",    required=True, help="逗号分隔的目标名称")
    p_dwo.add_argument("--model",     default=_default_model_world)
    p_dwo.add_argument("--conf",      type=float, default=0.1)
    p_dwo.add_argument("--calib-dir", default=None)

    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args)
    elif args.cmd == "detect":
        detect(args)
    elif args.cmd == "detect_world":
        detect_world(args)
    elif args.cmd == "capture_world":
        capture_world(args)
    elif args.cmd == "detect_world_offline":
        detect_world_offline(args)


if __name__ == "__main__":
    main()
