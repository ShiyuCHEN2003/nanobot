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

    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args)
    elif args.cmd == "detect":
        detect(args)


if __name__ == "__main__":
    main()
