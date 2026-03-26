#!/usr/bin/env python3
"""
从 ROS2 话题获取一帧图像（color + aligned depth），保存到磁盘。

前提：multi_realsense 已启动并发布以下话题：
  /cam_chest/color/image_raw
  /cam_chest/aligned_depth_to_color/image_raw

用法：
  source /opt/ros/humble/setup.bash
  python3 ros2_snapshot.py --output /tmp/nanobot_chest.jpg [--timeout 8]

输出 JSON（stdout）：
  {"ok": true, "image": "...", "depth": "..."}
  {"error": "..."}
"""

import argparse
import json
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",       default="/tmp/nanobot_chest.jpg")
    parser.add_argument("--color-topic",  default="/cam_chest/cam_chest/color/image_raw")
    parser.add_argument("--depth-topic",  default="/cam_chest/cam_chest/aligned_depth_to_color/image_raw")
    parser.add_argument("--timeout",      type=float, default=8.0)
    parser.add_argument("--no-depth",     action="store_true", help="只采集彩色图，跳过深度帧")
    args = parser.parse_args()

    depth_path = args.output.replace(".jpg", "_depth.npy")

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        import numpy as np
        import cv2
    except ImportError as e:
        print(json.dumps({"error": f"import failed: {e}"}))
        sys.exit(1)

    rclpy.init()
    node = rclpy.create_node("nanobot_snapshot")
    bridge = CvBridge()

    color_img = None
    depth_img = None

    def color_cb(msg):
        nonlocal color_img
        color_img = bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_cb(msg):
        nonlocal depth_img
        depth_img = bridge.imgmsg_to_cv2(msg, "passthrough")  # uint16 mm

    node.create_subscription(Image, args.color_topic, color_cb, 1)
    node.create_subscription(Image, args.depth_topic, depth_cb, 1)

    import time
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if color_img is not None and depth_img is not None:
            break

    node.destroy_node()
    rclpy.shutdown()

    if color_img is None:
        print(json.dumps({"error": f"no color frame on {args.color_topic} within {args.timeout}s"}))
        sys.exit(1)
    if depth_img is None:
        print(json.dumps({"error": f"no depth frame on {args.depth_topic} within {args.timeout}s"}))
        sys.exit(1)

    cv2.imwrite(args.output, color_img)
    np.save(depth_path, depth_img)
    print(json.dumps({
        "ok": True,
        "image": args.output,
        "depth": depth_path,
        "width": color_img.shape[1],
        "height": color_img.shape[0],
    }))


if __name__ == "__main__":
    main()
