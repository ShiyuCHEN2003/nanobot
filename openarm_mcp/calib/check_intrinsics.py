#!/usr/bin/env python3
# check_intrinsics.py
import pyrealsense2 as rs, json, numpy as np

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_device("314422070707")
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)
for _ in range(5):
    pipeline.wait_for_frames()

frames  = pipeline.wait_for_frames()
intr    = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
pipeline.stop()

print("=== Realsense 实时内参 ===")
print(f"分辨率: {intr.width} x {intr.height}")
print(f"fx={intr.fx:.4f}  fy={intr.fy:.4f}")
print(f"cx={intr.ppx:.4f}  cy={intr.ppy:.4f}")
print(f"畸变模型: {intr.model}")
print(f"畸变系数: {intr.coeffs}")

K_rs = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
print(f"\nK =\n{K_rs.round(4)}")

# 对比 calib_chest.json
with open("/home/vlai-minipc/nanobot/openarm_mcp/calib/calib_chest.json") as f:
    c = json.load(f)
K_file = np.array(c["K"])
print(f"\n=== calib_chest.json 里的 K ===")
print(K_file.round(4))
print(f"\n差值 (应该接近0):")
print((K_rs - K_file).round(4))
