#!/usr/bin/env python3
"""
nanobot MCP server for XArm wrist camera.

Exposes two tools to the AI agent:
  - capture_image: grab a frame from the RealSense wrist camera
  - detect_objects: run YOLOv8 on a captured frame
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Path to camera_detect.py
SCRIPT = Path(__file__).parent / "camera_detect.py"

# Python interpreter that has pyrealsense2 + ultralytics
LEROBOT_PYTHON = os.environ.get(
    "LEROBOT_PYTHON",
    os.path.expanduser("~/miniconda3/envs/lerobot/bin/python3"),
)

# Default image save path
DEFAULT_IMAGE = "/tmp/nanobot_wrist.jpg"
DEFAULT_IMAGE_CHEST = "/tmp/nanobot_chest.jpg"

# Default YOLO model (downloads automatically on first use)
DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

# Chest camera serial number
CHEST_SERIAL = "314422070707"

DEFAULT_MODEL_WORLD = str(Path(__file__).parent.parent / "yolov8s-world.pt")
CALIB_DIR = str(Path(__file__).parent / "calib")
DEFAULT_DEPTH_CHEST = "/tmp/nanobot_chest_depth.npy"


def _get_moonshot_key() -> str:
    try:
        cfg = Path.home() / ".nanobot" / "config.json"
        with open(cfg) as f:
            return json.load(f)["providers"]["moonshot"]["apiKey"]
    except Exception:
        return os.environ.get("MOONSHOT_API_KEY", "")


app = Server("xarm-wrist-camera")


def _run(cmd: list[str], timeout: int = 120) -> dict:
    """Run a subprocess command and return parsed JSON output."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "subprocess failed"}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"error": f"bad output: {result.stdout.strip()}"}


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="capture_image",
            description=(
                "Capture a color image from the RealSense wrist camera. "
                "Returns the saved image path. Call this before detect_objects."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "serial": {
                        "type": "string",
                        "description": "RealSense camera serial number (optional, auto-detect if omitted)",
                    },
                    "output": {
                        "type": "string",
                        "description": f"Path to save the image (default: {DEFAULT_IMAGE})",
                    },
                },
            },
        ),
        types.Tool(
            name="detect_objects",
            description=(
                "Run YOLOv8 object detection on an image. "
                "Returns a list of detected objects with label, confidence, and bounding box."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": f"Path to image file (default: {DEFAULT_IMAGE})",
                    },
                    "conf": {
                        "type": "number",
                        "description": "Confidence threshold 0-1 (default: 0.25)",
                    },
                },
            },
        ),
        types.Tool(
            name="capture_image_chest",
            description=(
                "Capture a color image from the chest RealSense camera (fixed, wide view). "
                "Use this to see the workspace in front of the robot. "
                "Returns the saved image path. Call this before detect_objects_chest."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="analyze_scene_chest",
            description=(
                "【首选工具，必须优先调用】胸部相机感知：拍照 → Kimi Vision 理解场景 → YOLO-World 定位 → 输出每个物体的3D世界坐标（相对机械臂基坐标系，单位米）。"
                "当用户问'能看到什么'、'识别物体'、'能抓什么'、'XXX在哪'、'XXX的位置'时必须使用此工具而非 detect_objects_chest。"
                "此工具直接提供可用于抓取的3D坐标，detect_objects_chest 只有像素坐标无法用于抓取。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "conf": {
                        "type": "number",
                        "description": "YOLO-World 置信度阈值（默认 0.1）",
                    },
                    "extra_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "额外指定要检测的物体名称（在视觉识别基础上追加）",
                    },
                },
            },
        ),
        # types.Tool(
        #     name="detect_objects_chest",
        #     description=(
        #         "【备用】胸部相机 YOLOv8 检测，仅支持 COCO 80 类常见物体，只输出像素坐标无3D坐标。"
        #         "优先使用 analyze_scene_chest，只有在 analyze_scene_chest 失败时才用此工具。"
        #     ),
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "conf": {
        #                 "type": "number",
        #                 "description": "Confidence threshold 0-1 (default: 0.4)",
        #             },
        #         },
        #     },
        # ),
        types.Tool(
            name="grasp_object",
            description=(
                "【抓取工具】执行机械臂抓取。"
                "如果已知目标的世界坐标（来自 analyze_scene_chest），直接传入 world_xyz 跳过感知，立即执行抓取。"
                "如果不知道坐标，传入 target 名称，工具会自动拍照识别再抓取。"
                "当用户说'帮我抓取XXX'、'抓住XXX'、'拿起XXX'时使用此工具。"
                "重要：如果刚才已经用 analyze_scene_chest 获取了坐标，必须把坐标传入 world_xyz，不要重新感知。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "world_xyz": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "目标3D世界坐标 [x, y, z]，单位米。已知坐标时传入此参数可跳过感知直接抓取。",
                    },
                    "target": {
                        "type": "string",
                        "description": "目标物体名称（中/英文），如'杯子'、'cup'、'bottle'。未知坐标时使用，留空则抓置信度最高的物体。",
                    },
                    "canport": {
                        "type": "string",
                        "description": "CAN总线端口（默认 can2）",
                    },
                    "conf": {
                        "type": "number",
                        "description": "YOLO-World 检测置信度阈值（默认 0.1），仅在未传 world_xyz 时使用",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "capture_image":
        output = arguments.get("output", DEFAULT_IMAGE)
        cmd = [LEROBOT_PYTHON, str(SCRIPT), "capture", "--output", output]
        if serial := arguments.get("serial"):
            cmd += ["--serial", serial]
        data = _run(cmd)
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "detect_objects":
        image = arguments.get("image", DEFAULT_IMAGE)
        conf = arguments.get("conf", 0.25)
        cmd = [
            LEROBOT_PYTHON, str(SCRIPT), "detect",
            "--image", image,
            "--model", DEFAULT_MODEL,
            "--conf", str(conf),
        ]
        data = _run(cmd)
        if "detections" in data:
            lines = [f"Found {len(data['detections'])} object(s):"]
            for d in data["detections"]:
                x1, y1, x2, y2 = d["bbox"]
                lines.append(
                    f"  - {d['label']} ({d['confidence']:.0%})  bbox:[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
                )
            return [types.TextContent(type="text", text="\n".join(lines))]
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "analyze_scene_chest":
        import base64
        import openai

        conf = arguments.get("conf", 0.05)
        extra_labels = arguments.get("extra_labels", [])

        # 1. 一次性采集 RGB + depth（相机只开一次）
        cap_cmd = [LEROBOT_PYTHON, str(SCRIPT), "capture_world",
                   "--output", DEFAULT_IMAGE_CHEST, "--serial", CHEST_SERIAL]
        cap_data = await asyncio.to_thread(_run, cap_cmd)
        if "error" in cap_data:
            return [types.TextContent(type="text", text=f"拍照失败: {cap_data['error']}")]
        depth_path = cap_data.get("depth", DEFAULT_DEPTH_CHEST)

        # 2. Vision API 识别物体
        with open(DEFAULT_IMAGE_CHEST, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        import httpx
        client = openai.AsyncOpenAI(
            api_key=_get_moonshot_key(),
            base_url="https://api.moonshot.cn/v1",
            http_client=httpx.AsyncClient(proxy=None, trust_env=False),
        )
        response = await client.chat.completions.create(
            model="moonshot-v1-8k-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text",
                     "text": (
                         "列出图片中所有可以被机械臂抓取的物体。"
                         "只回答逗号分隔的英文名称列表，不要其他内容。"
                         "例如：apple, plastic bottle, wooden block"
                     )},
                ],
            }],
            max_tokens=200,
        )
        vision_text = response.choices[0].message.content.strip()

        # 3. 解析标签
        labels = [l.strip().strip(".,。，") for l in vision_text.replace("\n", ",").split(",") if l.strip()]
        labels += extra_labels
        labels = list(dict.fromkeys(labels))  # 去重保序

        # 4. YOLO-World 检测 + 3D 坐标（离线，不重开相机）
        det_cmd = [
            LEROBOT_PYTHON, str(SCRIPT), "detect_world_offline",
            "--image", DEFAULT_IMAGE_CHEST,
            "--depth", depth_path,
            "--labels", ",".join(labels),
            "--model", DEFAULT_MODEL_WORLD,
            "--conf", str(conf),
            "--calib-dir", CALIB_DIR,
        ]
        det_data = await asyncio.to_thread(_run, det_cmd)

        # 5. 格式化输出
        lines = [f"Vision 识别到: {vision_text}", ""]
        if "error" in det_data:
            lines.append(f"YOLO-World 失败: {det_data['error']}")
        elif "detections" in det_data:
            lines.append(f"YOLO-World 定位到 {len(det_data['detections'])} 个物体:")
            for d in det_data["detections"]:
                xyz = d.get("world_xyz_m")
                pos = (f"world=({xyz[0]*100:.1f}, {xyz[1]*100:.1f}, {xyz[2]*100:.1f}) cm"
                       if xyz else "无深度数据")
                lines.append(f"  - {d['label']} ({d['confidence']:.0%})  depth={d['depth_m']*100:.1f}cm  {pos}")
        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "capture_image_chest":
        cmd = [LEROBOT_PYTHON, str(SCRIPT), "capture",
               "--output", DEFAULT_IMAGE_CHEST, "--serial", CHEST_SERIAL]
        data = _run(cmd)
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "detect_objects_chest":
        conf = arguments.get("conf", 0.4)
        # 先拍照
        cap_cmd = [LEROBOT_PYTHON, str(SCRIPT), "capture",
                   "--output", DEFAULT_IMAGE_CHEST, "--serial", CHEST_SERIAL]
        cap_data = _run(cap_cmd)
        if "error" in cap_data:
            return [types.TextContent(type="text", text=json.dumps(cap_data))]
        # 再检测
        det_cmd = [LEROBOT_PYTHON, str(SCRIPT), "detect",
                   "--image", DEFAULT_IMAGE_CHEST,
                   "--model", DEFAULT_MODEL, "--conf", str(conf)]
        data = _run(det_cmd)
        if "detections" in data:
            lines = [f"胸部相机检测到 {len(data['detections'])} 个物体:"]
            for d in data["detections"]:
                x1, y1, x2, y2 = d["bbox"]
                lines.append(
                    f"  - {d['label']} ({d['confidence']:.0%})  bbox:[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
                )
            return [types.TextContent(type="text", text="\n".join(lines))]
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "grasp_object":
        import base64
        import openai
        import httpx
        from xarm_mcp.grasp_utils import match_target, execute_grasp

        target = arguments.get("target", "")
        canport = arguments.get("canport", "can2")
        conf = arguments.get("conf", 0.05)
        world_xyz = arguments.get("world_xyz")  # 已知坐标时直接跳过感知

        if world_xyz is not None:
            # ── 直接抓取模式：跳过感知 ──
            xyz = list(world_xyz)
            pos_cm = f"({xyz[0]*100:.1f}, {xyz[1]*100:.1f}, {xyz[2]*100:.1f}) cm"
            result = await asyncio.to_thread(execute_grasp, xyz, canport)
            status = "成功" if result["success"] else "失败"
            return [types.TextContent(type="text", text=(
                f"抓取{status}\n"
                f"目标: {target or '未指定'}\n"
                f"世界坐标: {pos_cm}\n"
                f"详情: {result['message']}"
            ))]

        # ── 感知+抓取模式：拍照识别再抓取 ──

        # 1a. 一次性采集 RGB + depth（相机只开一次）
        cap_cmd = [LEROBOT_PYTHON, str(SCRIPT), "capture_world",
                   "--output", DEFAULT_IMAGE_CHEST, "--serial", CHEST_SERIAL]
        cap_data = await asyncio.to_thread(_run, cap_cmd)
        if "error" in cap_data:
            return [types.TextContent(type="text", text=f"抓取失败 - 拍照错误: {cap_data['error']}")]
        depth_path = cap_data.get("depth", DEFAULT_DEPTH_CHEST)

        # 1b. Kimi Vision API 识别可抓取物体
        with open(DEFAULT_IMAGE_CHEST, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        client = openai.AsyncOpenAI(
            api_key=_get_moonshot_key(),
            base_url="https://api.moonshot.cn/v1",
            http_client=httpx.AsyncClient(proxy=None, trust_env=False),
        )
        response = await client.chat.completions.create(
            model="moonshot-v1-8k-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text",
                     "text": (
                         "列出图片中所有可以被机械臂抓取的物体。"
                         "只回答逗号分隔的英文名称列表，不要其他内容。"
                         "例如：apple, plastic bottle, wooden block"
                     )},
                ],
            }],
            max_tokens=200,
        )
        vision_text = response.choices[0].message.content.strip()
        labels = [l.strip().strip(".,。，") for l in vision_text.replace("\n", ",").split(",") if l.strip()]
        labels = list(dict.fromkeys(labels))

        # 1c. YOLO-World 检测 + 3D 坐标（离线，不重开相机）
        det_cmd = [
            LEROBOT_PYTHON, str(SCRIPT), "detect_world_offline",
            "--image", DEFAULT_IMAGE_CHEST,
            "--depth", depth_path,
            "--labels", ",".join(labels),
            "--model", DEFAULT_MODEL_WORLD,
            "--conf", str(conf),
            "--calib-dir", CALIB_DIR,
        ]
        det_data = await asyncio.to_thread(_run, det_cmd)
        if "error" in det_data:
            return [types.TextContent(type="text",
                text=f"抓取失败 - 检测错误: {det_data['error']}\n(Vision看到: {vision_text})")]

        detections = det_data.get("detections", [])
        if not detections:
            return [types.TextContent(type="text",
                text=f"抓取失败 - 未检测到任何物体\n(Vision看到: {vision_text})")]

        # ── 目标匹配 ──
        best = match_target(detections, target)
        if best is None:
            detected_names = ", ".join(d["label"] for d in detections)
            return [types.TextContent(type="text",
                text=f"抓取失败 - 未找到目标 '{target}'\n检测到的物体: {detected_names}")]

        xyz = best["world_xyz_m"]
        pos_cm = f"({xyz[0]*100:.1f}, {xyz[1]*100:.1f}, {xyz[2]*100:.1f}) cm"

        # ── 执行抓取 ──
        result = await asyncio.to_thread(execute_grasp, xyz, canport)

        status = "成功" if result["success"] else "失败"
        return [types.TextContent(type="text", text=(
            f"抓取{status}\n"
            f"目标: {best['label']} (置信度 {best['confidence']:.0%})\n"
            f"世界坐标: {pos_cm}\n"
            f"详情: {result['message']}"
        ))]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
