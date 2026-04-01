#!/usr/bin/env python3
"""
nanobot MCP server for XArm wrist camera.

Exposes two tools to the AI agent:
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

# 使用当前环境的 Python（系统或 nanobot env）
NANOBOT_PYTHON = sys.executable

# ROS2 快照脚本（从 /cam_chest topic 拍一帧）
ROS2_SNAPSHOT = str(Path(__file__).parent / "ros2_snapshot.py")
ROS_SETUP = "/opt/ros/humble/setup.bash"


def _ros2_snapshot_cmd(output: str) -> list[str]:
    """构建 ros2_snapshot.py 命令（需要 source ROS2 环境）。
    必须用系统 Python 3.10，rclpy/cv_bridge 只有 3.10 的 C 扩展。
    """
    return ["bash", "-c",
            f"source {ROS_SETUP} && /usr/bin/python3 {ROS2_SNAPSHOT} --output {output}"]

# Default image save path
DEFAULT_IMAGE = "/tmp/nanobot_wrist.jpg"
DEFAULT_IMAGE_CHEST = "/tmp/nanobot_chest.jpg"

# Default YOLO model (downloads automatically on first use)
DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

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


def _parse_vision_bboxes(raw: str) -> list[dict] | None:
    """从 Vision 模型响应中提取 bbox JSON 列表，容忍 markdown 代码块包裹。"""
    import re
    text = raw.strip()
    # 去掉 ```json ... ``` 或 ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    # 找第一个 [ ... ]
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                item for item in data
                if isinstance(item, dict)
                and "label" in item
                and isinstance(item.get("bbox"), list)
                and len(item["bbox"]) == 4
            ]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


async def _vision_detect(img_b64: str, depth_path: str, hint: str = "") -> tuple[str, list[dict]]:
    """Kimi Vision 输出 bbox → bbox_to_3d 计算 3D 坐标。
    返回 (vision_raw_text, detections列表)。
    detections 每项: {label, bbox, depth_m, world_xyz_m}
    """
    import openai, httpx

    extra = f"重点关注: {hint}。" if hint else ""
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
                     f"检测图像中所有可被机械臂抓取的物体。{extra}"
                     "图像尺寸为 640x480 像素。"
                     "只返回 JSON 数组，不要任何其他文字，格式：\n"
                     '[{"label":"颜色+英文名称","bbox":[x1,y1,x2,y2]}, ...]\n'
                     '例：[{"label":"green block","bbox":[456,430,522,480]},'
                     '{"label":"pink plate","bbox":[122,358,415,480]}]'
                 )},
            ],
        }],
        max_tokens=400,
    )
    vision_raw = response.choices[0].message.content.strip()
    bboxes = _parse_vision_bboxes(vision_raw)
    if not bboxes:
        return vision_raw, []

    det_cmd = [
        NANOBOT_PYTHON, str(SCRIPT), "bbox_to_3d",
        "--depth", depth_path,
        "--bboxes", json.dumps(bboxes),
        "--calib-dir", CALIB_DIR,
    ]
    det_data = await asyncio.to_thread(_run, det_cmd)
    detections = det_data.get("detections", []) if "error" not in det_data else []
    return vision_raw, detections


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
                "【主要感知工具】拍照 → 视觉AI识别所有物体 → YOLO-World精确定位 → 返回每个物体的名称、置信度和3D世界坐标（相对机械臂基坐标系，单位米）。"
                "用户问'你能看到什么'、'有哪些物体'、'坐标是多少'时必须调用此工具。"
                "抓取前也必须先调用此工具获取坐标。"
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
                "执行机械臂抓取。传入 world_xyz 直接抓取已知3D坐标位置；传入 target 名称则自动感知后抓取。"
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
        types.Tool(
            name="place_object",
            description=(
                "执行机械臂放置：将夹爪中的物体放到指定位置。传入 world_xyz 直接放置到已知坐标；传入 target 名称则感知目标位置后放置。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "world_xyz": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "放置目标位置3D世界坐标 [x, y, z]，单位米（如盘子中心）。",
                    },
                    "target": {
                        "type": "string",
                        "description": "放置目标名称（中/英文），如'盘子'、'plate'、'box'。未知坐标时使用。",
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
    if name == "detect_objects":
        image = arguments.get("image", DEFAULT_IMAGE)
        conf = arguments.get("conf", 0.25)
        cmd = [
            NANOBOT_PYTHON, str(SCRIPT), "detect",
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

        extra_labels = arguments.get("extra_labels", [])
        hint = ", ".join(extra_labels) if extra_labels else ""

        # 1. 采集 RGB + depth
        cap_data = await asyncio.to_thread(_run, _ros2_snapshot_cmd(DEFAULT_IMAGE_CHEST))
        if "error" in cap_data:
            return [types.TextContent(type="text", text=f"拍照失败: {cap_data['error']}")]
        depth_path = cap_data.get("depth", DEFAULT_DEPTH_CHEST)

        # 2. Vision → bbox → 3D
        with open(DEFAULT_IMAGE_CHEST, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        vision_raw, detections = await _vision_detect(img_b64, depth_path, hint=hint)

        # 3. 格式化输出
        lines = [f"Vision 识别到: {vision_raw}", ""]
        if not detections:
            lines.append("未能获取物体坐标（Vision 未返回有效 bbox）")
        else:
            lines.append(f"检测到 {len(detections)} 个物体:")
            for d in detections:
                xyz = d.get("world_xyz_m")
                pos = (f"world=({xyz[0]*100:.1f}, {xyz[1]*100:.1f}, {xyz[2]*100:.1f}) cm"
                       if xyz else "无深度数据")
                lines.append(f"  - {d['label']}  depth={d['depth_m']*100:.1f}cm  {pos}")
        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "capture_image_chest":
        data = _run(_ros2_snapshot_cmd(DEFAULT_IMAGE_CHEST))
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "detect_objects_chest":
        conf = arguments.get("conf", 0.4)
        # 先从 ROS2 话题抓取一帧（不直接启动相机）
        cap_data = _run(_ros2_snapshot_cmd(DEFAULT_IMAGE_CHEST))
        if "error" in cap_data:
            return [types.TextContent(type="text", text=json.dumps(cap_data))]
        # 再检测
        det_cmd = [NANOBOT_PYTHON, str(SCRIPT), "detect",
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
        from xarm_mcp.grasp_utils import match_target, execute_grasp

        target = arguments.get("target", "")
        canport = arguments.get("canport", "can2")
        world_xyz = arguments.get("world_xyz")  # 已知坐标时直接跳过感知

        def _format_grasp_result(result: dict) -> str:
            lines = [f"详情: {result.get('message', '')}"]
            ik = result.get("ik")
            if not isinstance(ik, dict):
                return "\n".join(lines)

            pre = ik.get("pre_grasp", {})
            grasp = ik.get("grasp", {})
            pre_deg = pre.get("q_deg")
            grasp_deg = grasp.get("q_deg")
            pre_err = pre.get("pos_error_mm")
            grasp_err = grasp.get("pos_error_mm")

            if isinstance(pre_deg, list):
                lines.append("预抓取关节角(deg): " + ", ".join(f"{v:.1f}" for v in pre_deg))
            if isinstance(grasp_deg, list):
                lines.append("抓取点关节角(deg): " + ", ".join(f"{v:.1f}" for v in grasp_deg))
            if pre_err is not None:
                lines.append(f"预抓取IK误差: {float(pre_err):.1f} mm")
            if grasp_err is not None:
                lines.append(f"抓取点IK误差: {float(grasp_err):.1f} mm")
            return "\n".join(lines)

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
                f"{_format_grasp_result(result)}"
            ))]

        # ── 感知+抓取模式：拍照识别再抓取 ──

        # 1a. 采集 RGB + depth
        cap_data = await asyncio.to_thread(_run, _ros2_snapshot_cmd(DEFAULT_IMAGE_CHEST))
        if "error" in cap_data:
            return [types.TextContent(type="text", text=f"抓取失败 - 拍照错误: {cap_data['error']}")]
        depth_path = cap_data.get("depth", DEFAULT_DEPTH_CHEST)

        # 1b. Vision → bbox → 3D
        with open(DEFAULT_IMAGE_CHEST, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        vision_raw, detections = await _vision_detect(img_b64, depth_path, hint=target)

        if not detections:
            return [types.TextContent(type="text",
                text=f"抓取失败 - 未检测到任何物体\n(Vision看到: {vision_raw})")]

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
            f"{_format_grasp_result(result)}"
        ))]

    elif name == "place_object":
        import base64
        from xarm_mcp.grasp_utils import match_target, execute_place

        target = arguments.get("target", "")
        canport = arguments.get("canport", "can2")
        world_xyz = arguments.get("world_xyz")

        if world_xyz is not None:
            # ── 直接放置模式 ──
            xyz = list(world_xyz)
            pos_cm = f"({xyz[0]*100:.1f}, {xyz[1]*100:.1f}, {xyz[2]*100:.1f}) cm"
            result = await asyncio.to_thread(execute_place, xyz, canport)
            status = "成功" if result["success"] else "失败"
            return [types.TextContent(type="text", text=(
                f"放置{status}\n"
                f"目标: {target or '未指定'}\n"
                f"世界坐标: {pos_cm}\n"
                f"详情: {result.get('message', '')}"
            ))]

        # ── 感知+放置模式 ──
        cap_data = await asyncio.to_thread(_run, _ros2_snapshot_cmd(DEFAULT_IMAGE_CHEST))
        if "error" in cap_data:
            return [types.TextContent(type="text", text=f"放置失败 - 拍照错误: {cap_data['error']}")]
        depth_path = cap_data.get("depth", DEFAULT_DEPTH_CHEST)

        with open(DEFAULT_IMAGE_CHEST, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        vision_raw, detections = await _vision_detect(img_b64, depth_path, hint=target)

        if not detections:
            return [types.TextContent(type="text",
                text=f"放置失败 - 未检测到放置目标\n(Vision看到: {vision_raw})")]

        best = match_target(detections, target)
        if best is None:
            detected_names = ", ".join(d["label"] for d in detections)
            return [types.TextContent(type="text",
                text=f"放置失败 - 未找到目标 '{target}'\n检测到: {detected_names}")]

        xyz = best["world_xyz_m"]
        pos_cm = f"({xyz[0]*100:.1f}, {xyz[1]*100:.1f}, {xyz[2]*100:.1f}) cm"
        result = await asyncio.to_thread(execute_place, xyz, canport)
        status = "成功" if result["success"] else "失败"
        return [types.TextContent(type="text", text=(
            f"放置{status}\n"
            f"目标: {best['label']}\n"
            f"世界坐标: {pos_cm}\n"
            f"详情: {result.get('message', '')}"
        ))]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
