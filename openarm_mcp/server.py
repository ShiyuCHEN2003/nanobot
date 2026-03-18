#!/usr/bin/env python3
"""
nanobot MCP server for OpenArm wrist camera.

Exposes two tools to the AI agent:
  - capture_image: grab a frame from the RealSense wrist camera
  - detect_objects: run YOLOv8 on a captured frame
"""

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

# Default YOLO model (downloads automatically on first use)
DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

app = Server("openarm-wrist-camera")


def _run(cmd: list[str]) -> dict:
    """Run a subprocess command and return parsed JSON output."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
