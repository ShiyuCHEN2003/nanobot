---
name: xarm-grasp
description: Grasp a target object with the XArm robot arm. Use when user says 抓取, 拿起, 抓住, grasp, pick up, or asks to grab a specific object.
metadata: {"nanobot":{"emoji":"🦾","requires":{"mcp":["xarm-wrist-camera"]}}}
---

# XArm Grasp Workflow

## 步骤

1. 调用 `analyze_scene_chest` 感知场景，获取所有可抓取物体的3D坐标
2. 从结果中找到目标物体的 `world_xyz`
3. 调用 `grasp_object(world_xyz=[x, y, z], target="目标名称")` 执行抓取

## 规则

- 已有 `world_xyz` 时直接传入 `grasp_object`，不要重新调用 `analyze_scene_chest`
- 用户未指定目标时，先 `analyze_scene_chest`，列出检测到的物体，让用户确认后再抓取
- 用户明确指定目标时，直接走完整流程无需确认
- `grasp_object` 的 `target` 参数支持中英文（如"杯子"或"cup"）

## 示例

用户："帮我抓杯子"
→ analyze_scene_chest
→ 找到 cup world_xyz=[0.3, -0.1, 0.05]
→ grasp_object(world_xyz=[0.3, -0.1, 0.05], target="cup")

用户："抓一个东西"
→ analyze_scene_chest
→ 告知用户检测到哪些物体
→ 用户确认目标后 → grasp_object(world_xyz=[...], target="...")
