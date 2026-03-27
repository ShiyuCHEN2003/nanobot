---
name: xarm-startup
description: Start up the XArm robot system. Use when user says 启动机械臂, 启动系统, start robot, start cameras, 开机, or asks to initialize the robot arm system.
metadata: {"nanobot":{"emoji":"🚀","requires":{"bins":["tmux","ros2"]}}}
---

# XArm 系统启动

启动顺序：CAN 接口配置 → 相机检测 → ROS2 相机节点启动

## 执行

```bash
bash /home/vlai-minipc/nanobot/nanobot/skills/xarm-startup/scripts/startup.sh
```

## 步骤说明

1. **配置 CAN 接口**：运行 `setup_can_interfaces.sh`，配置 can0–can3 为 CAN-FD 模式（1Mbps/5Mbps）
2. **检测相机**：`ros2 run multi_realsense check_cameras` 枚举已连接的 RealSense 设备，打印序列号
3. **启动相机节点**：在 tmux 会话 `xarm_cameras` 中以 `auto_detect:=true` 启动 `multi_cameras.launch.py`

## 启动后验证

运行以下命令，**必须同时出现以下三个话题**才算启动成功：

```bash
source /opt/ros/humble/setup.bash && ros2 topic list | grep cam
```

预期输出（缺少任意一条均视为启动失败）：
```
/cam_chest/cam_chest/color/image_raw
/cam_chest/cam_chest/depth/image_rect_raw
/cam_chest/cam_chest/aligned_depth_to_color/image_raw
```

- 只有 `color/image_raw`，没有深度话题 → **未成功**，需重新启动
- 三条话题全部出现 → **启动成功**，可以执行抓取

```bash
# 重启相机节点
source /opt/ros/humble/setup.bash && \
ros2 launch multi_realsense multi_cameras.launch.py auto_detect:=true
```

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| CAN 接口未启动 | 硬件未连接 | 检查 USB-CAN 设备 |
| 相机检测失败 | USB 未插好 | 重新插拔，用 USB 3.0 口 |
| 只有 color 没有 depth | launch 旧版本或节点未重启 | 重新运行 launch 命令 |
| 三条话题均不出现 | 序列号不匹配或节点崩溃 | 检查 launch 终端输出 |
