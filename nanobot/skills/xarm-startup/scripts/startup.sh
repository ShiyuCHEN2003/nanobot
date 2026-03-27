#!/bin/bash
# XArm 系统启动脚本
# 步骤：1) 配置 CAN 接口  2) 检测相机序列号  3) 启动 ROS2 相机节点

set -e

SETUP_CAN=/home/vlai-minipc/x_air/src/xarm_can/setup/setup_can_interfaces.sh
ROS_SETUP=/opt/ros/humble/setup.bash

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── 1. CAN 接口配置 ──────────────────────────────────────────────
log "配置 CAN 接口..."
[ -f "$SETUP_CAN" ] || die "找不到 $SETUP_CAN"
sudo bash "$SETUP_CAN"

if ! ip link show | grep -q "can.*UP"; then
    die "CAN 接口未成功启动，请检查硬件连接"
fi
log "CAN 接口就绪"

# ── 2. 检测相机序列号 ─────────────────────────────────────────────
log "检测 RealSense 相机序列号..."
SERIALS=$(rs-enumerate-devices -s 2>/dev/null | awk '/Intel RealSense/ {print $(NF-1)}')
COUNT=$(echo "$SERIALS" | grep -c . || true)

if [ "$COUNT" -eq 0 ]; then
    die "未检测到 RealSense 相机，请检查 USB 连接"
fi
log "检测到 $COUNT 个相机: $(echo $SERIALS | tr '\n' ' ')"

# 取第一个作为 chest 相机（D435）
SERIAL_CHEST=$(echo "$SERIALS" | head -1)
log "胸部相机序列号: $SERIAL_CHEST"

# ── 3. 启动 ROS2 相机节点（后台）──────────────────────────────────
log "启动 ROS2 相机节点（后台）..."
source "$ROS_SETUP"

# 杀掉旧的 launch 进程
pkill -f "multi_cameras.launch.py" 2>/dev/null || true
sleep 1

nohup ros2 launch multi_realsense multi_cameras.launch.py \
    serial_chest:="_${SERIAL_CHEST}" \
    > /tmp/xarm_cameras.log 2>&1 &
LAUNCH_PID=$!
echo $LAUNCH_PID > /tmp/xarm_cameras.pid
log "launch PID: $LAUNCH_PID，日志: /tmp/xarm_cameras.log"

# ── 4. 验证启动结果 ───────────────────────────────────────────────
log "等待相机节点启动（15s）..."
sleep 15

COLOR_OK=$(ros2 topic list 2>/dev/null | grep -c "cam_chest.*color/image_raw" || true)
DEPTH_OK=$(ros2 topic list 2>/dev/null | grep -c "aligned_depth_to_color" || true)

if [ "$COLOR_OK" -gt 0 ] && [ "$DEPTH_OK" -gt 0 ]; then
    log "✅ 启动成功！color + depth 话题均已就绪"
    ros2 topic list 2>/dev/null | grep cam_chest
elif [ "$COLOR_OK" -gt 0 ]; then
    log "⚠️  只有 color 话题，depth 未出现。查看日志: tail -f /tmp/xarm_cameras.log"
    exit 1
else
    log "❌ 相机话题未出现。查看日志: tail -f /tmp/xarm_cameras.log"
    exit 1
fi
