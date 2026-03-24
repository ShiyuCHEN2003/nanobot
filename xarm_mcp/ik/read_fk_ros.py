#!/usr/bin/env python3
"""
从 ROS2 话题读取关节角并计算正解。

用法（在 ROS2 环境下）：
  source /opt/ros/humble/setup.bash
  ./script/launch_unilateral_ros2.sh right_arm can2 can1
  python xarm_mcp/read_fk_ros.py --topic /xarm_follower/joint_states
"""

import os, sys, argparse
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver

SCENE_XML            = os.path.join(_HERE, "xml/scene.xml")
RIGHT_ARM_QPOS_SLICE = slice(9, 16)


def build_fk():
    solver = XarmIKSolver(xml_path=SCENE_XML,
                          right_ee_name="xarm_right_hand_tcp")
    base_qpos = solver.data.qpos.copy()

    def fk(q7):
        qpos = base_qpos.copy()
        qpos[RIGHT_ARM_QPOS_SLICE] = q7
        solver.reset(qpos)
        pos, rot = solver.get_ee_pose("right")
        rpy = Rot.from_matrix(rot).as_euler('xyz', degrees=True)
        return pos, rot, rpy

    return fk


class FKNode(Node):
    def __init__(self, topic, fk_fn):
        super().__init__('fk_reader')
        self.fk = fk_fn
        self.sub = self.create_subscription(
            JointState, topic, self.cb, 10)
        self.get_logger().info(f"订阅话题: {topic}")

    def cb(self, msg: JointState):
        # 只取前 7 个关节（忽略夹爪）
        q7 = np.array(msg.position[:7])
        pos, rot, rpy = self.fk(q7)

        print(f"\n关节角 (deg): {np.round(np.degrees(q7), 2).tolist()}")
        print(f"末端位置 (m): x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
        print(f"末端 RPY (deg): r={rpy[0]:.1f}  p={rpy[1]:.1f}  y={rpy[2]:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/xarm_follower/joint_states")
    args = parser.parse_args()

    print("加载 FK 解算器...")
    fk = build_fk()

    rclpy.init()
    node = FKNode(args.topic, fk)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
