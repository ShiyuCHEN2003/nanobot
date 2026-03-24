#!/usr/bin/env python3
"""
Helper: reads ROS2 joint states and prints JSON lines to stdout.
Must run with Python 3.10 (system python3) that has rclpy built for it.

Usage (internal, called by handeye_chest.py via subprocess):
  /usr/bin/python3 _ros_joint_helper.py /right_arm_controller/joint_states right
"""
import sys, json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def _pick_q7(msg: JointState, arm_side: str):
    if len(msg.position) < 7:
        return None

    if len(msg.name) == len(msg.position) and len(msg.name) > 0:
        pairs = list(zip(msg.name, msg.position))
        non_gripper = [p for p in pairs if not any(k in p[0].lower() for k in ["gripper", "finger", "tool"])]

        if arm_side.lower() == "right":
            arm_pairs = [p for p in non_gripper if "right" in p[0].lower() or p[0].lower().startswith("r_")]
        elif arm_side.lower() == "left":
            arm_pairs = [p for p in non_gripper if "left" in p[0].lower() or p[0].lower().startswith("l_")]
        else:
            arm_pairs = non_gripper

        if len(arm_pairs) >= 7:
            return [float(v) for _, v in arm_pairs[:7]]

        if len(non_gripper) >= 7:
            return [float(v) for _, v in non_gripper[:7]]

    return [float(v) for v in msg.position[:7]]


class _Node(Node):
    def __init__(self, topic: str, arm_side: str):
        super().__init__("_joint_reader_helper")
        self.arm_side = arm_side
        self.create_subscription(JointState, topic, self._cb, 10)

    def _cb(self, msg: JointState):
        q7 = _pick_q7(msg, self.arm_side)
        if q7 is not None:
            print(json.dumps(q7), flush=True)


topic = sys.argv[1] if len(sys.argv) > 1 else "/right_arm_controller/joint_states"
arm_side = sys.argv[2] if len(sys.argv) > 2 else "right"
rclpy.init()
rclpy.spin(_Node(topic, arm_side))
