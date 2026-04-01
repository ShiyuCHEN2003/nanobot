#!/usr/bin/env python3

import argparse
import numpy as np

from xarm_mcp.arm_control import Arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    args = parser.parse_args()

    q_rad = np.deg2rad([0, 20, 0, 60, 0, 0, 0])

    print("Target joints (deg):", np.rad2deg(q_rad).tolist())
    arm = Arm(args.canport)
    arm.move_joints(q_rad)

    print("Closing gripper...")
    arm.gripper_close()

    q_now = arm.read_q7()
    print("Current joints (deg):", np.round(np.rad2deg(q_now), 2).tolist())
    print("Gripper: closed (0 deg)")


if __name__ == "__main__":
    main()
