#!/usr/bin/env python3

import argparse
import numpy as np

from xarm_mcp.arm_control import build_arm, move_joints, read_q7, gripper_close
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    args = parser.parse_args()

    q_deg = np.array([0.0, 20.0, 0.0, 60.0, 0.0, 0.0, 0.0], dtype=float)
    q_rad = np.deg2rad([0, 20, 0, 60, 0, 0, 0])

    print("Target joints (deg):", q_deg.tolist())
    ctrl = build_arm(args.canport)
    move_joints(ctrl, q_rad,canport=args.canport)
    # time.sleep(0.5)

    print("Closing gripper...")
    gripper_close(ctrl, args.canport)

    q_now = read_q7(ctrl, args.canport)
    print("Current joints (deg):", np.round(np.rad2deg(q_now), 2).tolist())
    print("Gripper: closed (0 deg)")


if __name__ == "__main__":
    main()
