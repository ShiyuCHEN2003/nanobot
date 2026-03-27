#!/usr/bin/env python3

import argparse
import numpy as np

from xarm_mcp.arm_control import build_arm, hold, move_joints, read_q7, gripper_close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canport", default="can2")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()

    q_deg = np.array([0.0, 20.0, 0.0, 60.0, 0.0, 0.0, 0.0], dtype=float)
    q_rad = np.deg2rad(q_deg)

    print("Target joints (deg):", q_deg.tolist())
    xarm, arm, gripper = build_arm(args.canport)
    move_joints(xarm, arm, q_rad, duration=args.duration, n_steps=args.steps)
    hold(xarm, arm, duration=0.5)

    print("Closing gripper...")
    gripper_close(xarm, gripper)

    xarm.refresh_all()
    xarm.recv_all()
    q_now = read_q7(arm)
    print("Current joints (deg):", np.round(np.rad2deg(q_now), 2).tolist())
    print("Gripper: closed (0 deg)")


if __name__ == "__main__":
    main()
