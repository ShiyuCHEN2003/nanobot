---
name: xarm-home
description: Move the XArm to the initial home pose. Use when user says 回到初始位置, 回零位, home pose, or asks to move to 0,20,0,60,0,0,0 degrees.
---

# XArm Home Pose

Move the arm to the initial pose (degrees):

`[0, 20, 0, 60, 0, 0, 0]`

## When to use

Use this skill when user asks to:
- 回到初始位置
- 回零位 / Home
- move to home pose
- move joints to `0,20,0,60,0,0,0`

## Execute

From repository root, run:

```bash
python nanobot/skills/xarm-home/scripts/move_home.py --canport can2
```

If user provides another CAN port, replace `can2`.

## Behavior

- Sends target joints with smooth interpolation.
- Holds briefly after motion.
- Prints readback joint angles in degrees for confirmation.
