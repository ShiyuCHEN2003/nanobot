"""
grasp_utils.py — 机械臂状态管理与抓取执行

在 nanobot conda 环境中运行，通过 xarm_mcp/server.py 的 grasp_object 工具调用。

主要接口：
  match_target(detections, target_label) → best detection dict or None
  execute_grasp(world_xyz, canport)       → {"success": bool, "message": str}
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SCENE_XML            = os.path.join(_HERE, "xml/scene.xml")
RIGHT_ARM_QPOS_SLICE = slice(9, 16)
PRE_GRASP_Z          = 0.10   # 预抓取点比目标高 10cm
CANPORT_DEFAULT      = "can2"

# 中英文标签同义词表
LABEL_SYNONYMS = {
    "杯子": "cup",
    "水杯": "cup",
    "茶杯": "cup",
    "马克杯": "cup",
    "瓶子": "bottle",
    "水瓶": "bottle",
    "矿泉水": "bottle",
    "苹果": "apple",
    "香蕉": "banana",
    "橙子": "orange",
    "碗": "bowl",
    "盒子": "box",
    "纸盒": "box",
    "罐子": "can",
    "易拉罐": "can",
    "遥控器": "remote",
    "手机": "cell phone",
    "书": "book",
    "剪刀": "scissors",
    "鼠标": "mouse",
    "键盘": "keyboard",
}

# 模块级缓存（进程生命周期内只初始化一次）
_arm_state = None   # {"xarm": ..., "arm": ..., "gripper": ...}
_ik_fn = None       # ik() callable


def _build_ik():
    """构建 IK 求解器（从 grasp_pipeline.py 提取，无需修改逻辑）。"""
    from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver

    solver = XarmIKSolver(
        xml_path=SCENE_XML,
        right_ee_name="xarm_right_hand_tcp",
        orientation_cost=0.0,
        solver="daqp",
    )
    base_qpos = solver.data.qpos.copy()

    SEEDS = [
        np.array([0,  0,  0,  np.pi/2, 0, 0, 0]),
        np.array([0,  0.3,  0,  np.pi/2, 0, 0, 0]),
        np.array([0.3, 0,  0,  np.pi/2, 0, 0, 0]),
        np.array([-0.3, 0, 0,  np.pi/2, 0, 0, 0]),
        np.array([0,  0,  0,  np.deg2rad(70), 0, 0, 0]),
    ]

    def ik(target_pos, q7_hint, pos_tol_mm=15.0):
        """尝试多个初始种子，返回误差最小的解。误差 < pos_tol_mm 视为成功。"""
        seeds = [q7_hint] + SEEDS
        best_result, best_q7, best_err = None, None, float('inf')

        for seed in seeds:
            qpos = base_qpos.copy()
            qpos[RIGHT_ARM_QPOS_SLICE] = seed
            solver.reset(qpos)
            result = solver.solve_until_converged(
                {"right": (target_pos, None)},
                max_steps=1000, tol=1e-4,
                pos_tol_mm=pos_tol_mm,
            )
            q7 = solver.data.qpos[RIGHT_ARM_QPOS_SLICE].copy()
            if result.success:
                return result, q7
            err = result.pos_errors.get('right', float('inf'))
            if err < best_err:
                best_err = err
                best_result = result
                best_q7 = q7.copy()

        if best_err < pos_tol_mm:
            best_result.success = True
            best_result.message = f"OK (pos_err={best_err:.1f}mm)"
        return best_result, best_q7

    return ik


def get_arm_resources(canport: str = CANPORT_DEFAULT):
    """懒初始化机械臂和 IK 求解器（进程内只执行一次）。"""
    global _arm_state, _ik_fn
    if _arm_state is None:
        from xarm_mcp.arm_control import build_arm, gripper_open
        xarm, arm, gripper = build_arm(canport)
        gripper_open(xarm, gripper)
        _arm_state = {"xarm": xarm, "arm": arm, "gripper": gripper}
    if _ik_fn is None:
        _ik_fn = _build_ik()
    return _arm_state, _ik_fn


def match_target(detections: list, target_label: str):
    """
    在检测列表中找到最匹配的目标。

    Args:
        detections: analyze_scene_chest 返回的检测列表，
                    每项含 {"label", "confidence", "world_xyz_m", ...}
        target_label: 用户指定的目标名称（中/英文），空字符串则取置信度最高的

    Returns:
        最匹配的 detection dict，或 None（未找到）
    """
    # 过滤掉没有3D坐标的检测
    valid = [d for d in detections if d.get("world_xyz_m") is not None]
    if not valid:
        return None

    # 无目标时返回置信度最高的
    if not target_label:
        return max(valid, key=lambda d: d["confidence"])

    # 中文 → 英文 同义词转换
    normalized = LABEL_SYNONYMS.get(target_label, target_label).lower().strip()

    # 子串匹配（双向）：用户输入包含检测标签，或检测标签包含用户输入
    matched = [
        d for d in valid
        if normalized in d["label"].lower() or d["label"].lower() in normalized
    ]
    if matched:
        return max(matched, key=lambda d: d["confidence"])

    return None


def execute_grasp(world_xyz, canport: str = CANPORT_DEFAULT) -> dict:
    """
    执行完整抓取动作序列（阻塞调用，在 asyncio.to_thread 中运行）。

    Args:
        world_xyz: 目标物体3D世界坐标 [x, y, z]，单位米
        canport:   CAN总线端口，默认 can2

    Returns:
        {"success": bool, "message": str}
    """
    from xarm_mcp.arm_control import read_q7, move_joints, hold, gripper_close, gripper_open

    try:
        arm_state, ik = get_arm_resources(canport)
    except Exception as e:
        return {"success": False, "message": f"机械臂初始化失败: {e}"}

    xarm = arm_state["xarm"]
    arm = arm_state["arm"]
    gripper = arm_state["gripper"]

    pos_world = np.array(world_xyz, dtype=float)

    # 张开夹爪
    try:
        gripper_open(xarm, gripper)
    except Exception as e:
        return {"success": False, "message": f"夹爪张开失败: {e}"}

    # 读取当前关节角
    try:
        xarm.refresh_all()
        xarm.recv_all()
        q7_now = read_q7(arm)
    except Exception as e:
        return {"success": False, "message": f"读取关节角失败: {e}"}

    # 计算预抓取点（目标上方 PRE_GRASP_Z）
    pre_grasp = pos_world.copy()
    pre_grasp[2] += PRE_GRASP_Z

    # IK 求解预抓取点
    result_pre, q7_pre = ik(pre_grasp, q7_now)
    if not result_pre.success:
        pre_err = result_pre.pos_errors.get('right', float('inf'))
        return {"success": False,
                "message": f"预抓取点 IK 失败（误差 {pre_err:.1f}mm）: {result_pre.message}"}

    # IK 求解抓取点
    result_grasp, q7_grasp = ik(pos_world, q7_pre)
    if not result_grasp.success:
        grasp_err = result_grasp.pos_errors.get('right', float('inf'))
        return {"success": False,
                "message": f"抓取点 IK 失败（误差 {grasp_err:.1f}mm）: {result_grasp.message}"}

    # 执行运动序列
    try:
        # 1. 移动到预抓取点（目标上方）
        move_joints(xarm, arm, q7_pre, duration=3.0)
        hold(xarm, arm, 0.5)

        # 2. 下降到抓取点
        move_joints(xarm, arm, q7_grasp, duration=2.0)
        hold(xarm, arm, 0.3)

        # 3. 夹爪闭合
        gripper_close(xarm, gripper)

        # 4. 抬起（回到预抓取点）
        move_joints(xarm, arm, q7_pre, duration=2.0)
        hold(xarm, arm, 1.0)

    except Exception as e:
        return {"success": False, "message": f"运动执行失败: {e}"}

    pre_err_mm = result_pre.pos_errors.get('right', 0)
    grasp_err_mm = result_grasp.pos_errors.get('right', 0)
    return {
        "success": True,
        "message": (
            f"抓取完成。"
            f"预抓取IK误差={pre_err_mm:.1f}mm，"
            f"抓取IK误差={grasp_err_mm:.1f}mm"
        ),
    }
