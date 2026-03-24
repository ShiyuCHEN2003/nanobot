"""
独立 IK 求解器，从 xarm_mujoco_teleop_mit.py 中提取。

用法示例:
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from xarm_ik_solver import XarmIKSolver

    solver = XarmIKSolver(xml_path="/path/to/scene.xml")

    # 目标位姿：位置 + 旋转矩阵
    pos_left  = np.array([0.3, 0.2, 0.5])
    rot_left  = R.from_euler("xyz", [0, 0, 0]).as_matrix()

    pos_right = np.array([0.3, -0.2, 0.5])
    rot_right = R.from_euler("xyz", [0, 0, 0]).as_matrix()

    targets = {
        "left":  (pos_left,  rot_left),
        "right": (pos_right, rot_right),
    }

    # 单步求解，返回完整关节角 (qpos)
    q = solver.solve_step(targets)
    print("joint positions:", q)

    # 迭代求解到收敛（适合离线规划）
    q = solver.solve_until_converged(targets, max_steps=500, tol=1e-3)
"""

import warnings
import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning, module="qpsolvers")


@dataclass
class IKResult:
    """IK 求解结果。"""
    success: bool
    q: np.ndarray                        # 求解得到的关节角
    pos_errors: dict = field(default_factory=dict)   # mm, per side
    rot_errors: dict = field(default_factory=dict)   # deg, per side
    iterations: int = 0
    converged: bool = False
    message: str = ""

    def __str__(self):
        lines = [f"IKResult(success={self.success}, iters={self.iterations}, converged={self.converged})"]
        for side in self.pos_errors:
            lines.append(
                f"  [{side}] pos_err={self.pos_errors[side]:.2f} mm  "
                f"rot_err={self.rot_errors[side]:.2f} deg"
            )
        if self.message:
            lines.append(f"  msg: {self.message}")
        return "\n".join(lines)


class XarmIKSolver:
    """
    基于 mink 的双臂 IK 求解器（无 ROS / XR 依赖）。

    Parameters
    ----------
    xml_path : str
        MuJoCo XML 场景路径。
    left_ee_name : str
        左臂末端执行器 body 名称。
    right_ee_name : str
        右臂末端执行器 body 名称。
    qpos_init : np.ndarray | None
        初始关节角。None 时使用 XML 默认值。
    dt : float
        积分步长（秒）。
    position_cost : float
        位置任务权重。
    orientation_cost : float
        姿态任务权重。
    posture_cost : float
        姿态保持任务权重（防止关节漂移）。
    solver : str
        QP 求解器，"osqp" 或 "quadprog"。
    """

    def __init__(
        self,
        xml_path: str,
        left_ee_name: str = "xarm_left_hand_tcp",
        right_ee_name: str = "xarm_right_hand_tcp",
        qpos_init: np.ndarray | None = None,
        dt: float = 1.0 / 200.0,
        position_cost: float = 1.0,
        orientation_cost: float = 0.1,
        posture_cost: float = 0.01,
        solver: str = "osqp",
    ):
        self.dt = dt
        self.solver = solver

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]

        # 初始化关节角
        if qpos_init is not None:
            self.qpos_init = np.array(qpos_init, dtype=float)
        else:
            self.qpos_init = self.data.qpos.copy()

        self._reset_state()

        # mink 配置
        self.configuration = mink.Configuration(self.model)
        self.configuration.update(self.data.qpos.copy())

        self.left_ee_task = mink.FrameTask(
            frame_name=left_ee_name,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=0.05,
        )
        self.right_ee_task = mink.FrameTask(
            frame_name=right_ee_name,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=0.05,
        )
        self.posture_task = mink.PostureTask(model=self.model, cost=posture_cost)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def solve_step(
        self,
        targets: dict,
        update_state: bool = True,
    ) -> np.ndarray:
        """
        单步 IK 求解，返回关节角（轻量接口，适合实时控制循环）。
        求解异常时保持当前状态不变并打印警告。
        """
        if not targets:
            return self.data.qpos.copy()

        current_q = self.data.qpos.copy()
        self.configuration.update(current_q)
        self.posture_task.set_target(current_q)

        tasks = [self.posture_task]
        if "left" in targets:
            pos, rot = targets["left"]
            if rot is None:
                _, rot = self.get_ee_pose("left")
            self.left_ee_task.set_target(mink.SE3.from_matrix(self._make_T(pos, rot)))
            tasks.append(self.left_ee_task)
        if "right" in targets:
            pos, rot = targets["right"]
            if rot is None:
                _, rot = self.get_ee_pose("right")
            self.right_ee_task.set_target(mink.SE3.from_matrix(self._make_T(pos, rot)))
            tasks.append(self.right_ee_task)

        try:
            vel = mink.solve_ik(
                self.configuration,
                tasks=tasks,
                dt=self.dt,
                solver=self.solver,
            )
        except Exception as e:
            print(f"[IK] QP solver failed: {e}")
            return current_q

        self.configuration.integrate_inplace(vel, self.dt)
        q_result = self.configuration.q.copy()

        if update_state:
            self.data.qpos[:] = q_result
            self.data.ctrl[:] = q_result
            mujoco.mj_forward(self.model, self.data)

        return q_result

    def solve_until_converged(
        self,
        targets: dict,
        max_steps: int = 500,
        tol: float = 1e-3,
        pos_tol_mm: float = 5.0,
        rot_tol_deg: float = 2.0,
        verbose: bool = False,
    ) -> IKResult:
        """
        迭代求解直到收敛，返回 IKResult（含误差、收敛状态、失败原因）。

        Parameters
        ----------
        targets       : {"left"/"right": (pos, rot_matrix)}
        max_steps     : 最大迭代步数
        tol           : qpos L∞ 变化量收敛阈值（rad）
        pos_tol_mm    : 位置误差判定阈值（mm），超出则 success=False
        rot_tol_deg   : 姿态误差判定阈值（deg），超出则 success=False
        verbose       : True 时每步打印误差
        """
        q_prev = self.data.qpos.copy()
        converged = False

        for i in range(max_steps):
            q = self.solve_step(targets, update_state=True)
            delta = np.max(np.abs(q - q_prev))
            if verbose:
                print(f"  iter {i:3d}  Δq={delta:.2e}")
            if delta < tol:
                converged = True
                break
            q_prev = q

        # 计算各臂末端误差
        pos_errors, rot_errors = {}, {}
        fail_msgs = []
        for side, (pos_tgt, rot_tgt) in targets.items():
            pos_fk, rot_fk = self.get_ee_pose(side)
            pos_err = np.linalg.norm(pos_fk - pos_tgt) * 1000
            pos_errors[side] = pos_err
            if pos_err > pos_tol_mm:
                fail_msgs.append(f"[{side}] pos_err={pos_err:.1f}mm > {pos_tol_mm}mm")
            if rot_tgt is not None:
                rot_err = self._rotation_error_deg(rot_tgt, rot_fk)
                rot_errors[side] = rot_err
                if rot_err > rot_tol_deg:
                    fail_msgs.append(f"[{side}] rot_err={rot_err:.1f}° > {rot_tol_deg}°")
            else:
                rot_errors[side] = 0.0

        if not converged:
            fail_msgs.insert(0, f"未收敛（达到 max_steps={max_steps}）")

        success = len(fail_msgs) == 0
        message = "  |  ".join(fail_msgs) if fail_msgs else "OK"
        if not success:
            print(f"[IK] 解算失败: {message}")
            # 失败时附加限位信息，帮助判断原因
            for ji in range(self.model.njnt):
                if self.model.jnt_type[ji] != 3:
                    continue
                lo, hi = self.model.jnt_range[ji]
                if hi <= lo:
                    continue
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, ji) or f"joint{ji}"
                if "finger" in name:
                    continue
                buf = (hi - lo) * 0.10
                qval = q[self.model.jnt_qposadr[ji]]
                if qval - lo < buf:
                    print(f"[IK]   ↳ {name} 接近下限 ({np.degrees(qval):.1f}° / min={np.degrees(lo):.1f}°)")
                elif hi - qval < buf:
                    print(f"[IK]   ↳ {name} 接近上限 ({np.degrees(qval):.1f}° / max={np.degrees(hi):.1f}°)")

        return IKResult(
            success=success,
            q=q.copy(),
            pos_errors=pos_errors,
            rot_errors=rot_errors,
            iterations=i + 1,
            converged=converged,
            message=message,
        )

    def reset(self, qpos: np.ndarray | None = None):
        """将内部状态重置到 qpos_init 或指定关节角。"""
        if qpos is not None:
            self.qpos_init = np.array(qpos, dtype=float)
        self._reset_state()
        self.configuration.update(self.data.qpos.copy())

    def get_ee_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """
        读取当前末端执行器位姿。

        Parameters
        ----------
        side : "left" | "right"

        Returns
        -------
        pos : np.ndarray, shape (3,)
        rot_matrix : np.ndarray, shape (3,3)
        """
        task = self.left_ee_task if side == "left" else self.right_ee_task
        frame_name = task.frame_name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        pos = self.data.xpos[body_id].copy()
        quat_wxyz = self.data.xquat[body_id].copy()  # wxyz
        rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
        return pos, rot

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _reset_state(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos_init
        self.data.ctrl[:] = self.qpos_init
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _make_T(pos: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = pos
        return T

    @staticmethod
    def _rotation_error_deg(rot_target: np.ndarray, rot_actual: np.ndarray) -> float:
        rel = rot_target @ rot_actual.T
        trace = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace)))


# ------------------------------------------------------------------
# 快速测试入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    xml_path = sys.argv[1] if len(sys.argv) > 1 else "/home/lft-vlai2/xarm_vr_teleop/xarm_teleop/xarm/scene.xml"

    solver = XarmIKSolver(xml_path=xml_path)

    # 读取当前末端位姿作为目标（验证零误差情况）
    pos_l, rot_l = solver.get_ee_pose("left")
    pos_r, rot_r = solver.get_ee_pose("right")

    print(f"当前位姿:")
    print(f"  Left EE:")
    print(f"    pos={np.round(pos_l, 4)}")
    print(f"    rot=\n{np.round(rot_l, 4)}")
    print(f"  Right EE:")
    print(f"    pos={np.round(pos_r, 4)}")
    print(f"    rot=\n{np.round(rot_r, 4)}")

    targets = {"left": (pos_l, rot_l), "right": (pos_r, rot_r)}

    result = solver.solve_until_converged(targets, max_steps=200, tol=1e-4)
    print(f"\n求解结果:\n{result}")
    print(f"Solved qpos: {np.round(result.q, 4)}")
