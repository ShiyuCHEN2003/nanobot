"""
零空间顺应控制。

允许人推动机械臂冗余关节（肘部），末端位姿由 IK 闭环维持不变。
外力通过电机力矩回读减去重力补偿估计。
"""
from __future__ import annotations

import threading
import time

import mujoco
import numpy as np

from xarm_mcp.arm_control import Arm
from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver
from xarm_mcp.xarm_controller.XArmcontroller import MITParam, ARM_7DOF_JOINT_LIMITS


class SafetyViolation(RuntimeError):
    """Raised when a safety limit is exceeded."""


class NullSpaceCompliance:
    def __init__(
        self,
        arm: Arm,
        solver: XarmIKSolver,
        ee_side: str = "right",
        alpha: float = 0.05,
        torque_threshold: float = 1.0,
        dt: float = 1.0 / 200.0,
    ):
        self._arm = arm
        self._solver = solver
        self._ee_side = ee_side
        self._alpha = alpha
        self._torque_threshold = torque_threshold
        self._dt = dt

        self._target_pos: np.ndarray | None = None
        self._target_rot: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def _estimate_ext_torque(
        self, q7: np.ndarray, tau_measured: np.ndarray
    ) -> np.ndarray:
        """τ_ext = τ_measured − gravity(q).  Returns zeros if no dynamics model."""
        if self._arm._dynamics is None:
            return np.zeros(7)
        gravity = self._arm._dynamics.get_gravity(q7)
        return tau_measured - gravity

    def _get_arm_dof_indices(self) -> np.ndarray:
        """Return the nv (velocity) column indices for this arm's 7 joints."""
        model = self._solver.model
        prefix = f"xarm_{self._ee_side}_joint"
        indices = []
        for i in range(1, 8):
            joint_name = f"{prefix}{i}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise RuntimeError(f"Joint not found in model: {joint_name}")
            indices.append(model.jnt_dofadr[jid])
        return np.array(indices, dtype=int)

    def _get_jacobian(self, q7: np.ndarray) -> np.ndarray:
        """
        Compute the 6×7 end-effector Jacobian for the arm joints.

        Syncs the arm's q7 into the solver's MuJoCo data, runs mj_forward,
        then extracts only the arm-joint columns from the full 6×nv Jacobian.
        """
        model = self._solver.model
        data = self._solver.data

        # Sync arm joints into the full model qpos
        arm_indices = self._get_arm_dof_indices()
        data.qpos[arm_indices] = q7
        mujoco.mj_forward(model, data)

        # Look up the end-effector body
        ee_name = (
            self._solver.left_ee_task.frame_name
            if self._ee_side == "left"
            else self._solver.right_ee_task.frame_name
        )
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if body_id < 0:
            raise RuntimeError(f"EE body not found: {ee_name}")

        nv = model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        J_full = np.vstack([jacp, jacr])          # (6, nv)
        return J_full[:, arm_indices]              # (6, 7)

    def _null_projector(self, J: np.ndarray) -> np.ndarray:
        """N = I₇ − J⁺·J  (7×7 null space projector)."""
        J_pinv = np.linalg.pinv(J)                # (7, 6)
        return np.eye(7) - J_pinv @ J

    _EE_DRIFT_LIMIT_M = 0.020   # 20mm

    def _safety_check(self, q7: np.ndarray) -> None:
        """
        Raise SafetyViolation if:
          - any joint is outside ARM_7DOF_JOINT_LIMITS
          - EE position drifted more than 20mm from target
        """
        for j, (lo, hi) in enumerate(ARM_7DOF_JOINT_LIMITS):
            if not (lo <= q7[j] <= hi):
                raise SafetyViolation(
                    f"joint limit violated: joint {j+1} = {np.degrees(q7[j]):.1f}° "
                    f"(limits [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°])"
                )

        if self._target_pos is not None:
            ee_pos, _ = self._solver.get_ee_pose(self._ee_side)
            drift = float(np.linalg.norm(ee_pos - self._target_pos))
            if drift > self._EE_DRIFT_LIMIT_M:
                raise SafetyViolation(
                    f"EE drift too large: {drift*1000:.1f}mm > "
                    f"{self._EE_DRIFT_LIMIT_M*1000:.0f}mm"
                )

    def _adaptive_alpha(self, tau_ext_norm: float) -> float:
        """Scale alpha down when external torque is very large."""
        if tau_ext_norm > 3.0:
            return self._alpha * 3.0 / tau_ext_norm
        return self._alpha

    def _control_step(self) -> None:
        """One 200Hz tick: sense → estimate → project → IK → actuate."""
        # 1. Read current state from CAN
        self._arm._ctrl.recv_all(self._arm.canport, timeout=0.005)
        all_pos    = self._arm._ctrl.get_positions(self._arm.canport)
        all_torque = self._arm._ctrl.get_torques_measured(self._arm.canport)

        q7         = np.array(all_pos[:7])

        # Safety check — raises SafetyViolation on limit breach
        self._safety_check(q7)

        tau_meas   = np.array(all_torque[:7])
        gripper_q  = float(all_pos[7]) if len(all_pos) > 7 else 0.0

        # 2. Estimate external torque
        tau_ext      = self._estimate_ext_torque(q7, tau_meas)
        tau_ext_norm = float(np.linalg.norm(tau_ext))

        # 3. Null space projection (only when external force is detectable)
        if tau_ext_norm > self._torque_threshold:
            J         = self._get_jacobian(q7)
            N         = self._null_projector(J)
            alpha_eff = self._adaptive_alpha(tau_ext_norm)
            delta_q   = alpha_eff * N @ tau_ext               # (7,)

            # Update PostureTask: push the "desired posture" in null space direction
            full_qpos = self._solver.data.qpos.copy()
            arm_idx   = self._get_arm_dof_indices()
            full_qpos[arm_idx] += delta_q
            self._solver.posture_task.set_target(full_qpos)
        else:
            # No external force: keep posture at current joint angles (don't snap back)
            full_qpos = self._solver.data.qpos.copy()
            self._solver.posture_task.set_target(full_qpos)

        # 4. IK solve step — end-effector pose is the hard constraint
        with self._lock:
            target_pos = self._target_pos
            target_rot = self._target_rot

        if target_pos is None:
            return

        q_result = self._solver.solve_step(
            {self._ee_side: (target_pos, target_rot)},
            update_state=True,
        )

        # 5. Extract arm q7 from full qpos result and send MIT command
        arm_idx  = self._get_arm_dof_indices()
        q7_cmd   = q_result[arm_idx]

        # Gravity feedforward
        tau_ff = (
            self._arm._dynamics.get_gravity(q7_cmd)
            if self._arm._dynamics is not None
            else np.zeros(7)
        )

        from xarm_mcp.arm_control import KP, KD, GRIPPER_KP, GRIPPER_KD
        params = [
            MITParam(kp=KP[j], kd=KD[j], q=float(q7_cmd[j]), tau=float(tau_ff[j]))
            for j in range(7)
        ]
        params.append(MITParam(kp=GRIPPER_KP, kd=GRIPPER_KD, q=gripper_q))

        self._arm._ctrl.mit_control_all(self._arm.canport, params)

    def start(self, target_pos: np.ndarray, target_rot: np.ndarray) -> None:
        """Set end-effector target and start the 200Hz control loop thread."""
        with self._lock:
            self._target_pos = np.array(target_pos, dtype=float)
            self._target_rot = np.array(target_rot, dtype=float)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the control loop and disable all motors."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._arm._ctrl.disable_all(self._arm.canport)

    def update_target(self, target_pos: np.ndarray, target_rot: np.ndarray) -> None:
        """Thread-safe update of the end-effector target pose."""
        with self._lock:
            self._target_pos = np.array(target_pos, dtype=float)
            self._target_rot = np.array(target_rot, dtype=float)

    def _run_loop(self) -> None:
        """Main 200Hz control loop — runs in a background thread."""
        while self._running:
            t0 = time.perf_counter()
            try:
                self._control_step()
            except SafetyViolation as e:
                print(f"[NullSpaceCompliance] SAFETY STOP: {e}")
                self._running = False
                break
            except Exception as e:
                print(f"[NullSpaceCompliance] control_step error: {e}")
                self._running = False
                break
            elapsed = time.perf_counter() - t0
            sleep_t = self._dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
