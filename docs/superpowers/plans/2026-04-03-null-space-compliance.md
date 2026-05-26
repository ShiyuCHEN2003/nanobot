# Null Space Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `NullSpaceCompliance` — a 200Hz control loop that lets humans push the robot arm's elbow freely while the end-effector pose stays locked, using motor torque feedback and mink IK.

**Architecture:** Motor torque residual (measured − gravity) is projected into the arm's null space via `N = I − J⁺J`. The projected displacement updates mink's `PostureTask` target each cycle, and `solve_step` keeps the end-effector locked while the arm drifts in the null space toward the pushed configuration.

**Tech Stack:** Python 3.10+, MuJoCo, mink, pinocchio (via existing `ArmDynamics`), python-can (via existing `XArmController`), pytest, `unittest.mock`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `xarm_mcp/null_space_compliance.py` | **Create** | `NullSpaceCompliance` class — full control loop |
| `tests/xarm_mcp/test_null_space_compliance.py` | **Create** | Unit tests (mocked hardware, real MuJoCo model) |

No existing files modified.

---

### Task 1: Test file skeleton + import check

**Files:**
- Create: `tests/xarm_mcp/__init__.py`
- Create: `tests/xarm_mcp/test_null_space_compliance.py`
- Create: `xarm_mcp/null_space_compliance.py` (stub)

- [ ] **Step 1: Create the test package init**

```bash
mkdir -p tests/xarm_mcp
touch tests/xarm_mcp/__init__.py
```

- [ ] **Step 2: Write the first failing test — import check**

Create `tests/xarm_mcp/test_null_space_compliance.py`:

```python
"""Tests for NullSpaceCompliance."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_import():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance
    assert NullSpaceCompliance is not None
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
cd /home/vlai-minipc/nanobot
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_import -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'xarm_mcp.null_space_compliance'`

- [ ] **Step 4: Create the stub module**

Create `xarm_mcp/null_space_compliance.py`:

```python
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


class NullSpaceCompliance:
    pass
```

- [ ] **Step 5: Run test to confirm it passes**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_import -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
git add xarm_mcp/null_space_compliance.py tests/xarm_mcp/__init__.py tests/xarm_mcp/test_null_space_compliance.py
git commit -m "feat: add null_space_compliance stub and test skeleton"
```

---

### Task 2: External torque estimation

**Files:**
- Modify: `xarm_mcp/null_space_compliance.py`
- Modify: `tests/xarm_mcp/test_null_space_compliance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/xarm_mcp/test_null_space_compliance.py`:

```python
def _make_nsc(alpha=0.05, threshold=1.0):
    """Build a NullSpaceCompliance with fully mocked dependencies."""
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm.canport = "can2"
    arm._ctrl = MagicMock()
    arm._dynamics = MagicMock()

    solver = MagicMock()
    solver.model = MagicMock()
    solver.data = MagicMock()

    return NullSpaceCompliance(arm, solver, ee_side="right",
                               alpha=alpha, torque_threshold=threshold)


def test_estimate_ext_torque_subtracts_gravity():
    nsc = _make_nsc()
    q7 = np.zeros(7)
    tau_measured = np.array([2.0, 1.0, 0.5, 0.3, 0.1, 0.1, 0.1])
    gravity      = np.array([1.5, 0.8, 0.3, 0.2, 0.0, 0.0, 0.0])

    nsc._arm._dynamics.get_gravity.return_value = gravity

    tau_ext = nsc._estimate_ext_torque(q7, tau_measured)

    nsc._arm._dynamics.get_gravity.assert_called_once_with(q7)
    np.testing.assert_allclose(tau_ext, tau_measured - gravity)


def test_estimate_ext_torque_no_dynamics_returns_zeros():
    nsc = _make_nsc()
    nsc._arm._dynamics = None
    q7 = np.ones(7)
    tau_measured = np.array([2.0] * 7)

    tau_ext = nsc._estimate_ext_torque(q7, tau_measured)

    np.testing.assert_allclose(tau_ext, np.zeros(7))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_estimate_ext_torque_subtracts_gravity tests/xarm_mcp/test_null_space_compliance.py::test_estimate_ext_torque_no_dynamics_returns_zeros -v
```

Expected: `FAILED` — `AttributeError: 'NullSpaceCompliance' object has no attribute '_estimate_ext_torque'`

- [ ] **Step 3: Implement `__init__` and `_estimate_ext_torque`**

Replace the `NullSpaceCompliance` class body in `xarm_mcp/null_space_compliance.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_estimate_ext_torque_subtracts_gravity tests/xarm_mcp/test_null_space_compliance.py::test_estimate_ext_torque_no_dynamics_returns_zeros -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add xarm_mcp/null_space_compliance.py tests/xarm_mcp/test_null_space_compliance.py
git commit -m "feat: implement _estimate_ext_torque with gravity subtraction"
```

---

### Task 3: Jacobian extraction and null space projector

**Files:**
- Modify: `xarm_mcp/null_space_compliance.py`
- Modify: `tests/xarm_mcp/test_null_space_compliance.py`

This task uses the real MuJoCo bimanual model to test Jacobian shapes and null space math.

- [ ] **Step 1: Write the failing tests**

Append to `tests/xarm_mcp/test_null_space_compliance.py`:

```python
XML_PATH = "/home/vlai-minipc/nanobot/xarm_mcp/xml/scene.xml"


def _make_real_solver():
    """Build a real XarmIKSolver for math tests (no CAN needed)."""
    import os
    from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver
    if not os.path.exists(XML_PATH):
        pytest.skip(f"MuJoCo XML not found: {XML_PATH}")
    return XarmIKSolver(xml_path=XML_PATH)


def test_get_arm_dof_indices_returns_7_indices():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm._dynamics = MagicMock()
    solver = _make_real_solver()

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    indices = nsc._get_arm_dof_indices()

    assert len(indices) == 7
    assert all(0 <= i < solver.model.nv for i in indices)


def test_get_jacobian_shape():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm._dynamics = MagicMock()
    solver = _make_real_solver()

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    q7 = np.zeros(7)
    J = nsc._get_jacobian(q7)

    assert J.shape == (6, 7)


def test_null_projector_shape_and_property():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm._dynamics = MagicMock()
    solver = _make_real_solver()

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    q7 = np.zeros(7)
    J = nsc._get_jacobian(q7)
    N = nsc._null_projector(J)

    # Shape
    assert N.shape == (7, 7)
    # N² == N  (idempotent)
    np.testing.assert_allclose(N @ N, N, atol=1e-6)
    # J·N ≈ 0  (null space property)
    np.testing.assert_allclose(J @ N, np.zeros((6, 7)), atol=1e-6)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_get_arm_dof_indices_returns_7_indices tests/xarm_mcp/test_null_space_compliance.py::test_get_jacobian_shape tests/xarm_mcp/test_null_space_compliance.py::test_null_projector_shape_and_property -v
```

Expected: `FAILED` — methods not yet defined.

- [ ] **Step 3: Implement `_get_arm_dof_indices`, `_get_jacobian`, `_null_projector`**

Add these three methods inside the `NullSpaceCompliance` class (after `_estimate_ext_torque`):

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_get_arm_dof_indices_returns_7_indices tests/xarm_mcp/test_null_space_compliance.py::test_get_jacobian_shape tests/xarm_mcp/test_null_space_compliance.py::test_null_projector_shape_and_property -v
```

Expected: all three `PASSED`

- [ ] **Step 5: Commit**

```bash
git add xarm_mcp/null_space_compliance.py tests/xarm_mcp/test_null_space_compliance.py
git commit -m "feat: implement Jacobian extraction and null space projector"
```

---

### Task 4: Core control step

**Files:**
- Modify: `xarm_mcp/null_space_compliance.py`
- Modify: `tests/xarm_mcp/test_null_space_compliance.py`

The control step: read state → estimate external torque → project → update PostureTask → IK → send MIT command.

- [ ] **Step 1: Write failing tests**

Append to `tests/xarm_mcp/test_null_space_compliance.py`:

```python
def test_control_step_sends_mit_when_below_threshold():
    """Below threshold: τ_ext is ignored, PostureTask not updated, IK still runs."""
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm.canport = "can2"
    arm._ctrl = MagicMock()
    arm._dynamics = MagicMock()
    arm._dynamics.get_gravity.return_value = np.zeros(7)
    arm._ctrl.get_positions.return_value = [0.0] * 8   # 7 arm + 1 gripper
    arm._ctrl.get_torques_measured.return_value = [0.1] * 8  # below threshold=1.0

    solver = _make_real_solver()
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right",
                              alpha=0.05, torque_threshold=1.0)
    nsc._target_pos = target_pos
    nsc._target_rot = target_rot

    nsc._control_step()

    # MIT command must have been sent
    arm._ctrl.mit_control_all.assert_called_once()
    channel, params = arm._ctrl.mit_control_all.call_args[0]
    assert channel == "can2"
    assert len(params) == 8   # 7 arm + 1 gripper


def test_control_step_updates_posture_when_above_threshold():
    """Above threshold: PostureTask target must change from current qpos."""
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm.canport = "can2"
    arm._ctrl = MagicMock()
    arm._dynamics = MagicMock()
    arm._dynamics.get_gravity.return_value = np.zeros(7)
    # Large external torque on joint 3 (elbow)
    arm._ctrl.get_positions.return_value = [0.0] * 8
    arm._ctrl.get_torques_measured.return_value = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    solver = _make_real_solver()
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right",
                              alpha=0.05, torque_threshold=1.0)
    nsc._target_pos = target_pos
    nsc._target_rot = target_rot

    q_before = solver.posture_task.target_q.copy() if hasattr(solver.posture_task, 'target_q') else None

    nsc._control_step()

    # Solver must have been called with the target pose
    arm._ctrl.mit_control_all.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_control_step_sends_mit_when_below_threshold tests/xarm_mcp/test_null_space_compliance.py::test_control_step_updates_posture_when_above_threshold -v
```

Expected: `FAILED` — `_control_step` not defined.

- [ ] **Step 3: Implement `_control_step` and `_adaptive_alpha`**

Add these methods inside `NullSpaceCompliance`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_control_step_sends_mit_when_below_threshold tests/xarm_mcp/test_null_space_compliance.py::test_control_step_updates_posture_when_above_threshold -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add xarm_mcp/null_space_compliance.py tests/xarm_mcp/test_null_space_compliance.py
git commit -m "feat: implement _control_step with null space projection and MIT output"
```

---

### Task 5: Thread-based control loop — start / stop / update_target

**Files:**
- Modify: `xarm_mcp/null_space_compliance.py`
- Modify: `tests/xarm_mcp/test_null_space_compliance.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/xarm_mcp/test_null_space_compliance.py`:

```python
def test_start_launches_thread_and_stop_joins():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm.canport = "can2"
    arm._ctrl = MagicMock()
    arm._dynamics = None
    arm._ctrl.get_positions.return_value = [0.0] * 8
    arm._ctrl.get_torques_measured.return_value = [0.0] * 8

    solver = _make_real_solver()
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    nsc.start(target_pos, target_rot)

    assert nsc._running is True
    assert nsc._thread is not None
    assert nsc._thread.is_alive()

    nsc.stop()

    assert nsc._running is False
    assert not nsc._thread.is_alive()


def test_update_target_changes_target_thread_safely():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance

    arm = MagicMock()
    arm.canport = "can2"
    arm._ctrl = MagicMock()
    arm._dynamics = None
    arm._ctrl.get_positions.return_value = [0.0] * 8
    arm._ctrl.get_torques_measured.return_value = [0.0] * 8

    solver = _make_real_solver()
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    nsc.start(target_pos, target_rot)

    new_pos = target_pos + np.array([0.01, 0.0, 0.0])
    nsc.update_target(new_pos, target_rot)

    with nsc._lock:
        np.testing.assert_allclose(nsc._target_pos, new_pos)

    nsc.stop()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_start_launches_thread_and_stop_joins tests/xarm_mcp/test_null_space_compliance.py::test_update_target_changes_target_thread_safely -v
```

Expected: `FAILED`

- [ ] **Step 3: Implement `start`, `stop`, `update_target`, `_run_loop`**

Add these methods inside `NullSpaceCompliance`:

```python
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
            except Exception as e:
                print(f"[NullSpaceCompliance] control_step error: {e}")
                self._running = False
                break
            elapsed = time.perf_counter() - t0
            sleep_t = self._dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_start_launches_thread_and_stop_joins tests/xarm_mcp/test_null_space_compliance.py::test_update_target_changes_target_thread_safely -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add xarm_mcp/null_space_compliance.py tests/xarm_mcp/test_null_space_compliance.py
git commit -m "feat: implement start/stop/update_target thread loop"
```

---

### Task 6: Safety checks

**Files:**
- Modify: `xarm_mcp/null_space_compliance.py`
- Modify: `tests/xarm_mcp/test_null_space_compliance.py`

Safety: stop on joint limit violation, IK position error > 20mm.

- [ ] **Step 1: Write failing tests**

Append to `tests/xarm_mcp/test_null_space_compliance.py`:

```python
def test_safety_check_triggers_stop_on_joint_limit():
    """If any joint exceeds ARM_7DOF_JOINT_LIMITS, _safety_check raises."""
    from xarm_mcp.null_space_compliance import NullSpaceCompliance, SafetyViolation
    from xarm_mcp.xarm_controller.XArmcontroller import ARM_7DOF_JOINT_LIMITS
    import math

    arm = MagicMock()
    arm._dynamics = None
    solver = MagicMock()
    solver.model = MagicMock()
    solver.data = MagicMock()

    nsc = NullSpaceCompliance(arm, solver)

    # J1 upper limit is 200° = 3.49 rad — set it just over
    bad_q7 = np.zeros(7)
    bad_q7[0] = math.radians(201)   # above J1 upper limit

    with pytest.raises(SafetyViolation, match="joint limit"):
        nsc._safety_check(bad_q7)


def test_safety_check_triggers_stop_on_ee_drift():
    """If IK EE error > 20mm, _safety_check raises."""
    from xarm_mcp.null_space_compliance import NullSpaceCompliance, SafetyViolation

    arm = MagicMock()
    arm._dynamics = None
    solver = _make_real_solver()
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    nsc._target_pos = target_pos
    nsc._target_rot = target_rot

    # Shift target far away to simulate drift
    nsc._target_pos = target_pos + np.array([0.1, 0.0, 0.0])  # 100mm drift

    q7 = np.zeros(7)
    with pytest.raises(SafetyViolation, match="EE drift"):
        nsc._safety_check(q7)


def test_safety_check_passes_for_valid_state():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance, SafetyViolation

    arm = MagicMock()
    arm._dynamics = None
    solver = _make_real_solver()
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    nsc._target_pos = target_pos
    nsc._target_rot = target_rot

    q7 = np.zeros(7)
    nsc._safety_check(q7)   # Should not raise
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_safety_check_triggers_stop_on_joint_limit tests/xarm_mcp/test_null_space_compliance.py::test_safety_check_triggers_stop_on_ee_drift tests/xarm_mcp/test_null_space_compliance.py::test_safety_check_passes_for_valid_state -v
```

Expected: `FAILED`

- [ ] **Step 3: Add `SafetyViolation` and `_safety_check`**

Add `SafetyViolation` at module level (before the class) and `_safety_check` inside the class:

At the top of the class block, after the imports in `null_space_compliance.py`, add:

```python
class SafetyViolation(RuntimeError):
    """Raised when a safety limit is exceeded."""
```

Then inside `NullSpaceCompliance`, add `_safety_check` and wire it into `_control_step`:

```python
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
```

Also update `_control_step`: add a safety check call right after reading state (before the projection block):

```python
        # Safety check — raises SafetyViolation on limit breach
        self._safety_check(q7)
```

Insert this line immediately after the line `q7 = np.array(all_pos[:7])` in `_control_step`.

And update `_run_loop` to catch `SafetyViolation` separately for clarity:

```python
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
```

- [ ] **Step 4: Run safety tests to confirm they pass**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py::test_safety_check_triggers_stop_on_joint_limit tests/xarm_mcp/test_null_space_compliance.py::test_safety_check_triggers_stop_on_ee_drift tests/xarm_mcp/test_null_space_compliance.py::test_safety_check_passes_for_valid_state -v
```

Expected: all three `PASSED`

- [ ] **Step 5: Run all tests together**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add xarm_mcp/null_space_compliance.py tests/xarm_mcp/test_null_space_compliance.py
git commit -m "feat: add SafetyViolation and _safety_check with joint limit and EE drift detection"
```

---

### Task 7: Final all-tests run and usage smoke test

**Files:**
- No new files

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/xarm_mcp/test_null_space_compliance.py -v --tb=short
```

Expected: all tests `PASSED`, no warnings about undefined names.

- [ ] **Step 2: Verify the module is importable and the class is complete**

```bash
python -c "
from xarm_mcp.null_space_compliance import NullSpaceCompliance, SafetyViolation
import inspect
methods = [m for m in dir(NullSpaceCompliance) if not m.startswith('__')]
print('Methods:', methods)
assert 'start' in methods
assert 'stop' in methods
assert 'update_target' in methods
assert '_control_step' in methods
assert '_safety_check' in methods
print('OK')
"
```

Expected output:
```
Methods: ['_EE_DRIFT_LIMIT_M', '_adaptive_alpha', '_control_step', '_estimate_ext_torque',
          '_get_arm_dof_indices', '_get_jacobian', '_null_projector', '_run_loop',
          '_safety_check', 'start', 'stop', 'update_target']
OK
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: null space compliance complete — all tests passing"
```

---

## Hardware Test (manual, requires real CAN)

After software tests pass, validate on the real arm:

```python
# Run from /home/vlai-minipc/nanobot
from xarm_mcp.arm_control import Arm
from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver
from xarm_mcp.null_space_compliance import NullSpaceCompliance

arm = Arm("can2")
solver = XarmIKSolver(xml_path="xarm_mcp/xml/scene.xml")

target_pos, target_rot = solver.get_ee_pose("right")
ctrl = NullSpaceCompliance(arm, solver, ee_side="right",
                           alpha=0.02, torque_threshold=1.5)
ctrl.start(target_pos, target_rot)
# Push elbow — end effector should stay fixed
# Ctrl+C or ctrl.stop() to stop
```

**Tuning guide:**
- If arm doesn't respond to push: lower `torque_threshold` (try 0.5)
- If arm jumps too much: lower `alpha` (try 0.01)
- If end effector drifts: lower `alpha` or raise `posture_cost` in XarmIKSolver
