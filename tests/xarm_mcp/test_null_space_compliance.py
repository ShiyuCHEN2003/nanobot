"""Tests for NullSpaceCompliance."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_import():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance
    assert NullSpaceCompliance is not None


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


XML_PATH = "/home/vlai-minipc/nanobot/xarm_mcp/xml/scene.xml"


def _make_real_solver():
    """Build a real XarmIKSolver for math tests (no CAN needed)."""
    import os
    from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver
    if not os.path.exists(XML_PATH):
        raise RuntimeError(f"MuJoCo XML not found: {XML_PATH}")
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
    import numpy as np2
    np2.testing.assert_allclose(N @ N, N, atol=1e-6)
    # J·N ≈ 0  (null space property)
    np2.testing.assert_allclose(J @ N, np.zeros((6, 7)), atol=1e-6)


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

    nsc._control_step()

    # Solver must have been called with the target pose
    arm._ctrl.mit_control_all.assert_called_once()


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


def test_safety_check_triggers_stop_on_joint_limit():
    """If any joint exceeds ARM_7DOF_JOINT_LIMITS, _safety_check raises."""
    from xarm_mcp.null_space_compliance import NullSpaceCompliance, SafetyViolation
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

    try:
        nsc._safety_check(bad_q7)
        assert False, "Expected SafetyViolation"
    except SafetyViolation as e:
        assert "joint limit" in str(e).lower()
    print("test_safety_check_triggers_stop_on_joint_limit: PASSED")


def test_safety_check_triggers_stop_on_ee_drift():
    """If IK EE error > 20mm, _safety_check raises."""
    from xarm_mcp.null_space_compliance import NullSpaceCompliance, SafetyViolation
    from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver

    arm = MagicMock()
    arm._dynamics = None
    solver = XarmIKSolver(xml_path="xarm_mcp/xml/scene.xml")
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    nsc._target_pos = target_pos + np.array([0.1, 0.0, 0.0])  # 100mm drift

    q7 = np.zeros(7)
    try:
        nsc._safety_check(q7)
        assert False, "Expected SafetyViolation"
    except SafetyViolation as e:
        assert "ee drift" in str(e).lower()
    print("test_safety_check_triggers_stop_on_ee_drift: PASSED")


def test_safety_check_passes_for_valid_state():
    from xarm_mcp.null_space_compliance import NullSpaceCompliance
    from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver

    arm = MagicMock()
    arm._dynamics = None
    solver = XarmIKSolver(xml_path="xarm_mcp/xml/scene.xml")
    target_pos, target_rot = solver.get_ee_pose("right")

    nsc = NullSpaceCompliance(arm, solver, ee_side="right")
    nsc._target_pos = target_pos
    nsc._target_rot = target_rot

    q7 = np.zeros(7)
    nsc._safety_check(q7)   # Should not raise
    print("test_safety_check_passes_for_valid_state: PASSED")
