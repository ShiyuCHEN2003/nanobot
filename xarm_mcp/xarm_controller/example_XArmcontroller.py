"""
XArmController 使用示例

运行前确保 CAN 接口已启动：
  sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
  sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on
"""

import time
import signal
import threading
import sys
import numpy as np

from XArmcontroller import (
    XArmController,
    MITParam,
    ARM_7DOF_LIMITS,
    ARM_8DOF_LIMITS,
    ARM_8DOF_SLAVE_IDS,
    ARM_8DOF_MASTER_IDS,
)
from gravity_comp import (ArmDynamics, GRAVITY_SCALE, RIGHT_KD,
                            compute_friction, _Fc, _k, _Fv, _Fo,
                            CORIOLIS_GAIN, FRICTION_GAIN)

_stop = threading.Event()
signal.signal(signal.SIGINT, lambda *_: _stop.set())


# ============================================================
# 示例 1：只使用右臂 (can0)，读取状态
# ============================================================

def example_read_state():
    print("=== 示例1：读取电机状态 ===")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    try:
        ctrl.enable_all()

        for _ in range(5):
            ctrl.recv_all()
            pos = ctrl.get_positions("can0")         # [rad] x7
            vel = ctrl.get_velocities("can0")        # [rad/s] x7
            tau = ctrl.get_torques_measured("can0")  # [Nm] x7

            print(f"  pos: {[f'{p:.3f}' for p in pos]}")
            print(f"  vel: {[f'{v:.3f}' for v in vel]}")
            print(f"  tau: {[f'{t:.3f}' for t in tau]}")
            print()
            time.sleep(0.2)

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 2：读取全部电机完整状态（get_all_states / num_motors）
# ============================================================

def example_read_all_states():
    print("=== 示例2：读取全部电机完整状态 ===")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    try:
        ctrl.enable_all()
        ctrl.recv_all(timeout=0.05)

        n = ctrl.num_motors("can0")
        print(f"  总线 can0 电机数: {n}")

        states = ctrl.get_all_states("can0")
        for i, s in enumerate(states):
            print(f"  motor{i+1} (ID=0x{s.can_id:02X}): "
                  f"pos={s.position:.4f} rad  vel={s.velocity:.4f} rad/s  "
                  f"tau={s.torque:.4f} Nm")

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 3：单关节力矩控制（对 motor 0 施加 2 Nm）
# ============================================================

def example_torque_one():
    print("=== 示例3：单关节力矩控制 ===")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    try:
        ctrl.enable_all()
        print("对 motor0 施加 2 Nm，持续 3 秒...")

        start = time.perf_counter()
        while time.perf_counter() - start < 3.0 and not _stop.is_set():
            ctrl.torque_control_one("can0", motor_idx=0, tau=2.0)
            ctrl.recv_all()

            s = ctrl.get_state("can0", motor_idx=0)
            print(f"  motor0  pos={s.position:.4f} rad  vel={s.velocity:.4f} rad/s  tau={s.torque:.4f} Nm")

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 4：全部电机力矩控制
# ============================================================

def example_torque_all():
    print("=== 示例4：全部电机力矩控制 ===")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    FIXED_TAU = [0.0, 2.0, 0.0, 0.5, 0.0, 0.0, 0.0]

    try:
        ctrl.enable_all()
        print("发送固定力矩，持续 5 秒（按 Ctrl+C 停止）...")

        start = time.perf_counter()
        while time.perf_counter() - start < 5.0 and not _stop.is_set():
            ctrl.torque_control_all("can0", taus=FIXED_TAU)
            ctrl.recv_all()

            pos = ctrl.get_positions("can0")
            print(f"  pos: {[f'{p:.3f}' for p in pos]}")

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 5：位置控制（所有关节回零位）
# ============================================================

def example_position_home():
    print("=== 示例5：位置控制 → 回零位 ===")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    KP = 20.0
    KD = 0.5

    try:
        ctrl.enable_all()
        print("所有关节缓慢运动到零位，持续 5 秒...")

        start = time.perf_counter()
        while time.perf_counter() - start < 5.0 and not _stop.is_set():
            ctrl.position_control_all(
                "can0",
                kp=KP, kd=KD,
                positions=[0.0] * 7,
            )
            ctrl.recv_all()

            pos = ctrl.get_positions("can0")
            print(f"  pos: {[f'{p:.3f}' for p in pos]}")
            time.sleep(0.05)

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 6：位置控制 + 重力前馈（MITParam + position_control_all tau_ff）
# ============================================================

def example_position_with_ff():
    print("=== 示例6：位置控制 + 重力前馈 ===")
    URDF = "/home/lft-vlai2/damiao_python/xarm_description/urdf/robot/xarm_bimanual.urdf"
    dyn  = ArmDynamics(URDF, "xarm_right_")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    KP     = [50.0, 50.0, 30.0, 30.0, 10.0, 10.0, 10.0]
    KD     = [ 1.0,  1.0,  0.5,  0.5,  0.2,  0.2,  0.2]
    TARGET = [0.0, 0.0, 0.0, 1.283, 0.0, 0.0, 0.0]

    try:
        ctrl.enable_all()
        ctrl.recv_all(timeout=0.05)
        print("位置控制到目标 + 重力前馈，持续 5 秒...")

        start = time.perf_counter()
        while time.perf_counter() - start < 5.0 and not _stop.is_set():
            q      = np.array(ctrl.get_positions("can0"))
            tau_ff = dyn.get_gravity(q).tolist()

            params = [MITParam(kp=KP[i], kd=KD[i], q=TARGET[i], tau=tau_ff[i])
                      for i in range(7)]
            ctrl.mit_control_all("can0", params)
            ctrl.recv_all()

            pos = ctrl.get_positions("can0")
            print(f"  pos: {[f'{p:.3f}' for p in pos]}")
            time.sleep(0.02)

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 7：纯重力补偿（悬浮模式）
# ============================================================

def example_gravity_comp():
    print("=== 示例7：重力补偿（悬浮模式）===")

    URDF = "/home/lft-vlai2/damiao_python/xarm_description/urdf/robot/xarm_bimanual.urdf"
    dyn  = ArmDynamics(URDF, "xarm_right_")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    frame_count = 0
    start_time  = time.monotonic()
    last_print  = start_time
    N = 7

    try:
        ctrl.enable_all()
        ctrl.recv_all(channel="can0", timeout=0.05)
        print("重力补偿已启动，用手拨动手臂，感受「悬浮」效果，按 Ctrl+C 停止\n")
        print(f"  GRAVITY_SCALE={GRAVITY_SCALE}  RIGHT_KD={RIGHT_KD}")

        while not _stop.is_set():
            frame_count += 1

            q  = np.array(ctrl.get_positions("can0"))
            dq = np.array(ctrl.get_velocities("can0"))

            g_tau   = dyn.get_gravity(q)
            cor_tau = dyn.get_coriolis(q, dq)

            params = []
            for i in range(N):
                fric  = compute_friction(dq[i], _Fc[i], _k[i], _Fv[i], _Fo[i])
                total = g_tau[i] * GRAVITY_SCALE + cor_tau[i] * CORIOLIS_GAIN + fric * FRICTION_GAIN
                params.append(MITParam(kp=0.0, kd=RIGHT_KD[i], q=0.0, dq=0.0, tau=total))

            ctrl.mit_control_all("can0", params)
            ctrl.recv_all(channel="can0")

            now = time.monotonic()
            if now - last_print >= 1.0:
                hz = frame_count / (now - start_time)
                tau_meas = ctrl.get_torques_measured("can0")
                print(f"  {hz:.1f} Hz | pos:      {[f'{v:.3f}' for v in q.tolist()]}")
                print(f"           | tau_cmd:  {[f'{v:.3f}' for v in g_tau.tolist()]}")
                print(f"           | tau_meas: {[f'{v:.3f}' for v in tau_meas]}")
                last_print = now

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 8：双臂同步控制（mit_control_bimanual / torque_control_bimanual）
# ============================================================

def example_bimanual():
    print("=== 示例8：双臂同步控制 ===")
    ctrl = XArmController({
        "can0": ARM_7DOF_LIMITS,   # 右臂
        "can1": ARM_7DOF_LIMITS,   # 左臂
    })

    try:
        ctrl.enable_all()
        ctrl.recv_all(timeout=0.05)
        print("双臂同时发送零阻尼指令，持续 3 秒...")

        start = time.perf_counter()
        while time.perf_counter() - start < 3.0 and not _stop.is_set():
            zero_params = [MITParam(kp=0, kd=0.1) for _ in range(7)]
            ctrl.mit_control_bimanual({
                "can0": zero_params,
                "can1": zero_params,
            })
            ctrl.recv_all()   # 顺序接收两路

            r_pos = ctrl.get_positions("can0")
            l_pos = ctrl.get_positions("can1")
            print(f"  right: {[f'{p:.3f}' for p in r_pos]}")
            print(f"  left:  {[f'{p:.3f}' for p in l_pos]}")
            time.sleep(0.1)

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 9：夹爪控制（8-DOF，ARM_8DOF_LIMITS）
# ============================================================

def example_gripper():
    print("=== 示例9：夹爪控制（8-DOF）===")
    ctrl = XArmController(
        {"can0": ARM_8DOF_LIMITS},
        slave_ids=ARM_8DOF_SLAVE_IDS,
        master_ids=ARM_8DOF_MASTER_IDS,
    )

    try:
        ctrl.enable_all()
        ctrl.recv_all(timeout=0.05)

        print("夹爪张开（pos=-1.0）...")
        start = time.perf_counter()
        while time.perf_counter() - start < 2.0 and not _stop.is_set():
            ctrl.gripper_position_control("can0", pos=-1.0, kp=50.0, kd=1.0)
            ctrl.recv_all(channel="can0")
        s = ctrl.get_gripper_state("can0")
        print(f"  夹爪状态: pos={s.position:.4f}  vel={s.velocity:.4f}  tau={s.torque:.4f}")

        print("夹爪闭合（pos=0.0）...")
        start = time.perf_counter()
        while time.perf_counter() - start < 2.0 and not _stop.is_set():
            ctrl.gripper_position_control("can0", pos=0.0, kp=50.0, kd=1.0)
            ctrl.recv_all(channel="can0")
        print(f"  夹爪位置: {ctrl.get_gripper_position('can0'):.4f} rad")

    finally:
        ctrl.disable_all()
        ctrl.close()


# ============================================================
# 示例 10：单电机使能/失能 + 设置零点（set_zero）
# ============================================================

def example_single_enable_and_zero():
    print("=== 示例10：单电机使能 + 设置零点 ===")
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})

    try:
        ctrl.enable_one("can0", motor_idx=0)
        ctrl.recv_all(timeout=0.05)

        s = ctrl.get_state("can0", motor_idx=0)
        print(f"  motor0 当前位置: {s.position:.4f} rad")

        # 将当前位置设为零点（谨慎使用！会改变电机零点标定）
        # ctrl.set_zero("can0", motor_idx=0)
        print("  （set_zero 已注释，取消注释后会将当前位置标定为零点）")

        ctrl.disable_one("can0", motor_idx=0)

    finally:
        ctrl.close()


# ============================================================
# 入口
# ============================================================

EXAMPLES = {
    "1":  ("读取电机状态",                          example_read_state),
    "2":  ("读取全部电机完整状态（get_all_states）",  example_read_all_states),
    "3":  ("单关节力矩控制",                         example_torque_one),
    "4":  ("全部电机力矩控制",                       example_torque_all),
    "5":  ("位置控制 → 回零位",                      example_position_home),
    "6":  ("位置控制 + 重力前馈（MITParam）",         example_position_with_ff),
    "7":  ("纯重力补偿（悬浮模式）",                  example_gravity_comp),
    "8":  ("双臂同步控制（bimanual）",                example_bimanual),
    "9":  ("夹爪控制（8-DOF）",                       example_gripper),
    "10": ("单电机使能/失能 + 设置零点",               example_single_enable_and_zero),
}

if __name__ == "__main__":
    print("XArmController 使用示例\n")
    for k, (desc, _) in EXAMPLES.items():
        print(f"  {k:>2}. {desc}")
    print()

    choice = sys.argv[1] if len(sys.argv) > 1 else input("选择示例 (1-10): ").strip()

    if choice in EXAMPLES:
        EXAMPLES[choice][1]()
    else:
        print(f"无效选择：{choice}")
        sys.exit(1)
