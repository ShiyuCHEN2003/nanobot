"""
PD 参数扫描测试脚本 — 用于找到最优 KP/KD 以减少电机抖动。

用法：
    cd /home/vlai-minipc/nanobot
    python xarm_mcp/pid_sweep_test.py

结果保存到 /tmp/pid_sweep_results.csv，并在终端打印最优参数。

测试轨迹覆盖全工作空间：
  HOME → POSE_A(肩肘大幅摆动) → POSE_B(腕部大幅运动) → POSE_C(负向+伸展) → HOME
"""

import sys
import os
import time
import csv
import itertools

import numpy as np

# 将 xarm_mcp 目录加入路径（直接执行时需要）
sys.path.insert(0, os.path.dirname(__file__))
from arm_control import build_arm, read_q7, _clip_q7, JOINT_LIMITS, KP, KD, GRIPPER_KP, GRIPPER_KD

try:
    from xarm_mcp.xarm_controller.XArmcontroller import MITParam
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "xarm_controller"))
    from XArmcontroller import MITParam  # type: ignore

# ── 测试路径点 (度) ──────────────────────────────────────────────────────
#   各关节限位：J1[-80,200] J2[-10,190] J3[-90,90] J4[0,140] J5[-90,90] J6[-45,45] J7[-90,90]
#   取限位的约 50% 幅度，留安全余量
#
#   路径：HOME → A → B → C → HOME
#
#   POSE_A  肩+肘大幅摆动（重载关节 J1-J4 为主）
#           J1: 0→50°  J2: 20→70°  J3: 0→40°  J4: 60→110°  J5-7: 不动
POSE_A_DEG = np.array([ 50.0,  70.0,  40.0, 110.0,   0.0,  0.0,   0.0])

#   POSE_B  腕部大幅运动（轻载关节 J5-J7 为主）+ 手臂适度弯曲
#           J1: 50→15°  J2: 70→45°  J3: 40→-25°  J4: 110→80°
#           J5: 0→55°   J6: 0→30°   J7: 0→55°
POSE_B_DEG = np.array([ 15.0,  45.0, -25.0,  80.0,  55.0, 30.0,  55.0])

#   POSE_C  负向 + 伸展（测试另一侧极性）
#           J1: 15→-35°  J2: 45→55°  J3: -25→35°  J4: 80→100°
#           J5: 55→-50°  J6: 30→-25° J7: 55→-50°
POSE_C_DEG = np.array([-35.0,  55.0,  35.0, 100.0, -50.0, -25.0, -50.0])

HOME_DEG = np.array([0.0, 20.0, 0.0, 60.0, 0.0, 0.0, 0.0])

# 转换为弧度
HOME_RAD   = np.deg2rad(HOME_DEG)
WAYPOINTS  = [np.deg2rad(d) for d in [POSE_A_DEG, POSE_B_DEG, POSE_C_DEG, HOME_DEG]]
WAYPOINT_NAMES = ["POSE_A(肩肘)", "POSE_B(腕部)", "POSE_C(负向)", "HOME"]

# ── 扫描网格 ──────────────────────────────────────────────────────────────
KP_SCALES = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
KD_SCALES = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

# ── 运动参数 ──────────────────────────────────────────────────────────────
DURATION   = 4.0    # 每段运动时间 (s)
N_STEPS    = 800    # 插值步数 → dt ≈ 5ms
HOLD_SECS  = 0.4    # 到达目标后保持时间 (s)
CANPORT    = "can2"
MAX_ERR_RAD = np.deg2rad(10.0)  # 超过此误差则中止试验（覆盖更大范围，放宽到 10°）

# ── 输出文件 ──────────────────────────────────────────────────────────────
RESULTS_CSV = "/tmp/pid_sweep_results.csv"
_PER_JOINT  = [f"jitter_j{i}" for i in range(7)]
CSV_FIELDS  = (["kp_scale", "kd_scale", "status",
                "rms_error_rad", "max_overshoot_rad",
                "jitter_rad_s2", "combined_score"]
               + _PER_JOINT)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _make_params(q7: np.ndarray, kp_vec: np.ndarray, kd_vec: np.ndarray,
                 gripper_q: float) -> list:
    """构建 8 个 MITParam（7 臂 + 1 夹爪保持）。"""
    params = [MITParam(kp=float(kp_vec[j]), kd=float(kd_vec[j]), q=float(q7[j]))
              for j in range(7)]
    params.append(MITParam(kp=GRIPPER_KP, kd=GRIPPER_KD, q=float(gripper_q)))
    return params


def move_and_record(ctrl, q_start: np.ndarray, q_end: np.ndarray,
                    kp_vec: np.ndarray, kd_vec: np.ndarray,
                    duration: float = DURATION, n_steps: int = N_STEPS):
    """
    线性插值运动，记录每步的实际与目标关节角。

    Returns
    -------
    list of dict  — 每步 {"q_actual": (7,), "q_target": (7,), "dt": float}

    Raises
    ------
    RuntimeError  — 任意关节误差超过 MAX_ERR_RAD
    """
    ctrl.recv_all(CANPORT, timeout=0.05)
    q_end_clipped = _clip_q7(q_end)
    g_q = ctrl.get_positions(CANPORT)[7]   # 夹爪保持当前位置
    dt = duration / n_steps
    records = []

    for i in range(n_steps + 1):
        alpha = i / n_steps
        q_tgt = q_start + (q_end_clipped - q_start) * alpha

        ctrl.mit_control_all(CANPORT, _make_params(q_tgt, kp_vec, kd_vec, g_q))
        ctrl.recv_all(CANPORT, timeout=0.005)

        q_act = read_q7(ctrl, CANPORT)

        # 安全检查
        err = np.abs(q_act - q_tgt)
        if np.any(err > MAX_ERR_RAD):
            raise RuntimeError(
                f"安全中止：步骤 {i}，最大误差 {np.max(err):.4f} rad "
                f"({np.rad2deg(np.max(err)):.1f}°)"
            )

        records.append({"q_actual": q_act.copy(),
                         "q_target": q_tgt.copy(),
                         "dt": dt})
        time.sleep(dt)

    return records


def hold_position(ctrl, kp_vec: np.ndarray, kd_vec: np.ndarray,
                  duration: float = HOLD_SECS):
    """原地保持（使用测试增益），捕获保持阶段的数据。"""
    ctrl.recv_all(CANPORT, timeout=0.05)
    q = read_q7(ctrl, CANPORT)
    g_q = ctrl.get_positions(CANPORT)[7]
    dt = 0.005
    t_end = time.time() + duration
    records = []
    while time.time() < t_end:
        ctrl.mit_control_all(CANPORT, _make_params(q, kp_vec, kd_vec, g_q))
        ctrl.recv_all(CANPORT, timeout=0.005)
        q_act = read_q7(ctrl, CANPORT)
        records.append({"q_actual": q_act.copy(),
                         "q_target": q.copy(),
                         "dt": dt})
        time.sleep(dt)
    return records


def compute_metrics(records: list) -> dict:
    """
    计算三项评估指标：
    - rms_error_rad      : RMS 跟踪误差
    - max_overshoot_rad  : 末端最大超调（最后 20 步）
    - jitter_rad_s2      : 平均绝对加速度（抖动核心指标）
    - combined_score     : 综合得分（越低越好）
    """
    q_act = np.stack([r["q_actual"] for r in records])  # (N, 7)
    q_tgt = np.stack([r["q_target"] for r in records])  # (N, 7)
    dt = records[0]["dt"]

    errors = q_act - q_tgt
    rms_error = float(np.sqrt(np.mean(errors ** 2)))

    settle = q_act[-20:] - q_tgt[-20:]
    max_overshoot = float(np.max(np.abs(settle)))

    vel   = np.diff(q_act, axis=0) / dt   # (N-1, 7)
    accel = np.diff(vel,   axis=0) / dt   # (N-2, 7)
    jitter_per_joint = np.mean(np.abs(accel), axis=0)   # (7,)
    jitter = float(np.mean(jitter_per_joint))

    combined = rms_error + 0.01 * jitter + 10.0 * max_overshoot

    result = {
        "rms_error_rad":     rms_error,
        "max_overshoot_rad": max_overshoot,
        "jitter_rad_s2":     jitter,
        "combined_score":    combined,
    }
    for i, v in enumerate(jitter_per_joint):
        result[f"jitter_j{i}"] = float(v)
    return result


def run_trial(ctrl, kp_scale: float, kd_scale: float) -> dict:
    """
    执行一组完整试验：
      基准归位 → HOME → POSE_A → POSE_B → POSE_C → HOME
    覆盖全工作空间，返回指标 dict（含 kp_scale, kd_scale, status）。
    """
    kp_vec = np.array(KP) * kp_scale
    kd_vec = np.array(KD) * kd_scale

    # 步骤 1：用基准增益安全归位（防止上轮残余振荡）
    print("    归位（基准增益）...", end=" ", flush=True)
    q_now = read_q7(ctrl, CANPORT)
    move_and_record(ctrl, q_now, HOME_RAD,
                    np.array(KP), np.array(KD), duration=4.0)
    time.sleep(0.2)
    print("完成")

    # 步骤 2：按顺序经过所有路径点（测试增益）
    all_records = []
    q_now = read_q7(ctrl, CANPORT)
    for wp, name in zip(WAYPOINTS, WAYPOINT_NAMES):
        print(f"    → {name} ...", end=" ", flush=True)
        recs = move_and_record(ctrl, q_now, wp, kp_vec, kd_vec)
        rec_hold = hold_position(ctrl, kp_vec, kd_vec, HOLD_SECS)
        all_records.extend(recs + rec_hold)
        q_now = read_q7(ctrl, CANPORT)
        print("完成")

    metrics = compute_metrics(all_records)
    metrics["kp_scale"] = kp_scale
    metrics["kd_scale"] = kd_scale
    metrics["status"]   = "ok"
    return metrics


def save_csv(results: list, path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in CSV_FIELDS} for r in results])


def print_summary(results: list):
    ok = [r for r in results if r.get("status") == "ok"]
    print("\n" + "=" * 65)
    print(f"扫描完成  共 {len(results)} 组，成功 {len(ok)} 组")

    if not ok:
        print("没有成功的试验！")
        return

    # 过滤超调超过 3° 的组合
    safe = [r for r in ok if r["max_overshoot_rad"] < np.deg2rad(3.0)]
    pool = safe if safe else ok
    best = min(pool, key=lambda r: r["combined_score"])

    best_kp = [round(kp * best["kp_scale"], 1) for kp in KP]
    best_kd = [round(kd * best["kd_scale"], 3) for kd in KD]

    print(f"\n最优参数  (KP_scale={best['kp_scale']:.2f}, KD_scale={best['kd_scale']:.2f})")
    print(f"  KP = {best_kp}")
    print(f"  KD = {best_kd}")
    print(f"  RMS误差:   {best['rms_error_rad']:.5f} rad  "
          f"({np.rad2deg(best['rms_error_rad']):.3f}°)")
    print(f"  最大超调:  {best['max_overshoot_rad']:.5f} rad  "
          f"({np.rad2deg(best['max_overshoot_rad']):.3f}°)")
    print(f"  抖动分数:  {best['jitter_rad_s2']:.4f} rad/s²")
    print(f"  综合得分:  {best['combined_score']:.5f}")

    # 逐关节抖动分解
    print("\n逐关节抖动 (rad/s²)：")
    joint_names = ["J1(肩旋)", "J2(肩仰)", "J3(肘旋)", "J4(肘仰)", "J5(腕旋)", "J6(腕仰)", "J7(腕转)"]
    for i, jname in enumerate(joint_names):
        v = best.get(f"jitter_j{i}", float("nan"))
        bar = "█" * int(v / 0.5)
        print(f"  {jname}: {v:7.3f}  {bar}")

    print(f"\n结果已保存到 {RESULTS_CSV}")
    print("\n将以下两行更新到 arm_control.py 第 26-27 行：")
    print(f"KP = {best_kp}")
    print(f"KD = {best_kd}")
    print("=" * 65)

    # 打印完整排行（前 10 名）
    top10 = sorted(ok, key=lambda r: r["combined_score"])[:10]
    print("\nTop-10 排行：")
    print(f"{'KP_s':>5} {'KD_s':>5}  {'RMS(rad)':>10} {'超调(rad)':>10} {'抖动(r/s²)':>12} {'综合':>10}")
    for r in top10:
        print(f"{r['kp_scale']:>5.2f} {r['kd_scale']:>5.2f}  "
              f"{r['rms_error_rad']:>10.5f} {r['max_overshoot_rad']:>10.5f} "
              f"{r['jitter_rad_s2']:>12.4f} {r['combined_score']:>10.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("初始化机械臂...")
    ctrl = build_arm(CANPORT)

    # 先用基准增益运动到 home
    print("运动到初始位置（基准增益）...")
    q_now = read_q7(ctrl, CANPORT)
    move_and_record(ctrl, q_now, HOME_RAD,
                    np.array(KP), np.array(KD), duration=5.0, n_steps=1000)
    time.sleep(0.5)

    grid = list(itertools.product(KP_SCALES, KD_SCALES))
    total = len(grid)
    results = []

    print(f"\n开始扫描，共 {total} 组参数\n")

    for idx, (kp_s, kd_s) in enumerate(grid):
        print(f"[{idx+1:02d}/{total}] KP_scale={kp_s:.2f}  KD_scale={kd_s:.2f}")
        try:
            metrics = run_trial(ctrl, kp_s, kd_s)
            print(f"    ✓ 综合={metrics['combined_score']:.4f}  "
                  f"抖动={metrics['jitter_rad_s2']:.4f} r/s²  "
                  f"RMS={np.rad2deg(metrics['rms_error_rad']):.3f}°")
        except RuntimeError as e:
            print(f"    ✗ 中止: {e}")
            metrics = {
                "kp_scale": kp_s, "kd_scale": kd_s, "status": "aborted",
                "rms_error_rad": 999.0, "max_overshoot_rad": 999.0,
                "jitter_rad_s2": 999.0, "combined_score": 999.0,
            }

        results.append(metrics)
        save_csv(results, RESULTS_CSV)   # 增量保存，防止中断丢数据

    print_summary(results)


if __name__ == "__main__":
    main()
