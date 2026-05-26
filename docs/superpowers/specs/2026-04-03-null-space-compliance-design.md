# 零空间顺应控制设计文档

**日期**：2026-04-03  
**目标**：在 XArm v10 7-DOF 机械臂上实现零空间顺应控制，允许人推动冗余关节（肘部避障），末端位姿保持不动，提升人机交互安全性。

---

## 1. 背景与目标

XArm v10 是 7 自由度机械臂，末端位姿只需 6 个自由度，因此有 1 个冗余自由度——即零空间。  
利用该零空间，可以在不改变末端位姿的前提下改变关节构型。

**需求**：人用手推肘部等关节时，机械臂顺应该推力在零空间内自由移动，末端位姿由 IK 闭环维持不变。传感来源为达妙电机自带的力矩回读（无额外传感器）。

---

## 2. 架构与数据流

```
Motors (CAN)
  │  τ_measured[7], q[7]
  ▼
外力估计
  τ_ext = τ_measured - ArmDynamics.get_gravity(q)
  │
  ▼ (‖τ_ext‖ > torque_threshold 才激活)
零空间投影
  J   = mujoco.mj_jacBody(...)     # 6×7 雅可比（位置3行 + 姿态3行）
  J⁺  = np.linalg.pinv(J)          # 7×6 伪逆
  N   = I₇ - J⁺ · J               # 7×7 零空间投影矩阵
  α_eff = α · min(1.0, 3.0/‖τ_ext‖)   # 自适应增益
  δq  = α_eff · N · τ_ext          # 零空间方向关节位移
  │
  ▼
PostureTask 目标更新
  q_posture = q_current + δq
  solver.posture_task.set_target(q_posture)
  │
  ▼
mink IK solve_step({ee_side: (target_pos, target_rot)})
  末端位姿为硬约束，posture 为软约束
  mink QP 自动处理关节限位
  │
  ▼
MIT control_all + 重力补偿前馈 → Motors
```

**控制频率**：200 Hz，独立线程运行。

---

## 3. 新增文件

**`xarm_mcp/null_space_compliance.py`**  
新增独立模块，不修改任何现有文件。

---

## 4. 类接口

### `NullSpaceCompliance`

```python
class NullSpaceCompliance:
    def __init__(
        self,
        arm: Arm,
        solver: XarmIKSolver,
        ee_side: str = "right",          # "left" | "right"
        alpha: float = 0.05,             # 零空间响应基础增益
        torque_threshold: float = 1.0,   # Nm，低于此值视为噪声，不激活
        dt: float = 1/200,
    )

    def start(self, target_pos: np.ndarray, target_rot: np.ndarray) -> None
        # 设定末端目标（世界坐标系），启动 200Hz 控制循环线程

    def stop(self) -> None
        # 停止控制循环，失能电机

    def update_target(self, target_pos: np.ndarray, target_rot: np.ndarray) -> None
        # 运行中更新末端目标（线程安全）
```

### 关键内部方法

| 方法 | 说明 |
|------|------|
| `_get_jacobian(q7)` | 调用 `mujoco.mj_jacBody`，返回 shape (6,7) 的雅可比矩阵 |
| `_null_projector(J)` | `N = I - pinv(J)·J`，返回 shape (7,7) |
| `_estimate_ext_torque(q7, tau_measured)` | `τ_ext = τ_measured - gravity(q7)` |
| `_control_step()` | 单步主循环：读状态→估外力→投影→更新 PostureTask→IK→发 MIT 指令 |

---

## 5. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | 0.05 | 零空间响应增益，建议从 0.02 开始调大，值越大响应越灵敏但末端扰动越大 |
| `torque_threshold` | 1.0 Nm | 低于此值的 τ_ext 视为重力补偿残差噪声，不激活零空间运动 |
| `dt` | 1/200 s | 控制周期 |

**自适应增益**：当 `‖τ_ext‖ > 3.0 Nm` 时，`α_eff = α * 3.0 / ‖τ_ext‖`，防止单步 δq 过大。

---

## 6. 安全与边界处理

触发以下任一条件时，立即调用 `stop()`：

1. **关节超限**：任一关节角超出 `ARM_7DOF_JOINT_LIMITS`（主动检测，不静默截断）
2. **IK 失败**：`solve_step` 抛出异常
3. **末端漂移过大**：末端实际位置与目标偏差 > 20mm（IK 跟不上时的保护）
4. **CAN 通信超时**：`recv_all` 未收到完整回包

**松手后行为**：`‖τ_ext‖ < threshold` 时，`q_posture` 保持当前关节角（不弹回原始构型），顺应结果自然保持。

**奇异位形**：不额外实现，mink 内置 `lm_damping=0.05` 已处理。

---

## 7. 不在本次范围内

- 双臂同步（`ee_side` 参数预留接口）
- 相机感知障碍物（本次只用力矩感知）
- 雅可比速度层直接控制（方案 B，工程量大，留后续）

---

## 8. 用法示例

```python
from xarm_mcp.arm_control import Arm
from xarm_mcp.ik.xarm_ik_solver import XarmIKSolver
from xarm_mcp.null_space_compliance import NullSpaceCompliance
import numpy as np
from scipy.spatial.transform import Rotation as R

arm = Arm("can2")
solver = XarmIKSolver(xml_path="xarm_mcp/xml/scene.xml")

# 读取当前末端位姿作为目标
target_pos, target_rot = solver.get_ee_pose("right")

ctrl = NullSpaceCompliance(arm, solver, ee_side="right", alpha=0.05)
ctrl.start(target_pos, target_rot)

# 此后人可以推动肘部，末端自动维持在 target_pos/target_rot
# ctrl.stop() 停止
```
