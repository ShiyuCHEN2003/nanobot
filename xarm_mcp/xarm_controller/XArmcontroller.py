"""
XArmController —— 达妙电机控制器模块
适用于 PEAK PCAN-USB Pro FD，SocketCAN (can0/can1) 接口

支持：
  - 单路 CAN：控制指定电机或该路全部电机
  - 双路 CAN：同时控制左/右臂

电机限制参数（必须与实际型号一致，填错会导致力矩编码错误）：
  DM4310:  p_max=12.5rad  v_max=30 rad/s  t_max=10 Nm
  DM4340:  p_max=12.5rad  v_max=8  rad/s  t_max=28 Nm
  DM8009:  p_max=12.5rad  v_max=45 rad/s  t_max=54 Nm
  DM8006:  p_max=12.5rad  v_max=45 rad/s  t_max=40 Nm
  DM6006:  p_max=12.5rad  v_max=45 rad/s  t_max=12 Nm

用法示例：
    from XArmcontroller import XArmController, ARM_7DOF_LIMITS

    ctrl = XArmController({
        "can0": ARM_7DOF_LIMITS,   # 右臂
        "can1": ARM_7DOF_LIMITS,   # 左臂
    })
    try:
        ctrl.enable_all()
        while running:
            taus = compute_gravity(ctrl.get_positions("can0"))
            ctrl.torque_control_all("can0", taus)
            ctrl.torque_control_all("can1", taus)
            ctrl.recv_all()
    finally:
        ctrl.disable_all()
        ctrl.close()
"""

import can
import time
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ===================== 常用电机限制参数 =====================

#  [p_max(rad), v_max(rad/s), t_max(Nm)]
LIMIT_DM4310  = [12.5, 30,  10]
LIMIT_DM4340  = [12.5,  8,  28]
LIMIT_DM8006  = [12.5, 45,  40]
LIMIT_DM8009  = [12.5, 45,  54]
LIMIT_DM6006  = [12.5, 45,  12]

# XArm v10 七自由度手臂电机配置（关节1~7）
# 关节1-2: DM8009，关节3-4: DM4340，关节5-7: DM4310
ARM_7DOF_LIMITS: List[List[float]] = [
    LIMIT_DM8009,   # joint 1  (CAN ID 0x01)
    LIMIT_DM8009,   # joint 2  (CAN ID 0x02)
    LIMIT_DM4340,   # joint 3  (CAN ID 0x03)
    LIMIT_DM4340,   # joint 4  (CAN ID 0x04)
    LIMIT_DM4310,   # joint 5  (CAN ID 0x05)
    LIMIT_DM4310,   # joint 6  (CAN ID 0x06)
    LIMIT_DM4310,   # joint 7  (CAN ID 0x07)
]

# 夹爪电机（CAN ID 0x08），位置：0=夹紧，-1=张开
LIMIT_GRIPPER = LIMIT_DM4310   # 根据实际型号调整

# 含夹爪的 8 自由度配置（7 个臂关节 + 1 个夹爪）
ARM_8DOF_LIMITS: List[List[float]] = ARM_7DOF_LIMITS + [LIMIT_GRIPPER]

# 从机 CAN ID 和主机 ID（主机 ID = 从机 ID + 0x10）
ARM_SLAVE_IDS  = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
ARM_MASTER_IDS = [sid + 0x10 for sid in ARM_SLAVE_IDS]

ARM_8DOF_SLAVE_IDS  = ARM_SLAVE_IDS  + [0x08]
ARM_8DOF_MASTER_IDS = ARM_MASTER_IDS + [0x18]


# ===================== 数据结构 =====================

@dataclass
class MotorState:
    can_id:    int
    master_id: int
    limit:     List[float]
    position:  float = 0.0        # rad
    velocity:  float = 0.0        # rad/s
    torque:    float = 0.0        # Nm
    last_update: float = field(default_factory=time.monotonic)

@dataclass
class MITParam:
    kp:  float = 0.0
    kd:  float = 0.0
    q:   float = 0.0   # 期望位置 (rad)
    dq:  float = 0.0   # 期望速度 (rad/s)
    tau: float = 0.0   # 前馈力矩 (Nm)

class _SingleBus:
    """单路 CAN FD 总线，管理该总线上的所有电机。"""
    def __init__(self, channel: str, motor_limits: List[List[float]],
                 slave_ids: List[int] = ARM_SLAVE_IDS,
                 master_ids: List[int] = ARM_MASTER_IDS):
        assert len(motor_limits) == len(slave_ids), \
            "motor_limits 与 slave_ids 数量必须一致"

        self.channel = channel
        self._motors: List[MotorState] = []
        self._id_map: Dict[int, MotorState] = {}
        self._master_ids: Set[int] = set()

        for limit, sid, mid in zip(motor_limits, slave_ids, master_ids):
            m = MotorState(can_id=sid, master_id=mid, limit=limit)
            self._motors.append(m)
            self._id_map[sid] = m
            self._id_map[mid] = m
            self._master_ids.add(mid)

        self._bus = can.Bus(channel=channel, interface="socketcan", fd=True)

    # ---- 编解码 ----

    @staticmethod
    def _f2u(x: float, xmin: float, xmax: float, bits: int) -> int:
        return int((x - xmin) / (xmax - xmin) * ((1 << bits) - 1))

    @staticmethod
    def _u2f(x: int, xmin: float, xmax: float, bits: int) -> float:
        return x / ((1 << bits) - 1) * (xmax - xmin) + xmin

    def _pack_mit(self, m: MotorState, p: MITParam) -> bytes:
        p_max, v_max, t_max = m.limit
        q_u   = self._f2u(p.q,   -p_max, p_max, 16)
        dq_u  = self._f2u(p.dq,  -v_max, v_max, 12)
        kp_u  = self._f2u(p.kp,  0, 500, 12)
        kd_u  = self._f2u(p.kd,  0, 5,   12)
        tau_u = self._f2u(p.tau, -t_max, t_max, 12)
        d = bytearray(8)
        d[0] = (q_u >> 8) & 0xFF
        d[1] = q_u & 0xFF
        d[2] = dq_u >> 4
        d[3] = ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF)
        d[4] = kp_u & 0xFF
        d[5] = kd_u >> 4
        d[6] = ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF)
        d[7] = tau_u & 0xFF
        return bytes(d)

    def _parse_frame(self, msg: can.Message):
        d = msg.data
        if len(d) < 6:
            return
        m = self._id_map.get(msg.arbitration_id)
        if m is None:
            return
        p_max, v_max, t_max = m.limit
        q_u   = (d[1] << 8) | d[2]
        dq_u  = (d[3] << 4) | (d[4] >> 4)
        tau_u = ((d[4] & 0xF) << 8) | d[5]
        m.position    = self._u2f(q_u,   -p_max, p_max, 16)
        m.velocity    = self._u2f(dq_u,  -v_max, v_max, 12)
        m.torque      = self._u2f(tau_u, -t_max, t_max, 12)
        m.last_update = time.monotonic()

    # ---- 发送 ----

    def _send(self, arb_id: int, data: bytes):
        msg = can.Message(arbitration_id=arb_id, data=data,
                          is_fd=True, is_extended_id=False)
        self._bus.send(msg)

    def _send_cmd(self, slave_id: int, cmd: int):
        self._send(slave_id, bytes([0xFF] * 7 + [cmd]))

    # ---- 接收 ----

    def recv_all(self, timeout: float = 0.01):
        """等待所有电机回包（阻塞，最多 timeout 秒）。"""
        received: Set[int] = set()
        deadline = time.perf_counter() + timeout
        while received != self._master_ids:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            msg = self._bus.recv(timeout=remaining)
            if msg is None:
                break
            if msg.arbitration_id in self._master_ids:
                self._parse_frame(msg)
                received.add(msg.arbitration_id)

    # ---- 控制接口 ----

    def enable(self, motor_idx: int, repeat: int = 3):
        """使能单个电机（按索引 0~N-1）。"""
        sid = self._motors[motor_idx].can_id
        for _ in range(repeat):
            self._send_cmd(sid, 0xFC)
            time.sleep(0.002)

    def disable(self, motor_idx: int, repeat: int = 3):
        """失能单个电机（按索引 0~N-1）。"""
        sid = self._motors[motor_idx].can_id
        for _ in range(repeat):
            self._send_cmd(sid, 0xFD)
            time.sleep(0.002)

    def set_zero(self, motor_idx: int):
        """设置当前位置为零点。"""
        sid = self._motors[motor_idx].can_id
        self._send_cmd(sid, 0xFE)
        time.sleep(0.002)

    def enable_all(self, repeat: int = 3):
        """使能该总线上全部电机。"""
        for m in self._motors:
            for _ in range(repeat):
                self._send_cmd(m.can_id, 0xFC)
                time.sleep(0.002)
        time.sleep(0.2)

    def disable_all(self, repeat: int = 3):
        """失能该总线上全部电机。"""
        for m in self._motors:
            for _ in range(repeat):
                self._send_cmd(m.can_id, 0xFD)
                time.sleep(0.002)

    def mit_control_one(self, motor_idx: int, p: MITParam):
        """对指定电机发送 MIT 控制指令（按索引）。"""
        m = self._motors[motor_idx]
        data = self._pack_mit(m, p)
        self._send(m.can_id, data)

    def mit_control_all(self, params: List[MITParam]):
        """对该总线全部电机发送 MIT 控制指令。"""
        assert len(params) == len(self._motors), \
            f"params 数量({len(params)}) 与电机数量({len(self._motors)})不符"
        for m, p in zip(self._motors, params):
            self._send(m.can_id, self._pack_mit(m, p))

    def torque_control_one(self, motor_idx: int, tau: float, kd: float = 0.0):
        """对指定电机发送纯力矩指令。"""
        self.mit_control_one(motor_idx, MITParam(kp=0, kd=kd, tau=tau))

    def torque_control_all(self, taus: List[float], kd: float = 0.0):
        """对该总线全部电机发送力矩指令（列表长度需与电机数一致）。"""
        params = [MITParam(kp=0, kd=kd, tau=tau) for tau in taus]
        self.mit_control_all(params)

    def position_control_one(self, motor_idx: int,
                             kp: float, kd: float,
                             pos: float, tau_ff: float = 0.0):
        """
        对指定电机发送位置控制指令（MIT 模式，期望速度固定为 0）。
          motor_idx : 电机索引 0~N-1
          kp        : 位置刚度  [0, 500]
          kd        : 速度阻尼  [0, 5]
          pos       : 期望位置  (rad)
          tau_ff    : 前馈力矩  (Nm)，默认 0
        """
        self.mit_control_one(motor_idx,
                             MITParam(kp=kp, kd=kd, q=pos, dq=0.0, tau=tau_ff))

    def position_control_all(self,
                             kp: float, kd: float,
                             positions: List[float],
                             tau_ff: Optional[List[float]] = None):
        """
        对该总线全部 7 个电机发送位置控制指令。
          kp        : 位置刚度（所有电机统一），[0, 500]
          kd        : 速度阻尼（所有电机统一），[0, 5]
          positions : 各电机期望位置列表 (rad)，长度需与电机数一致
          tau_ff    : 各电机前馈力矩列表 (Nm)，默认全为 0
        """
        n = len(self._motors)
        assert len(positions) == n, \
            f"positions 长度({len(positions)}) 与电机数({n})不符"
        tff = tau_ff if tau_ff is not None else [0.0] * n
        params = [
            MITParam(kp=kp, kd=kd, q=pos, dq=0.0, tau=tf)
            for pos, tf in zip(positions, tff)
        ]
        self.mit_control_all(params)

    # ---- 状态查询 ----

    def get_positions(self) -> List[float]:
        return [m.position for m in self._motors]

    def get_velocities(self) -> List[float]:
        return [m.velocity for m in self._motors]

    def get_torques(self) -> List[float]:
        return [m.torque for m in self._motors]

    def get_state(self, motor_idx: int) -> MotorState:
        return self._motors[motor_idx]

    def get_all_states(self) -> List[MotorState]:
        return list(self._motors)

    def num_motors(self) -> int:
        return len(self._motors)

    def close(self):
        self._bus.shutdown()


# ===================== 多路 CAN 控制器 =====================

class XArmController:
    """
    达妙电机多路 CAN 控制器。

    参数
    ----
    bus_configs : dict
        键为 CAN 接口名（如 "can0"/"can1"），
        值为该总线的电机限制列表（每条 [p_max, v_max, t_max]）。

    示例
    ----
    # 双臂：右臂 can0，左臂 can1
    ctrl = XArmController({
        "can0": ARM_7DOF_LIMITS,
        "can1": ARM_7DOF_LIMITS,
    })
    """

    def __init__(
        self,
        bus_configs: Dict[str, List[List[float]]],
        slave_ids:  List[int] = ARM_SLAVE_IDS,
        master_ids: List[int] = ARM_MASTER_IDS,
    ):
        self._buses: Dict[str, _SingleBus] = {}
        for channel, limits in bus_configs.items():
            self._buses[channel] = _SingleBus(
                channel, limits, slave_ids, master_ids)
        print(f"[XArmController] 已初始化 {list(self._buses.keys())}")

    def _bus(self, channel: str) -> _SingleBus:
        b = self._buses.get(channel)
        if b is None:
            raise KeyError(f"CAN 接口 '{channel}' 未注册，可用: {list(self._buses)}")
        return b

    # ---- 使能 / 失能 ----

    def enable_all(self, channel: Optional[str] = None):
        """使能全部电机；channel=None 时使能所有总线。"""
        targets = [self._buses[channel]] if channel else self._buses.values()
        for bus in targets:
            print(f"  使能 {bus.channel} ...")
            bus.enable_all()
        print("[XArmController] 所有电机已使能")

    def disable_all(self, channel: Optional[str] = None):
        """失能全部电机；channel=None 时失能所有总线。"""
        targets = [self._buses[channel]] if channel else self._buses.values()
        for bus in targets:
            print(f"  失能 {bus.channel} ...")
            bus.disable_all()

    def enable_one(self, channel: str, motor_idx: int):
        self._bus(channel).enable(motor_idx)

    def disable_one(self, channel: str, motor_idx: int):
        self._bus(channel).disable(motor_idx)

    def set_zero(self, channel: str, motor_idx: int):
        self._bus(channel).set_zero(motor_idx)

    # ---- 接收 ----

    def recv_all(self, channel: Optional[str] = None, timeout: float = 0.01):
        """接收回包；channel=None 时接收所有总线（顺序阻塞）。"""
        targets = [self._buses[channel]] if channel else self._buses.values()
        for bus in targets:
            bus.recv_all(timeout=timeout)

    # ---- 单电机控制 ----

    def mit_control_one(self, channel: str, motor_idx: int, p: MITParam):
        """对指定总线的指定电机发送 MIT 指令。"""
        self._bus(channel).mit_control_one(motor_idx, p)

    def torque_control_one(self, channel: str, motor_idx: int,
                           tau: float, kd: float = 0.0):
        """对指定总线的指定电机发送力矩指令。"""
        self._bus(channel).torque_control_one(motor_idx, tau, kd)

    # ---- 单总线全部电机控制 ----

    def mit_control_all(self, channel: str, params: List[MITParam]):
        """对指定总线全部电机发送 MIT 指令。"""
        self._bus(channel).mit_control_all(params)

    def torque_control_all(self, channel: str, taus: List[float],
                           kd: float = 0.0):
        """对指定总线全部电机发送力矩指令。"""
        self._bus(channel).torque_control_all(taus, kd)

    def position_control_one(self, channel: str, motor_idx: int,
                             kp: float, kd: float,
                             pos: float, tau_ff: float = 0.0):
        """
        对指定总线的指定电机发送位置控制指令。
          channel   : CAN 接口名（"can0" / "can1"）
          motor_idx : 电机索引 0~6
          kp        : 位置刚度  [0, 500]
          kd        : 速度阻尼  [0, 5]
          pos       : 期望位置  (rad)
          tau_ff    : 前馈力矩  (Nm)，默认 0
        """
        self._bus(channel).position_control_one(motor_idx, kp, kd, pos, tau_ff)

    def position_control_all(self, channel: str,
                             kp: float, kd: float,
                             positions: List[float],
                             tau_ff: Optional[List[float]] = None):
        """
        对指定总线全部 7 个电机发送位置控制指令。
          channel   : CAN 接口名（"can0" / "can1"）
          kp        : 位置刚度（所有电机统一），[0, 500]
          kd        : 速度阻尼（所有电机统一），[0, 5]
          positions : 各电机期望位置 (rad)，列表长度需为 7
          tau_ff    : 各电机前馈力矩 (Nm)，默认全为 0

        示例：所有电机运动到零位
          ctrl.position_control_all("can0", kp=10.0, kd=0.5,
                                    positions=[0.0]*7)
        """
        self._bus(channel).position_control_all(kp, kd, positions, tau_ff)

    # ---- 双臂同时控制 ----

    def mit_control_bimanual(
        self,
        params: Dict[str, List[MITParam]],
    ):
        """
        同时对多路总线发送 MIT 指令。
        params = {"can0": [...], "can1": [...]}
        """
        for channel, ps in params.items():
            self._bus(channel).mit_control_all(ps)

    def torque_control_bimanual(
        self,
        taus: Dict[str, List[float]],
        kd: float = 0.0,
    ):
        """
        同时对多路总线发送力矩指令。
        taus = {"can0": [tau1..tau7], "can1": [tau1..tau7]}
        """
        for channel, t in taus.items():
            self._bus(channel).torque_control_all(t, kd)

    # ---- 状态查询 ----

    def get_positions(self, channel: str) -> List[float]:
        return self._bus(channel).get_positions()

    def get_velocities(self, channel: str) -> List[float]:
        return self._bus(channel).get_velocities()

    def get_torques_measured(self, channel: str) -> List[float]:
        return self._bus(channel).get_torques()

    def get_state(self, channel: str, motor_idx: int) -> MotorState:
        return self._bus(channel).get_state(motor_idx)

    def get_all_states(self, channel: str) -> List[MotorState]:
        return self._bus(channel).get_all_states()

    def num_motors(self, channel: str) -> int:
        return self._bus(channel).num_motors()

    # ---- 夹爪控制（电机索引 7，CAN ID 0x08）----

    def gripper_position_control(self, channel: str, pos: float,
                                  kp: float = 50.0, kd: float = 1.0):
        """
        夹爪位置控制（MIT 模式）。
          pos : 目标位置，0.0=夹紧，-1.0=完全张开（已标定零点）
          kp  : 位置刚度 [0, 500]，默认 50
          kd  : 速度阻尼 [0,   5]，默认 1
        """
        self._bus(channel).position_control_one(7, kp, kd, pos)

    def get_gripper_position(self, channel: str) -> float:
        """读取夹爪电机当前位置（0=夹紧，-1=张开）。"""
        return self._bus(channel).get_state(7).position

    def get_gripper_state(self, channel: str) -> "MotorState":
        """读取夹爪电机完整状态（位置/速度/力矩）。"""
        return self._bus(channel).get_state(7)

    # ---- 关闭 ----

    def close(self):
        for bus in self._buses.values():
            bus.close()
        print("[XArmController] 已关闭所有 CAN 总线")

if __name__ == "__main__":
    ctrl = XArmController({"can0": ARM_7DOF_LIMITS})
    ctrl.enable_all()
    ctrl.recv_all()
    ctrl.torque_control_one("can0", 0, 2.0)
