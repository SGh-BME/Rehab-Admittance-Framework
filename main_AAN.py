####### --------------------------------------------------------
#AAN ___ version 3
####### --------------------------------------------------------

import pybullet as p
import pybullet_data
import time
import os
import math
import threading
import queue
from collections import deque
from enum import Enum, auto
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any, Tuple
import sys

from admittance_controller import (
    AdmittanceParams,
    TargetAdmittanceState,
    step_target_admittance_AAN,   # NEW
    AANParams,                    # NEW
    AANState,                     # NEW
    reset_target_admittance_state,
)


from functions import DataLogger, TrialState, update_trial_state
import datetime

# =============================================================================
# === CONFIG
# =============================================================================
CONFIG = {
    "admittance": {
        "Mx": 2.0, "My": 2.0,
        "Bx": 40.0, "By": 40.0,
        "Kx": 5, "Ky": 5,
        "force_deadzone_N": 0.5,
        "force_max_N": 25.0,
        "v_max_mps": 0.35
    },
    "robot": {
        "urdf_path": "D:/Sima-uni/CodesPycharm/HitBot_2026_V1/Joystick control/Joystick control/robot32_description/urdf/robot32_for_pybullet.urdf",
        "home_ext_mm": 420.0,
        "max_ext_mm": 830.0,
        "physical_home_angles_deg": {'j2': -42.0, 'j3': -44.0, 'j4': 0.0},

        "j_limits_deg": {
            'j1': [-9999.0, 9999.0],
            'j2': [-90.0, 90.0],
            'j3': [-164.0, 164.0],
            'j4': [-1080.0, 1080.0]
        },

        "j1_z_height_m": 0.064,
        "j1_z_limits_m": [0.064, 0.064 + 0.410],
        "ee_home_xy_m": [0.4, 0.27],

        "xy_angle_min_deg": 180.0,
        "xy_angle_max_deg": 90.0,
    },
    "simulation": {
        "time_step": 1/480.0,
        "gravity": [0, 0, -9.81]
    },
    "control": {
        "ik_max_iters": 200,
        "ik_residual_thresh": 1e-4,
        "ik_damping": 0.05,
        "position_gains": [0.35, 0.35, 0.35, 0.25],
        "forces": [2500]*4,
        "collision_active": True,

        "j_max_speed_rad_s": {
            'j2': math.radians(150.0),
            'j3': math.radians(150.0),
            'j4': math.radians(1000.0)
        }
    },
    "joystick": {
        "port": "COM6",
        "baud": 115200,
        "intent_axis_snap_deg": 18.0,
    },
    "smooth": {
        "ma_window": 3,
        "deadzone_inner": 0.08,
        "deadzone_outer": 0.12,
        "snap_to_center_thresh": 0.005,
        "curve_pow": 1.6,
        "max_mag_kg": 2.0,
        "angle_filt_alpha": 0.25,
        "mag_slew_max_delta": 10.0,
    },
    "ee": {
        "max_speed_control_deg_s": 150.0,
        "bounds": {"xmin": -0.60, "xmax": 0.60, "ymin": -0.60, "ymax": 0.60},
        "vel_slew_mps2": 20.0,
        "joint_slew_radps2": 50.0,
    },
    "ui": {
        "xy_canvas_px": 360,
        "joy_canvas_px": 220,
        "xy_extent_mm": 800.0,  # radius of grid labels, center to edge (for ticks)
        "refresh_hz": 120.0,
    },
}

# =============================================================================
# === HELPERS
# =============================================================================
class MovingAverage:
    def __init__(self, n: int = 10):
        self.q = deque(maxlen=max(2, n))
        self.sum = 0.0
    def add(self, x: float) -> float:
        if len(self.q) == self.q.maxlen:
            self.sum -= self.q[0]
        self.q.append(x)
        self.sum += x
        return x
    def value(self) -> float:
        return self.sum/len(self.q) if self.q else 0.0

class HysteresisDeadzone:
    def __init__(self, inner: float, outer: float):
        self.inner, self.outer, self.active = float(inner), float(outer), False
    def apply(self, x: float) -> float:
        ax = abs(x)
        if self.active:
            if ax <= self.inner:
                self.active = False
                return 0.0
            return x
        else:
            if ax >= self.outer:
                self.active = True
                return x
            return 0.0

class SlewLimiter:
    def __init__(self, max_delta_per_sec: float, initial: float = 0.0):
        self.max_delta = float(max_delta_per_sec)
        self.prev = float(initial)
        self.last_t = time.perf_counter()
    def reset(self, value: float = 0.0):
        self.prev = float(value); self.last_t = time.perf_counter()
    def step(self, target: float) -> float:
        now = time.perf_counter()
        dt = max(1e-4, now - self.last_t)
        self.last_t = now
        max_step = self.max_delta * dt
        delta = target - self.prev
        delta = max(-max_step, min(max_step, delta))
        self.prev += delta
        return self.prev

class AngleFilter:
    def __init__(self, alpha: float = 0.50, initial: float = 0.0):
        self.alpha = float(alpha); self.value = float(initial)
    @staticmethod
    def wrap180(d):
        w = (d + 180.0) % 360.0 - 180.0
        return w if w != -180.0 else 180.0
    def update(self, new_deg: float) -> float:
        e = self.wrap180(new_deg - self.value)
        self.value = self.wrap180(self.value + self.alpha*e)
        return self.value

def shaped_mag(mag: float, max_mag: float, power: float) -> float:
    m = max(0.0, min(1.0, mag/max(max_mag,1e-6)))
    return (m**max(1.0,power))*max_mag

# ================= FORCE SENSOR HELPERS =================
def hex_to_force_moment(hex_bytes, scale_factor, mid_value=0x80000, half_range=0x40000):
    try:
        decimal_value = int(hex_bytes.decode('ascii'), 16)
        return ((decimal_value - mid_value) / half_range) * scale_factor
    except Exception:
        return 0.0

class ForceSensorReader:
    def __init__(self, port="COM3", baud=921600, scale_N=300.0):
        self.port = port
        self.baud = baud
        self.scale_N = scale_N
        self._stop = threading.Event()
        self.lock = threading.RLock()
        self.fx = 0.0
        self.fy = 0.0

        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(0.2)
        # optional: zero
        self.zero()

        threading.Thread(target=self._loop, daemon=True).start()

    def zero(self):
        try:
            self.ser.write(b'O')  # your sensor zero-reset command
            self.ser.flush()
            time.sleep(0.1)
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        try:
            self.ser.close()
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.ser.write(b'P')        # request packet
                self.ser.flush()
                resp = self.ser.read(36)    # expected length
                if len(resp) >= 11:
                    fx_hex = resp[1:6]
                    fy_hex = resp[6:11]
                    fx = hex_to_force_moment(fx_hex, self.scale_N)
                    fy = hex_to_force_moment(fy_hex, self.scale_N)
                    with self.lock:
                        self.fx, self.fy = fx, fy
            except Exception:
                # keep last values on error
                time.sleep(0.01)

    def get_fx_fy(self):
        with self.lock:
            return self.fx, self.fy
# =============================================================================


# =============================================================================
# === JOYSTICK
# =============================================================================
try:
    import serial
except ImportError:
    serial = None

class JoystickReader:
    def __init__(self, cfg: Dict, sm: Dict, shared: Dict, lock: threading.RLock):
        self.cfg, self.sm, self.shared, self.lock = cfg, sm, shared, lock
        self.state = {"mag_kg": 0.0, "ang_deg_360": 0.0, "raw_angle_360": 0.0}
        self.ma_mag = MovingAverage(1)
        self.hyst_xy = HysteresisDeadzone(sm["deadzone_inner"], sm["deadzone_outer"])
        self.last_angle = 0.0
        self.angle_filt = AngleFilter(sm["angle_filt_alpha"], 0.0)
        self.mag_slew = SlewLimiter(sm["mag_slew_max_delta"], 0.0)
        self._stop = threading.Event()
        self.ser = None
        self.buffer = ""
        if serial is None:
            print("[Joystick] pyserial not found; joystick disabled.")
        else:
            try:
                self.ser = serial.Serial(cfg["port"], cfg["baud"], timeout=0.001)
                print(f"[Joystick] Connected on {cfg['port']}")
                self.tare()
            except Exception as e:
                print(f"[Joystick] Serial open failed: {e}")
        threading.Thread(target=self._loop, daemon=True).start()

    def tare(self):
        if self.ser:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b't'); self.ser.flush()
                time.sleep(0.25)
                print("[Joystick] Tare sent")
            except Exception as e:
                print(f"[Joystick] Tare failed: {e}")

    def stop(self):
        self._stop.set()
        if self.ser:
            try: self.ser.close()
            except: pass

    def _loop(self):
        while not self._stop.is_set():
            if not self.ser:
                time.sleep(0.05); continue
            try:
                raw_data = self.ser.read(self.ser.in_waiting or 1).decode('utf-8', errors='ignore')
                self.buffer += raw_data
                if '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    line = line.strip()
                    if not line.startswith("J,"): continue
                    data_str = line[2:]
                    nums = data_str.split(',')
                    if len(nums) < 2: continue
                    raw_mag = float(nums[0])
                    raw_angle = float(nums[1])
                    r = raw_mag
                    self.ma_mag.add(r)
                    r_h = self.hyst_xy.apply(self.ma_mag.value())
                    r_snap_center = r_h
                    if r_h > 0.0 and r_h < self.sm['snap_to_center_thresh']:
                        r_snap_center = 0.0
                        self.hyst_xy.active = False
                    r_s = self.mag_slew.step(r_snap_center)
                    if r_s > 0.0:
                        ang180 = AngleFilter.wrap180(raw_angle)
                        self.last_angle = self.angle_filt.update(ang180)
                    ang360_disp = (self.last_angle + 360.0) % 360.0
                    with self.lock:
                        self.shared["joystick_state"] = {
                            "mag_kg": r_s,
                            "ang_deg_360": ang360_disp,
                            "raw_angle_360": raw_angle
                        }
            except Exception as ex:
                print(f"[Joystick] Read error: {ex}")
                time.sleep(0.005)

    def get(self) -> Dict:
        with self.lock:
            return self.shared.get('joystick_state', self.state)


# =============================================================================
# === SIMULATION CONTROLLER
# =============================================================================
class SimulationController:
    def __init__(self, cfg: Dict, cmd_q: queue.Queue, shared: Dict, lock: threading.RLock, app):
        self.cfg, self.cmd_q, self.shared, self.lock, self.app = cfg, cmd_q, shared, lock, app
        self._stop = threading.Event()
        self.robot = None
        self.plane = None
        self.collision_state_color = None
        self.joint_indices = []
        self.ll = []; self.ul = []
        self.ee_link = -1
        self.base_link = -1
        self.physical_home = None
        self.ee_home = [0.0,0.0,0.0]
        self.ee_target = [0.0,0.0,0.0]
        self.fixed_z = 0.0
        self.vx_slew = SlewLimiter(self.cfg["ee"]["vel_slew_mps2"], 0.0)
        self.vy_slew = SlewLimiter(self.cfg["ee"]["vel_slew_mps2"], 0.0)
        self.j_slews = [SlewLimiter(self.cfg["ee"]["joint_slew_radps2"], 0.0) for _ in range(4)]
        self._last_ee = None
        self._last_t = None
        self.ee_speed_mps = 0.0

        # ====================================================AAN
        adm = dict(self.cfg.get("admittance", {}))

        # We still read Kx/Ky from config, but they become K_max for AAN scheduling
        Kx_cfg = float(adm.pop("Kx", 5.0))
        Ky_cfg = float(adm.pop("Ky", 5.0))

        # Admittance base params (Mx,My,Bx,By, force shaping, v_max)
        self.adm_params = AdmittanceParams(**adm) if adm else AdmittanceParams()

        # Target-based state (stores x,y,vx,vy)
        self.target_state = TargetAdmittanceState()

        # -------------------------
        # AAN parameters + state
        # -------------------------
        # Choose “professional” defaults: small K_min so it never feels dead, and K_max from your old Kx/Ky
        self.aan_params = AANParams(
            F0_N=max(self.adm_params.force_deadzone_N, 0.5),
            F1_N=2.0,  # tune
            e_ref_m=0.05,  # 5 cm error saturates difficulty
            Kx_min=2.0, Ky_min=2.0,
            Kx_max=Kx_cfg, Ky_max=Ky_cfg,
            # Keep damping constant by default (recommended initially)
            Bx_min=self.adm_params.Bx, By_min=self.adm_params.By,
            Bx_max=self.adm_params.Bx, By_max=self.adm_params.By,
            need_alpha=0.15,
            need_deadband=0.02,
        )
        self.aan_state = AANState()

        # =====================================================

        # =======================
        # Logging setup
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/admittance_run_{ts}.csv"
        self.logger = DataLogger(self.log_path)

        self.trial = TrialState()
        self.t0 = time.perf_counter()
        # =======================

        # smoothing filter for EE speed to make GUI speed stable
        self.speed_ma = MovingAverage(20)

        self.ik_lower_limits = []
        self.ik_upper_limits = []
        self.ik_joint_ranges = []
        self.is_returning_home = False  # State flag for the return home sequence

        # --- Target marker: small sphere at EE z, visual only (no collision) ---
        self.target_marker_id = None

        self._setup_bullet()
        self._load_robot()
        self._identify_by_name()
        self._calculate_ik_limits()
        a = self._initialize_pose()

        for i, slew in enumerate(self.j_slews):
            slew.reset(self.physical_home[i])

        with self.lock:
            self.fixed_z = self.ee_home[2]
            self.ee_target[2] = self.fixed_z
            self.ee_home[2] = self.fixed_z
            self.shared['tuning']['j1_z_height_m'] = self.fixed_z

        self._create_target_marker()

    def _create_target_marker(self):
        """Create small spheres above EE z for target visualization (visual only, no collision)."""
        self.target_marker_radius = 0.015
        self.target_marker_z_offset = 0.06
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.target_marker_radius, rgbaColor=[1.0, 0.2, 0.2, 0.9])
        self.target_marker_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_id,
            basePosition=[0, 0, -10], baseOrientation=[0, 0, 0, 1],
        )
        # Four markers for the Targets section (orange when shown)
        orange = [1.0, 0.5, 0.0, 0.9]
        vis_id_4 = p.createVisualShape(p.GEOM_SPHERE, radius=self.target_marker_radius, rgbaColor=orange)
        self.target_marker_ids = []
        for _ in range(4):
            mid = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_id_4,
                basePosition=[0, 0, -10], baseOrientation=[0, 0, 0, 1],
            )
            self.target_marker_ids.append(mid)
        # Pink sphere at home position (center)
        pink = [1.0, 0.4, 0.6, 0.9]
        vis_id_home = p.createVisualShape(p.GEOM_SPHERE, radius=self.target_marker_radius, rgbaColor=pink)
        hx, hy = self.ee_home[0], self.ee_home[1]
        hz = self.ee_home[2] + self.target_marker_z_offset
        self.home_marker_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_id_home,
            basePosition=[hx, hy, hz], baseOrientation=[0, 0, 0, 1],
        )

    def _update_target_marker(self):
        """Update single click target and the 4 Targets markers from shared state; selected target turns green when EE reaches it."""
        z = self.fixed_z + self.target_marker_z_offset
        with self.lock:
            tgt = self.shared.get("target_xy_m", None)
            targets_xy = self.shared.get("targets_xy", [(0.3, 0.0)] * 4)
            targets_visible = self.shared.get("targets_visible", [False] * 4)
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        ee_xy = (ls[4][0], ls[4][1]) if ls else (0.0, 0.0)
        close_radius = 0.025
        red = [1.0, 0.2, 0.2, 0.9]
        green = [0.2, 0.8, 0.2, 0.9]
        if self.target_marker_id is not None:
            if tgt is not None:
                x, y = float(tgt[0]), float(tgt[1])
                p.resetBasePositionAndOrientation(self.target_marker_id, [x, y, z], [0, 0, 0, 1])
                if math.hypot(ee_xy[0] - x, ee_xy[1] - y) < close_radius:
                    p.changeVisualShape(self.target_marker_id, -1, rgbaColor=green)
                else:
                    p.changeVisualShape(self.target_marker_id, -1, rgbaColor=red)
            else:
                p.resetBasePositionAndOrientation(self.target_marker_id, [0, 0, -10], [0, 0, 0, 1])
        for i in range(4):
            if i >= len(self.target_marker_ids):
                break
            if targets_visible[i] and i < len(targets_xy):
                x, y = float(targets_xy[i][0]), float(targets_xy[i][1])
                p.resetBasePositionAndOrientation(self.target_marker_ids[i], [x, y, z], [0, 0, 0, 1])
            else:
                p.resetBasePositionAndOrientation(self.target_marker_ids[i], [0, 0, -10], [0, 0, 0, 1])
        # Home marker follows current Z height
        if self.home_marker_id is not None:
            hz = self.fixed_z + self.target_marker_z_offset
            p.resetBasePositionAndOrientation(
                self.home_marker_id,
                [self.ee_home[0], self.ee_home[1], hz],
                [0, 0, 0, 1],
            )

    def _setup_bullet(self):
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.cfg['simulation']['gravity'])
        self.plane = p.loadURDF('plane.urdf')

    def _load_robot(self):
        urdf = self.cfg['robot']['urdf_path']
        if not os.path.exists(urdf):
            raise FileNotFoundError(f"URDF not found: {urdf}")
        p.setAdditionalSearchPath(os.path.dirname(urdf))
        self.robot = p.loadURDF(urdf, [0, 0, 0.01], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.base_link = -1

    def _identify_by_name(self):
        name_to_idx = {}
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            name_to_idx[info[1].decode()] = j
        try:
            j1 = name_to_idx["base_to_shoulder_joint"]
            j2 = name_to_idx["shoulder_to_elbow_joint"]
            j3 = name_to_idx["elbow_to_wrist_joint"]
            j4 = name_to_idx["wrist_to_end_joint"]
        except KeyError as e:
            raise RuntimeError(f"Missing joint in URDF: {e}")

        self.joint_indices = [j1,j2,j3,j4]
        self.ee_link = j4
        infos = [p.getJointInfo(self.robot, i) for i in self.joint_indices]
        self.ll = [inf[8] for inf in infos]
        self.ul = [inf[9] for inf in infos]

    def _calculate_ik_limits(self):
        cfg_robot = self.cfg['robot']
        ll, ul = [], []
        j1_info = p.getJointInfo(self.robot, self.joint_indices[0])
        ll.append(j1_info[8])
        ul.append(j1_info[9])
        for i, key in enumerate(['j2', 'j3', 'j4']):
            home_angle = cfg_robot['physical_home_angles_deg'][key]
            log_min, log_max = cfg_robot['j_limits_deg'][key]
            if key == 'j3':
                phys_at_log_min = -log_min + home_angle
                phys_at_log_max = -log_max + home_angle
            else:
                phys_at_log_min = log_min + home_angle
                phys_at_log_max = log_max + home_angle
            ll.append(math.radians(min(phys_at_log_min, phys_at_log_max)))
            ul.append(math.radians(max(phys_at_log_min, phys_at_log_max)))
        self.ik_lower_limits = ll
        self.ik_upper_limits = ul
        self.ik_joint_ranges = [u - l for l, u in zip(ll, ul)]

    def _logical_to_physical(self, logical) -> List[float]:
        cfg = self.cfg['robot']
        home = cfg['physical_home_angles_deg']
        rng = cfg['max_ext_mm'] - cfg['home_ext_mm']
        percent = (logical['j1'] - cfg['home_ext_mm'])/rng if rng else 0.0
        j1 = self.ll[0] + percent*(self.ul[0]-self.ll[0])
        j2 = math.radians(logical['j2'] + home['j2'])
        j3 = math.radians(-logical['j3'] + home['j3'])
        j4 = math.radians(logical['j4'] + home['j4'])
        return [j1,j2,j3,j4]

    def _initialize_pose(self):
        logical_home = {'j1': self.cfg['robot']['home_ext_mm'], 'j2': 0.0, 'j3': 0.0, 'j4': 0.0}
        self.physical_home = self._logical_to_physical(logical_home)
        for idx, j in enumerate(self.joint_indices):
            p.resetJointState(self.robot, j, self.physical_home[idx])
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        ee = list(ls[4]) if ls else [0.0,0.0,self.cfg['robot']['j1_z_height_m']]
        self.ee_home = ee[:]
        home_xy = self.cfg['robot'].get('ee_home_xy_m')
        if home_xy is not None and len(home_xy) >= 2:
            self.ee_home[0] = float(home_xy[0])
            self.ee_home[1] = float(home_xy[1])
        self.ee_target = self.ee_home[:]
        for i in range(-1, p.getNumJoints(self.robot)):
            p.changeVisualShape(self.robot, i, rgbaColor=[1, 1, 1, 1])
        self.collision_state_color = "WHITE"
        with self.lock:
            self.shared['last_safe_positions'] = [s[0] for s in p.getJointStates(self.robot, self.joint_indices)]

    def stop(self):
        self._stop.set()
        #==========================================================
        if hasattr(self, "logger") and self.logger:
            self.logger.close()
            if self.app:
                self.app.log(f"Saved log: {self.log_path}")
        #===========================================================

    def _check_collision(self) -> bool:
        is_colliding = False
        env_contacts = p.getClosestPoints(self.robot, self.plane, distance=0.0)
        if env_contacts:
             is_colliding = True
        if not is_colliding:
            ee_contacts = p.getClosestPoints(self.robot, self.robot, distance=0.0,
                                             linkIndexA=self.base_link, linkIndexB=self.ee_link)
            if ee_contacts:
                if any(c[8] < 0 for c in ee_contacts):
                    is_colliding = True
        return is_colliding

    def _process_cmds(self):
        try:
            cmd,val = self.cmd_q.get_nowait()
            if cmd == 'recalibrate_center':
                samples=[]; t0=time.perf_counter()
                while time.perf_counter()-t0 < 0.3:
                    with self.lock:
                        samples.append(self.shared.get('joystick_state',{}).get('raw_angle_360',0.0))
                    time.sleep(0.01)
                raw = sum(samples)/max(1,len(samples))
                with self.lock:
                    self.shared['tuning']['heading_offset_360'] = raw % 360.0
                    if 'joystick' in self.shared: self.shared['joystick'].tare()
                print(f"[Simulation] Heading offset set to {self.shared['tuning']['heading_offset_360']:.1f}°")
            elif cmd == 'return_home':
                print("[Simulation] Return Home sequence initiated.")
                self.is_returning_home = True
                # -------------------------------------------------------
                ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
                ee = ls[4] if ls else self.ee_home
                reset_target_admittance_state(self.target_state, x=ee[0], y=ee[1], zero_velocity=True)
                self.aan_state.need = 0.0
                # -----------------------------------------------------
            elif cmd == 'reset_slews':
                self.vx_slew.reset(0.0)
                self.vy_slew.reset(0.0)
                current_j_pos = [s[0] for s in p.getJointStates(self.robot, self.joint_indices)]
                for i, slew in enumerate(self.j_slews):
                    slew.reset(current_j_pos[i])
                print("[Simulation] Slews reset")
        except queue.Empty:
            pass

    def run(self):
        print("[Simulation] IK EE control with unified snap-back response.")
        dt = self.cfg['simulation']['time_step']
        while not self._stop.is_set():
            if not p.isConnected(): break
            try:
                # --- State updates and command processing ---
                # with self.lock:
                #     js_mag = self.shared.get('joystick_state', {}).get('mag_kg', 0.0)
                # if self.is_returning_home and js_mag > 0.1:
                #     print("[Simulation] Return Home cancelled by joystick input.")
                #     self.is_returning_home = False
                # --- TEMP: substitute force sensor for joystick ---

                # 1) Process GUI / button commands FIRST
                self._process_cmds()
                Fx, Fy = (0.0, 0.0)
                with self.lock:
                    fs = self.shared.get('force_sensor', None)
                if fs is not None:
                    Fx, Fy = fs.get_fx_fy()

                # Optional: deadzone in N (use your CONFIG values)
                adm_cfg = self.cfg.get("admittance", {})
                dz = float(adm_cfg.get("force_deadzone_N", 0.0))
                Fx = 0.0 if abs(Fx) < dz else Fx
                Fy = 0.0 if abs(Fy) < dz else Fy

                # Optional: saturate in N (so spikes don’t explode velocity)
                fmax = float(adm_cfg.get("force_max_N", 1e9))
                mag = math.hypot(Fx, Fy)
                if mag > fmax and mag > 1e-9:
                    s = fmax / mag
                    Fx *= s
                    Fy *= s

                with self.lock:
                    # Store real forces for admittance
                    self.shared["force_xy_N"] = (Fx, Fy)

                    # Keep joystick_state only for GUI arrow display (optional)
                    # (This does NOT drive admittance anymore; it's just visualization.)
                    ang_deg = (math.degrees(math.atan2(Fx, Fy)) + 360.0) % 360.0 if (abs(Fx) + abs(Fy)) > 0 else 0.0
                    mag_kg_equiv = min(
                        CONFIG['smooth']['max_mag_kg'],
                        (mag / max(fmax, 1e-6)) * CONFIG['smooth']['max_mag_kg']
                    )
                    self.shared["joystick_state"] = {
                        "mag_kg": mag_kg_equiv,
                        "ang_deg_360": ang_deg,
                        "raw_angle_360": ang_deg
                    }

                 # # deadzone in N (tune)
                # if abs(Fx) < 1.0: Fx = 0.0
                # if abs(Fy) < 1.0: Fy = 0.0
                #
                # mag_N = math.hypot(Fx, Fy)
                # angle_deg = (math.degrees(math.atan2(Fx, Fy)) + 360.0) % 360.0
                #
                # F_full_N = 20.0  # tune this
                # mag_kg_equiv = min(
                #     CONFIG['smooth']['max_mag_kg'],
                #     (mag_N / F_full_N) * CONFIG['smooth']['max_mag_kg']
                # )
                # with self.lock:
                #     self.shared["joystick_state"] = {
                #         "mag_kg": mag_kg_equiv,
                #         "ang_deg_360": angle_deg,
                #         "raw_angle_360": angle_deg
                #     }
                    js_mag = mag_kg_equiv  # keep existing logic below working
                    # Cancel Return Home if user applies force
                    if self.is_returning_home and js_mag > 0.1:
                        print("[Simulation] Return Home cancelled by input.")
                        self.is_returning_home = False
                #=========================================================================================

                # self._process_cmds()

                # --- Target Calculation ---
                if self.is_returning_home:
                    self.ee_target = self.ee_home[:]
                    ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
                    ee_current = ls[4] if ls else self.ee_home
                    dist_to_home = math.hypot(ee_current[0] - self.ee_home[0], ee_current[1] - self.ee_home[1])
                    if dist_to_home < 0.01: # 1cm arrival threshold
                        print("[Simulation] Arrived home.")
                        self.is_returning_home = False
                        self.vx_slew.reset(0.0)
                        self.vy_slew.reset(0.0)
                else:
                    with self.lock:
                        self.fixed_z = self.shared['tuning']['j1_z_height_m']
                    self._update_target(dt)

                self._update_target_marker()

                # --- Simulation Step ---
                self._ik_and_drive()
                p.stepSimulation()

                # --- Fault Checking and Response ---
                collision_active = self.shared['tuning']['collision_active']
                collision_detected = self._check_collision() if collision_active else False

                ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
                ee_current = ls[4] if ls else (0.0, 0.0, self.fixed_z)
                dist_to_target = math.hypot(ee_current[0] - self.ee_target[0], ee_current[1] - self.ee_target[1])
                is_unreachable = dist_to_target > 0.05

                # admittance_active = True  # or a flag you add later
                #
                # fault_detected = collision_detected or (
                #         (not admittance_active) and is_unreachable and not self.is_returning_home
                # )

                # fault_detected = collision_detected or (is_unreachable and not self.is_returning_home) ===== sima
                fault_detected = collision_detected

                with self.lock:
                    self.shared['collision_detected'] = collision_detected

                if fault_detected:
                    if collision_detected and self.collision_state_color != "RED":
                        if self.app: self.app.log("Collision! Snapping back.")
                        for i in range(-1, p.getNumJoints(self.robot)):
                            p.changeVisualShape(self.robot, i, rgbaColor=[1, 0, 0, 1])
                        self.collision_state_color = "RED"
                    elif is_unreachable:
                        if self.app: self.app.log("Target unreachable! Snapping back.")

                    with self.lock:
                        last_safe_pos = self.shared.get('last_safe_positions', self.physical_home)

                    for idx, j in enumerate(self.joint_indices):
                        p.resetJointState(self.robot, j, last_safe_pos[idx])

                    ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
                    safe_ee_pos = list(ls[4])
                    self.ee_target = safe_ee_pos[:]
                    # ------------------------------------------------------------------------
                    reset_target_admittance_state(
                        self.target_state,
                        x=safe_ee_pos[0],
                        y=safe_ee_pos[1],
                        zero_velocity=True
                    )
                    self.aan_state.need = 0.0
                    # ---------------------------------------------------------------------------
                    self.vx_slew.reset(0.0)
                    self.vy_slew.reset(0.0)
                    for i, slew in enumerate(self.j_slews):
                        slew.reset(last_safe_pos[i])
                else:
                    if self.collision_state_color != "WHITE":
                        for i in range(-1, p.getNumJoints(self.robot)):
                            p.changeVisualShape(self.robot, i, rgbaColor=[1, 1, 1, 1])
                        self.collision_state_color = "WHITE"
                    with self.lock:
                        self.shared['last_safe_positions'] = [s[0] for s in p.getJointStates(self.robot, self.joint_indices)]

                self._update_shared()
                #=============================================================================================================================
                # -------------------------
                # Event marking (force on/off)
                # -------------------------
                # Event marking (force on/off)
                force_on_thresh = max(1.5 * float(self.cfg["admittance"].get("force_deadzone_N", 0.5)), 0.5)
                self.trial = update_trial_state(self.trial, Fx, Fy, force_on_thresh, False)

                # -------------------------
                # Gather states for logging
                # -------------------------
                t_now = time.perf_counter() - self.t0

                # EE current
                ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
                ee_cur = ls[4] if ls else (0.0, 0.0, self.fixed_z)

                # joints
                qs = [p.getJointState(self.robot, j)[0] for j in self.joint_indices]


                # velocities for logging (ensure attributes exist)
                vx_adm = getattr(self, "_vx_adm", 0.0)
                vy_adm = getattr(self, "_vy_adm", 0.0)
                vx_cmd = getattr(self, "_vx_cmd", 0.0)
                vy_cmd = getattr(self, "_vy_cmd", 0.0)


                row = {
                    "t": t_now,
                    "Fx_N": Fx,
                    "Fy_N": Fy,
                    "Fmag_N": math.hypot(Fx, Fy),

                    "vx_adm_mps": vx_adm,
                    "vy_adm_mps": vy_adm,
                    "v_adm_mag_mps": math.hypot(vx_adm, vy_adm),

                    "vx_cmd_mps": vx_cmd,
                    "vy_cmd_mps": vy_cmd,
                    "v_cmd_mag_mps": math.hypot(vx_cmd, vy_cmd),

                    "x_m": ee_cur[0], "y_m": ee_cur[1], "z_m": ee_cur[2],
                    "x_tgt_m": self.ee_target[0], "y_tgt_m": self.ee_target[1], "z_tgt_m": self.ee_target[2],

                    "q1": qs[0], "q2": qs[1], "q3": qs[2], "q4": qs[3],

                    "force_on": int(self.trial.force_on),
                    "contact_on": int(self.trial.contact_on),
                    "force_event": self.trial.force_event,
                    "contact_event": self.trial.contact_event,

                    "need": getattr(self, "_need_f", 0.0),
                    "Kx_eff": getattr(self, "_Kx_eff", 0.0),
                    "Ky_eff": getattr(self, "_Ky_eff", 0.0),

                }
                self.logger.write(row)

                # Flush occasionally (not every step, for speed)
                if int(t_now * 10) % 10 == 0:  # about once per second
                    self.logger.flush()
                #==============================================================================================================================

                time.sleep(dt)
            except Exception as e:
                if "Not connected" in str(e): break
                print(f"[Simulation] Step error: {e}")
                time.sleep(0.005)

    def _ik_and_drive(self):
        rest = [p.getJointState(self.robot, j)[0] for j in self.joint_indices]
        q = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=self.ee_target,
            targetOrientation=None,
            lowerLimits=self.ik_lower_limits,
            upperLimits=self.ik_upper_limits,
            jointRanges=self.ik_joint_ranges,
            restPoses=rest,
            residualThreshold=self.cfg['control']['ik_residual_thresh'],
            maxNumIterations=self.cfg['control']['ik_max_iters'],
            jointDamping=[self.cfg['control']['ik_damping']] * len(self.joint_indices)
        )
        q = list(q[:4])
        q_smooth = [self.j_slews[i].step(val) for i, val in enumerate(q)]
        p.setJointMotorControlArray(
            self.robot, self.joint_indices, p.POSITION_CONTROL,
            targetPositions=q_smooth,
            positionGains=self.cfg['control']['position_gains'],
            forces=self.cfg['control']['forces']
        )

    def _update_target(self, dt: float):
        # --- current EE pose ---
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        ee = ls[4] if ls else (self.ee_home[0], self.ee_home[1], self.fixed_z)
        x, y = float(ee[0]), float(ee[1])

        # --- pull current force + target from shared ---
        with self.lock:
            Fx, Fy = self.shared.get("force_xy_N", (0.0, 0.0))
            tgt = self.shared.get("target_xy_m", None)

        # If no target selected, hold current pose and keep state consistent
        if tgt is None:
            # optional: keep internal admittance state aligned to current EE
            reset_target_admittance_state(self.target_state, x=x, y=y, zero_velocity=True)

            self._vx_adm = 0.0
            self._vy_adm = 0.0
            self._vx_cmd = 0.0
            self._vy_cmd = 0.0

            self.ee_target[0] = x
            self.ee_target[1] = y
            self.ee_target[2] = self.fixed_z
            return

        x_d, y_d = float(tgt[0]), float(tgt[1])

        # --- Run AAN target admittance (variable stiffness) ---
        x_next, y_next, vx_next, vy_next, Kx_eff, Ky_eff, Bx_eff, By_eff, need_f = step_target_admittance_AAN(
            x=x, y=y,
            x_d=x_d, y_d=y_d,
            Fx=float(Fx), Fy=float(Fy),
            dt=float(dt),
            params=self.adm_params,
            adm_state=self.target_state,
            aan_params=self.aan_params,
            aan_state=self.aan_state,
        )

        # (Optional) expose for GUI/logging/debug
        self._Kx_eff = float(Kx_eff)
        self._Ky_eff = float(Ky_eff)
        self._need_f = float(need_f)

        # Log velocities for CSV
        self._vx_adm = float(vx_next)
        self._vy_adm = float(vy_next)

        # If you still want slew on velocity, apply it here (optional)
        vx_cmd = self.vx_slew.step(vx_next)
        vy_cmd = self.vy_slew.step(vy_next)
        self._vx_cmd = float(vx_cmd)
        self._vy_cmd = float(vy_cmd)

        # IMPORTANT:
        # Your target-based function already integrated x,y internally.
        # So command the integrated compliant position (x_next, y_next).
        self.ee_target[0] = float(x_next)
        self.ee_target[1] = float(y_next)
        self.ee_target[2] = float(self.fixed_z)

        # --- Shared safety checks (KEEP THESE) ---
        r_target = math.hypot(self.ee_target[0], self.ee_target[1])
        R_max = CONFIG['robot']['max_ext_mm'] / 1000.0
        if r_target > R_max:
            scale = R_max / r_target
            self.ee_target[0] *= scale
            self.ee_target[1] *= scale

        b = self.cfg['ee']['bounds']
        self.ee_target[0] = max(b['xmin'], min(b['xmax'], self.ee_target[0]))
        self.ee_target[1] = max(b['ymin'], min(b['ymax'], self.ee_target[1]))

    def _update_shared(self):
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        ee = ls[4] if ls else (0.0,0.0,0.0)
        t_now = time.perf_counter()
        if self._last_ee is not None and self._last_t is not None:
            dt = max(1e-4, t_now - self._last_t)
            # use XY distance only and smooth it for a stable speed readout
            dx = ee[0] - self._last_ee[0]
            dy = ee[1] - self._last_ee[1]
            dist_xy = math.hypot(dx, dy)
            raw_speed = dist_xy / dt  # m/s in XY plane
            self.speed_ma.add(raw_speed)
            smoothed = self.speed_ma.value()
            # small threshold to kill jitter when basically stopped
            if smoothed < 1e-3:   # < 1 mm/s
                smoothed = 0.0
            self.ee_speed_mps = smoothed
        self._last_ee, self._last_t = ee, t_now

        with self.lock:
            js = self.shared.get('joystick_state', {})
            mag_kg = js.get('mag_kg', 0.0)
            norm_shaped_kg = shaped_mag(
                mag_kg,
                CONFIG['smooth']['max_mag_kg'],
                self.shared['tuning'].get('curve_pow', CONFIG['smooth']['curve_pow'])
            )
            self.shared['ee_pos_mm'] = [c*1000.0 for c in ee]
            self.shared['ee_speed_mm_s'] = self.ee_speed_mps*1000.0
            self.shared['ee_target_mm'] = [c*1000.0 for c in self.ee_target]
            self.shared['joystick_norm_mag'] = norm_shaped_kg
            joint_states = p.getJointStates(self.robot, self.joint_indices)
            logical_angles_deg = {
                'j1_mm': (joint_states[0][0] - self.ll[0]) / (self.ul[0] - self.ll[0]) * (self.cfg['robot']['max_ext_mm'] - self.cfg['robot']['home_ext_mm']) + self.cfg['robot']['home_ext_mm'],
                'j2_deg': math.degrees(joint_states[1][0]) - self.cfg['robot']['physical_home_angles_deg']['j2'],
                'j3_deg': -(math.degrees(joint_states[2][0]) - self.cfg['robot']['physical_home_angles_deg']['j3']),
                'j4_deg': math.degrees(joint_states[3][0]) - self.cfg['robot']['physical_home_angles_deg']['j4']
            }
            self.shared['joint_angles_deg'] = logical_angles_deg

# =============================================================================
# === GUI
# =============================================================================
class Gui(tk.Tk):
    def __init__(self, cmd_q: queue.Queue, shared: Dict, lock: threading.RLock):
        super().__init__()
        self.cmd_q, self.shared, self.lock = cmd_q, shared, lock
        self.title("Robot EE Control (IK · Responsive)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_ms = int(1000.0/CONFIG['ui']['refresh_hz'])
        self.xy_px    = CONFIG['ui']['xy_canvas_px']
        self.joy_px   = CONFIG['ui']['joy_canvas_px']
        self.xy_half = self.xy_px//2
        self.joy_half= self.joy_px//2
        self.mm_extent = CONFIG['ui']['xy_extent_mm']
        self.mm_to_px = self.xy_half / self.mm_extent  # px per mm
        self.collision_var = tk.BooleanVar(value=CONFIG['control']['collision_active'])
        self.collision_var.trace_add('write', self._update_collision_state)
        self.ee_trail = deque(maxlen=50)
        self.force_history = deque(maxlen=200)  # Store (Fx, Fy) tuples
        self.cursor_xy_m: Tuple[float, float] = (0.3, 0.0)
        self.cursor_marker = None
        self._build()
        self._draw_workspace()
        self._tick()

    def _zero_force_sensor(self): #Sima for force =============
        with self.lock:
            fs = self.shared.get('force_sensor', None)
        if fs is not None:
            fs.zero()
            self.log("Force sensor zeroed (tare).")
        else:
            self.log("Force sensor not connected.")

    def _update_collision_state(self, *args):
        with self.lock:
            self.shared['tuning']['collision_active'] = self.collision_var.get()
        print(f"[GUI] Collision Detection Toggled: {'ON' if self.collision_var.get() else 'OFF'}")

    def _draw_workspace(self):
        R_max_mm = CONFIG['robot']['max_ext_mm']
        R_outer_px = int(R_max_mm * self.mm_to_px)
        x_min_outer, y_min_outer = self.xy_half - R_outer_px, self.xy_half - R_outer_px
        x_max_outer, y_max_outer = self.xy_half + R_outer_px, self.xy_half + R_outer_px
        self.xy_canvas.create_oval(x_min_outer, y_min_outer, x_max_outer, y_max_outer, outline='#b0b0ff', width=2)
        R_home_px = int(CONFIG['robot']['home_ext_mm'] * self.mm_to_px)
        x_min_home, y_min_home = self.xy_half - R_home_px, self.xy_half - R_home_px
        x_max_home, y_max_home = self.xy_half + R_home_px, self.xy_half + R_home_px
        self.xy_canvas.create_oval(x_min_home, y_min_home, x_max_home, y_max_home, outline='#b0b0ff', width=1)
        self.xy_canvas.create_line(self.xy_half, 0, self.xy_half, self.xy_px, fill="#ddd")
        self.xy_canvas.create_line(0, self.xy_half, self.xy_px, self.xy_half, fill="#ddd")
        self.xy_canvas.create_oval(self.xy_half-2, self.xy_half-2, self.xy_half+2, self.xy_half+2, fill='gray', outline='')
        self.xy_canvas.create_text(self.xy_px - 10, self.xy_half, text="+X", anchor='e', font=('Arial',8))
        self.xy_canvas.create_text(self.xy_half, 10, text="+Y", anchor='n', font=('Arial',8))
        for dist_mm in range(200, int(self.mm_extent)+1, 200):
            px = int(dist_mm * self.mm_to_px)
            self.xy_canvas.create_text(self.xy_half + px, self.xy_half + 5, text=f"{dist_mm}", font=('Arial',6), anchor='n')
            self.xy_canvas.create_text(self.xy_half + 5, self.xy_half - px, text=f"{dist_mm}", font=('Arial',6), anchor='w')

        # Initial cursor marker
        self._draw_cursor_marker()

    def _build(self):
        main = ttk.Frame(self, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsw", padx=(0,10))
        left.columnconfigure(0, weight=0)
        left.columnconfigure(1, weight=1)

        controls_col = ttk.Frame(left)
        controls_col.grid(row=0, column=0, sticky="nsw", padx=(0,10))
        controls = ttk.LabelFrame(controls_col, text="Controls", padding=8)
        controls.pack(fill='x')

        ttk.Button(controls, text="Zero Force Sensor", command=self._zero_force_sensor).pack(fill='x', pady=4)# Sima for force ########

        ttk.Button(controls, text="Recalibrate Center", command=lambda: self.cmd_q.put(('recalibrate_center', None))).pack(fill='x', pady=4)
        ttk.Button(controls, text="Return Home", command=lambda: self.cmd_q.put(('return_home', None))).pack(fill='x', pady=4)
        ttk.Button(controls, text="Reset Slews", command=lambda: self.cmd_q.put(('reset_slews', None))).pack(fill='x', pady=4)
        ttk.Checkbutton(controls, text="Enable Collision Stop", variable=self.collision_var).pack(fill='x', pady=4)

        tuning = ttk.LabelFrame(controls_col, text="Tuning", padding=8)
        tuning.pack(fill='x', pady=(8,0))
        self.vars = {}
        self.value_labels = {}
        max_speed_deg_s = CONFIG['ee']['max_speed_control_deg_s']
        self._slider(tuning, 'ee_speed_mm_s', f"Joint Max Speed ({chr(176)}/s)", 10.0, max_speed_deg_s, max_speed_deg_s, 5.0)
        self._slider(tuning, 'curve_pow', "Curve Power", 1.0, 3.0, CONFIG['smooth']['curve_pow'], 0.05)
        j1_min_m, j1_max_m = CONFIG['robot']['j1_z_limits_m']
        j1_default_m = CONFIG['robot']['j1_z_height_m']
        self._slider(tuning, 'j1_z_height_m', "Z Height (m)", j1_min_m, j1_max_m, j1_default_m, 0.01)

        targets_frame = ttk.LabelFrame(controls_col, text="Targets", padding=8)
        targets_frame.pack(fill='x', pady=(8,0))
        self.target_x_vars = []
        self.target_y_vars = []
        self.target_visible_vars = []
        b = CONFIG['ee']['bounds']
        home_xy = CONFIG['robot'].get('ee_home_xy_m', [0.4, 0.27])
        hx, hy = float(home_xy[0]), float(home_xy[1])
        d = 0.25
        c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
        offsets = [(0, d), (0, -d), (-d, 0), (d, 0)]
        defaults_xy = [
            (hx + (dx * c - dy * s), hy + (dx * s + dy * c))
            for dx, dy in offsets
        ]
        for i in range(4):
            row = ttk.Frame(targets_frame)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=f"T{i+1}", width=3, anchor='w').pack(side='left', padx=(0,4))
            vx = tk.StringVar(value=f"{defaults_xy[i][0]:.3f}")
            vy = tk.StringVar(value=f"{defaults_xy[i][1]:.3f}")
            self.target_x_vars.append(vx)
            self.target_y_vars.append(vy)
            ttk.Label(row, text="X:").pack(side='left', padx=(0,2))
            ttk.Entry(row, textvariable=vx, width=7).pack(side='left', padx=(0,6))
            ttk.Label(row, text="Y:").pack(side='left', padx=(0,2))
            ttk.Entry(row, textvariable=vy, width=7).pack(side='left', padx=(0,6))
            vis = tk.BooleanVar(value=(i < 2))
            self.target_visible_vars.append(vis)
            ttk.Checkbutton(row, text="Show", variable=vis).pack(side='left')

        status = ttk.LabelFrame(controls_col, text="Status", padding=8)
        status.pack(fill='x', pady=(8,0))
        self.pos_var = tk.StringVar(value="X=0.0 Y=0.0 Z=0.0")
        self.spd_var = tk.StringVar(value="Speed=0.0 mm/s")
        self.joy_var = tk.StringVar(value="Mag=0.00 kg  Ang=0.0°")
        self.collision_status_var = tk.StringVar(value=f"Collision: {'ACTIVE' if CONFIG['control']['collision_active'] else 'INACTIVE'}")
        self.joints_var = tk.StringVar(value="J1=0.0mm J2=0.0° J3=0.0° J4=0.0°")
        for var in (self.pos_var, self.spd_var, self.joy_var, self.collision_status_var, self.joints_var):
            ttk.Label(status, textvariable=var, font=('Courier',10)).pack(anchor='w', pady=2)

        log_frame = ttk.LabelFrame(controls_col, text="Log", padding=8)
        log_frame.pack(fill='x', pady=(8,0))
        self.log_text = tk.Text(log_frame, height=5, width=30, state='disabled')
        self.log_text.pack(fill='x')

        plots_frame = ttk.LabelFrame(left, text="Plots", padding=8)
        plots_frame.grid(row=0, column=1, sticky="nw", padx=(0,0))
        self.xy_canvas = tk.Canvas(plots_frame, width=self.xy_px, height=self.xy_px, bg="white", highlightthickness=1, highlightbackground="#999")
        self.xy_canvas.grid(row=0, column=0, pady=(0,8))
        self.xy_canvas.bind("<Button-1>", self._on_xy_click)
        self.joy_canvas = tk.Canvas(plots_frame, width=self.joy_px, height=self.joy_px, bg="white", highlightthickness=1, highlightbackground="#999")
        self.joy_canvas.grid(row=1, column=0, pady=(0, 8))

        self.force_canvas = tk.Canvas(plots_frame, width=self.joy_px, height=self.joy_px, bg="white", highlightthickness=1, highlightbackground="#999")
        self.force_canvas.grid(row=2, column=0)

        self.ee_dot = self.xy_canvas.create_oval(0,0,0,0, fill="#22a", outline="")
        self.ee_target_dot = self.xy_canvas.create_oval(0,0,0,0, fill="#f00", outline="", tags="target")
        r = self.joy_half-10
        self.joy_canvas.create_oval(self.joy_half-r, self.joy_half-r, self.joy_half+r, self.joy_half+r, outline="#ddd")
        self.joy_canvas.create_line(self.joy_half, 8, self.joy_half, self.joy_px-8, fill="#eee")
        self.joy_canvas.create_line(8, self.joy_half, self.joy_px-8, self.joy_half, fill="#eee")
        self.joy_arrow = self.joy_canvas.create_line(0,0,0,0, width=3, fill="#c33", arrow=tk.LAST)
        
        # Setup force plot: draw axes and labels
        force_half = self.joy_px // 2
        self.force_canvas.create_line(20, force_half, self.joy_px-20, force_half, fill="#ddd", width=1)  # X axis
        self.force_canvas.create_line(20, 20, 20, self.joy_px-20, fill="#ddd", width=1)  # Y axis
        self.force_canvas.create_text(self.joy_px-10, force_half+5, text="t", anchor='e', font=('Arial', 8))
        self.force_canvas.create_text(15, 15, text="F (N)", anchor='nw', font=('Arial', 8))

    def _draw_cursor_marker(self):
        """Draw or update a crosshair showing the cursor world location on XY canvas."""
        x_m, y_m = self.cursor_xy_m
        x_px = int(self.xy_half + (x_m*1000.0)*self.mm_to_px)
        y_px = int(self.xy_half - (y_m*1000.0)*self.mm_to_px)
        if self.cursor_marker is not None:
            self.xy_canvas.delete(self.cursor_marker)
        # small crosshair
        size = 6
        self.cursor_marker = self.xy_canvas.create_line(x_px-size, y_px, x_px+size, y_px, fill="#0a0", width=2)
        self.xy_canvas.create_line(x_px, y_px-size, x_px, y_px+size, fill="#0a0", width=2, tags=("cursor_aux",))
        # remove previous aux lines
        self.xy_canvas.delete("cursor_aux")

    def _on_xy_click(self, event):
        """Map canvas click to world meters, store as cursor, and show marker."""
        dx_px = event.x - self.xy_half
        dy_px = self.xy_half - event.y
        x_mm = dx_px / self.mm_to_px
        y_mm = dy_px / self.mm_to_px
        # Convert to meters
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        # Clamp to EE bounds rect just to keep it reasonable
        b = CONFIG['ee']['bounds']
        x_m = max(b['xmin'], min(b['xmax'], x_m))
        y_m = max(b['ymin'], min(b['ymax'], y_m))
        self.cursor_xy_m = (x_m, y_m)
        self._draw_cursor_marker()
        # ============================================================== sima target
        with self.lock:
            self.shared['target_xy_m'] = (x_m, y_m)
        # ==============================================================
        self.log(f"Cursor set to ({x_m:.3f}, {y_m:.3f}) m")

    def _slider(self, parent, key, label, vmin, vmax, v0, step):
        var = tk.DoubleVar(value=v0)
        self.vars[key] = var
        ttk.Label(parent, text=label).pack(anchor='w', pady=(4,0))
        tk.Scale(parent, variable=var, from_=vmin, to=vmax, orient='horizontal', resolution=step, length=260).pack(fill='x')
        value_label = ttk.Label(parent, text=f"{v0:.2f}")
        value_label.pack(anchor='w')
        self.value_labels[key] = value_label
        var.trace_add('write', lambda *a, v=var: value_label.config(text=f"{v.get():.2f}"))

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def _tick(self):
        with self.lock:
            for key, var in self.vars.items():
                self.shared['tuning'][key] = var.get()
            pos = self.shared.get('ee_pos_mm',[0.0,0.0,0.0])
            spd = self.shared.get('ee_speed_mm_s', 0.0)
            js  = self.shared.get('joystick_state', {})

            collision_active = self.shared.get('tuning', {}).get('collision_active', False)
            collision_detected = self.shared.get('collision_detected', False)
            joints = self.shared.get('joint_angles_deg', {'j1_mm':0, 'j2_deg':0, 'j3_deg':0, 'j4_deg':0})

        targets_xy = []
        targets_visible = []
        b = CONFIG['ee']['bounds']
        for i in range(4):
            try:
                x_m = max(b['xmin'], min(b['xmax'], float(self.target_x_vars[i].get())))
                y_m = max(b['ymin'], min(b['ymax'], float(self.target_y_vars[i].get())))
            except (ValueError, TypeError):
                x_m, y_m = 0.3, 0.0
            targets_xy.append((x_m, y_m))
            targets_visible.append(self.target_visible_vars[i].get())
        with self.lock:
            self.shared['targets_xy'] = targets_xy
            self.shared['targets_visible'] = targets_visible
        self.pos_var.set(f"X={pos[0]:.1f}  Y={pos[1]:.1f}  Z={pos[2]:.1f}")
        self.spd_var.set(f"Speed={spd:.1f} mm/s")
        self.joy_var.set(f"Mag={js.get('mag_kg',0.0):.2f} kg  Ang={js.get('ang_deg_360',0.0):.1f}°")
        self.collision_status_var.set(f"Collision: {'DETECTED!' if collision_detected else 'CLEAR'} ({'ACTIVE' if collision_active else 'INACTIVE'})")
        self.joints_var.set(f"J1={joints['j1_mm']:.1f}mm J2={joints['j2_deg']:.1f}° J3={joints['j3_deg']:.1f}° J4={joints['j4_deg']:.1f}°")
        x_px = int(self.xy_half + (pos[0]*self.mm_to_px))
        y_px = int(self.xy_half - (pos[1]*self.mm_to_px))
        self.xy_canvas.coords(self.ee_dot, x_px-5, y_px-5, x_px+5, y_px+5)
        self.ee_trail.append((x_px, y_px))
        self.xy_canvas.delete("trail")
        for i, (px, py) in enumerate(self.ee_trail):
            alpha = (i / len(self.ee_trail)) ** 2
            color = f"#{int(0x22 + alpha*0xdd):02x}{int(0xaa + alpha*0x55):02x}{int(0xaa):02x}"
            self.xy_canvas.create_oval(px-2, py-2, px+2, py+2, fill="", outline=color, width=1, tags="trail")

        with self.lock:
            tgt = self.shared.get('target_xy_m', None)

        if tgt is not None:
            tx_px = int(self.xy_half + (tgt[0] * 1000.0) * self.mm_to_px)
            ty_px = int(self.xy_half - (tgt[1] * 1000.0) * self.mm_to_px)
            self.xy_canvas.coords(self.ee_target_dot, tx_px - 3, ty_px - 3, tx_px + 3, ty_px + 3)
        else:
            self.xy_canvas.coords(self.ee_target_dot, 0, 0, 0, 0)

        self.xy_canvas.delete("target_dots")
        for i in range(4):
            if not targets_visible[i]:
                continue
            x_m, y_m = targets_xy[i]
            tx_px = int(self.xy_half + (x_m * 1000.0) * self.mm_to_px)
            ty_px = int(self.xy_half - (y_m * 1000.0) * self.mm_to_px)
            self.xy_canvas.create_oval(tx_px - 4, ty_px - 4, tx_px + 4, ty_px + 4, fill="#f80", outline="#c60", width=1, tags="target_dots")

        ang = js.get('ang_deg_360',0.0)
        mag = js.get('mag_kg',0.0)
        mag01 = max(0.0, min(1.0, mag/CONFIG['smooth']['max_mag_kg']))
        R = (self.joy_half-16) * mag01
        rad = math.radians(ang)
        x0,y0 = self.joy_half, self.joy_half
        x1 = x0 + R*math.sin(rad)
        y1 = y0 - R*math.cos(rad)
        color = f"#{int(0xff * (1 - mag01)):02x}{int(0xff * mag01):02x}00"
        self.joy_canvas.coords(self.joy_arrow, x0,y0, x1,y1)
        self.joy_canvas.itemconfig(self.joy_arrow, fill=color)
        
        # Update force plot
        with self.lock:
            Fx, Fy = self.shared.get("force_xy_N", (0.0, 0.0))
        self.force_history.append((Fx, Fy))
        
        # Clear previous force plot lines
        self.force_canvas.delete("force_line")
        
        if len(self.force_history) > 1:
            force_half = self.joy_px // 2
            plot_width = self.joy_px - 40  # Leave margins
            plot_height = self.joy_px - 40
            force_max = 30.0  # Max force in N for scaling
            
            # Draw Fx line (red)
            points_fx = []
            points_fy = []
            for i, (fx, fy) in enumerate(self.force_history):
                x_px = 20 + int((i / max(1, len(self.force_history) - 1)) * plot_width)
                y_fx = force_half - int((fx / force_max) * (plot_height / 2))
                y_fy = force_half - int((fy / force_max) * (plot_height / 2))
                y_fx = max(20, min(self.joy_px - 20, y_fx))
                y_fy = max(20, min(self.joy_px - 20, y_fy))
                points_fx.append((x_px, y_fx))
                points_fy.append((x_px, y_fy))
            
            # Draw Fx line
            if len(points_fx) > 1:
                for i in range(len(points_fx) - 1):
                    self.force_canvas.create_line(
                        points_fx[i][0], points_fx[i][1],
                        points_fx[i+1][0], points_fx[i+1][1],
                        fill="#c33", width=2, tags="force_line"
                    )
            
            # Draw Fy line (blue)
            if len(points_fy) > 1:
                for i in range(len(points_fy) - 1):
                    self.force_canvas.create_line(
                        points_fy[i][0], points_fy[i][1],
                        points_fy[i+1][0], points_fy[i+1][1],
                        fill="#33c", width=2, tags="force_line"
                    )
            
            # Draw zero line
            self.force_canvas.create_line(20, force_half, self.joy_px-20, force_half, fill="#aaa", width=1, tags="force_line")
        
        self.after(self.refresh_ms, self._tick)

    # def _on_close(self):
    #     print("Closing application...")
    #     with self.lock:
    #         if 'joystick' in self.shared: self.shared['joystick'].stop()
    #         if 'sim' in self.shared: self.shared['sim'].stop()
    #     if p.isConnected(): p.disconnect()
    #     self.destroy()
    def _on_close(self):
        print("Closing application...")
        with self.lock:
            if 'joystick' in self.shared:
                self.shared['joystick'].stop()
            if 'force_sensor' in self.shared:
                self.shared['force_sensor'].stop()
            if 'sim' in self.shared:
                self.shared['sim'].stop()
        if p.isConnected():
            p.disconnect()
        self.destroy()


# =============================================================================
# === MAIN
# =============================================================================
if __name__ == "__main__":
    cmd_q = queue.Queue()
    lock = threading.RLock()
    shared: Dict[str, Any] = {
        'joystick_state': {},
        'joystick_norm_mag': 0.0,

        'target_xy_m': None,
        'targets_xy': [(0.30, 0.20), (0.30, -0.20), (0.45, 0.20), (0.45, -0.20)],
        'targets_visible': [True, True, False, False],
        'tuning': {
            'curve_pow': CONFIG['smooth']['curve_pow'],
            'heading_offset_360': 0.0,
            'ee_speed_mm_s': CONFIG['ee']['max_speed_control_deg_s'],
            'collision_active': CONFIG['control']['collision_active'],
            'j1_z_height_m': CONFIG['robot']['j1_z_height_m'],
        },
        'collision_detected': False
    }

    try:
        # shared['joystick'] = JoystickReader(CONFIG['joystick'], CONFIG['smooth'], shared, lock)

        # --- FORCE SENSOR (TEMP joystick substitute) ---
        shared['force_sensor'] = ForceSensorReader(port="COM3", baud=921600, scale_N=300.0)

        app = Gui(cmd_q, shared, lock)
        shared['sim'] = SimulationController(CONFIG, cmd_q, shared, lock, app)
        threading.Thread(target=shared['sim'].run, daemon=True).start()
        app.mainloop()
    except Exception as e:
        print(f"A fatal error occurred: {e}", file=sys.stderr)
    finally:
        if 'joystick' in shared and shared.get('joystick'): shared['joystick'].stop()
        if p.isConnected(): p.disconnect()
