# functions.py
from __future__ import annotations
import csv
import os
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def hypot2(x: float, y: float) -> float:
    return math.hypot(x, y)


@dataclass
class TrialState:
    """
    Minimal state machine for event marking (force on/off, contact on/off, etc.)
    """
    force_on: bool = False
    contact_on: bool = False
    last_force_on: bool = False
    last_contact_on: bool = False
    force_event: str = ""      # "force_on" / "force_off" / ""
    contact_event: str = ""    # "contact_on" / "contact_off" / ""


class DataLogger:
    """
    Streams rows to CSV with a fixed header.
    Keeps main loop clean.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

        self._f = open(csv_path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)

        self.header = [
            # time
            "t", #Elapsed time since the run started (seconds). Use this for the x-axis in plots.

            # force input (N)
            "Fx_N", "Fy_N", "Fmag_N", #Force magnitude √(Fx² + Fy²) (N). How hard the user is pushing, regardless of direction.

            # admittance output (m/s) (the velocity that comes out of admittance)
            "vx_adm_mps", "vy_adm_mps", "v_adm_mag_mps",
            # Velocity in x from the admittance controller (m/s), before any extra smoothing.
            #Velocity in y from the admittance controller (m/s).
            #Magnitude of that velocity (m/s).

            # smoothed/applied EE vel used to integrate target (m/s)
            "vx_cmd_mps", "vy_cmd_mps", "v_cmd_mag_mps",
            #vx_cmd_mps — Velocity in x actually used to move the target (after slew/smoothing), in m/s.
            #vy_cmd_mps — Same for y (m/s).
            #v_cmd_mag_mps — Magnitude of commanded velocity (m/s).

            # EE current pose (m)
            "x_m", "y_m", "z_m", # Current end-effector position in world frame (m). This is the “real” or measured EE position.

            # EE target pose (m)
            "x_tgt_m", "y_tgt_m", "z_tgt_m", #Target position the controller is trying to reach (m). Same as the selected target (click or T1–T4).

            # joint states (rad) (optional but useful)
            "q1", "q2", "q3", "q4", #Joint positions (rad). Raw robot configuration for analysis or debugging.

            # events
            "force_on", #1 if force magnitude is above threshold, 0 otherwise. “User is pushing.”
            "contact_on", #1 if contact/collision is detected, 0 otherwise.
            "force_event", #Edge event: "force_on", "force_off", or empty.
            "contact_event", # Edge event: "contact_on", "contact_off", or empty.

            # AAN: need (eta_f), effective gains #AAN (control4 only; empty in adaptive)
            "need", # Filtered activation η_f ∈ [0, 1]. Drives stiffness: high need → more assistance (higher K). From force magnitude and distance to target.
            "Kx_eff", "Ky_eff", #Effective stiffness used in the admittance law (N/m). Time-varying in AAN.
                                # Effective stiffness (N/m). Time-varying from w_c. (adaptive)
            "Bx_eff", "By_eff", #Effective damping (N·s/m). In your AAN setup these are usually constant.
                                #  Effective damping (N·s/m). Time-varying from w_c and w_p.(adaptive)

            # Adaptive: cooperation & proximity weights
            "w_c_filt", #Filtered cooperation weight ∈ [0, 1]. High when force is toward target, low when opposing. Stiffness is high when w_c is low.
            "w_p_filt", #Filtered proximity weight ∈ [0, 1]. High when EE is near target (task phase). Used for damping.
        ]
        self._w.writerow(self.header)
        self._f.flush()

    def write(self, row: Dict[str, Any]):
        # Write in header order; missing values become empty
        out = [row.get(k, "") for k in self.header]
        self._w.writerow(out)

    def flush(self):
        self._f.flush()

    def close(self):
        try:
            self._f.flush()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass


def update_trial_state(
    trial: TrialState,
    Fx: float,
    Fy: float,
    force_on_thresh_N: float,
    contact_on: bool
) -> TrialState:
    """
    Produces stable boolean markers and event edges:
      - force_on: |F| >= threshold
      - contact_on: provided by sim (e.g., EE touching box)

    Also generates edge events:
      force_event: "force_on"/"force_off"
      contact_event: "contact_on"/"contact_off"
    """
    fmag = hypot2(Fx, Fy)
    trial.force_on = (fmag >= force_on_thresh_N)
    trial.contact_on = bool(contact_on)

    trial.force_event = ""
    trial.contact_event = ""

    if trial.force_on and (not trial.last_force_on):
        trial.force_event = "force_on"
    elif (not trial.force_on) and trial.last_force_on:
        trial.force_event = "force_off"

    if trial.contact_on and (not trial.last_contact_on):
        trial.contact_event = "contact_on"
    elif (not trial.contact_on) and trial.last_contact_on:
        trial.contact_event = "contact_off"

    trial.last_force_on = trial.force_on
    trial.last_contact_on = trial.contact_on
    return trial


# --------------------------
# Offline metric helpers
# --------------------------

def compute_settling_time(
    t: List[float],
    vmag: List[float],
    t_release: float,
    eps: float = 0.002,          # 2 mm/s
    hold_time: float = 0.25      # must stay below eps for this long
) -> Optional[float]:
    """
    Settling time after release: first time after t_release where |v|<eps
    and stays below for hold_time.
    Returns seconds or None.
    """
    if not t:
        return None

    # find start index
    start_idx = None
    for i, ti in enumerate(t):
        if ti >= t_release:
            start_idx = i
            break
    if start_idx is None:
        return None

    # scan forward
    for i in range(start_idx, len(t)):
        if vmag[i] < eps:
            # check hold
            t0 = t[i]
            ok = True
            for j in range(i, len(t)):
                if t[j] - t0 >= hold_time:
                    break
                if vmag[j] >= eps:
                    ok = False
                    break
            if ok:
                return t[i] - t_release
    return None


def compute_drift(
    x: List[float], y: List[float],
    idx_settle: int,
    idx_end: int
) -> float:
    """
    Drift distance from settle point to end (meters).
    """
    if idx_settle < 0 or idx_end <= idx_settle:
        return float("nan")
    dx = x[idx_end] - x[idx_settle]
    dy = y[idx_end] - y[idx_settle]
    return hypot2(dx, dy)


def compute_path_length(x: List[float], y: List[float]) -> float:
    """
    Total path length in XY (meters).
    """
    if len(x) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(x)):
        s += hypot2(x[i] - x[i-1], y[i] - y[i-1])
    return s


def rms(values: List[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v*v for v in values) / len(values))
