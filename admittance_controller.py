# admittance_controller.py
import math
from dataclasses import dataclass
from typing import Tuple


####################################################################################################### Free floating
@dataclass
class AdmittanceParams: #########used for both free floating and target based
    # Task-space admittance parameters in XY
    Mx: float = 2.0   # "virtual mass" in x
    My: float = 2.0   # "virtual mass" in y
    Bx: float = 40.0  # "virtual damping" in x
    By: float = 40.0  # "virtual damping" in y

    # Input shaping / safety
    force_deadzone_N: float = 0.5
    force_max_N: float = 25.0
    v_max_mps: float = 0.35


@dataclass
class AdmittanceState:
    vx: float = 0.0
    vy: float = 0.0


def joystick_mag_angle_to_force_xy( #########used for both free floating and target based
    mag_kg: float,
    ang_deg_360: float,
    heading_offset_360: float,
    kg_to_N: float = 9.81
) -> Tuple[float, float]:
    """
    Convert joystick (mag_kg, angle) to task-space forces (Fx, Fy) in Newtons.
    Uses your existing direction convention:
      ux = sin(rad), uy = cos(rad)
    """
    theta = math.radians((ang_deg_360 - heading_offset_360) % 360.0)
    F = max(0.0, mag_kg) * kg_to_N
    Fx = F * math.sin(theta)
    Fy = F * math.cos(theta)
    return Fx, Fy


def clamp_mag(x: float, y: float, max_mag: float) -> Tuple[float, float]: #########used for both free floating and target based
    m = math.hypot(x, y)
    if m <= max_mag or m < 1e-12:
        return x, y
    s = max_mag / m
    return x * s, y * s


def step_free_floating_admittance(
    Fx: float,
    Fy: float,
    dt: float,
    params: AdmittanceParams,
    state: AdmittanceState
) -> Tuple[float, float]:
    """
    Implements: M vdot + B v = F   (K=0)
    Returns the commanded EE velocity (vx, vy) in m/s.
    """
    # Deadzone on force magnitude
    if math.hypot(Fx, Fy) < params.force_deadzone_N:
        Fx, Fy = 0.0, 0.0

    # Saturate force
    Fx, Fy = clamp_mag(Fx, Fy, params.force_max_N)

    # vdot = (F - Bv) / M  (component-wise)
    ax = (Fx - params.Bx * state.vx) / max(params.Mx, 1e-9)
    ay = (Fy - params.By * state.vy) / max(params.My, 1e-9)

    # Integrate velocity
    state.vx += ax * dt
    state.vy += ay * dt

    # Saturate velocity
    state.vx, state.vy = clamp_mag(state.vx, state.vy, params.v_max_mps)

    return state.vx, state.vy
####################################################################################################### target based admittance
@dataclass
class TargetAdmittanceState:
    """
    State for target-based (spring) task-space admittance.

    We keep velocities (vx, vy) and optionally the current integrated position (x, y).
    If you prefer, you can ignore x/y here and integrate x/y in main.py using the
    returned velocities. I include x/y because it makes the controller self-contained.
    """
    vx: float = 0.0
    vy: float = 0.0
    x: float = 0.0
    y: float = 0.0
    initialized: bool = False


def step_target_admittance_to_equilibrium(
    x: float,
    y: float,
    x_d: float,
    y_d: float,
    Fx: float,
    Fy: float,
    dt: float,
    params: AdmittanceParams,
    state: TargetAdmittanceState,
    Kx: float,
    Ky: float,
) -> Tuple[float, float, float, float]:
    """
    Target-based task-space admittance (fixed equilibrium / target).

    Implements (per-axis):
        M * x_ddot + D * x_dot + K * (x - x_d) = F_ext

    State-space:
        v_dot = (F_ext - D*v - K*(x - x_d)) / M
        x_dot = v

    Returns:
        (x_next, y_next, vx_next, vy_next)

    Notes:
    - Uses the same deadzone + force saturation and velocity saturation as your free-floating function.
    - Kx, Ky are the virtual stiffness values (N/m). Set them >0 for target behavior.
    - If Kx=Ky=0, this reduces to free-floating (up to numerical differences).
    """

    # Initialize state position from current EE (first call)
    if not state.initialized:
        state.x = float(x)
        state.y = float(y)
        state.vx = 0.0
        state.vy = 0.0
        state.initialized = True

    # --- Force shaping (reuse your same policies) ---
    # Deadzone on force magnitude
    if math.hypot(Fx, Fy) < params.force_deadzone_N:
        Fx, Fy = 0.0, 0.0

    # Saturate force magnitude
    Fx, Fy = clamp_mag(Fx, Fy, params.force_max_N)

    # --- Compute acceleration from admittance dynamics ---
    ex = state.x - float(x_d)
    ey = state.y - float(y_d)

    # vdot = (F - D*v - K*e) / M
    ax = (Fx - params.Bx * state.vx - float(Kx) * ex) / max(params.Mx, 1e-9)
    ay = (Fy - params.By * state.vy - float(Ky) * ey) / max(params.My, 1e-9)

    # Integrate velocity
    state.vx += ax * dt
    state.vy += ay * dt

    # Saturate velocity magnitude
    state.vx, state.vy = clamp_mag(state.vx, state.vy, params.v_max_mps)

    # Integrate position using the *new* velocity (semi-implicit Euler style)
    state.x += state.vx * dt
    state.y += state.vy * dt

    return state.x, state.y, state.vx, state.vy


def reset_target_admittance_state(
    state: TargetAdmittanceState,
    x: float = 0.0,
    y: float = 0.0,
    zero_velocity: bool = True,
):
    """
    Utility: hard reset the target-admittance state.
    Useful when you press Return Home, teleport, or after collision snap-back.
    """
    state.x = float(x)
    state.y = float(y)
    if zero_velocity:
        state.vx = 0.0
        state.vy = 0.0
    state.initialized = True

# ----------------------------------------------------------------------------------------------------------------------------------------
# Assist-as-Needed (AAN) helpers
# -----------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class AANParams:
    """Parameters for AAN: F_0, F_1 (engagement), e_ref (task error), K_min/K_max, LPF alpha."""

    # --- Engagement from force magnitude (F_0, F_1 in LaTeX) ---
    F0_N: float = 0.5   # F_0: min engagement threshold, >= force_deadzone_N
    F1_N: float = 2.0   # F_1: reference force for full engagement

    # --- Task difficulty from target error (e_ref in LaTeX) ---
    e_ref_m: float = 0.05  # e_ref: distance at which w_e saturates to 1 (m)

    # --- Variable stiffness bounds K_min, K_max (N/m) ---
    Kx_min: float = 2.0
    Ky_min: float = 2.0
    Kx_max: float = 25.0
    Ky_max: float = 25.0

    # --- Optional variable damping bounds (N·s/m); set B*_max = B*_min to disable ---
    Bx_min: float = 40.0
    By_min: float = 40.0
    Bx_max: float = 40.0
    By_max: float = 40.0

    # --- LPF: dot(eta_f) = alpha*(eta - eta_f); alpha > 0 = adaptation rate (1/alpha = time constant in s) ---
    need_alpha: float = 0.15
    need_deadband: float = 0.02  # ignore tiny changes


@dataclass
class AANState:
    need: float = 0.0  # filtered need in [0,1]


def sat01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else float(x)


def compute_engagement_weight(Fx: float, Fy: float, F0_N: float, F1_N: float) -> float:
    """
    Engagement weight w_F in [0,1] from force magnitude.
    Matches: w_F(t) = sat_{[0,1]}( (||F_ext|| - F_0) / (F_1 - F_0) ), with F_0=F0_N, F_1=F1_N.
    """
    Fmag = math.hypot(Fx, Fy)
    denom = max(F1_N - F0_N, 1e-9)
    return sat01((Fmag - F0_N) / denom)


def compute_error_weight(x: float, y: float, x_d: float, y_d: float, e_ref_m: float) -> float:
    """
    Task error weight w_e in [0,1] from distance-to-target.
    Matches: w_e(t) = sat_{[0,1]}( ||x - x_d|| / e_ref ), with e_ref=e_ref_m.
    """
    e = math.hypot(x - x_d, y - y_d)
    denom = max(e_ref_m, 1e-9)
    return sat01(e / denom)


def lowpass_need(
    need_raw: float, aan_state: AANState, alpha: float, deadband: float, dt: float = 0.0
) -> float:
    """
    Low-pass filter for activation signal: dot(eta_f) = alpha * (eta - eta_f).
    Discrete equivalent: eta_f += (1 - exp(-alpha*dt)) * (eta - eta_f), so time constant = 1/alpha (s).
    If dt <= 0, uses legacy blend a = alpha (step-invariant).
    """
    need_raw = sat01(need_raw)
    if abs(need_raw - aan_state.need) < float(deadband):
        return aan_state.need
    if dt > 0:
        a = min(1.0, 1.0 - math.exp(-float(alpha) * dt))
    else:
        a = sat01(alpha)
    aan_state.need = sat01((1.0 - a) * aan_state.need + a * need_raw)
    return aan_state.need


def lerp(a: float, b: float, t: float) -> float:
    t = sat01(t)
    return float(a) + (float(b) - float(a)) * t


def schedule_KB_from_need(need: float, aan_params: AANParams) -> tuple[float, float, float, float]:
    """
    K_d(t) = K_min + (K_max - K_min) * eta_f(t); same for B (optional).
    need = eta_f (filtered activation). Returns (Kx_eff, Ky_eff, Bx_eff, By_eff).
    """
    Kx_eff = lerp(aan_params.Kx_min, aan_params.Kx_max, need)
    Ky_eff = lerp(aan_params.Ky_min, aan_params.Ky_max, need)
    Bx_eff = lerp(aan_params.Bx_min, aan_params.Bx_max, need)
    By_eff = lerp(aan_params.By_min, aan_params.By_max, need)
    return Kx_eff, Ky_eff, Bx_eff, By_eff


def step_target_admittance_AAN(
    x: float,
    y: float,
    x_d: float,
    y_d: float,
    Fx: float,
    Fy: float,
    dt: float,
    params: AdmittanceParams,
    adm_state: TargetAdmittanceState,
    aan_params: AANParams,
    aan_state: AANState,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Assist-as-needed extension of target-based admittance (matches LaTeX formulation).

    Dynamics: M_d x_ddot + D_d x_dot + K_d(t)(x - x_d) = F_ext  [K_d(t) time-varying].
    Stiffness schedule: K_d(t) = K_min + (K_max - K_min) * eta_f(t).
    Activation: eta = w_F(force) * w_e(error); eta_f = LPF(eta) with dot(eta_f) = alpha*(eta - eta_f).

    Returns:
        x_next, y_next, vx_next, vy_next,
        Kx_eff, Ky_eff, Bx_eff, By_eff,
        need_f (eta_f)
    """

    # Initialize state on first call
    if not adm_state.initialized:
        adm_state.x = float(x)
        adm_state.y = float(y)
        adm_state.vx = 0.0
        adm_state.vy = 0.0
        adm_state.initialized = True

    # --- Force shaping (same as your other steps) ---
    if math.hypot(Fx, Fy) < params.force_deadzone_N:
        Fx, Fy = 0.0, 0.0
    Fx, Fy = clamp_mag(Fx, Fy, params.force_max_N)

    # --- Compute need (engagement × difficulty): eta = w_F * w_e ---
    wF = compute_engagement_weight(Fx, Fy, aan_params.F0_N, aan_params.F1_N)
    wE = compute_error_weight(x, y, x_d, y_d, aan_params.e_ref_m)  # use measured position
    need_raw = wF * wE
    need_f = lowpass_need(
        need_raw, aan_state, aan_params.need_alpha, aan_params.need_deadband, dt=dt
    )

    # --- Schedule K and (optionally) B ---
    Kx_eff, Ky_eff, Bx_eff, By_eff = schedule_KB_from_need(need_f, aan_params)

    # --- Admittance dynamics with time-varying K,B ---
    ex = adm_state.x - float(x_d)
    ey = adm_state.y - float(y_d)

    ax = (Fx - Bx_eff * adm_state.vx - Kx_eff * ex) / max(params.Mx, 1e-9)
    ay = (Fy - By_eff * adm_state.vy - Ky_eff * ey) / max(params.My, 1e-9)

    adm_state.vx += ax * dt
    adm_state.vy += ay * dt

    adm_state.vx, adm_state.vy = clamp_mag(adm_state.vx, adm_state.vy, params.v_max_mps)

    adm_state.x += adm_state.vx * dt
    adm_state.y += adm_state.vy * dt

    return (
        adm_state.x, adm_state.y, adm_state.vx, adm_state.vy,
        Kx_eff, Ky_eff, Bx_eff, By_eff,
        need_f
    )



######################################################################################################## Assist As Needed: force based for Joysticksimulationv4_FHsensor_control3_log_tgt
# @dataclass
# class AANForceParams:
#     # Force reference: above this, assistance goes near minimum
#     F_ref_N: float = 8.0
#
#     # Assistance stiffness range (N/m)
#     Kx_min: float = 0.0
#     Kx_max: float = 20.0
#     Ky_min: float = 0.0
#     Ky_max: float = 20.0
#
#     # Smooth / stability
#     need_alpha: float = 0.15   # 0..1, higher = faster change
#     need_deadband: float = 0.05  # ignore tiny changes
#
# @dataclass
# class AANForceState:
#     need: float = 1.0  # filtered need in [0,1]
#
# def compute_need_from_force(Fx: float, Fy: float, F_ref_N: float) -> float:
#     """
#     Returns need in [0,1]:
#       - need ~ 1 when force ~ 0  (max assistance)
#       - need ~ 0 when force >= F_ref_N (min assistance)
#     """
#     Fmag = math.hypot(Fx, Fy)
#     if F_ref_N <= 1e-9:
#         return 1.0
#     need = 1.0 - max(0.0, min(1.0, Fmag / F_ref_N))
#     return need
#
# def stiffness_from_need(need: float, K_min: float, K_max: float) -> float:
#     need = max(0.0, min(1.0, float(need)))
#     return float(K_min) + (float(K_max) - float(K_min)) * need
#
# def step_target_admittance_AAN_force(
#     x: float,
#     y: float,
#     x_d: float,
#     y_d: float,
#     Fx: float,
#     Fy: float,
#     dt: float,
#     params: AdmittanceParams,
#     adm_state: TargetAdmittanceState,
#     aan_params: AANForceParams,
#     aan_state: AANForceState,
# ) -> Tuple[float, float, float, float, float, float, float]:
#     """
#     Assist-as-needed wrapper around step_target_admittance_to_equilibrium().
#
#     Returns:
#       x_next, y_next, vx_next, vy_next, Kx_eff, Ky_eff, need_filt
#     """
#
#     # --- 1) raw need from force ---
#     # need_raw = compute_need_from_force(Fx, Fy, aan_params.F_ref_N)
#
#     # ------------------
#     need_raw = None
#     Fmag = math.hypot(Fx, Fy)
#     F_engage = 1.0  # N (tune: must be > deadzone)
#
#     if Fmag < F_engage:
#         # user not engaging → no assist
#         aan_state.need = 0.0
#         need_f = 0.0
#     else:
#         need_raw = compute_need_from_force(Fx, Fy, aan_params.F_ref_N)
#     # ------------------
#
#     # --- 2) filter need (low-pass) ---
#     # deadband prevents tiny fluctuations
#     if abs(need_raw - aan_state.need) < aan_params.need_deadband:
#         need_f = aan_state.need
#     else:
#         a = max(0.0, min(1.0, float(aan_params.need_alpha)))
#         need_f = (1.0 - a) * aan_state.need + a * need_raw
#         need_f = max(0.0, min(1.0, need_f))
#         aan_state.need = need_f
#
#     # --- 3) compute effective stiffness from filtered need ---
#     Kx_eff = stiffness_from_need(need_f, aan_params.Kx_min, aan_params.Kx_max)
#     Ky_eff = stiffness_from_need(need_f, aan_params.Ky_min, aan_params.Ky_max)
#
#     # --- 4) call your existing target-admittance step ---
#     x_next, y_next, vx_next, vy_next = step_target_admittance_to_equilibrium(
#         x=x, y=y,
#         x_d=x_d, y_d=y_d,
#         Fx=Fx, Fy=Fy,
#         dt=dt,
#         params=params,
#         state=adm_state,
#         Kx=Kx_eff,
#         Ky=Ky_eff
#     )
#
#     return x_next, y_next, vx_next, vy_next, Kx_eff, Ky_eff, need_f

