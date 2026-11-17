"""Dynamical equations of motion, perturbations (J2 and SRP), costates, and Hamiltonian.

Supports both physical SI units and canonical dimensionless units.
"""

import numpy as np
from .constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    MU_SUN, AU_M, K_SRP, MU_EFF_CANON,
    T_MAX_CANON, C_CANON, MDOT_MAX_CANON,
    SCALE_L, SCALE_V, SCALE_T, SCALE_M,
    R_MARS_CANON, V_MARS_CANON
)


# ==============================================================================
# Gravity and Perturbation Accelerations
# ==============================================================================

def central_gravity(r: np.ndarray, mu: float = MU_EFF_CANON) -> np.ndarray:
    """Central two-body gravitational acceleration g(r) = -mu * r / ||r||^3."""
    r_norm = np.linalg.norm(r)
    return -mu * r / (r_norm ** 3)


def central_gravity_jacobian(r: np.ndarray, mu: float = MU_EFF_CANON) -> np.ndarray:
    """Jacobian dg/dr = -(mu / r^3) * (I - 3 r_hat r_hat^T)."""
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm
    I = np.eye(3)
    return -(mu / (r_norm ** 3)) * (I - 3.0 * np.outer(r_hat, r_hat))


def srp_acceleration(r: np.ndarray, k_srp: float = K_SRP) -> np.ndarray:
    """Solar Radiation Pressure acceleration directed radially outward: a_srp = K * r / ||r||^3."""
    r_norm = np.linalg.norm(r)
    return k_srp * r / (r_norm ** 3)


def srp_jacobian(r: np.ndarray, k_srp: float = K_SRP) -> np.ndarray:
    """Jacobian d(a_srp)/dr = (K / r^3) * (I - 3 r_hat r_hat^T)."""
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm
    I = np.eye(3)
    return (k_srp / (r_norm ** 3)) * (I - 3.0 * np.outer(r_hat, r_hat))


def j2_acceleration(r: np.ndarray, mu: float = MU_EARTH, r_e: float = R_EARTH, j2: float = J2_EARTH) -> np.ndarray:
    """Earth J2 gravitational perturbation acceleration.
    
    a_J2 = -3/2 * J2 * mu * (R_e^2 / r^5) * [x(1 - 5z^2/r^2), y(1 - 5z^2/r^2), z(3 - 5z^2/r^2)]
    """
    x, y, z = r[0], r[1], r[2]
    r_norm = np.linalg.norm(r)
    r2 = r_norm ** 2
    z2_r2 = (z ** 2) / r2
    factor = -1.5 * j2 * mu * (r_e ** 2) / (r_norm ** 5)
    
    a_j2 = factor * np.array([
        x * (1.0 - 5.0 * z2_r2),
        y * (1.0 - 5.0 * z2_r2),
        z * (3.0 - 5.0 * z2_r2)
    ])
    return a_j2


def j2_jacobian(r: np.ndarray, mu: float = MU_EARTH, r_e: float = R_EARTH, j2: float = J2_EARTH) -> np.ndarray:
    """Analytical Jacobian da_J2 / dr."""
    x, y, z = r[0], r[1], r[2]
    r_norm = np.linalg.norm(r)
    r2 = r_norm ** 2
    r5 = r_norm ** 5
    r7 = r_norm ** 7
    k = -1.5 * j2 * mu * (r_e ** 2)
    
    # Components of f = [x(1 - 5z^2/r^2), y(1 - 5z^2/r^2), z(3 - 5z^2/r^2)]
    z2 = z ** 2
    f0 = x * (1.0 - 5.0 * z2 / r2)
    f1 = y * (1.0 - 5.0 * z2 / r2)
    f2 = z * (3.0 - 5.0 * z2 / r2)
    
    # d(r^-5)/dr_j = -5 r_j / r^7
    # df0/dx = 1 - 5z^2/r^2 + 10 x^2 z^2 / r^4
    # df0/dy = 10 x y z^2 / r^4
    # df0/dz = -10 x z / r^2 + 10 x z^3 / r^4 = -10 x z / r^2 * (1 - z^2/r^2)
    r4 = r2 ** 2
    
    df_dr = np.zeros((3, 3))
    df_dr[0, 0] = 1.0 - 5.0 * z2 / r2 + 10.0 * (x ** 2) * z2 / r4
    df_dr[0, 1] = 10.0 * x * y * z2 / r4
    df_dr[0, 2] = -10.0 * x * z / r2 + 10.0 * x * (z ** 3) / r4
    
    df_dr[1, 0] = df_dr[0, 1]
    df_dr[1, 1] = 1.0 - 5.0 * z2 / r2 + 10.0 * (y ** 2) * z2 / r4
    df_dr[1, 2] = -10.0 * y * z / r2 + 10.0 * y * (z ** 3) / r4
    
    df_dr[2, 0] = -5.0 * x * z / r2 * (2.0 - 2.0 * z2 / r2)  # simplified
    df_dr[2, 0] = 10.0 * x * (z ** 3) / r4 - 6.0 * x * z / r2
    df_dr[2, 1] = 10.0 * y * (z ** 3) / r4 - 6.0 * y * z / r2
    df_dr[2, 2] = 3.0 - 15.0 * z2 / r2 + 10.0 * (z ** 4) / r4
    
    # d(a_j2)/dr = k * (df_dr / r^5 - 5 * outer(f, r) / r^7)
    f_vec = np.array([f0, f1, f2])
    da_dr = k * (df_dr / r5 - 5.0 * np.outer(f_vec, r) / r7)
    return da_dr


# ==============================================================================
# Optimal Control Laws & Hamiltonian
# ==============================================================================

def optimal_thrust_direction(lambda_v: np.ndarray) -> np.ndarray:
    """Pontryagin optimal thrust direction u = -lambda_v / ||lambda_v||."""
    lv_norm = np.linalg.norm(lambda_v)
    if lv_norm < 1e-14:
        return np.zeros_like(lambda_v)
    return -lambda_v / lv_norm


def switching_function(lambda_v: np.ndarray, lambda_m: float, m: float, c: float = C_CANON) -> float:
    """Switching function S = (1 - lambda_m)/c - ||lambda_v|| / m."""
    lv_norm = np.linalg.norm(lambda_v)
    return (1.0 - lambda_m) / c - (lv_norm / m)


def optimal_throttle(s: float, t_max: float = T_MAX_CANON, smoothing_eps: float = 0.05) -> float:
    """Smooth throttle control between 0 and T_max.
    
    When smoothing_eps -> 0, recovers bang-bang control:
    T = T_max if S < 0, else 0.
    With smoothing_eps > 0, provides stable continuous derivative for gradient/collocation methods.
    """
    if smoothing_eps <= 0:
        return t_max if s < 0 else 0.0
    # Sigmoidal smooth throttle
    val = -s / smoothing_eps
    val = np.clip(val, -30.0, 30.0)
    return t_max / (1.0 + np.exp(-val))


# ==============================================================================
# Heliocentric State and Costate Dynamics (Canonical Units)
# ==============================================================================

def heliocentric_dynamics(
    t: float,
    state: np.ndarray,
    u_thrust: np.ndarray,
    thrust_mag: float,
    mu_eff: float = MU_EFF_CANON,
    c: float = C_CANON
) -> np.ndarray:
    """Heliocentric state equations: [dr/dt, dv/dt, dm/dt].
    
    state = [rx, ry, rz, vx, vy, vz, m]
    """
    r = state[0:3]
    v = state[3:6]
    m = state[6]
    
    g_eff = central_gravity(r, mu=mu_eff)
    a_thrust = (thrust_mag / m) * u_thrust
    
    rdot = v
    vdot = g_eff + a_thrust
    mdot = -thrust_mag / c
    
    return np.concatenate([rdot, vdot, [mdot]])


def heliocentric_state_costate_ode(
    t: float,
    y: np.ndarray,
    mu_eff: float = MU_EFF_CANON,
    t_max: float = T_MAX_CANON,
    c: float = C_CANON,
    smoothing_eps: float = 0.05
) -> np.ndarray:
    """Full 14-state ODE for Indirect Shooting and PoNN verification.
    
    y = [r(3), v(3), m(1), lambda_r(3), lambda_v(3), lambda_m(1)]
    """
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    lam_r = y[7:10]
    lam_v = y[10:13]
    lam_m = y[13]
    
    u = optimal_thrust_direction(lam_v)
    s = switching_function(lam_v, lam_m, m, c=c)
    t_mag = optimal_throttle(s, t_max=t_max, smoothing_eps=smoothing_eps)
    
    # State derivatives
    g_eff = central_gravity(r, mu=mu_eff)
    a_thrust = (t_mag / m) * u
    rdot = v
    vdot = g_eff + a_thrust
    mdot = -t_mag / c
    
    # Costate derivatives
    dg_dr = central_gravity_jacobian(r, mu=mu_eff)
    d_lam_r = -dg_dr.T @ lam_v
    d_lam_v = -lam_r
    d_lam_m = (t_mag / (m ** 2)) * np.dot(lam_v, u)
    
    ydot = np.zeros(14)
    ydot[0:3] = rdot
    ydot[3:6] = vdot
    ydot[6] = mdot
    ydot[7:10] = d_lam_r
    ydot[10:13] = d_lam_v
    ydot[13] = d_lam_m
    return ydot


# ==============================================================================
# Mars Target State and Ephemerides
# ==============================================================================

def get_mars_state(theta: float, r_mars: float = R_MARS_CANON, mu_eff: float = MU_EFF_CANON) -> tuple[np.ndarray, np.ndarray]:
    """Target position and velocity of Mars at polar orbital angle theta."""
    v_mars = np.sqrt(mu_eff / r_mars)
    r_target = np.array([r_mars * np.cos(theta), r_mars * np.sin(theta), 0.0])
    v_target = np.array([-v_mars * np.sin(theta), v_mars * np.cos(theta), 0.0])
    return r_target, v_target


def mars_arrival_residual(
    r_final: np.ndarray,
    v_final: np.ndarray,
    r_mars: np.ndarray,
    v_mars: np.ndarray
) -> float:
    """Mars arrival constraint residual in dimensionless units:
    
    eps_M = || [(r(tf) - r_Mars(tf)) / L, (v(tf) - v_Mars(tf)) / V] ||_2
    (In canonical units, L=1 and V=1).
    """
    pos_err = r_final - r_mars
    vel_err = v_final - v_mars
    residual_vec = np.concatenate([pos_err, vel_err])
    return float(np.linalg.norm(residual_vec))


# ==============================================================================
# Specific Mechanical Energy and Work Integral
# ==============================================================================

def specific_mechanical_energy(r: np.ndarray, v: np.ndarray, mu_eff: float = MU_EFF_CANON) -> float:
    """Specific mechanical energy E = 0.5 * ||v||^2 - mu_eff / ||r||."""
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    return 0.5 * (v_norm ** 2) - (mu_eff / r_norm)


def compute_energy_conservation_error(
    t_span: np.ndarray,
    r_hist: np.ndarray,
    v_hist: np.ndarray,
    m_hist: np.ndarray,
    thrust_vec_hist: np.ndarray,
    mu_eff: float = MU_EFF_CANON
) -> tuple[float, np.ndarray]:
    """Computes relative energy conservation error over the trajectory:
    
    eps_E(t) = |E(t) - E(t0) - W(t)| / |E(t0)|
    where W(t) = integral_t0^t a_thrust · v dtau.
    """
    n_pts = len(t_span)
    energy_hist = np.zeros(n_pts)
    work_integrand = np.zeros(n_pts)
    
    for i in range(n_pts):
        energy_hist[i] = specific_mechanical_energy(r_hist[i], v_hist[i], mu_eff=mu_eff)
        a_thrust = thrust_vec_hist[i] / m_hist[i]
        work_integrand[i] = np.dot(a_thrust, v_hist[i])
        
    # Trapezoidal integration for cumulative work
    work_hist = np.zeros(n_pts)
    for i in range(1, n_pts):
        dt = t_span[i] - t_span[i - 1]
        work_hist[i] = work_hist[i - 1] + 0.5 * (work_integrand[i] + work_integrand[i - 1]) * dt
        
    e0 = energy_hist[0]
    rel_error_profile = np.abs(energy_hist - e0 - work_hist) / np.abs(e0)
    max_energy_error = float(np.max(rel_error_profile))
    
    return max_energy_error, rel_error_profile
