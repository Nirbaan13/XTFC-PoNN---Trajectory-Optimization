"""Heliocentric state / costate dynamics for the common fuel-optimal problem.

All functions use canonical units (see problem.py).  SRP is kept as an
explicit perturbing acceleration a_p = k r / r^3 with its own analytical
Jacobian, exactly as written in the paper (it is not folded into mu).
"""

from __future__ import annotations

import math
import numpy as np

from .problem import HelioProblem


# ---------------------------------------------------------------------------
# Throttle law
# ---------------------------------------------------------------------------
def smooth_throttle(S: float, t_max: float, eps: float) -> float:
    """Logistic smoothing of the bang-bang law T = Tmax * H(-S).

    T(S; eps) = Tmax / (1 + exp(S/eps)) = Tmax/2 * (1 - tanh(S / (2 eps))).
    eps -> 0 recovers T = Tmax for S < 0 and T = 0 for S > 0.
    """
    if eps <= 0.0:
        return t_max if S < 0.0 else 0.0
    return 0.5 * t_max * (1.0 - math.tanh(0.5 * S / eps))


def switching_function(lam_v_norm: float, lam_m: float, m: float, c: float) -> float:
    return (1.0 - lam_m) / c - lam_v_norm / m


# ---------------------------------------------------------------------------
# Accelerations and Jacobians
# ---------------------------------------------------------------------------
def accel_grav_srp(r: np.ndarray, mu: float, k: float) -> np.ndarray:
    rn = np.linalg.norm(r)
    return (-mu + k) * r / rn ** 3


def grav_jacobian(r: np.ndarray, mu: float) -> np.ndarray:
    rn = np.linalg.norm(r)
    rh = r / rn
    return -(mu / rn ** 3) * (np.eye(3) - 3.0 * np.outer(rh, rh))


def srp_jacobian(r: np.ndarray, k: float) -> np.ndarray:
    rn = np.linalg.norm(r)
    rh = r / rn
    return (k / rn ** 3) * (np.eye(3) - 3.0 * np.outer(rh, rh))


# ---------------------------------------------------------------------------
# 14-state ODE (fast scalar implementation for solve_ivp)
# ---------------------------------------------------------------------------
def make_rhs14(prob: HelioProblem, eps: float, counter: list | None = None):
    """Return f(t, y) for y = [r, v, m, lam_r, lam_v, lam_m] (14)."""
    mu, k, tmax, c = prob.mu, prob.k_srp, prob.t_max, prob.c
    mk = mu - k  # g + a_p = -(mu - k) r / r^3 ; Jacobian = -(mu-k)/r^3 (I - 3 rr^T)

    def rhs(t, y):
        if counter is not None:
            counter[0] += 1
        x, yy, z, vx, vy, vz, m, lx, ly, lz, ax, ay, az, lm = y
        r2 = x * x + yy * yy + z * z
        rn = math.sqrt(r2)
        ir3 = 1.0 / (r2 * rn)
        lvn = math.sqrt(ax * ax + ay * ay + az * az)
        S = (1.0 - lm) / c - lvn / m
        T = smooth_throttle(S, tmax, eps)
        if lvn > 0.0:
            ux, uy, uz = -ax / lvn, -ay / lvn, -az / lvn
        else:
            ux = uy = uz = 0.0
        ta = T / m
        gx, gy, gz = -mk * x * ir3, -mk * yy * ir3, -mk * z * ir3
        # (d(g+a_p)/dr)^T lam_v ; matrix symmetric: -(mk/r^3)(lam_v - 3 rhat (rhat.lam_v))
        rdl = (x * ax + yy * ay + z * az) / r2
        jx = -mk * ir3 * (ax - 3.0 * x * rdl)
        jy = -mk * ir3 * (ay - 3.0 * yy * rdl)
        jz = -mk * ir3 * (az - 3.0 * z * rdl)
        return np.array([
            vx, vy, vz,
            gx + ta * ux, gy + ta * uy, gz + ta * uz,
            -T / c,
            -jx, -jy, -jz,
            -lx, -ly, -lz,
            -ta / m * lvn,
        ])

    return rhs


def rhs14_reference(prob: HelioProblem, eps: float, y: np.ndarray) -> np.ndarray:
    """Straightforward matrix implementation (used in tests to verify make_rhs14)."""
    r, v, m = y[0:3], y[3:6], y[6]
    lr, lv, lm = y[7:10], y[10:13], y[13]
    lvn = np.linalg.norm(lv)
    u = -lv / lvn
    S = switching_function(lvn, lm, m, prob.c)
    T = smooth_throttle(S, prob.t_max, eps)
    a = accel_grav_srp(r, prob.mu, prob.k_srp)
    J = grav_jacobian(r, prob.mu) + srp_jacobian(r, prob.k_srp)
    return np.concatenate([v, a + T / m * u, [-T / prob.c], -J.T @ lv, -lr,
                           [(T / m ** 2) * np.dot(lv, u)]])


def hamiltonian(prob: HelioProblem, eps: float, y: np.ndarray) -> float:
    """H = T/c + lam_r.v + lam_v.(g + a_p + T/m u) - lam_m T/c (paper form).

    For eps > 0 the smoothed problem's true Hamiltonian contains the extra
    entropy-like term that generates the logistic law; the paper form is
    reported as the diagnostic (it is exactly conserved only for eps -> 0).
    """
    r, v, m = y[0:3], y[3:6], y[6]
    lr, lv, lm = y[7:10], y[10:13], y[13]
    lvn = np.linalg.norm(lv)
    u = -lv / lvn if lvn > 0 else np.zeros(3)
    S = switching_function(lvn, lm, m, prob.c)
    T = smooth_throttle(S, prob.t_max, eps)
    a = accel_grav_srp(r, prob.mu, prob.k_srp)
    return float(T / prob.c + lr @ v + lv @ (a + T / m * u) - lm * T / prob.c)


def smoothed_hamiltonian(prob: HelioProblem, eps: float, y: np.ndarray) -> float:
    """Hamiltonian of the smoothed problem, conserved exactly along extremals.

    The logistic law T = Tmax/(1+exp(S/eps)) is the minimiser over
    T in [0, Tmax] of  S*T + eps*Tmax*[p ln p + (1-p) ln(1-p)],  p = T/Tmax.
    Hence H_eps = H_paper + eps*Tmax*[p ln p + (1-p) ln(1-p)] is constant.
    """
    H = hamiltonian(prob, eps, y)
    if eps <= 0:
        return H
    m, lv, lm = y[6], y[10:13], y[13]
    S = switching_function(np.linalg.norm(lv), lm, m, prob.c)
    p = 1.0 / (1.0 + math.exp(min(S / eps, 700.0)))
    ent = 0.0
    if 0.0 < p < 1.0:
        ent = p * math.log(p) + (1 - p) * math.log(1 - p)
    return H + eps * prob.t_max * ent


# ---------------------------------------------------------------------------
# 7-state ODE driven by an explicit control history
# ---------------------------------------------------------------------------
def make_rhs7_control(prob: HelioProblem, thrust_vec_fn, counter: list | None = None):
    """f(t, x) for x = [r, v, m] with thrust vector T*u = thrust_vec_fn(t)."""
    mk = prob.mu - prob.k_srp
    c = prob.c

    def rhs(t, x):
        if counter is not None:
            counter[0] += 1
        r = x[0:3]
        rn = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        Tv = thrust_vec_fn(t)
        T = math.sqrt(Tv[0] ** 2 + Tv[1] ** 2 + Tv[2] ** 2)
        a = -mk * r / rn ** 3 + Tv / x[6]
        return np.array([x[3], x[4], x[5], a[0], a[1], a[2], -T / c])

    return rhs


# ---------------------------------------------------------------------------
# Metrics shared by all methods
# ---------------------------------------------------------------------------
def arrival_residual(prob: HelioProblem, r_f: np.ndarray, v_f: np.ndarray) -> float:
    """eps_M = ||[(r - r_M)/L, (v - v_M)/V]||_2 (canonical units: L = V = 1)."""
    return float(np.linalg.norm(np.concatenate([r_f - prob.rf, v_f - prob.vf])))


def energy_balance_error(prob: HelioProblem, t: np.ndarray, r: np.ndarray, v: np.ndarray,
                         thrust_acc: np.ndarray) -> tuple[float, np.ndarray]:
    """eps_E = max_t |E(t) - E(t0) - W(t)| / |E(t0)|, E = v^2/2 - mu_S/r,
    W = int (a_T + a_p).v dt (trapezoid on the supplied dense grid)."""
    rn = np.linalg.norm(r, axis=1)
    E = 0.5 * np.sum(v * v, axis=1) - prob.mu / rn
    a_p = prob.k_srp * r / rn[:, None] ** 3
    integrand = np.sum((thrust_acc + a_p) * v, axis=1)
    W = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
    prof = np.abs(E - E[0] - W) / abs(E[0])
    return float(np.max(prof)), prof
