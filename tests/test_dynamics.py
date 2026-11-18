"""Tests for dynamics, Jacobians, optimal control laws, and energy conservation."""

import numpy as np
import pytest

from trajectory_optimization.dynamics import (
    central_gravity, central_gravity_jacobian,
    srp_acceleration, srp_jacobian,
    j2_acceleration, j2_jacobian,
    optimal_thrust_direction, optimal_throttle, switching_function,
    get_mars_state, mars_arrival_residual, compute_energy_conservation_error
)
from trajectory_optimization.constants import (
    MU_EFF_CANON, T_MAX_CANON, C_CANON, R_MARS_CANON, V_MARS_CANON
)


def test_central_gravity_jacobian():
    r = np.array([1.2, -0.4, 0.1])
    j_ana = central_gravity_jacobian(r, mu=MU_EFF_CANON)
    j_num = np.zeros((3, 3))
    eps = 1e-7
    for col in range(3):
        r_p = r.copy(); r_p[col] += eps
        r_m = r.copy(); r_m[col] -= eps
        j_num[:, col] = (central_gravity(r_p, mu=MU_EFF_CANON) - central_gravity(r_m, mu=MU_EFF_CANON)) / (2 * eps)
    assert np.allclose(j_ana, j_num, atol=1e-6)


def test_j2_jacobian():
    r = np.array([6800e3, 1500e3, -2500e3])
    j_ana = j2_jacobian(r)
    j_num = np.zeros((3, 3))
    eps = 1e-5
    for col in range(3):
        r_p = r.copy(); r_p[col] += eps
        r_m = r.copy(); r_m[col] -= eps
        j_num[:, col] = (j2_acceleration(r_p) - j2_acceleration(r_m)) / (2 * eps)
    assert np.allclose(j_ana, j_num, atol=1e-7)


def test_optimal_control_laws():
    lam_v = np.array([0.0, -2.0, 0.0])
    u = optimal_thrust_direction(lam_v)
    assert np.allclose(u, np.array([0.0, 1.0, 0.0]))
    assert np.isclose(np.linalg.norm(u), 1.0)

    # Switching function
    s_neg = switching_function(lam_v, lambda_m=0.0, m=1.0, c=C_CANON)
    assert s_neg < 0  # Thrust should be active
    t_smooth = optimal_throttle(s_neg, t_max=T_MAX_CANON, smoothing_eps=0.05)
    assert t_smooth > 0.9 * T_MAX_CANON


def test_mars_residual_definition():
    r_target, v_target = get_mars_state(theta=np.pi / 2.0)
    # Exact match should give 0.0
    res_zero = mars_arrival_residual(r_target, v_target, r_target, v_target)
    assert np.isclose(res_zero, 0.0)

    # Small error
    dr = np.array([1e-6, 0.0, 0.0])
    res_err = mars_arrival_residual(r_target + dr, v_target, r_target, v_target)
    assert np.isclose(res_err, 1e-6)


def test_srp_jacobian():
    r = np.array([1.2, 0.5, -0.2])
    j_ana = srp_jacobian(r, k_srp=1.84589e-5)
    j_num = np.zeros((3, 3))
    eps = 1e-6
    for col in range(3):
        r_p = r.copy(); r_p[col] += eps
        r_m = r.copy(); r_m[col] -= eps
        j_num[:, col] = (srp_acceleration(r_p, k_srp=1.84589e-5) - srp_acceleration(r_m, k_srp=1.84589e-5)) / (2 * eps)
    assert np.allclose(j_ana, j_num, atol=1e-8)


def test_hamiltonian_conservation():
    from scipy.integrate import solve_ivp
    from trajectory_optimization.dynamics import heliocentric_state_costate_ode
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -0.1, -0.2, 0.0, -0.3, -0.4, 0.0, 0.0])
    sol = solve_ivp(lambda t, y: heliocentric_state_costate_ode(t, y, smoothing_eps=0.005), (0.0, 0.5), y0, method='DOP853', rtol=1e-11, atol=1e-12)
    def calc_H(y):
        r = y[0:3]; v = y[3:6]; m = y[6]; lam_r = y[7:10]; lam_v = y[10:13]; lam_m = y[13]
        g = central_gravity(r, mu=MU_EFF_CANON)
        u = optimal_thrust_direction(lam_v)
        s = switching_function(lam_v, lam_m, m, c=C_CANON)
        T = optimal_throttle(s, t_max=T_MAX_CANON, smoothing_eps=0.005)
        return (T / C_CANON) * (1.0 - lam_m) + np.dot(lam_r, v) + np.dot(lam_v, g + (T / m) * u)
    H_vals = np.array([calc_H(sol.y[:, i]) for i in range(sol.y.shape[1])])
    assert np.max(np.abs(H_vals - H_vals[0])) < 1e-6

