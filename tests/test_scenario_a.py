"""Unit tests for Scenario A (D'Ambrosio & Furfaro 2024, Aerospace 11, 228) — Phases 0, A1, A2, C."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from trajectory_optimization.ponn_paper import (
    PaperPoNNModel,
    PaperPoNNSolver,
    chebyshev_basis,
    make_collocation_grid,
    omega_table_a6,
)
from trajectory_optimization.scenario_a import (
    ScenarioAProblem,
    ScenarioAShootingSolver,
    make_scenario_a_rhs14,
    srp_accel_and_jacobian,
)


def _hamiltonian(prob: ScenarioAProblem, y14: np.ndarray, rho: float) -> float:
    r = y14[0:3]
    v = y14[3:6]
    m = y14[6]
    lr = y14[7:10]
    lv = y14[10:13]
    lm = y14[13]
    rn = np.linalg.norm(r)
    lvn = max(np.linalg.norm(lv), 1e-15)
    alpha = -lv / lvn
    S = prob.c * lvn / m + lm - 1.0
    delta = 0.5 * (1.0 + np.tanh(S / rho))
    a_p, _ = srp_accel_and_jacobian(prob, r, alpha)
    a = -prob.mu * r / (rn ** 3) + a_p + (prob.t_max * delta / m) * alpha
    return float(
        prob.t_max * delta / prob.c
        + np.dot(lr, v)
        + np.dot(lv, a)
        - lm * prob.t_max * delta / prob.c
    )


@pytest.mark.parametrize("srp_mode,a_over_m", [("none", 0.0), ("radial", 0.5), ("attitude_flat_plate", 0.5)])
def test_srp_jacobian_and_costate_ode_finite_differences(srp_mode: str, a_over_m: float):
    """Verify analytical SRP Jacobian da_p/dr and Hamiltonian costate ODEs by central FD."""
    prob = ScenarioAProblem(srp_mode=srp_mode, srp_area_to_mass=a_over_m)
    rng = np.random.default_rng(42)
    r = np.array([-0.8, 0.7, 0.05])
    v = np.array([-0.5, -0.6, 0.02])
    m = 0.78
    lr = rng.normal(size=3)
    lv = rng.normal(size=3)
    lm = 0.35
    alpha = -lv / np.linalg.norm(lv)

    # 1. Verify da_p/dr by central FD (holding alpha fixed, matching PMP partial derivative wrt r)
    a_p, J_p = srp_accel_and_jacobian(prob, r, alpha)
    eps = 1e-7
    J_fd = np.zeros((3, 3))
    for j in range(3):
        dr = np.zeros(3)
        dr[j] = eps
        ap_plus, _ = srp_accel_and_jacobian(prob, r + dr, alpha)
        ap_minus, _ = srp_accel_and_jacobian(prob, r - dr, alpha)
        J_fd[:, j] = (ap_plus - ap_minus) / (2.0 * eps)
    assert np.max(np.abs(J_p - J_fd)) < 1e-9

    # 2. Verify costate ODEs (-dH/dr, -dH/dv, -dH/dm) against RHS14 (at fixed control / smooth rho)
    # Note: Envelope theorem ensures partial H / partial delta = 0 at optimal bang-bang, while at finite rho
    # the PMP ODE uses partial H / partial x at fixed control (delta, alpha) per D'Ambrosio & Furfaro Eqs. 39-41.
    y14 = np.concatenate([r, v, [m], lr, lv, [lm]])
    rhs = make_scenario_a_rhs14(prob, rho=1e-2)
    dy = rhs(0.0, y14)

    # Check lam_v' = -lam_r
    assert np.allclose(dy[10:13], -lr, atol=1e-14)
    # Check lam_r' = -(d/dr)(lv . (-mu r/r^3 + a_p)) via central FD with fixed alpha
    dH_dr = np.zeros(3)
    for j in range(3):
        dr = np.zeros(3)
        dr[j] = eps
        rp, rm = r + dr, r - dr
        ap_p, _ = srp_accel_and_jacobian(prob, rp, alpha)
        ap_m, _ = srp_accel_and_jacobian(prob, rm, alpha)
        hp = np.dot(lv, -prob.mu * rp / (np.linalg.norm(rp) ** 3) + ap_p)
        hm = np.dot(lv, -prob.mu * rm / (np.linalg.norm(rm) ** 3) + ap_m)
        dH_dr[j] = (hp - hm) / (2.0 * eps)
    assert np.max(np.abs(dy[7:10] + dH_dr)) < 1e-8

    # Check lam_m' = -(d/dm)(lv . (Tmax delta / m) alpha)
    lvn = np.linalg.norm(lv)
    S = prob.c * lvn / m + lm - 1.0
    delta = 0.5 * (1.0 + np.tanh(S / 1e-2))
    expected_lmd = -(prob.t_max * delta / (m ** 2)) * lvn
    assert abs(dy[13] - expected_lmd) < 1e-14


@pytest.mark.parametrize("basis_type", ["chebyshev", "chebyshev_shifted", "tanh_elm"])
def test_ponn_ce_boundary_exactness_and_derivatives(basis_type: str):
    """Verify exact boundary satisfaction and FD derivatives of Table A6/A1 constrained expressions."""
    prob = ScenarioAProblem()
    model = PaperPoNNModel(prob, n_neurons=20, basis_type=basis_type, elm_seed=7)
    z = np.linspace(-1.0, 1.0, 41)
    ops = model.operators(z)
    rng = np.random.default_rng(123)
    xi = rng.uniform(-0.2, 0.2, 8 * model.L)
    ev = model.evaluate(xi, ops=ops)

    # Exact boundary conditions (r0, v0, rf, vf, m0, lam_m(tf)=0)
    assert np.allclose(ev["r"][0], prob.r0, atol=1e-12)
    assert np.allclose(ev["v"][0], prob.v0, atol=1e-12)
    assert np.allclose(ev["r"][-1], prob.rf, atol=1e-12)
    assert np.allclose(ev["v"][-1], prob.vf, atol=1e-12)
    assert abs(ev["m"][0] - prob.m0) < 1e-13
    assert abs(ev["lm"][-1]) < 1e-13

    # Central FD check of Chebyshev & Omega derivatives
    z_test = np.array([-0.63, 0.12, 0.77])
    eps = 1e-6
    T_p, dT_p, _ = chebyshev_basis(z_test + eps, 15)
    T_m, dT_m, _ = chebyshev_basis(z_test - eps, 15)
    _, dT, ddT = chebyshev_basis(z_test, 15)
    assert np.max(np.abs(dT - (T_p - T_m) / (2 * eps))) < 1e-8
    assert np.max(np.abs(ddT - (dT_p - dT_m) / (2 * eps))) < 1e-7

    Om_p, dOm_p, _ = omega_table_a6(z_test + eps)
    Om_m, dOm_m, _ = omega_table_a6(z_test - eps)
    _, dOm, ddOm = omega_table_a6(z_test)
    assert np.max(np.abs(dOm - (Om_p - Om_m) / (2 * eps))) < 1e-8
    assert np.max(np.abs(ddOm - (dOm_p - dOm_m) / (2 * eps))) < 1e-7

    # Central FD check of time derivatives v=dr/dt, a=dv/dt, m'=dm/dt, lv'=dlv/dt, lv''=dlv'/dt, lm'=dlm/dt
    dt = 1e-6
    dz = dt * model.b2
    ev0 = model.evaluate(xi, z=z_test)
    ev_p = model.evaluate(xi, z=z_test + dz)
    ev_m = model.evaluate(xi, z=z_test - dz)
    assert np.max(np.abs(ev0["v"] - (ev_p["r"] - ev_m["r"]) / (2 * dt))) < 1e-7
    assert np.max(np.abs(ev0["a"] - (ev_p["v"] - ev_m["v"]) / (2 * dt))) < 1e-6
    assert np.max(np.abs(ev0["md"] - (ev_p["m"] - ev_m["m"]) / (2 * dt))) < 1e-7
    assert np.max(np.abs(ev0["lvd"] - (ev_p["lv"] - ev_m["lv"]) / (2 * dt))) < 1e-7
    assert np.max(np.abs(ev0["lvdd"] - (ev_p["lvd"] - ev_m["lvd"]) / (2 * dt))) < 1e-6
    assert np.max(np.abs(ev0["lmd"] - (ev_p["lm"] - ev_m["lm"]) / (2 * dt))) < 1e-7


@pytest.mark.parametrize("srp_mode,a_over_m", [("none", 0.0), ("radial", 0.5), ("attitude_flat_plate", 0.5)])
def test_ponn_analytical_jacobian_matches_fd(srp_mode: str, a_over_m: float):
    """Verify full 8n x 8L JAX analytical Jacobian against central finite differences."""
    prob = ScenarioAProblem(srp_mode=srp_mode, srp_area_to_mass=a_over_m)
    model = PaperPoNNModel(prob, n_neurons=10, basis_type="chebyshev_shifted")
    z = make_collocation_grid(18, "cgl")
    ops = model.operators(z)
    rng = np.random.default_rng(99)
    xi = 1e-2 * rng.normal(size=8 * model.L)
    r0, J_ana = model.residual_and_jacobian(xi, ops, rho=0.2)

    eps = 1e-7
    J_fd = np.zeros_like(J_ana)
    for k in range(len(xi)):
        dx = np.zeros_like(xi)
        dx[k] = eps
        rp = model.residual_only(xi + dx, ops, rho=0.2)
        rm = model.residual_only(xi - dx, ops, rho=0.2)
        J_fd[:, k] = (rp - rm) / (2.0 * eps)
    assert np.max(np.abs(J_ana - J_fd)) < 1e-7


def test_phase_a1_shooting_and_phase_a2_projection():
    """Phase A1 independent check (~396.06 kg, 4 switches) and Phase A2 basis projection test."""
    prob = ScenarioAProblem()
    sh = ScenarioAShootingSolver(stop_at_rho=1e-5).solve(prob, seed=1000)
    assert sh.success
    assert sh.arrival_residual < 1e-8
    assert len(sh.switches) == 4
    assert sh.thrust_sequence == "ON-OFF-ON-OFF-ON"
    assert abs(sh.fuel_kg - 396.06) < 0.05

    # Strongest Phase A2 test: project smooth shooting solution onto L=70 Chebyshev basis
    sh_s = ScenarioAShootingSolver(stop_at_rho=1e-1).solve(prob, seed=1000)
    assert sh_s.success
    rho_s = sh_s.rho_reached
    y0 = np.concatenate([prob.r0, prob.v0, [prob.m0], sh_s.lam0])
    sol = solve_ivp(
        make_scenario_a_rhs14(prob, rho_s), (0.0, prob.tf), y0,
        method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True,
    )
    model = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted")
    z_fit = make_collocation_grid(400, "cgl")
    ops_fit = model.operators(z_fit)
    Y_fit = sol.sol(ops_fit["t_tu"]).T
    xi = model.project_trajectory(ops_fit, Y_fit)
    z_cgl = make_collocation_grid(140, "cgl")
    ops = model.operators(z_cgl)
    r_vec = model.residual_only(xi, ops, rho=rho_s)
    assert float(np.sqrt(np.mean(r_vec ** 2))) < 1e-4
