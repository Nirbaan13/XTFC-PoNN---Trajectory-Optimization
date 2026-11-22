"""180-node LGL pseudospectral direct collocation solver for Scenario A (CasADi + IPOPT).

3-D heliocentric fuel-optimal transfer with explicit mass state, bounded thrust
0 <= T <= T_max, unit thrust direction ||alpha|| = 1, hard initial and terminal
boundary conditions r(t0)=r0, v(t0)=v0, m(t0)=m0, r(tf)=rf, v(tf)=vf, minimizing
propellant mass m0 - m(tf). Supports 'none', 'radial', and 'attitude_flat_plate' SRP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from .pseudospectral import (
    lgl_differentiation_matrix,
    lgl_nodes_and_weights,
)
from .scenario_a import (
    ScenarioAProblem,
    detect_switches,
    energy_balance_error_scenario_a,
    propagate_control_scenario_a,
)


@dataclass
class ScenarioACollocationResult:
    success: bool
    status: str
    n_nodes: int
    t_days: np.ndarray
    t_tu: np.ndarray
    r: np.ndarray
    v: np.ndarray
    m: np.ndarray
    delta: np.ndarray
    alpha: np.ndarray
    fuel_kg: float
    final_mass_kg: float
    terminal_error: float
    reprop_arrival_residual: float
    energy_error: float
    switches: list[dict]
    thrust_sequence: str
    iterations: int
    n_func_evals: int
    n_jac_evals: int
    runtime_s: float
    extra: dict = field(default_factory=dict)


def _detect_throttle_switches(t_days: np.ndarray, delta: np.ndarray, threshold: float = 0.5) -> tuple[list[dict], str]:
    """Detect ON/OFF switches from direct-collocation throttle profile delta(t) in [0, 1]."""
    S_proxy = delta - threshold
    # Exclude tiny boundary-node artifacts within 1 day of t0 or tf
    tf_d = float(t_days[-1])
    raw = detect_switches(t_days, S_proxy)
    switches = [sw for sw in raw if 1.0 < sw["t_days"] < tf_d - 1.0]
    if not switches:
        seq = "ON" if float(np.mean(delta)) >= threshold else "OFF"
    else:
        # Initial arc state from interior of [0, first_switch]
        mask0 = (t_days >= 1.0) & (t_days <= switches[0]["t_days"])
        init_on = float(np.mean(delta[mask0])) >= threshold if np.any(mask0) else (delta[0] >= threshold)
        parts = ["ON" if init_on else "OFF"]
        for _ in switches:
            parts.append("OFF" if parts[-1] == "ON" else "ON")
        seq = "-".join(parts)
    return switches, seq


class ScenarioACollocationSolver:
    """Global LGL pseudospectral direct collocation for Scenario A (CasADi + IPOPT)."""

    def __init__(
        self,
        n_nodes: int = 180,
        max_iter: int = 1500,
        tol: float = 1e-7,
    ):
        self.n_nodes = int(n_nodes)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.tau, self.weights = lgl_nodes_and_weights(self.n_nodes)
        self.D = lgl_differentiation_matrix(self.tau)
        self._nlp_cache: dict[tuple, tuple] = {}

    def _build_nlp(self, prob: ScenarioAProblem):
        key = (
            self.n_nodes, round(prob.mu, 12), round(prob.c, 12),
            round(prob.t_max, 12), round(prob.tf, 12),
            round(prob.k_srp, 12), prob.srp_mode,
            tuple(np.round(prob.r0, 12)), tuple(np.round(prob.v0, 12)),
            tuple(np.round(prob.rf, 12)), tuple(np.round(prob.vf, 12)),
            round(prob.m0, 12),
        )
        if key in self._nlp_cache:
            return self._nlp_cache[key]

        N = self.n_nodes
        half_tf = 0.5 * prob.tf
        D_ca = ca.DM(self.D)

        R = ca.MX.sym("R", N, 3)
        V = ca.MX.sym("V", N, 3)
        M = ca.MX.sym("M", N, 1)
        Dlt = ca.MX.sym("Dlt", N, 1)    # throttle delta in [0, 1]
        A = ca.MX.sym("A", N, 3)        # thrust unit vector alpha

        # Dynamics RHS at all N nodes
        r2 = ca.sum2(R * R)
        rn = ca.sqrt(r2 + 1e-18)
        ir3 = 1.0 / (r2 * rn)

        if prob.srp_mode == "radial" and prob.k_srp > 0.0:
            grav_srp = -(prob.mu - prob.k_srp) * (R * ca.repmat(ir3, 1, 3))
        elif prob.srp_mode == "attitude_flat_plate" and prob.k_srp > 0.0:
            rhat = R / ca.repmat(rn, 1, 3)
            udot = ca.sum2(rhat * A)
            phi = 0.5 + 0.5 * (udot * udot)
            a_srp = prob.k_srp * ca.repmat(phi * ir3, 1, 3) * R
            grav_srp = -prob.mu * (R * ca.repmat(ir3, 1, 3)) + a_srp
        else:
            grav_srp = -prob.mu * (R * ca.repmat(ir3, 1, 3))

        acc = grav_srp + ca.repmat(prob.t_max * Dlt / M, 1, 3) * A
        mdot = -(prob.t_max / prob.c) * Dlt

        # Defect constraints: D X - (tf/2) F = 0
        def_r = D_ca @ R - half_tf * V
        def_v = D_ca @ V - half_tf * acc
        def_m = D_ca @ M - half_tf * mdot
        unit_a = ca.sum2(A * A) - 1.0

        # Boundary conditions (hard)
        bc_r0 = R[0, :].T - ca.DM(prob.r0)
        bc_v0 = V[0, :].T - ca.DM(prob.v0)
        bc_m0 = M[0] - prob.m0
        bc_rf = R[N - 1, :].T - ca.DM(prob.rf)
        bc_vf = V[N - 1, :].T - ca.DM(prob.vf)

        x_vec = ca.vertcat(
            ca.reshape(R, 3 * N, 1),
            ca.reshape(V, 3 * N, 1),
            M,
            Dlt,
            ca.reshape(A, 3 * N, 1),
        )
        g_vec = ca.vertcat(
            ca.reshape(def_r, 3 * N, 1),
            ca.reshape(def_v, 3 * N, 1),
            def_m,
            unit_a,
            bc_r0, bc_v0, bc_m0, bc_rf, bc_vf,
        )

        # Objective: minimize propellant m0 - m(tf)
        obj = prob.m0 - M[N - 1]

        lbx = np.concatenate([
            np.full(3 * N, -5.0),
            np.full(3 * N, -5.0),
            np.full(N, 0.05),
            np.zeros(N),
            np.full(3 * N, -1.0),
        ])
        ubx = np.concatenate([
            np.full(3 * N, 5.0),
            np.full(3 * N, 5.0),
            np.full(N, prob.m0 * 1.05),
            np.ones(N),
            np.full(3 * N, 1.0),
        ])
        ng = int(g_vec.size1())
        lbg = np.zeros(ng)
        ubg = np.zeros(ng)

        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.max_iter,
            "ipopt.tol": self.tol,
            "ipopt.acceptable_tol": 1e-5,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "limited-memory",
        }
        solver = ca.nlpsol("scenario_a_lgl", "ipopt", {"x": x_vec, "f": obj, "g": g_vec}, opts)
        pack = (solver, lbx, ubx, lbg, ubg)
        self._nlp_cache[key] = pack
        return pack

    def _initial_guess(self, prob: ScenarioAProblem, seed: int | None = None) -> np.ndarray:
        """Build smooth orbital interpolation guess (seed=None) or randomized polar/3D guess (seed!=None)."""
        N = self.n_nodes
        s = 0.5 * (self.tau + 1.0)   # in [0, 1]

        # Polar coordinates of r0, rf in xy plane + linear z
        r0_rad = float(np.linalg.norm(prob.r0[:2]))
        rf_rad = float(np.linalg.norm(prob.rf[:2]))
        th0 = float(np.arctan2(prob.r0[1], prob.r0[0]))
        thf = float(np.arctan2(prob.rf[1], prob.rf[0]))
        # Prograde transfer angle (Earth -> Mars ~ 349 days is ~ 200-260 deg CCW, check v0 cross r0!)
        # Note: r0=[-0.94,-0.34], v0=[+0.33,-0.94] -> r0 x v0 = (-0.94)(-0.94)-(-0.34)(0.33) = +1.0 > 0 (CCW!)
        dth = (thf - th0) % (2.0 * np.pi)
        if dth < np.pi:
            dth += 2.0 * np.pi  # wait: check actual angle: th0=-160 deg, thf=+134 deg -> dth = 294 deg (~1.63 pi)
        # Wait: (-160 deg) + 294 deg = +134 deg, so (thf - th0) % 2pi = 294 deg is ALREADY > pi!
        dth = (thf - th0) % (2.0 * np.pi)

        if seed is None:
            r_rad = r0_rad + (rf_rad - r0_rad) * s
            th = th0 + dth * s
            rz = prob.r0[2] + (prob.rf[2] - prob.r0[2]) * s
            m_guess = prob.m0 * (1.0 - 0.35 * s)
            dlt_guess = np.full(N, 0.5)
            R = np.column_stack([r_rad * np.cos(th), r_rad * np.sin(th), rz])
            v_tan = np.sqrt(prob.mu / np.maximum(r_rad, 0.2))
            V = np.column_stack([-v_tan * np.sin(th), v_tan * np.cos(th), np.zeros(N)])
            A = np.column_stack([-np.sin(th), np.cos(th), np.zeros(N)])
        else:
            rng = np.random.default_rng(seed)
            # Random smooth perturbation on radial, angular, out-of-plane, mass, throttle, and thrust angle
            bump = np.sin(np.pi * s)
            r_rad = np.clip(
                r0_rad + (rf_rad - r0_rad) * s + rng.uniform(-0.35, 0.45) * bump
                + rng.uniform(-0.15, 0.15) * np.sin(2.0 * np.pi * s),
                0.35, 3.0,
            )
            th = th0 + dth * s + rng.uniform(-0.8, 0.8) * bump
            rz = prob.r0[2] + (prob.rf[2] - prob.r0[2]) * s + rng.uniform(-0.2, 0.2) * bump
            R = np.column_stack([r_rad * np.cos(th), r_rad * np.sin(th), rz])
            v_scale = rng.uniform(0.7, 1.3)
            v_tan = v_scale * np.sqrt(prob.mu / np.maximum(r_rad, 0.2))
            v_rad = rng.uniform(-0.3, 0.3) * bump
            V = np.column_stack([
                v_rad * np.cos(th) - v_tan * np.sin(th),
                v_rad * np.sin(th) + v_tan * np.cos(th),
                rng.uniform(-0.15, 0.15) * bump,
            ])
            m_end = rng.uniform(0.35, 0.85) * prob.m0
            m_guess = prob.m0 + (m_end - prob.m0) * s
            dlt_guess = np.clip(rng.uniform(0.1, 0.9, N), 0.05, 0.95)
            # Random thrust unit vectors
            A_raw = rng.normal(size=(N, 3))
            A = A_raw / np.maximum(np.linalg.norm(A_raw, axis=1, keepdims=True), 1e-8)

        # CasADi reshapes Fortran (column-major) order
        return np.concatenate([
            R.reshape(-1, order="F"),
            V.reshape(-1, order="F"),
            m_guess,
            dlt_guess,
            A.reshape(-1, order="F"),
        ])

    def solve(self, prob: ScenarioAProblem, seed: int | None = None) -> ScenarioACollocationResult:
        t0 = time.perf_counter()
        solver, lbx, ubx, lbg, ubg = self._build_nlp(prob)
        x0 = self._initial_guess(prob, seed=seed)

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        runtime = time.perf_counter() - t0
        stats = solver.stats()
        ret = str(stats.get("return_status", "unknown"))
        iters = int(stats.get("iter_count", 0))
        nfev = int(stats.get("n_call_nlp_f", 0)) + int(stats.get("n_call_nlp_g", 0))
        njev = int(stats.get("n_call_nlp_jac_g", 0))

        N = self.n_nodes
        x_opt = np.asarray(sol["x"]).ravel()
        R = x_opt[0:3 * N].reshape((N, 3), order="F")
        V = x_opt[3 * N:6 * N].reshape((N, 3), order="F")
        M = x_opt[6 * N:7 * N]
        Dlt = np.clip(x_opt[7 * N:8 * N], 0.0, 1.0)
        A = x_opt[8 * N:11 * N].reshape((N, 3), order="F")
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

        t_tu = 0.5 * (self.tau + 1.0) * prob.tf
        t_days = t_tu * prob.t_ref_days
        term_err = float(np.linalg.norm(np.concatenate([R[-1] - prob.rf, V[-1] - prob.vf])))
        final_mass_kg = float(M[-1] * prob.m_ref_kg)
        fuel_kg = float(prob.m0_kg - final_mass_kg)
        switches, seq = _detect_throttle_switches(t_days, Dlt)

        # Independent DOP853 re-propagation of the open-loop control profile
        sol_rp = propagate_control_scenario_a(prob, t_tu, Dlt, A)
        if sol_rp.status == 0 and np.all(np.isfinite(sol_rp.y[:, -1])):
            xf = sol_rp.y[:, -1]
            reprop_err = float(np.linalg.norm(np.concatenate([xf[0:3] - prob.rf, xf[3:6] - prob.vf])))
        else:
            reprop_err = float("nan")

        e_err, _ = energy_balance_error_scenario_a(prob, t_tu, R, V, Dlt, A, M)
        ok_status = ret in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        success = bool(ok_status and term_err < 1e-4 and np.isfinite(reprop_err) and reprop_err < 5e-2)
        return ScenarioACollocationResult(
            success=success,
            status=ret,
            n_nodes=N,
            t_days=t_days,
            t_tu=t_tu,
            r=R,
            v=V,
            m=M,
            delta=Dlt,
            alpha=A,
            fuel_kg=fuel_kg,
            final_mass_kg=final_mass_kg,
            terminal_error=term_err,
            reprop_arrival_residual=reprop_err,
            energy_error=e_err,
            switches=switches,
            thrust_sequence=seq,
            iterations=iters,
            n_func_evals=nfev,
            n_jac_evals=njev,
            runtime_s=runtime,
        )
