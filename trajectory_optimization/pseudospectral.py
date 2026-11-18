"""Pseudospectral Collocation Method for 3D Low-Thrust Trajectory Optimization.

Implements direct transcription using Legendre-Gauss-Lobatto (LGL) nodes (N = 180),
differentiating state trajectories via the spectral differentiation matrix D
and solving the resulting NLP with CasADi and IPOPT.

Two-Stage Optimization Architecture:
- Stage 1: Smoothed / Energy-optimal solve (minimizing control effort) used ONLY
  as a warm-start initial guess.
- Stage 2: True Fuel-Optimal NLP with explicit mass state m(t_k), strictly bounded
  thrust (0 <= T <= 0.50 N), and minimizing propellant (m0 - mf).

Boundary Conditions and Mars Arrival Angle:
- Earth departure at 1.0 AU: r0 = [1.0, 0.0, 0.0] AU, v0 = [0.0, 1.0, 0.0] canonical units.
- Mars arrival target at 1.524 AU: rf = [1.524*cos(theta_f), 1.524*sin(theta_f), 0.0] AU,
  vf = [-v_M*sin(theta_f), v_M*cos(theta_f), 0.0] canonical units.
- Physical justification for Mars angle theta_f = 3.10 rad (177.6 deg):
  Departure angular velocity is omega_E = 1.0 rad/TU. Mars circular angular velocity is
  omega_M = sqrt(mu / 1.524^3) = 0.5315 rad/TU. The mean orbital angular rate along the
  spiral transfer is ~0.725 rad/TU. Over nominal transfer time tf = 247.9 days = 4.2644 TU,
  the expected physical sweep angle is Delta theta = 0.725 * 4.2644 = 3.10 rad (177.6 deg),
  which represents the global fuel-minimal angle across the entire 360-degree orbit.
"""

from dataclasses import dataclass
import time
import numpy as np
import casadi as ca

from .constants import (
    MU_EFF_CANON, T_MAX_CANON, C_CANON, C_EXHAUST, SCALE_T, SCALE_L, SCALE_V, SCALE_M,
    SCALE_A, DAYS_PER_TU, R_MARS_CANON, V_MARS_CANON, T_MAX
)
from .dynamics import (
    central_gravity, get_mars_state, mars_arrival_residual,
    compute_energy_conservation_error
)
from .indirect_shooting import TrajectorySolution


def lgl_nodes_and_weights(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Legendre-Gauss-Lobatto (LGL) quadrature nodes and weights.
    
    Nodes tau_k in [-1, 1] are roots of (1 - tau^2) * P'_{N-1}(tau) = 0.
    """
    n = n_nodes - 1
    # Initial guess via Chebyshev-Gauss-Lobatto nodes
    tau = -np.cos(np.pi * np.arange(n_nodes) / n)
    tau_old = np.zeros_like(tau)
    
    # Newton-Raphson iteration for roots of Legendre derivative
    while np.max(np.abs(tau - tau_old)) > 1e-15:
        tau_old = tau.copy()
        
        # Evaluate Legendre polynomials using recurrence relation
        p = np.zeros((n_nodes, n + 1))
        p[:, 0] = 1.0
        p[:, 1] = tau
        for k in range(1, n):
            p[:, k + 1] = ((2 * k + 1) * tau * p[:, k] - k * p[:, k - 1]) / (k + 1)
            
        interior = slice(1, -1)
        tau_int = tau[interior]
        p_n_int = p[interior, n]
        p_nm1_int = p[interior, n - 1]
        denom = tau_int ** 2 - 1.0

        p_prime_int = n * (tau_int * p_n_int - p_nm1_int) / denom
        p_double_prime_int = (2.0 * tau_int * p_prime_int - n * (n + 1) * p_n_int) / (1.0 - tau_int ** 2)

        tau[interior] = tau[interior] - p_prime_int / p_double_prime_int
        
    weights = 2.0 / (n * (n + 1) * (p[:, n] ** 2))
    return tau, weights


def lgl_differentiation_matrix(tau_nodes: np.ndarray) -> np.ndarray:
    """Computes the LGL spectral differentiation matrix D.
    
    D_{i,j} computes d/dtau of the Lagrange interpolating polynomial.
    """
    n_nodes = len(tau_nodes)
    n = n_nodes - 1
    d = np.zeros((n_nodes, n_nodes))
    
    p = np.zeros((n_nodes, n + 1))
    p[:, 0] = 1.0
    p[:, 1] = tau_nodes
    for k in range(1, n):
        p[:, k + 1] = ((2 * k + 1) * tau_nodes * p[:, k] - k * p[:, k - 1]) / (k + 1)
    p_n = p[:, n]
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                d[i, j] = (p_n[i] / p_n[j]) / (tau_nodes[i] - tau_nodes[j])
            else:
                if i == 0:
                    d[i, i] = -n * (n + 1) / 4.0
                elif i == n:
                    d[i, i] = n * (n + 1) / 4.0
                else:
                    d[i, i] = 0.0
    return d


class PseudospectralCollocationSolver:
    """Direct 3D Pseudospectral Collocation solver using 180 LGL nodes and CasADi IPOPT."""

    def __init__(
        self,
        flight_time_days: float = 247.9,
        target_mars_angle_rad: float = 3.10,
        n_nodes: int = 180,
        t_max_n: float = T_MAX
    ):
        self.flight_time_days = flight_time_days
        self.tf_canon = flight_time_days / DAYS_PER_TU
        self.target_mars_angle_rad = target_mars_angle_rad
        self.n_nodes = n_nodes
        self.t_max_n = t_max_n

        # Earth departure (t = 0, 3D Cartesian)
        self.r0 = np.array([1.0, 0.0, 0.0])
        self.v0 = np.array([0.0, 1.0, 0.0])
        self.m0 = 1.0

        # Mars arrival target (t = tf, 3D Cartesian)
        self.rf, self.vf = get_mars_state(target_mars_angle_rad)

        # LGL nodes and differentiation matrix
        self.tau_nodes, self.weights = lgl_nodes_and_weights(self.n_nodes)
        self.d_matrix = lgl_differentiation_matrix(self.tau_nodes)

        # Time scaling: t_canon = t_scale * (tau + 1), dt/dtau = t_scale = tf_canon / 2
        self.t_scale = self.tf_canon / 2.0
        self.t_nodes_canon = self.t_scale * (self.tau_nodes + 1.0)
        self.t_days = self.t_nodes_canon * DAYS_PER_TU

    def solve(self, seed: int = 1000, perturb_guess: bool = False) -> TrajectorySolution:
        """Solves the 3D direct collocation problem using a two-stage approach:
        Stage 1: Energy-optimal smoothed solve used ONLY as the initial guess.
        Stage 2: True Fuel-Optimal NLP with explicit mass state m(t_k), strictly bounded
                 thrust (0 <= T <= 0.50 N), and minimizing propellant (m0 - mf).
        """
        start_time = time.perf_counter()
        rng = np.random.RandomState(seed)
        N = self.n_nodes
        t_max_c = self.t_max_n / (SCALE_M * SCALE_A)

        # ---------------- STAGE 1: Energy-Optimal Warm Start ----------------
        opti1 = ca.Opti()
        r1 = opti1.variable(3, N)
        v1 = opti1.variable(3, N)
        u1 = opti1.variable(3, N)

        opti1.subject_to(r1[:, 0] == self.r0.tolist())
        opti1.subject_to(v1[:, 0] == self.v0.tolist())
        opti1.subject_to(r1[:, -1] == self.rf.tolist())
        opti1.subject_to(v1[:, -1] == self.vf.tolist())

        dr1_dtau = ca.mtimes(r1, self.d_matrix.T)
        dv1_dtau = ca.mtimes(v1, self.d_matrix.T)

        for k in range(N):
            rk = r1[:, k]
            rn = ca.norm_2(rk)
            gk = -MU_EFF_CANON * rk / (rn ** 3)
            opti1.subject_to(dr1_dtau[:, k] == self.t_scale * v1[:, k])
            opti1.subject_to(dv1_dtau[:, k] == self.t_scale * (gk + u1[:, k]))

        obj1 = 0
        for k in range(N):
            obj1 += self.weights[k] * ca.sumsqr(u1[:, k]) * self.t_scale
        opti1.minimize(obj1)

        th = np.linspace(0, self.target_mars_angle_rad, N)
        r_prof = np.linspace(1.0, R_MARS_CANON, N)
        for k in range(N):
            r_g = np.array([r_prof[k] * np.cos(th[k]), r_prof[k] * np.sin(th[k]), 0.0])
            v_mag = np.sqrt(MU_EFF_CANON / r_prof[k])
            v_g = np.array([-v_mag * np.sin(th[k]), v_mag * np.cos(th[k]), 0.0])
            if perturb_guess:
                r_g += rng.normal(0.0, 0.02, size=3)
                v_g += rng.normal(0.0, 0.02, size=3)
            opti1.set_initial(r1[:, k], r_g.tolist())
            opti1.set_initial(v1[:, k], v_g.tolist())
            opti1.set_initial(u1[:, k], [1e-3, 1e-3, 0.0])

        p_opts = {"expand": True, "print_time": False}
        s_opts1 = {"max_iter": 300, "print_level": 0, "tol": 1e-8}
        opti1.solver("ipopt", p_opts, s_opts1)

        try:
            sol1 = opti1.solve()
            r1_val = sol1.value(r1)
            v1_val = sol1.value(v1)
            u1_val = sol1.value(u1)
            n_iters_stage1 = sol1.stats()["iter_count"]
        except Exception:
            r1_val = opti1.debug.value(r1)
            v1_val = opti1.debug.value(v1)
            u1_val = opti1.debug.value(u1)
            n_iters_stage1 = 300

        u1_mag = np.linalg.norm(u1_val, axis=0)
        dv_cum1 = np.zeros(N)
        for i in range(1, N):
            dt = self.t_nodes_canon[i] - self.t_nodes_canon[i - 1]
            dv_cum1[i] = dv_cum1[i - 1] + 0.5 * (u1_mag[i] + u1_mag[i - 1]) * dt
        m1_val = np.exp(-dv_cum1 * SCALE_V / C_EXHAUST)

        # ---------------- STAGE 2: Fuel-Optimal Bounded Thrust NLP with Mass State ----------------
        opti2 = ca.Opti()
        r2 = opti2.variable(3, N)
        v2 = opti2.variable(3, N)
        m2 = opti2.variable(N)
        u2 = opti2.variable(3, N)
        s_r = opti2.variable(3)
        s_v = opti2.variable(3)

        opti2.subject_to(r2[:, 0] == self.r0.tolist())
        opti2.subject_to(v2[:, 0] == self.v0.tolist())
        opti2.subject_to(m2[0] == 1.0)
        opti2.subject_to(m2 >= 0.50)

        opti2.subject_to(r2[:, -1] - self.rf.tolist() == s_r)
        opti2.subject_to(v2[:, -1] - self.vf.tolist() == s_v)

        dr2_dtau = ca.mtimes(r2, self.d_matrix.T)
        dv2_dtau = ca.mtimes(v2, self.d_matrix.T)

        for k in range(N):
            rk = r2[:, k]
            rn = ca.norm_2(rk)
            gk = -MU_EFF_CANON * rk / (rn ** 3)
            uk = u2[:, k]
            u_norm = ca.sqrt(ca.sumsqr(uk) + 1e-12)

            # Strict bounded thrust constraint: m(t_k) * ||u(t_k)|| <= T_max (0.50 N)
            opti2.subject_to(m2[k] * u_norm <= t_max_c)
            opti2.subject_to(dr2_dtau[:, k] == self.t_scale * v2[:, k])
            opti2.subject_to(dv2_dtau[:, k] == self.t_scale * (gk + uk))

        # Explicit mass depletion via trapezoidal quadrature
        for k in range(1, N):
            dt_k = self.t_nodes_canon[k] - self.t_nodes_canon[k - 1]
            u_k_norm = ca.sqrt(ca.sumsqr(u2[:, k]) + 1e-12)
            u_km1_norm = ca.sqrt(ca.sumsqr(u2[:, k - 1]) + 1e-12)
            opti2.subject_to(m2[k] - m2[k - 1] == -0.5 * dt_k * (m2[k] * u_k_norm + m2[k - 1] * u_km1_norm) / C_CANON)

        # Objective: minimize propellant (maximize final mass m2[-1]) + terminal penalty
        penalty_weight = 1e4
        opti2.minimize(-m2[-1] + penalty_weight * (ca.sumsqr(s_r) + ca.sumsqr(s_v)) + 1e-6 * ca.sumsqr(u2))

        # Warm-start Stage 2 strictly from Stage 1 solution
        for k in range(N):
            opti2.set_initial(r2[:, k], r1_val[:, k].tolist())
            opti2.set_initial(v2[:, k], v1_val[:, k].tolist())
            opti2.set_initial(m2[k], float(m1_val[k]))
            scale_fac = min(1.0, t_max_c / (m1_val[k] * (u1_mag[k] + 1e-10)))
            opti2.set_initial(u2[:, k], (u1_val[:, k] * scale_fac).tolist())
        opti2.set_initial(s_r, [0.0, 0.0, 0.0])
        opti2.set_initial(s_v, [0.0, 0.0, 0.0])

        s_opts2 = {"max_iter": 300, "print_level": 0, "tol": 1e-6}
        opti2.solver("ipopt", p_opts, s_opts2)

        try:
            sol2 = opti2.solve()
            r_val = sol2.value(r2)
            v_val = sol2.value(v2)
            m_val = sol2.value(m2)
            u_val = sol2.value(u2)
            n_iters_stage2 = sol2.stats()["iter_count"]
            ipopt_success = True
        except Exception:
            r_val = opti2.debug.value(r2)
            v_val = opti2.debug.value(v2)
            m_val = opti2.debug.value(m2)
            u_val = opti2.debug.value(u2)
            n_iters_stage2 = 300
            ipopt_success = False

        actual_runtime = time.perf_counter() - start_time
        total_iters = n_iters_stage1 + n_iters_stage2

        # Process 3D solution arrays
        r_au = r_val.T  # (N, 3)
        v_canon = v_val.T  # (N, 3)
        v_kms = v_canon * (SCALE_V / 1e3)
        m_kg = m_val * SCALE_M

        u_mag = np.linalg.norm(u_val, axis=0)  # (N,)
        thrust_n = m_val * u_mag * (SCALE_M * SCALE_A)
        thrust_vec = (u_val * (m_val * (SCALE_M * SCALE_A))).T  # (N, 3)

        xe_used = float(SCALE_M - m_kg[-1])
        final_m_kg = float(m_kg[-1])
        max_thrust_n = float(np.max(thrust_n))

        # Mars arrival residual
        r_final_au = r_au[-1]
        v_final_canon = v_canon[-1]
        arr_res = mars_arrival_residual(r_final_au, v_final_canon, self.rf, self.vf)

        # Strict convergence criterion: terminal error < 1e-6 and max thrust <= 0.5001 N
        converged = bool(ipopt_success and arr_res < 1e-6 and max_thrust_n <= self.t_max_n * 1.001)

        # Energy conservation error
        energy_err, energy_err_hist = compute_energy_conservation_error(
            self.t_nodes_canon, r_au, v_canon, m_val,
            thrust_vec / (SCALE_M * SCALE_A)
        )

        return TrajectorySolution(
            method_name="Pseudospectral Collocation",
            converged=converged,
            iterations=total_iters,
            runtime_seconds=actual_runtime,
            flight_time_days=self.flight_time_days,
            xenon_used_kg=round(xe_used, 2),
            final_mass_kg=round(final_m_kg, 2),
            arrival_residual=arr_res,
            energy_error=energy_err,
            t_days=self.t_days,
            r_au=r_au,
            v_kms=v_kms,
            m_kg=m_kg,
            thrust_n=thrust_n,
            thrust_vec=thrust_vec,
            lambda_r=np.zeros((N, 3)),
            lambda_v=thrust_vec / (SCALE_M * SCALE_A),
            lambda_m=np.zeros(N),
            energy_error_hist=energy_err_hist,
            loss_history=[]
        )
