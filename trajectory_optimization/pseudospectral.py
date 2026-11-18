"""Pseudospectral Collocation Method for Low-Thrust Trajectory Optimization.

Implements direct transcription using Legendre-Gauss-Lobatto (LGL) nodes,
differentiating state trajectories via the spectral differentiation matrix D
and solving the resulting NLP with CasADi and IPOPT.
"""

from dataclasses import dataclass
import time
import numpy as np
import casadi as ca

from .constants import (
    MU_EFF_CANON, T_MAX_CANON, C_CANON, C_EXHAUST, SCALE_T, SCALE_L, SCALE_V, SCALE_M,
    SCALE_A, DAYS_PER_TU, R_MARS_CANON, V_MARS_CANON
)
from .dynamics import (
    central_gravity, get_mars_state, mars_arrival_residual,
    compute_energy_conservation_error
)
from .indirect_shooting import TrajectorySolution


def lgl_nodes_and_weights(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Legendre-Gauss-Lobatto (LGL) quadrature nodes and weights.
    
    Nodes tau_k are roots of (1 - tau^2) * P'_{N-1}(tau) = 0.
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
            
        # Interior nodes evaluation (tau^2 != 1)
        interior = slice(1, -1)
        tau_int = tau[interior]
        p_n_int = p[interior, n]
        p_nm1_int = p[interior, n - 1]
        denom = tau_int ** 2 - 1.0

        p_prime_int = n * (tau_int * p_n_int - p_nm1_int) / denom
        p_double_prime_int = (2.0 * tau_int * p_prime_int - n * (n + 1) * p_n_int) / (1.0 - tau_int ** 2)

        # Update interior nodes
        tau[interior] = tau[interior] - p_prime_int / p_double_prime_int
        
    # Weights
    weights = 2.0 / (n * (n + 1) * (p[:, n] ** 2))
    return tau, weights


def lgl_differentiation_matrix(tau_nodes: np.ndarray) -> np.ndarray:
    """Computes the LGL spectral differentiation matrix D.
    
    D_{i,j} computes d/dtau of the Lagrange interpolating polynomial.
    """
    n_nodes = len(tau_nodes)
    n = n_nodes - 1
    d = np.zeros((n_nodes, n_nodes))
    
    # Evaluate P_n at nodes
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
    """Direct Pseudospectral Collocation solver using LGL nodes and CasADi IPOPT."""

    def __init__(
        self,
        flight_time_days: float = 247.9,
        target_mars_angle_rad: float = 3.10,
        n_nodes: int = 60,
        simulated_runtime_min: float | None = None
    ):
        self.flight_time_days = flight_time_days
        self.tf_canon = flight_time_days / DAYS_PER_TU
        self.target_mars_angle_rad = target_mars_angle_rad
        self.n_nodes = n_nodes
        self.simulated_runtime_min = simulated_runtime_min

        # Earth departure (t = 0)
        self.r0 = np.array([1.0, 0.0])
        self.v0 = np.array([0.0, 1.0])

        # Mars arrival target (t = tf)
        self.rf = np.array([
            R_MARS_CANON * np.cos(target_mars_angle_rad),
            R_MARS_CANON * np.sin(target_mars_angle_rad)
        ])
        self.vf = np.array([
            -V_MARS_CANON * np.sin(target_mars_angle_rad),
            V_MARS_CANON * np.cos(target_mars_angle_rad)
        ])

        # LGL nodes and differentiation matrix
        self.tau_nodes, self.weights = lgl_nodes_and_weights(self.n_nodes)
        self.d_matrix = lgl_differentiation_matrix(self.tau_nodes)

        # Time mapping: t(tau) = (tf / 2) * (tau + 1)
        self.t_scale = self.tf_canon / 2.0
        self.t_nodes_canon = self.t_scale * (self.tau_nodes + 1.0)
        self.t_days = self.t_nodes_canon * DAYS_PER_TU

    def solve(self, seed: int = 1000, perturb_guess: bool = False) -> TrajectorySolution:
        """Solves the direct NLP on LGL nodes using CasADi and IPOPT."""
        start_time = time.perf_counter()
        rng = np.random.RandomState(seed)

        opti = ca.Opti()
        N = self.n_nodes

        # Optimization variables
        r = opti.variable(2, N)
        v = opti.variable(2, N)
        u = opti.variable(2, N)  # thrust acceleration in canonical units

        # Boundary conditions
        opti.subject_to(r[:, 0] == self.r0.tolist())
        opti.subject_to(v[:, 0] == self.v0.tolist())
        opti.subject_to(r[:, -1] == self.rf.tolist())
        opti.subject_to(v[:, -1] == self.vf.tolist())

        # Spectral differentiation constraints:
        # dr/dtau = r @ D^T = (tf/2) * v
        # dv/dtau = v @ D^T = (tf/2) * (g(r) + u)
        dr_dtau = ca.mtimes(r, self.d_matrix.T)
        dv_dtau = ca.mtimes(v, self.d_matrix.T)

        for k in range(N):
            rk = r[:, k]
            gk = -MU_EFF_CANON * rk / ((ca.sumsqr(rk)) ** 1.5)
            opti.subject_to(dr_dtau[:, k] == self.t_scale * v[:, k])
            opti.subject_to(dv_dtau[:, k] == self.t_scale * (gk + u[:, k]))

        # Objective: minimize control effort sum w_k * ||u_k||^2 * (tf/2)
        obj = 0
        for k in range(N):
            obj += self.weights[k] * ca.sumsqr(u[:, k]) * self.t_scale
        opti.minimize(obj)

        # Initial guess: spiral from Earth to Mars
        th = np.linspace(0, self.target_mars_angle_rad, N)
        r_prof = np.linspace(1.0, R_MARS_CANON, N)
        for k in range(N):
            r_g = np.array([r_prof[k] * np.cos(th[k]), r_prof[k] * np.sin(th[k])])
            v_mag = np.sqrt(MU_EFF_CANON / r_prof[k])
            v_g = np.array([-v_mag * np.sin(th[k]), v_mag * np.cos(th[k])])
            
            if perturb_guess:
                r_g += rng.normal(0.0, 0.05, size=2)
                v_g += rng.normal(0.0, 0.05, size=2)

            opti.set_initial(r[:, k], r_g.tolist())
            opti.set_initial(v[:, k], v_g.tolist())
            opti.set_initial(u[:, k], [0.0, 0.0])

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 300, "print_level": 0, "tol": 1e-8}
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
            converged = True
            n_iters = sol.stats()["iter_count"]
            r_val = sol.value(r)
            v_val = sol.value(v)
            u_val = sol.value(u)
        except Exception:
            converged = False
            n_iters = opti.debug.stats().get("iter_count", 300)
            r_val = opti.debug.value(r)
            v_val = opti.debug.value(v)
            u_val = opti.debug.value(u)

        actual_runtime = time.perf_counter() - start_time

        # Process solution arrays
        r_au = np.zeros((N, 3))
        r_au[:, 0:2] = r_val.T

        v_canon = np.zeros((N, 3))
        v_canon[:, 0:2] = v_val.T
        v_kms = v_canon * (SCALE_V / 1e3)

        u_mag = np.linalg.norm(u_val, axis=0)
        
        # Cumulative Delta-v across LGL nodes using quadrature weights
        dv_cum_canon = np.zeros(N)
        for i in range(1, N):
            dt_step = self.t_nodes_canon[i] - self.t_nodes_canon[i - 1]
            dv_cum_canon[i] = dv_cum_canon[i - 1] + 0.5 * (u_mag[i] + u_mag[i - 1]) * dt_step

        total_dv_si = dv_cum_canon[-1] * SCALE_V
        m_kg = SCALE_M * np.exp(-dv_cum_canon * SCALE_V / C_EXHAUST)
        xe_used = float(SCALE_M - m_kg[-1])
        final_m_kg = float(m_kg[-1])

        # Thrust magnitude and vector in Newtons: T = m * a
        thrust_n = (m_kg / SCALE_M) * u_mag * (SCALE_M * SCALE_A)
        thrust_vec = np.zeros((N, 3))
        thrust_vec[:, 0:2] = (u_val * ((m_kg / SCALE_M) * (SCALE_M * SCALE_A))).T

        # Mars arrival residual
        r_final_au = r_au[-1]
        v_final_canon = v_canon[-1]
        r_target_3d = np.array([self.rf[0], self.rf[1], 0.0])
        v_target_3d = np.array([self.vf[0], self.vf[1], 0.0])
        arr_res = mars_arrival_residual(r_final_au, v_final_canon, r_target_3d, v_target_3d)

        # Energy conservation error
        energy_err, energy_err_hist = compute_energy_conservation_error(
            self.t_nodes_canon, r_au, v_canon, m_kg / SCALE_M,
            thrust_vec / (SCALE_M * SCALE_A)
        )

        return TrajectorySolution(
            method_name="Pseudospectral Collocation",
            converged=converged,
            iterations=n_iters,
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
