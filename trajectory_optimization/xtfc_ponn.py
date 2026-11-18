"""Extreme Theory of Functional Connections Pontryagin Neural Network (X-TFC PoNN).

Solves the fuel-optimal low-thrust Earth-Mars transfer by analytically embedding
boundary conditions via TFC constrained expressions and optimizing Chebyshev neural
network weights on Pontryagin state-costate ODE residuals using nonlinear least-squares.
"""

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares

from .constants import (
    MU_EFF_CANON, T_MAX_CANON, C_CANON, C_EXHAUST, SCALE_T, SCALE_L, SCALE_V, SCALE_M,
    SCALE_A, DAYS_PER_TU, R_MARS_CANON, V_MARS_CANON
)
from .dynamics import (
    central_gravity, central_gravity_jacobian,
    get_mars_state, mars_arrival_residual,
    compute_energy_conservation_error
)
from .indirect_shooting import TrajectorySolution


class XTFC_PoNN_Solver:
    """X-TFC Pontryagin Neural Network solver for perturbed Earth-Mars transfers."""

    def __init__(
        self,
        flight_time_days: float = 247.6,
        target_mars_angle_rad: float = 3.10,
        n_collocation_nodes: int = 60,
        n_basis_terms: int = 16,
        n_epochs: int = 500
    ):
        self.flight_time_days = flight_time_days
        self.tf_canon = flight_time_days / DAYS_PER_TU
        self.target_mars_angle_rad = target_mars_angle_rad
        self.n_nodes = n_collocation_nodes
        self.n_basis = n_basis_terms
        self.n_epochs = n_epochs

        # Earth departure conditions (t = 0, tau = -1)
        self.r0 = np.array([1.0, 0.0])
        self.v0 = np.array([0.0, 1.0])

        # Mars arrival target (t = tf, tau = +1)
        self.rf = np.array([
            R_MARS_CANON * np.cos(target_mars_angle_rad),
            R_MARS_CANON * np.sin(target_mars_angle_rad)
        ])
        self.vf = np.array([
            -V_MARS_CANON * np.sin(target_mars_angle_rad),
            V_MARS_CANON * np.cos(target_mars_angle_rad)
        ])

        # Collocation nodes in [-1, 1] (Chebyshev-Gauss-Lobatto)
        i = np.arange(self.n_nodes)
        self.tau_nodes = -np.cos(np.pi * i / (self.n_nodes - 1))
        # Mapping: t = (tf / 2) * (tau + 1) => dtau/dt = 2 / tf
        self.c_tau = 2.0 / self.tf_canon
        self.t_nodes = (self.tf_canon / 2.0) * (self.tau_nodes + 1.0)
        self.t_days = self.t_nodes * DAYS_PER_TU

        # Chebyshev polynomials at nodes
        self.T, self.dT = self._chebyshev_basis(self.tau_nodes, self.n_basis)
        self.T_m1 = np.array([(-1.0) ** k for k in range(self.n_basis)])
        self.T_p1 = np.ones(self.n_basis)

        # 1st order TFC switching functions
        self.phi0 = 0.5 * (1.0 - self.tau_nodes)
        self.phi1 = 0.5 * (1.0 + self.tau_nodes)
        self.dphi0 = -0.5
        self.dphi1 = 0.5

    def _chebyshev_basis(self, tau: np.ndarray, num_terms: int) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates Chebyshev polynomials T_k(tau) and their derivatives dT_k/dtau."""
        n_pts = len(tau)
        T = np.zeros((n_pts, num_terms))
        dT = np.zeros((n_pts, num_terms))

        T[:, 0] = 1.0
        dT[:, 0] = 0.0
        if num_terms > 1:
            T[:, 1] = tau
            dT[:, 1] = 1.0

        for k in range(2, num_terms):
            T[:, k] = 2.0 * tau * T[:, k - 1] - T[:, k - 2]
            dT[:, k] = 2.0 * T[:, k - 1] + 2.0 * tau * dT[:, k - 1] - dT[:, k - 2]

        return T, dT

    def solve(self, seed: int = 1000, perturb_weights: bool = False) -> TrajectorySolution:
        """Trains the X-TFC PoNN using nonlinear least-squares and returns the solution."""
        start_time = time.perf_counter()
        rng = np.random.RandomState(seed)

        M = self.n_basis
        N = self.n_nodes

        # Initial baseline state profiles
        th_prof = np.linspace(0, self.target_mars_angle_rad, N)
        r_prof = np.linspace(1.0, R_MARS_CANON, N)
        r_init = np.column_stack([r_prof * np.cos(th_prof), r_prof * np.sin(th_prof)])
        v_init = np.column_stack([
            -np.sqrt(MU_EFF_CANON / r_prof) * np.sin(th_prof),
            np.sqrt(MU_EFF_CANON / r_prof) * np.cos(th_prof)
        ])

        # Baseline costates
        lam_v_init = np.column_stack([np.full(N, -0.05), np.full(N, -0.15)])
        lam_r_init = np.column_stack([np.full(N, -0.13), np.full(N, -0.05)])

        if perturb_weights:
            r_init += rng.normal(0.0, 0.05, size=r_init.shape)
            v_init += rng.normal(0.0, 0.05, size=v_init.shape)
            lam_v_init += rng.normal(0.0, 0.02, size=lam_v_init.shape)
            lam_r_init += rng.normal(0.0, 0.02, size=lam_r_init.shape)

        w_r0, _, _, _ = np.linalg.lstsq(self.T, r_init, rcond=None)
        w_v0, _, _, _ = np.linalg.lstsq(self.T, v_init, rcond=None)
        w_lr0, _, _, _ = np.linalg.lstsq(self.T, lam_r_init, rcond=None)
        w_lv0, _, _, _ = np.linalg.lstsq(self.T, lam_v_init, rcond=None)
        p0 = np.concatenate([w_r0.flatten(), w_v0.flatten(), w_lr0.flatten(), w_lv0.flatten()])

        loss_history = []

        def eval_system(p: np.ndarray) -> np.ndarray:
            W_r = p[0:2 * M].reshape((M, 2))
            W_v = p[2 * M:4 * M].reshape((M, 2))
            W_lr = p[4 * M:6 * M].reshape((M, 2))
            W_lv = p[6 * M:8 * M].reshape((M, 2))

            # TFC Constrained Expressions:
            # r(tau) = g_r(tau) + phi0 * (r0 - g_r(-1)) + phi1 * (rf - g_r(1))
            g_r = self.T @ W_r
            dg_r = self.dT @ W_r
            dr_0 = self.r0 - self.T_m1 @ W_r
            dr_f = self.rf - self.T_p1 @ W_r
            r_tau = g_r + np.outer(self.phi0, dr_0) + np.outer(self.phi1, dr_f)
            dr_dt = self.c_tau * (dg_r + np.outer(self.dphi0, dr_0) + np.outer(self.dphi1, dr_f))

            # v(tau) = g_v(tau) + phi0 * (v0 - g_v(-1)) + phi1 * (vf - g_v(1))
            g_v = self.T @ W_v
            dg_v = self.dT @ W_v
            dv_0 = self.v0 - self.T_m1 @ W_v
            dv_f = self.vf - self.T_p1 @ W_v
            v_tau = g_v + np.outer(self.phi0, dv_0) + np.outer(self.phi1, dv_f)
            dv_dt = self.c_tau * (dg_v + np.outer(self.dphi0, dv_0) + np.outer(self.dphi1, dv_f))

            # Costates
            lam_r = self.T @ W_lr
            dlam_r_dt = self.c_tau * (self.dT @ W_lr)

            lam_v = self.T @ W_lv
            dlam_v_dt = self.c_tau * (self.dT @ W_lv)

            # Dynamics residuals:
            # 1. dr/dt - v = 0
            res_r = dr_dt - v_tau

            # 2. dv/dt - (g(r) - lam_v) = 0
            rn = np.linalg.norm(r_tau, axis=1)
            g_eff = -MU_EFF_CANON * r_tau / (rn[:, None] ** 3)
            res_v = dv_dt - (g_eff - lam_v)

            # 3. dlam_r/dt + (dg/dr)^T @ lam_v = 0
            res_lr = np.zeros_like(lam_r)
            for i in range(N):
                r_i = r_tau[i]
                rn_i = rn[i]
                dg = -(MU_EFF_CANON / (rn_i ** 3)) * (np.eye(2) - 3.0 * np.outer(r_i / rn_i, r_i / rn_i))
                res_lr[i] = dlam_r_dt[i] + dg.T @ lam_v[i]

            # 4. dlam_v/dt + lam_r = 0
            res_lv = dlam_v_dt + lam_r

            vec = np.concatenate([res_r.flatten(), res_v.flatten(), res_lr.flatten(), res_lv.flatten()])
            loss = float(np.mean(vec ** 2))
            loss_history.append(loss)
            return vec

        # Optimize network weights via Levenberg-Marquardt
        res = least_squares(eval_system, p0, method="lm", max_nfev=self.n_epochs * 4)
        converged = bool(res.success and loss_history[-1] < 1e-4)

        # Evaluate final converged fields
        W_r = res.x[0:2 * M].reshape((M, 2))
        W_v = res.x[2 * M:4 * M].reshape((M, 2))
        W_lr = res.x[4 * M:6 * M].reshape((M, 2))
        W_lv = res.x[6 * M:8 * M].reshape((M, 2))

        # Reconstruct on fine evaluation grid
        n_eval = 250
        tau_eval = np.linspace(-1.0, 1.0, n_eval)
        t_eval_canon = (self.tf_canon / 2.0) * (tau_eval + 1.0)
        t_days = t_eval_canon * DAYS_PER_TU

        T_eval, dT_eval = self._chebyshev_basis(tau_eval, M)
        phi0_eval = 0.5 * (1.0 - tau_eval)
        phi1_eval = 0.5 * (1.0 + tau_eval)

        dr_0 = self.r0 - self.T_m1 @ W_r
        dr_f = self.rf - self.T_p1 @ W_r
        r_eval_2d = (T_eval @ W_r) + np.outer(phi0_eval, dr_0) + np.outer(phi1_eval, dr_f)

        dv_0 = self.v0 - self.T_m1 @ W_v
        dv_f = self.vf - self.T_p1 @ W_v
        v_eval_2d = (T_eval @ W_v) + np.outer(phi0_eval, dv_0) + np.outer(phi1_eval, dv_f)

        lam_v_eval = T_eval @ W_lv
        lam_r_eval = T_eval @ W_lr

        # Convert to 3D arrays
        r_au = np.zeros((n_eval, 3))
        r_au[:, 0:2] = r_eval_2d

        v_canon = np.zeros((n_eval, 3))
        v_canon[:, 0:2] = v_eval_2d
        v_kms = v_canon * (SCALE_V / 1e3)

        lam_v = np.zeros((n_eval, 3))
        lam_v[:, 0:2] = lam_v_eval

        lam_r = np.zeros((n_eval, 3))
        lam_r[:, 0:2] = lam_r_eval

        # Thrust acceleration: a = -lam_v
        a_mag_canon = np.linalg.norm(lam_v_eval, axis=1)

        # Cumulative Delta-v
        dv_cum_canon = np.zeros(n_eval)
        for i in range(1, n_eval):
            dt_step = t_eval_canon[i] - t_eval_canon[i - 1]
            dv_cum_canon[i] = dv_cum_canon[i - 1] + 0.5 * (a_mag_canon[i] + a_mag_canon[i - 1]) * dt_step

        total_dv_si = dv_cum_canon[-1] * SCALE_V
        m_kg = SCALE_M * np.exp(-dv_cum_canon * SCALE_V / C_EXHAUST)
        xe_used = float(SCALE_M - m_kg[-1])
        final_m_kg = float(m_kg[-1])

        # Thrust magnitude in Newtons: T = m * a
        thrust_n = (m_kg / SCALE_M) * a_mag_canon * (SCALE_M * SCALE_A)
        thrust_vec = np.zeros((n_eval, 3))
        for i in range(n_eval):
            if a_mag_canon[i] > 1e-12:
                thrust_vec[i, 0:2] = -(lam_v_eval[i] / a_mag_canon[i]) * thrust_n[i]

        # Mars arrival residual (analytically guaranteed to be small by TFC)
        r_final_au = r_au[-1]
        v_final_canon = v_canon[-1]
        r_target_3d = np.array([self.rf[0], self.rf[1], 0.0])
        v_target_3d = np.array([self.vf[0], self.vf[1], 0.0])
        arr_res = mars_arrival_residual(r_final_au, v_final_canon, r_target_3d, v_target_3d)

        # Energy conservation error
        energy_err, energy_err_hist = compute_energy_conservation_error(
            t_eval_canon, r_au, v_canon, m_kg / SCALE_M,
            thrust_vec / (SCALE_M * SCALE_A)
        )

        actual_runtime = time.perf_counter() - start_time

        return TrajectorySolution(
            method_name="X-TFC PoNN",
            converged=converged,
            iterations=len(loss_history),
            runtime_seconds=actual_runtime,
            flight_time_days=self.flight_time_days,
            xenon_used_kg=round(xe_used, 2),
            final_mass_kg=round(final_m_kg, 2),
            arrival_residual=arr_res,
            energy_error=energy_err,
            t_days=t_days,
            r_au=r_au,
            v_kms=v_kms,
            m_kg=m_kg,
            thrust_n=thrust_n,
            thrust_vec=thrust_vec,
            lambda_r=lam_r,
            lambda_v=lam_v,
            lambda_m=np.zeros(n_eval),
            energy_error_hist=energy_err_hist,
            loss_history=loss_history
        )
