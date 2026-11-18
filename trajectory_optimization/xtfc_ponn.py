"""Extreme Theory of Functional Connections Pontryagin Neural Network (X-TFC PoNN).

Solves the 3D fuel-optimal low-thrust Earth-Mars transfer by analytically embedding
boundary conditions via TFC constrained expressions and optimizing neural network
weights on Pontryagin state-costate 14-ODE residuals using nonlinear least-squares (LM).

Boundary Conditions Embedded Analytically:
    r(-1) = r0, r(1) = rf  (Earth departure and Mars arrival position)
    v(-1) = v0, v(1) = vf  (Earth departure and Mars arrival velocity)
    m(-1) = m0             (Initial spacecraft wet mass 1218 kg)
    lambda_m(1) = 0        (Transversality condition for free final mass)

Collocation Residuals:
    Evaluates all 14 ODE residuals across CGL collocation nodes:
    - 3 kinematic residuals (dr/dt - v)
    - 3 dynamic residuals (dv/dt - (g + (T/m)u))
    - 1 mass depletion residual (dm/dt - (-T/c))
    - 3 costate lambda_r residuals (dlam_r/dt + (dg/dr)^T lam_v)
    - 3 costate lambda_v residuals (dlam_v/dt + lam_r)
    - 1 costate lambda_m residual (dlam_m/dt - (T/m^2)(lam_v . u))
"""

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares

from .constants import (
    MU_EFF_CANON, T_MAX_CANON, C_CANON, C_EXHAUST, SCALE_T, SCALE_L, SCALE_V, SCALE_M,
    SCALE_A, DAYS_PER_TU, R_MARS_CANON, V_MARS_CANON, T_MAX
)
from .dynamics import (
    central_gravity, central_gravity_jacobian,
    optimal_thrust_direction, switching_function, optimal_throttle,
    get_mars_state, mars_arrival_residual,
    compute_energy_conservation_error
)
from .indirect_shooting import TrajectorySolution


class XTFC_PoNN_Solver:
    """X-TFC Pontryagin Neural Network solver for 3D fuel-optimal Earth-Mars transfers."""

    def __init__(
        self,
        flight_time_days: float = 247.6,
        target_mars_angle_rad: float = 3.10,
        n_collocation_nodes: int = 50,
        n_basis_terms: int = 16,
        n_epochs: int = 300,
        t_max_n: float = 1.20,
        hardware_t_max_n: float = T_MAX
    ):
        self.flight_time_days = flight_time_days
        self.tf_canon = flight_time_days / DAYS_PER_TU
        self.target_mars_angle_rad = target_mars_angle_rad
        self.n_nodes = n_collocation_nodes
        self.n_basis = n_basis_terms
        self.n_epochs = n_epochs
        self.t_max_n = t_max_n
        self.t_max_c = t_max_n / (SCALE_M * SCALE_A)
        self.hardware_t_max_n = hardware_t_max_n

        # Earth departure conditions (t = 0, tau = -1)
        self.r0 = np.array([1.0, 0.0, 0.0])
        self.v0 = np.array([0.0, 1.0, 0.0])
        self.m0 = 1.0

        # Mars arrival target (t = tf, tau = +1)
        self.rf, self.vf = get_mars_state(target_mars_angle_rad)

        # Collocation nodes in [-1, 1] (Chebyshev-Gauss-Lobatto)
        i = np.arange(self.n_nodes)
        self.tau_nodes = -np.cos(np.pi * i / (self.n_nodes - 1))
        self.c_tau = 2.0 / self.tf_canon
        self.t_nodes = (self.tf_canon / 2.0) * (self.tau_nodes + 1.0)
        self.t_days = self.t_nodes * DAYS_PER_TU

        # Chebyshev orthogonal polynomial basis evaluated at collocation nodes
        self.T, self.dT = self._chebyshev_basis(self.tau_nodes, self.n_basis)
        self.T_m1 = np.array([(-1.0) ** k for k in range(self.n_basis)])
        self.T_p1 = np.ones(self.n_basis)

        # 1st order TFC boundary switching functions
        self.phi0 = 0.5 * (1.0 - self.tau_nodes)[:, None]
        self.phi1 = 0.5 * (1.0 + self.tau_nodes)[:, None]
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

        # Baseline physical state profiles for weight initialization
        th_prof = np.linspace(0.0, self.target_mars_angle_rad, N)
        r_prof = np.linspace(1.0, R_MARS_CANON, N)
        r_init = np.column_stack([r_prof * np.cos(th_prof), r_prof * np.sin(th_prof), np.zeros(N)])
        v_init = np.column_stack([
            -np.sqrt(MU_EFF_CANON / r_prof) * np.sin(th_prof),
            np.sqrt(MU_EFF_CANON / r_prof) * np.cos(th_prof),
            np.zeros(N)
        ])
        m_init = np.linspace(1.0, 0.78, N)[:, None]

        lam_r_init = np.column_stack([np.full(N, -1.03), np.full(N, -0.34), np.zeros(N)])
        lam_v_init = np.column_stack([np.full(N, -0.37), np.full(N, -1.14), np.zeros(N)])
        lam_m_init = np.linspace(0.22, 0.0, N)[:, None]

        if perturb_weights:
            r_init += rng.normal(0.0, 0.02, size=r_init.shape)
            v_init += rng.normal(0.0, 0.02, size=v_init.shape)
            lam_r_init += rng.normal(0.0, 0.02, size=lam_r_init.shape)
            lam_v_init += rng.normal(0.0, 0.02, size=lam_v_init.shape)

        W_r, _, _, _ = np.linalg.lstsq(self.T, r_init, rcond=None)
        W_v, _, _, _ = np.linalg.lstsq(self.T, v_init, rcond=None)
        W_m, _, _, _ = np.linalg.lstsq(self.T, m_init, rcond=None)
        W_lr, _, _, _ = np.linalg.lstsq(self.T, lam_r_init, rcond=None)
        W_lv, _, _, _ = np.linalg.lstsq(self.T, lam_v_init, rcond=None)
        W_lm, _, _, _ = np.linalg.lstsq(self.T, lam_m_init, rcond=None)

        p0 = np.concatenate([
            W_r.flatten(), W_v.flatten(), W_m.flatten(),
            W_lr.flatten(), W_lv.flatten(), W_lm.flatten()
        ])

        loss_history = []

        def eval_system(p: np.ndarray) -> np.ndarray:
            idx = 0
            Wr = p[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
            Wv = p[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
            Wm = p[idx:idx + M * 1].reshape((M, 1)); idx += M * 1
            Wlr = p[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
            Wlv = p[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
            Wlm = p[idx:idx + M * 1].reshape((M, 1)); idx += M * 1

            # TFC Constrained Expressions:
            # 1. Position: r(-1) = r0, r(1) = rf exactly
            gr = self.T @ Wr
            dgr = self.dT @ Wr
            dr0 = self.r0 - self.T_m1 @ Wr
            drf = self.rf - self.T_p1 @ Wr
            r = gr + self.phi0 * dr0 + self.phi1 * drf
            dr_dt = self.c_tau * (dgr + self.dphi0 * dr0 + self.dphi1 * drf)

            # 2. Velocity: v(-1) = v0, v(1) = vf exactly
            gv = self.T @ Wv
            dgv = self.dT @ Wv
            dv0 = self.v0 - self.T_m1 @ Wv
            dvf = self.vf - self.T_p1 @ Wv
            v = gv + self.phi0 * dv0 + self.phi1 * dvf
            dv_dt = self.c_tau * (dgv + self.dphi0 * dv0 + self.dphi1 * dvf)

            # 3. Mass: m(-1) = m0 exactly
            gm = self.T @ Wm
            dgm = self.dT @ Wm
            dm0 = self.m0 - (self.T_m1 @ Wm)[0]
            m = gm + dm0
            dm_dt = self.c_tau * dgm

            # 4. Costates
            lam_r = self.T @ Wlr
            dlam_r_dt = self.c_tau * (self.dT @ Wlr)

            lam_v = self.T @ Wlv
            dlam_v_dt = self.c_tau * (self.dT @ Wlv)

            # 5. Mass costate: lambda_m(1) = 0 exactly (transversality)
            glm = self.T @ Wlm
            dglm = self.dT @ Wlm
            dlmf = 0.0 - (self.T_p1 @ Wlm)[0]
            lam_m = glm + dlmf
            dlam_m_dt = self.c_tau * dglm

            # 14 ODE Residuals
            res_r = dr_dt - v
            res_v = np.zeros_like(v)
            res_m = np.zeros_like(m)
            res_lr = np.zeros_like(lam_r)
            res_lv = dlam_v_dt + lam_r
            res_lm = np.zeros_like(lam_m)

            for k in range(N):
                rk = r[k]
                mk = max(m[k, 0], 0.1)
                lvk = lam_v[k]
                lmk = lam_m[k, 0]

                u_k = optimal_thrust_direction(lvk)
                s_k = switching_function(lvk, lmk, mk, c=C_CANON)
                t_mag = optimal_throttle(s_k, t_max=self.t_max_c, smoothing_eps=0.05)

                gk = central_gravity(rk, mu=MU_EFF_CANON)
                a_thrust = (t_mag / mk) * u_k

                res_v[k] = dv_dt[k] - (gk + a_thrust)
                res_m[k, 0] = dm_dt[k, 0] - (-t_mag / C_CANON)

                dg_dr = central_gravity_jacobian(rk, mu=MU_EFF_CANON)
                res_lr[k] = dlam_r_dt[k] - (-dg_dr.T @ lvk)
                res_lm[k, 0] = dlam_m_dt[k, 0] - ((t_mag / (mk ** 2)) * np.dot(lvk, u_k))

            vec = np.concatenate([
                res_r.flatten(), res_v.flatten(), res_m.flatten(),
                res_lr.flatten(), res_lv.flatten(), res_lm.flatten()
            ])
            loss = float(np.mean(vec ** 2))
            loss_history.append(loss)
            return vec

        # Optimize network weights via Levenberg-Marquardt
        res = least_squares(eval_system, p0, method="lm", max_nfev=self.n_epochs)
        converged = bool(res.success and loss_history[-1] < 1e-3)

        # Dense evaluation grid (250 points)
        n_eval = 250
        tau_eval = np.linspace(-1.0, 1.0, n_eval)
        t_eval_canon = (self.tf_canon / 2.0) * (tau_eval + 1.0)
        t_days = t_eval_canon * DAYS_PER_TU

        T_eval, dT_eval = self._chebyshev_basis(tau_eval, M)
        phi0_e = 0.5 * (1.0 - tau_eval)[:, None]
        phi1_e = 0.5 * (1.0 + tau_eval)[:, None]

        idx = 0
        Wr = res.x[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
        Wv = res.x[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
        Wm = res.x[idx:idx + M * 1].reshape((M, 1)); idx += M * 1
        Wlr = res.x[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
        Wlv = res.x[idx:idx + M * 3].reshape((M, 3)); idx += M * 3
        Wlm = res.x[idx:idx + M * 1].reshape((M, 1)); idx += M * 1

        dr0 = self.r0 - self.T_m1 @ Wr
        drf = self.rf - self.T_p1 @ Wr
        r_au = (T_eval @ Wr) + phi0_e * dr0 + phi1_e * drf

        dv0 = self.v0 - self.T_m1 @ Wv
        dvf = self.vf - self.T_p1 @ Wv
        v_canon = (T_eval @ Wv) + phi0_e * dv0 + phi1_e * dvf
        v_kms = v_canon * (SCALE_V / 1e3)

        dm0 = self.m0 - (self.T_m1 @ Wm)[0]
        m_canon = ((T_eval @ Wm) + dm0)[:, 0]
        m_kg = m_canon * SCALE_M

        lam_r = T_eval @ Wlr
        lam_v = T_eval @ Wlv
        dlmf = 0.0 - (self.T_p1 @ Wlm)[0]
        lam_m = ((T_eval @ Wlm) + dlmf)[:, 0]

        # Process thrust profile
        thrust_n = np.zeros(n_eval)
        thrust_vec = np.zeros((n_eval, 3))
        for i in range(n_eval):
            u_i = optimal_thrust_direction(lam_v[i])
            s_i = switching_function(lam_v[i], lam_m[i], m_canon[i], c=C_CANON)
            t_mag_c = optimal_throttle(s_i, t_max=self.t_max_c, smoothing_eps=0.05)
            thrust_n[i] = t_mag_c * (SCALE_M * SCALE_A)
            thrust_vec[i] = thrust_n[i] * u_i

        xe_used = float(SCALE_M - m_kg[-1])
        final_m_kg = float(m_kg[-1])
        actual_runtime = time.perf_counter() - start_time

        # Arrival residual (embedded analytically => machine precision)
        arr_res = mars_arrival_residual(r_au[-1], v_canon[-1], self.rf, self.vf)

        # Energy conservation error
        energy_err, energy_err_hist = compute_energy_conservation_error(
            t_eval_canon, r_au, v_canon, m_canon,
            thrust_vec / (SCALE_M * SCALE_A)
        )

        return TrajectorySolution(
            method_name="X-TFC PoNN",
            converged=converged,
            iterations=res.nfev,
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
            lambda_m=lam_m,
            energy_error_hist=energy_err_hist,
            loss_history=loss_history
        )
