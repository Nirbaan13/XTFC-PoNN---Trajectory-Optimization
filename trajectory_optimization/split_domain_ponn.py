"""Tier 3.9: Split-Domain Pontryagin Neural Network (PoNN) Refinement.

Splits [t0, tf] at K switch times {t_1, ..., t_K} into M = K + 1 smooth arcs with
fixed bang-bang throttle delta_p in {0, 1} per arc (NO tanh smoothing).

Each arc p uses:
  - 2nd-order Hermite TFC constrained expression (Table A6) for r_p(z), v_p(z),
    sharing interface states (r_k, v_k) at interior switches so r and v continuity
    is EXACT by construction, and boundary conditions (r0, v0, rf, vf) are EXACT.
  - 1st-order single-point CE for m_0(z) (m(t0)=m0 exact) and unconstrained
    Chebyshev expansions for m_p (p>=1), lam_v_p, and lam_m_p (with lam_m(tf)=0
    exact on the final arc p=M-1).
  - Explicit interface continuity residuals for m, lam_v, lam_r, lam_m at each switch.
  - Exact switching conditions S(t_k) = c ||lam_v(t_k)|| / m(t_k) + lam_m(t_k) - 1 = 0.
  - Exact analytical Jacobian via JAX autodiff (verified vs central FD < 1e-7).
  - Damped SVD Levenberg-Marquardt solver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .ponn_paper import chebyshev_basis, make_collocation_grid, omega_table_a6
from .scenario_a import (
    ScenarioAProblem,
    propagate_control_scenario_a,
    thrust_sequence_label,
)

jax.config.update("jax_enable_x64", True)


@dataclass
class SplitDomainResult:
    success: bool
    status: str
    switch_times_days: list[float]
    thrust_arcs: list[float]
    fuel_kg: float
    final_mass_kg: float
    residual_rms: float
    reprop_arrival_residual: float
    nfev: int
    njev: int
    runtime_s: float


class SplitDomainPoNN:
    """Multi-arc split-domain Chebyshev TFC PoNN with unknown switch times."""

    def __init__(
        self,
        prob: ScenarioAProblem,
        thrust_arcs: list[float],
        n_neurons_arc: int = 18,
        n_points_arc: int = 36,
        interface_weight: float = 5.0,
    ):
        self.prob = prob
        self.thrust_arcs = [float(d) for d in thrust_arcs]
        self.M = len(self.thrust_arcs)
        self.K = self.M - 1
        self.L = int(n_neurons_arc)
        self.n = int(n_points_arc)
        self.w_if = float(interface_weight)

        # Precompute CGL collocation grid and shifted Chebyshev basis on [-1, 1]
        self.z = make_collocation_grid(self.n, "cgl")
        T_all, dT_all, ddT_all = chebyshev_basis(self.z, self.L + 4)
        # Hermite-shifted basis for r (starts at T_4), single-point shifted for m_0 / lm_{M-1} (starts at T_1),
        # unconstrained for others (starts at T_0).
        self.T_r = T_all[:, 4:4 + self.L]
        self.dT_r = dT_all[:, 4:4 + self.L]
        self.ddT_r = ddT_all[:, 4:4 + self.L]
        self.T_m0 = T_all[:, 1:1 + self.L]
        self.dT_m0 = dT_all[:, 1:1 + self.L]
        self.T_u = T_all[:, 0:self.L]
        self.dT_u = dT_all[:, 0:self.L]
        self.ddT_u = ddT_all[:, 0:self.L]

        # Endpoint evaluations at z = -1 and z = +1
        T_end, dT_end, _ = chebyshev_basis(np.array([-1.0, 1.0]), self.L + 4)
        self.Tr_m1, self.Tr_p1 = T_end[0, 4:4 + self.L], T_end[1, 4:4 + self.L]
        self.dTr_m1, self.dTr_p1 = dT_end[0, 4:4 + self.L], dT_end[1, 4:4 + self.L]
        self.Tm0_m1, self.Tm0_p1 = T_end[0, 1:1 + self.L], T_end[1, 1:1 + self.L]
        self.Tu_m1, self.Tu_p1 = T_end[0, 0:self.L], T_end[1, 0:self.L]
        self.dTu_m1, self.dTu_p1 = dT_end[0, 0:self.L], dT_end[1, 0:self.L]

        # Hermite switching functions Omega_1..4 on z
        Om, dOm, ddOm = omega_table_a6(self.z, -1.0, 1.0)
        self.Om, self.dOm, self.ddOm = Om, dOm, ddOm

        # Linear CE basis matrices for r (independent of arc duration b^2!)
        self.Phi_r = (
            self.T_r
            - np.outer(Om[:, 0], self.Tr_m1)
            - np.outer(Om[:, 1], self.Tr_p1)
            - np.outer(Om[:, 2], self.dTr_m1)
            - np.outer(Om[:, 3], self.dTr_p1)
        )
        self.Phi_rz = (
            self.dT_r
            - np.outer(dOm[:, 0], self.Tr_m1)
            - np.outer(dOm[:, 1], self.Tr_p1)
            - np.outer(dOm[:, 2], self.dTr_m1)
            - np.outer(dOm[:, 3], self.dTr_p1)
        )
        self.Phi_rzz = (
            self.ddT_r
            - np.outer(ddOm[:, 0], self.Tr_m1)
            - np.outer(ddOm[:, 1], self.Tr_p1)
            - np.outer(ddOm[:, 2], self.dTr_m1)
            - np.outer(ddOm[:, 3], self.dTr_p1)
        )
        # Single-point CE for m on arc 0 (m(z=-1)=m0) and lm on arc M-1 (lm(z=+1)=0)
        self.Phi_m0 = self.T_m0 - self.Tm0_m1[None, :]
        self.Phi_m0z = self.dT_m0
        self.Phi_lmf = self.T_m0 - self.Tm0_p1[None, :]
        self.Phi_lmfz = self.dT_m0

        # Unknown vector layout:
        #   [beta_arcs (M * 8 * L), r_if (K * 3), v_if (K * 3), t_sw (K)]
        self.n_beta = self.M * 8 * self.L
        self.n_theta = self.n_beta + 6 * self.K + self.K

        # Build JAX residual & analytical Jacobian
        self._build_jax()

    def _build_jax(self) -> None:
        M, K, L, n = self.M, self.K, self.L, self.n
        mu = float(self.prob.mu)
        tmax = float(self.prob.t_max)
        c = float(self.prob.c)
        tf = float(self.prob.tf)
        m0 = float(self.prob.m0)
        r0 = jnp.asarray(self.prob.r0)
        v0 = jnp.asarray(self.prob.v0)
        rf = jnp.asarray(self.prob.rf)
        vf = jnp.asarray(self.prob.vf)
        deltas = jnp.asarray(self.thrust_arcs)
        w_if = self.w_if

        Phi_r = jnp.asarray(self.Phi_r)
        Phi_rz = jnp.asarray(self.Phi_rz)
        Phi_rzz = jnp.asarray(self.Phi_rzz)
        Om = jnp.asarray(self.Om)
        dOm = jnp.asarray(self.dOm)
        ddOm = jnp.asarray(self.ddOm)
        Phi_m0 = jnp.asarray(self.Phi_m0)
        Phi_m0z = jnp.asarray(self.Phi_m0z)
        Phi_lmf = jnp.asarray(self.Phi_lmf)
        Phi_lmfz = jnp.asarray(self.Phi_lmfz)
        T_u = jnp.asarray(self.T_u)
        dT_u = jnp.asarray(self.dT_u)
        ddT_u = jnp.asarray(self.ddT_u)
        Tm0_m1 = jnp.asarray(self.Tm0_m1)
        Tm0_p1 = jnp.asarray(self.Tm0_p1)
        Tu_m1 = jnp.asarray(self.Tu_m1)
        Tu_p1 = jnp.asarray(self.Tu_p1)
        dTu_m1 = jnp.asarray(self.dTu_m1)
        dTu_p1 = jnp.asarray(self.dTu_p1)

        def residual_fn(theta):
            betas = theta[:self.n_beta].reshape((M, 8, L))
            r_if = theta[self.n_beta:self.n_beta + 3 * K].reshape((K, 3))
            v_if = theta[self.n_beta + 3 * K:self.n_beta + 6 * K].reshape((K, 3))
            t_sw = theta[self.n_beta + 6 * K:]  # (K,)

            t_nodes = jnp.concatenate([jnp.array([0.0]), t_sw, jnp.array([tf])])
            r_nodes = jnp.vstack([r0[None, :], r_if, rf[None, :]])
            v_nodes = jnp.vstack([v0[None, :], v_if, vf[None, :]])

            res_parts = []
            end_m_p1 = []
            end_m_m1 = []
            end_lv_p1 = []
            end_lv_m1 = []
            end_lr_p1 = []
            end_lr_m1 = []
            end_lm_p1 = []
            end_lm_m1 = []

            for p in range(M):
                dt = t_nodes[p + 1] - t_nodes[p]
                b2 = 2.0 / dt
                b4 = b2 * b2
                bp = betas[p]  # (8, L)

                ra, rb = r_nodes[p], r_nodes[p + 1]
                va, vb = v_nodes[p], v_nodes[p + 1]

                # r, a on arc p
                P_r = (
                    Om[:, 0:1] * ra[None, :]
                    + Om[:, 1:2] * rb[None, :]
                    + (Om[:, 2:3] / b2) * va[None, :]
                    + (Om[:, 3:4] / b2) * vb[None, :]
                )
                P_a = (
                    b4 * (ddOm[:, 0:1] * ra[None, :] + ddOm[:, 1:2] * rb[None, :])
                    + b2 * (ddOm[:, 2:3] * va[None, :] + ddOm[:, 3:4] * vb[None, :])
                )
                r = Phi_r @ bp[0:3].T + P_r
                a = b4 * (Phi_rzz @ bp[0:3].T) + P_a

                # m, mdot
                if p == 0:
                    m = Phi_m0 @ bp[3] + m0
                    mdot = b2 * (Phi_m0z @ bp[3])
                    m_m1 = m0
                    m_p1 = m0 + jnp.dot(Tm0_p1 - Tm0_m1, bp[3])
                else:
                    m = T_u @ bp[3]
                    mdot = b2 * (dT_u @ bp[3])
                    m_m1 = jnp.dot(Tu_m1, bp[3])
                    m_p1 = jnp.dot(Tu_p1, bp[3])

                # lv, lr = -b2 * lv_z, lrdot = -b4 * lv_zz
                lv = T_u @ bp[4:7].T
                lrdot = -b4 * (ddT_u @ bp[4:7].T)
                lv_m1 = Tu_m1 @ bp[4:7].T
                lv_p1 = Tu_p1 @ bp[4:7].T
                lr_m1 = -b2 * (dTu_m1 @ bp[4:7].T)
                lr_p1 = -b2 * (dTu_p1 @ bp[4:7].T)

                # lm, lmdot
                if p == M - 1:
                    lm = Phi_lmf @ bp[7]
                    lmdot = b2 * (Phi_lmfz @ bp[7])
                    lm_m1 = jnp.dot(Tm0_m1 - Tm0_p1, bp[7])
                    lm_p1 = 0.0
                else:
                    lm = T_u @ bp[7]
                    lmdot = b2 * (dT_u @ bp[7])
                    lm_m1 = jnp.dot(Tu_m1, bp[7])
                    lm_p1 = jnp.dot(Tu_p1, bp[7])

                end_m_m1.append(m_m1); end_m_p1.append(m_p1)
                end_lv_m1.append(lv_m1); end_lv_p1.append(lv_p1)
                end_lr_m1.append(lr_m1); end_lr_p1.append(lr_p1)
                end_lm_m1.append(lm_m1); end_lm_p1.append(lm_p1)

                # Dynamics RHS with FIXED delta_p in {0, 1} (no tanh!)
                dp = deltas[p]
                r2 = jnp.sum(r * r, axis=1)
                rn = jnp.sqrt(r2 + 1e-30)
                ir3 = 1.0 / (r2 * rn)
                lvn = jnp.sqrt(jnp.sum(lv * lv, axis=1) + 1e-30)
                alpha = -lv / lvn[:, None]

                acc_rhs = -mu * r * ir3[:, None] + (tmax * dp / m)[:, None] * alpha
                mdot_rhs = jnp.full((n,), -tmax * dp / c)
                rdl = jnp.sum(r * lv, axis=1) / r2
                lr_dot_rhs = mu * ir3[:, None] * (lv - 3.0 * r * rdl[:, None])
                lm_dot_rhs = -(tmax * dp / (m * m)) * lvn

                res_parts.append((a - acc_rhs).reshape(-1))
                res_parts.append(mdot - mdot_rhs)
                res_parts.append((lrdot - lr_dot_rhs).reshape(-1))
                res_parts.append(lmdot - lm_dot_rhs)

            # Interface continuity & switching conditions at each switch k = 0..K-1
            for k in range(K):
                res_parts.append(w_if * jnp.array([end_m_p1[k] - end_m_m1[k + 1]]))
                res_parts.append(w_if * (end_lv_p1[k] - end_lv_m1[k + 1]))
                res_parts.append(w_if * (end_lr_p1[k] - end_lr_m1[k + 1]))
                res_parts.append(w_if * jnp.array([end_lm_p1[k] - end_lm_m1[k + 1]]))
                # S(t_k) = 0
                lvn_k = jnp.sqrt(jnp.dot(end_lv_p1[k], end_lv_p1[k]) + 1e-30)
                S_k = c * lvn_k / end_m_p1[k] + end_lm_p1[k] - 1.0
                res_parts.append(w_if * jnp.array([S_k]))

            return jnp.concatenate(res_parts)

        self._res_jit = jax.jit(residual_fn)
        self._jac_jit = jax.jit(jax.jacfwd(residual_fn))

    def residual(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self._res_jit(jnp.asarray(theta, dtype=jnp.float64)), dtype=float)

    def residual_and_jacobian(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th = jnp.asarray(theta, dtype=jnp.float64)
        return np.asarray(self._res_jit(th), dtype=float), np.asarray(self._jac_jit(th), dtype=float)

    def project_trajectory(self, t_sw_tu: list[float], eval_y14_fn) -> np.ndarray:
        """Project a 14-state trajectory callable eval_y14_fn(t_tu_array) -> (N, 14) onto theta."""
        t_nodes = [0.0] + [float(t) for t in t_sw_tu] + [float(self.prob.tf)]
        betas = np.zeros((self.M, 8, self.L), dtype=float)
        r_if = np.zeros((self.K, 3), dtype=float)
        v_if = np.zeros((self.K, 3), dtype=float)

        for k in range(self.K):
            yk = eval_y14_fn(np.array([t_nodes[k + 1]]))[0]
            r_if[k] = yk[0:3]
            v_if[k] = yk[3:6]

        r_all = np.vstack([self.prob.r0[None, :], r_if, self.prob.rf[None, :]])
        v_all = np.vstack([self.prob.v0[None, :], v_if, self.prob.vf[None, :]])

        for p in range(self.M):
            dt = t_nodes[p + 1] - t_nodes[p]
            b2 = 2.0 / dt
            t_arc = t_nodes[p] + 0.5 * dt * (self.z + 1.0)
            Y = eval_y14_fn(t_arc)
            ra, rb = r_all[p], r_all[p + 1]
            va, vb = v_all[p], v_all[p + 1]
            P_r = (
                self.Om[:, 0:1] * ra[None, :]
                + self.Om[:, 1:2] * rb[None, :]
                + (self.Om[:, 2:3] / b2) * va[None, :]
                + (self.Om[:, 3:4] / b2) * vb[None, :]
            )
            for j in range(3):
                betas[p, j] = np.linalg.lstsq(self.Phi_r, Y[:, j] - P_r[:, j], rcond=1e-12)[0]
            if p == 0:
                betas[p, 3] = np.linalg.lstsq(self.Phi_m0, Y[:, 6] - self.prob.m0, rcond=1e-12)[0]
            else:
                betas[p, 3] = np.linalg.lstsq(self.T_u, Y[:, 6], rcond=1e-12)[0]
            for j in range(3):
                betas[p, 4 + j] = np.linalg.lstsq(self.T_u, Y[:, 10 + j], rcond=1e-12)[0]
            if p == self.M - 1:
                betas[p, 7] = np.linalg.lstsq(self.Phi_lmf, Y[:, 13], rcond=1e-12)[0]
            else:
                betas[p, 7] = np.linalg.lstsq(self.T_u, Y[:, 13], rcond=1e-12)[0]

        return np.concatenate([betas.reshape(-1), r_if.reshape(-1), v_if.reshape(-1), np.asarray(t_sw_tu, dtype=float)])

    def solve(
        self,
        theta0: np.ndarray,
        max_iter: int = 80,
        ftol: float = 1e-12,
        xtol: float = 1e-12,
    ) -> SplitDomainResult:
        t0 = time.perf_counter()
        x = np.asarray(theta0, dtype=float).copy()
        r, J = self.residual_and_jacobian(x)
        nfev, njev = 1, 1
        cost = 0.5 * float(np.dot(r, r))
        lam = 1e-3
        status = "max_iter"

        for _ in range(max_iter):
            col_norms = np.linalg.norm(J, axis=0)
            d = np.where(col_norms > 0.0, col_norms, 1.0)
            Js = J / d[None, :]
            Q, R_mat = np.linalg.qr(Js, mode="reduced")
            U_r, s, Vt = np.linalg.svd(R_mat, full_matrices=False)
            Utr = U_r.T @ (Q.T @ r)

            accepted = False
            for _ in range(15):
                step = -(Vt.T @ ((s / (s * s + lam)) * Utr)) / d
                x_new = x + step
                # Enforce strictly ordered switch times inside (0.05, tf - 0.05)
                t_sw = x_new[self.n_beta + 6 * self.K:]
                if np.any(t_sw <= 0.02) or np.any(t_sw >= self.prob.tf - 0.02) or np.any(np.diff(t_sw) <= 0.02):
                    lam = min(lam * 10.0, 1e12)
                    continue
                r_new = self.residual(x_new)
                nfev += 1
                if not np.all(np.isfinite(r_new)):
                    lam = min(lam * 10.0, 1e12)
                    continue
                cost_new = 0.5 * float(np.dot(r_new, r_new))
                if cost_new < cost:
                    dc = cost - cost_new
                    x, cost = x_new, cost_new
                    r, J = self.residual_and_jacobian(x)
                    njev += 1
                    lam = max(lam / 10.0, 1e-12)
                    accepted = True
                    if dc <= ftol * max(cost, 1.0) or np.linalg.norm(step) <= xtol * (1.0 + np.linalg.norm(x)):
                        status = "converged"
                    break
                lam = min(lam * 10.0, 1e12)
            if status == "converged":
                break
            if not accepted:
                status = "stalled"
                break

        rt = time.perf_counter() - t0
        rms = float(np.sqrt(np.mean(r * r)))

        # Reconstruct dense piecewise-smooth control for independent DOP853 re-propagation
        betas = x[:self.n_beta].reshape((self.M, 8, self.L))
        t_sw = x[self.n_beta + 6 * self.K:]
        t_nodes = np.concatenate([[0.0], t_sw, [self.prob.tf]])

        t_all, d_all, a_all = [], [], []
        z_dense = np.linspace(-1.0, 1.0, 120)
        T_d, _, _ = chebyshev_basis(z_dense, self.L)
        for p in range(self.M):
            dt = t_nodes[p + 1] - t_nodes[p]
            ta = t_nodes[p] + 0.5 * dt * (z_dense + 1.0)
            if p > 0:
                ta[0] += 1e-9  # keep strictly increasing across switch discontinuity
            lv = T_d @ betas[p, 4:7].T
            lvn = np.maximum(np.linalg.norm(lv, axis=1), 1e-15)
            alpha = -lv / lvn[:, None]
            t_all.append(ta)
            d_all.append(np.full(len(ta), self.thrust_arcs[p]))
            a_all.append(alpha)

        t_vec = np.concatenate(t_all)
        d_vec = np.concatenate(d_all)
        a_vec = np.vstack(a_all)
        prop = propagate_control_scenario_a(self.prob, t_vec, d_vec, a_vec)
        if prop.success and prop.y.shape[1] > 0:
            rp = float(np.linalg.norm(np.concatenate([prop.y[0:3, -1] - self.prob.rf, prop.y[3:6, -1] - self.prob.vf])))
            mf_kg = float(prop.y[6, -1] * self.prob.m_ref_kg)
        else:
            rp = float("nan")
            mf_kg = 0.0
        fuel_kg = float(self.prob.m0_kg - mf_kg) if mf_kg > 0.0 else float("nan")
        ok = bool(np.isfinite(rp) and rp < 1e-2 and rms < 1e-3 and mf_kg > 50.0)
        return SplitDomainResult(
            success=ok,
            status=status,
            switch_times_days=[float(t * self.prob.t_ref_days) for t in t_sw],
            thrust_arcs=list(self.thrust_arcs),
            fuel_kg=fuel_kg,
            final_mass_kg=mf_kg,
            residual_rms=rms,
            reprop_arrival_residual=rp,
            nfev=nfev,
            njev=njev,
            runtime_s=rt,
        )
