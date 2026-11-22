"""Exact D'Ambrosio & Furfaro (Aerospace 2024, 11, 228) Pontryagin Neural Network (PoNN).

Implements Sections 2.2, 3.1, 3.3, 4.1, Table 2, and Appendix A (Tables A1 & A6):
- Domain z in [-1, 1], z0=-1, zf=1, dz=2, b^2 = (zf-z0)/(tf-t0) = 2/tf, z* = z - z0.
- Chebyshev NN free function sigma_q(z) = T_{q-1}(z), q=1..L, input weights w=1, biases b=0
  (plus optional 'tanh_elm' mode for Phase A4(b) ablation).
- 2nd-order Hermite TFC constrained expressions (Table A6 Om1..Om4) for r_i (i=1,2,3),
  1st-order single-point CEs (Table A1 Om1=1) for m(t0)=m0 and lam_m(tf)=0,
  unconstrained Chebyshev expansion for lam_v_i (eliminating lam_r = -lam_v', lam_r' = -lam_v'').
- Unknown vector Xi = [beta_1, beta_2, beta_3, beta_m, beta_lv1, beta_lv2, beta_lv3, beta_lm] (8*L).
- 8 residual components per collocation point (a_res(3), m_res(1), lv_res(3), lm_res(1)).
- Exact analytical Jacobian via JAX automatic differentiation of the 8-state point RHS
  combined with the linear TFC constrained basis matrices.
- Levenberg-Marquardt ('lm') and Trust-Region-Reflective ('trf') via scipy.optimize.least_squares,
  with rho continuation over 20 log-spaced values from 1 to 1e-10 and Sec. 4.1 switch refinement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from .scenario_a import (
    DEFAULT_RHO_SCHEDULE,
    ScenarioAProblem,
    detect_switches,
    energy_balance_error_scenario_a,
    propagate_control_scenario_a,
    throttle_paper,
    thrust_sequence_label,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Chebyshev polynomials T_0 .. T_{L-1} and exact 1st / 2nd derivatives on [-1, 1]
# ---------------------------------------------------------------------------
def chebyshev_basis(z: np.ndarray, n_terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate T_k(z), T_k'(z), T_k''(z) for k = 0 .. n_terms-1 via exact recurrence."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    T = np.zeros((n, n_terms), dtype=float)
    dT = np.zeros((n, n_terms), dtype=float)
    ddT = np.zeros((n, n_terms), dtype=float)
    if n_terms == 0:
        return T, dT, ddT
    T[:, 0] = 1.0
    if n_terms > 1:
        T[:, 1] = z
        dT[:, 1] = 1.0
    for k in range(2, n_terms):
        T[:, k] = 2.0 * z * T[:, k - 1] - T[:, k - 2]
        dT[:, k] = 2.0 * T[:, k - 1] + 2.0 * z * dT[:, k - 1] - dT[:, k - 2]
        ddT[:, k] = 4.0 * dT[:, k - 1] + 2.0 * z * ddT[:, k - 1] - ddT[:, k - 2]
    return T, dT, ddT


def tanh_elm_basis(
    z: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-weight tanh ELM basis and exact 1st / 2nd z-derivatives (for Phase A4 ablation)."""
    z = np.asarray(z, dtype=float)
    arg = np.outer(z, weights) + biases[None, :]
    h = np.tanh(arg)
    sech2 = 1.0 - h * h
    dh = sech2 * weights[None, :]
    ddh = -2.0 * h * sech2 * (weights[None, :] ** 2)
    return h, dh, ddh


# ---------------------------------------------------------------------------
# Appendix A Table A6 Hermite switching functions Omega_1 .. Omega_4
# ---------------------------------------------------------------------------
def omega_table_a6(z: np.ndarray, z0: float = -1.0, zf: float = 1.0):
    """Returns (Om, dOm, ddOm), each shape (len(z), 4), matching Appendix A Table A6."""
    z = np.asarray(z, dtype=float)
    dz = zf - z0
    zs = z - z0
    u = zs / dz

    om1 = 1.0 + 2.0 * u ** 3 - 3.0 * u ** 2
    om2 = -2.0 * u ** 3 + 3.0 * u ** 2
    om3 = zs + (zs ** 3) / (dz ** 2) - 2.0 * (zs ** 2) / dz
    om4 = (zs ** 3) / (dz ** 2) - (zs ** 2) / dz

    dom1 = (6.0 * (zs ** 2)) / (dz ** 3) - (6.0 * zs) / (dz ** 2)
    dom2 = -(6.0 * (zs ** 2)) / (dz ** 3) + (6.0 * zs) / (dz ** 2)
    dom3 = 1.0 + (3.0 * (zs ** 2)) / (dz ** 2) - (4.0 * zs) / dz
    dom4 = (3.0 * (zs ** 2)) / (dz ** 2) - (2.0 * zs) / dz

    ddom1 = (12.0 * zs) / (dz ** 3) - 6.0 / (dz ** 2)
    ddom2 = -(12.0 * zs) / (dz ** 3) + 6.0 / (dz ** 2)
    ddom3 = (6.0 * zs) / (dz ** 2) - 4.0 / dz
    ddom4 = (6.0 * zs) / (dz ** 2) - 2.0 / dz

    Om = np.column_stack([om1, om2, om3, om4])
    dOm = np.column_stack([dom1, dom2, dom3, dom4])
    ddOm = np.column_stack([ddom1, ddom2, ddom3, ddom4])
    return Om, dOm, ddOm


# ---------------------------------------------------------------------------
# JAX point-local 8-component nonlinear RHS & Jacobian (supports Phase C SRP)
# ---------------------------------------------------------------------------
def _rhs8_point_none(q, mu, tmax, c, rho, k_srp):
    """q = [r(3), m(1), lv(3), lm(1)] -> nonlinear RHS for [a(3), m'(1), -lv''(3), lm'(1)]."""
    r = q[0:3]
    m = q[3]
    lv = q[4:7]
    lm = q[7]
    r2 = jnp.dot(r, r)
    rn = jnp.sqrt(r2)
    ir3 = 1.0 / (r2 * rn)
    lvn = jnp.sqrt(jnp.dot(lv, lv) + 1e-30)
    alpha = -lv / lvn
    S = c * lvn / m + lm - 1.0
    delta = 0.5 * (1.0 + jnp.tanh(S / rho))

    acc_rhs = -mu * r * ir3 + (tmax * delta / m) * alpha
    mdot_rhs = -tmax * delta / c
    rdl = jnp.dot(r, lv) / r2
    lr_dot_rhs = mu * ir3 * (lv - 3.0 * r * rdl)
    lm_dot_rhs = -(tmax * delta / (m * m)) * lvn
    return jnp.concatenate([acc_rhs, jnp.array([mdot_rhs]), lr_dot_rhs, jnp.array([lm_dot_rhs])])


def _rhs8_point_radial_srp(q, mu, tmax, c, rho, k_srp):
    r = q[0:3]
    m = q[3]
    lv = q[4:7]
    lm = q[7]
    r2 = jnp.dot(r, r)
    rn = jnp.sqrt(r2)
    ir3 = 1.0 / (r2 * rn)
    lvn = jnp.sqrt(jnp.dot(lv, lv) + 1e-30)
    alpha = -lv / lvn
    S = c * lvn / m + lm - 1.0
    delta = 0.5 * (1.0 + jnp.tanh(S / rho))

    mk = mu - k_srp
    acc_rhs = -mk * r * ir3 + (tmax * delta / m) * alpha
    mdot_rhs = -tmax * delta / c
    rdl = jnp.dot(r, lv) / r2
    lr_dot_rhs = mk * ir3 * (lv - 3.0 * r * rdl)
    lm_dot_rhs = -(tmax * delta / (m * m)) * lvn
    return jnp.concatenate([acc_rhs, jnp.array([mdot_rhs]), lr_dot_rhs, jnp.array([lm_dot_rhs])])


def _rhs8_point_attitude_srp(q, mu, tmax, c, rho, k_srp):
    r = q[0:3]
    m = q[3]
    lv = q[4:7]
    lm = q[7]
    r2 = jnp.dot(r, r)
    rn = jnp.sqrt(r2)
    ir3 = 1.0 / (r2 * rn)
    rhat = r / rn
    lvn = jnp.sqrt(jnp.dot(lv, lv) + 1e-30)
    alpha = -lv / lvn
    S = c * lvn / m + lm - 1.0
    delta = 0.5 * (1.0 + jnp.tanh(S / rho))

    u_dot = jnp.dot(rhat, alpha)
    phi = 0.5 + 0.5 * (u_dot ** 2)
    a_p = k_srp * phi * ir3 * r
    dphi_dr = (u_dot / rn) * (alpha - u_dot * rhat)
    # (da_p/dr)^T lv = k*phi*ir3*(lv - 3*rhat*(rhat.lv)) + k*ir3*(r.lv)*dphi_dr
    rdl = jnp.dot(r, lv) / r2
    jp_t_lv = k_srp * phi * ir3 * (lv - 3.0 * r * rdl) + k_srp * ir3 * jnp.dot(r, lv) * dphi_dr

    acc_rhs = -mu * r * ir3 + a_p + (tmax * delta / m) * alpha
    mdot_rhs = -tmax * delta / c
    lr_dot_rhs = mu * ir3 * (lv - 3.0 * r * rdl) - jp_t_lv
    lm_dot_rhs = -(tmax * delta / (m * m)) * lvn
    return jnp.concatenate([acc_rhs, jnp.array([mdot_rhs]), lr_dot_rhs, jnp.array([lm_dot_rhs])])


_RHS8_BATCH = {
    "none": jax.jit(jax.vmap(_rhs8_point_none, in_axes=(0, None, None, None, None, None))),
    "radial": jax.jit(jax.vmap(_rhs8_point_radial_srp, in_axes=(0, None, None, None, None, None))),
    "attitude_flat_plate": jax.jit(jax.vmap(_rhs8_point_attitude_srp, in_axes=(0, None, None, None, None, None))),
}
_JRHS8_BATCH = {
    "none": jax.jit(jax.vmap(jax.jacfwd(_rhs8_point_none), in_axes=(0, None, None, None, None, None))),
    "radial": jax.jit(jax.vmap(jax.jacfwd(_rhs8_point_radial_srp), in_axes=(0, None, None, None, None, None))),
    "attitude_flat_plate": jax.jit(jax.vmap(jax.jacfwd(_rhs8_point_attitude_srp), in_axes=(0, None, None, None, None, None))),
}


# ---------------------------------------------------------------------------
# Paper PoNN Model (Constrained Expressions + Exact Residual & Jacobian)
# ---------------------------------------------------------------------------
class PaperPoNNModel:
    """8-group X-TFC PoNN model matching D'Ambrosio & Furfaro (Aerospace 2024, 11, 228)."""

    def __init__(
        self,
        prob: ScenarioAProblem,
        n_neurons: int = 70,
        basis_type: str = "chebyshev",     # "chebyshev" | "chebyshev_shifted" | "tanh_elm"
        elm_seed: int = 1000,
        elm_w_range: tuple[float, float] = (-1.0, 1.0),
        elm_b_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.prob = prob
        self.L = int(n_neurons)
        self.basis_type = basis_type
        self.z0 = -1.0
        self.zf = 1.0
        self.b2 = (self.zf - self.z0) / prob.tf
        self.b4 = self.b2 ** 2

        if basis_type == "tanh_elm":
            rng = np.random.default_rng(elm_seed)
            self.W = rng.uniform(elm_w_range[0], elm_w_range[1], self.L)
            self.b = rng.uniform(elm_b_range[0], elm_b_range[1], self.L)
        else:
            self.W = np.ones(self.L, dtype=float)
            self.b = np.zeros(self.L, dtype=float)

    def _eval_raw_basis(self, z: np.ndarray, n_terms: int | None = None):
        L_req = self.L if n_terms is None else n_terms
        if self.basis_type == "chebyshev":
            return chebyshev_basis(z, L_req)
        if self.basis_type == "chebyshev_shifted":
            return chebyshev_basis(z, L_req + 4)
        if self.basis_type == "tanh_elm":
            return tanh_elm_basis(z, self.W, self.b)
        raise ValueError(f"Unknown basis_type: {self.basis_type}")

    def operators(self, z: np.ndarray) -> dict:
        """Build linear constrained-expression operators on collocation grid z."""
        z = np.asarray(z, dtype=float)
        H_all, dH_all, ddH_all = self._eval_raw_basis(z)
        h_ends, dh_ends, _ = self._eval_raw_basis(np.array([self.z0, self.zf]))

        if self.basis_type == "chebyshev_shifted":
            # Exclude support polynomials linearly dependent on boundary constraints:
            # r_i uses T_4..T_{L+3}, m & lm use T_1..T_L, lv uses T_0..T_{L-1}
            Hr, dHr, ddHr = H_all[:, 4:4 + self.L], dH_all[:, 4:4 + self.L], ddH_all[:, 4:4 + self.L]
            hr0, hrf = h_ends[0, 4:4 + self.L], h_ends[1, 4:4 + self.L]
            dhr0, dhrf = dh_ends[0, 4:4 + self.L], dh_ends[1, 4:4 + self.L]

            Hm, dHm = H_all[:, 1:1 + self.L], dH_all[:, 1:1 + self.L]
            hm0, hmf = h_ends[0, 1:1 + self.L], h_ends[1, 1:1 + self.L]

            Hlv, dHlv, ddHlv = H_all[:, :self.L], dH_all[:, :self.L], ddH_all[:, :self.L]
        else:
            Hr, dHr, ddHr = H_all, dH_all, ddH_all
            hr0, hrf = h_ends[0], h_ends[1]
            dhr0, dhrf = dh_ends[0], dh_ends[1]
            Hm, dHm = H_all, dH_all
            hm0, hmf = h_ends[0], h_ends[1]
            Hlv, dHlv, ddHlv = H_all, dH_all, ddH_all

        Om, dOm, ddOm = omega_table_a6(z, self.z0, self.zf)
        om1, om2, om3, om4 = Om[:, 0:1], Om[:, 1:2], Om[:, 2:3], Om[:, 3:4]
        dom1, dom2, dom3, dom4 = dOm[:, 0:1], dOm[:, 1:2], dOm[:, 2:3], dOm[:, 3:4]
        ddom1, ddom2, ddom3, ddom4 = ddOm[:, 0:1], ddOm[:, 1:2], ddOm[:, 2:3], ddOm[:, 3:4]

        # Position, velocity, acceleration constrained bases
        Phi_r = Hr - om1 * hr0[None, :] - om2 * hrf[None, :] - om3 * dhr0[None, :] - om4 * dhrf[None, :]
        Phi_v = self.b2 * (dHr - dom1 * hr0[None, :] - dom2 * hrf[None, :] - dom3 * dhr0[None, :] - dom4 * dhrf[None, :])
        Phi_a = self.b4 * (ddHr - ddom1 * hr0[None, :] - ddom2 * hrf[None, :] - ddom3 * dhr0[None, :] - ddom4 * dhrf[None, :])

        # Particular parts for r, v, a (shape n x 3)
        p = self.prob
        P_r = om1 * p.r0[None, :] + om2 * p.rf[None, :] + (om3 * p.v0[None, :] + om4 * p.vf[None, :]) / self.b2
        P_v = self.b2 * (dom1 * p.r0[None, :] + dom2 * p.rf[None, :]) + dom3 * p.v0[None, :] + dom4 * p.vf[None, :]
        P_a = self.b4 * (ddom1 * p.r0[None, :] + ddom2 * p.rf[None, :]) + self.b2 * (ddom3 * p.v0[None, :] + ddom4 * p.vf[None, :])

        # Mass (Table A1: Om1=1, Om1'=0, m(z0)=m0)
        Phi_m = Hm - hm0[None, :]
        Phi_md = self.b2 * dHm
        P_m = np.full(len(z), p.m0, dtype=float)

        # Velocity costate (unconstrained; lam_r = -lam_v', lam_r' = -lam_v'')
        Phi_lv = Hlv
        Phi_lvd = self.b2 * dHlv
        Phi_lvdd = self.b4 * ddHlv

        # Mass costate (Table A1: Om1=1, Om1'=0, lam_m(zf)=0)
        Phi_lm = Hm - hmf[None, :]
        Phi_lmd = self.b2 * dHm

        # Stack into 8 value bases and 8 LHS-derivative bases for vectorized Jacobian
        # Groups: 0,1,2=r_i ; 3=m ; 4,5,6=lv_i ; 7=lm
        Phi_val = np.stack([Phi_r, Phi_r, Phi_r, Phi_m, Phi_lv, Phi_lv, Phi_lv, Phi_lm], axis=0)          # (8, n, L)
        Phi_lhs = np.stack([Phi_a, Phi_a, Phi_a, Phi_md, -Phi_lvdd, -Phi_lvdd, -Phi_lvdd, Phi_lmd], axis=0)  # (8, n, L)
        P_val = np.column_stack([P_r, P_m[:, None], np.zeros((len(z), 4))])                                 # (n, 8)
        P_lhs = np.column_stack([P_a, np.zeros((len(z), 5))])                                               # (n, 8)

        return {
            "z": z,
            "t_tu": (z - self.z0) / self.b2,
            "Phi_r": Phi_r, "Phi_v": Phi_v, "Phi_a": Phi_a,
            "P_r": P_r, "P_v": P_v, "P_a": P_a,
            "Phi_m": Phi_m, "Phi_md": Phi_md, "P_m": P_m,
            "Phi_lv": Phi_lv, "Phi_lvd": Phi_lvd, "Phi_lvdd": Phi_lvdd,
            "Phi_lm": Phi_lm, "Phi_lmd": Phi_lmd,
            "Phi_val": Phi_val, "Phi_lhs": Phi_lhs, "P_val": P_val, "P_lhs": P_lhs,
        }

    def evaluate(self, xi: np.ndarray, ops: dict | None = None, z: np.ndarray | None = None) -> dict:
        """Evaluate full 14-state trajectory (r, v, a, m, m', lam_r, lam_v, lam_m, S, alpha)."""
        if ops is None:
            if z is None:
                raise ValueError("Provide either ops or z")
            ops = self.operators(z)
        beta = np.asarray(xi, dtype=float).reshape(8, self.L)
        r = ops["Phi_r"] @ beta[0:3].T + ops["P_r"]
        v = ops["Phi_v"] @ beta[0:3].T + ops["P_v"]
        a = ops["Phi_a"] @ beta[0:3].T + ops["P_a"]
        m = ops["Phi_m"] @ beta[3] + ops["P_m"]
        md = ops["Phi_md"] @ beta[3]
        lv = ops["Phi_lv"] @ beta[4:7].T
        lvd = ops["Phi_lvd"] @ beta[4:7].T
        lvdd = ops["Phi_lvdd"] @ beta[4:7].T
        lr = -lvd
        lrd = -lvdd
        lm = ops["Phi_lm"] @ beta[7]
        lmd = ops["Phi_lmd"] @ beta[7]
        lvn = np.linalg.norm(lv, axis=1)
        alpha = -lv / np.maximum(lvn, 1e-12)[:, None]
        S = self.prob.c * lvn / m + lm - 1.0
        Y14 = np.column_stack([r, v, m[:, None], lr, lv, lm[:, None]])
        return {
            "z": ops["z"], "t_tu": ops["t_tu"], "t_days": ops["t_tu"] * self.prob.t_ref_days,
            "r": r, "v": v, "a": a, "m": m, "md": md,
            "lr": lr, "lrd": lrd, "lv": lv, "lvd": lvd, "lvdd": lvdd,
            "lm": lm, "lmd": lmd, "S": S, "alpha": alpha, "Y14": Y14,
        }

    def residual_and_jacobian(self, xi: np.ndarray, ops: dict, rho: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute residual L (8*n,) and exact analytical Jacobian dL/dXi (8*n, 8*L)."""
        beta = np.asarray(xi, dtype=float).reshape(8, self.L)
        q = np.einsum("inl,il->ni", ops["Phi_val"], beta) + ops["P_val"]      # (n, 8)
        lhs = np.einsum("inl,il->ni", ops["Phi_lhs"], beta) + ops["P_lhs"]    # (n, 8)

        mode = self.prob.srp_mode if self.prob.k_srp > 0.0 else "none"
        rhs_fn = _RHS8_BATCH[mode]
        jrhs_fn = _JRHS8_BATCH[mode]
        F = np.asarray(rhs_fn(jnp.asarray(q), self.prob.mu, self.prob.t_max, self.prob.c, float(rho), self.prob.k_srp))
        Jf = np.asarray(jrhs_fn(jnp.asarray(q), self.prob.mu, self.prob.t_max, self.prob.c, float(rho), self.prob.k_srp))

        # Residual layout: (8, n) flattened -> group-major [L_ax, L_ay, L_az, L_m, L_lvx, L_lvy, L_lvz, L_lm]
        R = (lhs - F).T.reshape(-1)

        # Jacobian: J[i, n_idx, j, l] = delta_{ij} * Phi_lhs[i, n_idx, l] - Jf[n_idx, i, j] * Phi_val[j, n_idx, l]
        J = -np.einsum("nij,jnl->injl", Jf, ops["Phi_val"])
        for i in range(8):
            J[i, :, i, :] += ops["Phi_lhs"][i]
        return R, J.reshape(8 * len(ops["z"]), 8 * self.L)

    def residual_only(self, xi: np.ndarray, ops: dict, rho: float) -> np.ndarray:
        beta = np.asarray(xi, dtype=float).reshape(8, self.L)
        q = np.einsum("inl,il->ni", ops["Phi_val"], beta) + ops["P_val"]
        lhs = np.einsum("inl,il->ni", ops["Phi_lhs"], beta) + ops["P_lhs"]
        mode = self.prob.srp_mode if self.prob.k_srp > 0.0 else "none"
        F = np.asarray(_RHS8_BATCH[mode](jnp.asarray(q), self.prob.mu, self.prob.t_max, self.prob.c, float(rho), self.prob.k_srp))
        return (lhs - F).T.reshape(-1)

    def project_trajectory(self, ops: dict, Y14: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
        """Project a reference 14-state trajectory Y14(z) onto the 8-group CE basis by least squares."""
        beta = np.zeros((8, self.L), dtype=float)
        for i in range(3):
            beta[i] = np.linalg.lstsq(ops["Phi_r"], Y14[:, i] - ops["P_r"][:, i], rcond=rcond)[0]
        beta[3] = np.linalg.lstsq(ops["Phi_m"], Y14[:, 6] - ops["P_m"], rcond=rcond)[0]
        for i in range(3):
            beta[4 + i] = np.linalg.lstsq(ops["Phi_lv"], Y14[:, 10 + i], rcond=rcond)[0]
        beta[7] = np.linalg.lstsq(ops["Phi_lm"], Y14[:, 13], rcond=rcond)[0]
        return beta.reshape(-1)

    def initial_guess_paper(
        self,
        ops: dict,
        seed: int,
        init_mode: str = "paper_u01",
    ) -> np.ndarray:
        """Paper Sec. 4.1 initialization:
        - paper_u01: exact Sec. 4.1 beta_1..3, beta_lv1..3 ~ U(0, 1) (seeded)
        - scaled_u01: beta_1..3 ~ 1e-3 * U(0, 1), beta_lv1..3 ~ 1e-2 * U(0, 1)
          (prevents initial Hermite derivative amplification |T_k'(+-1)|=k^2 -> ~25,000 AU)
        - beta_m by least squares so m(t) = m0 (exact => 0)
        - beta_lm by least squares so lam_m(t) = 1 (with exact lam_m(tf) = 0)
        """
        rng = np.random.default_rng(seed)
        beta = np.zeros((8, self.L), dtype=float)
        if init_mode == "paper_u01":
            beta[0:3] = rng.uniform(0.0, 1.0, (3, self.L))
            beta[4:7] = rng.uniform(0.0, 1.0, (3, self.L))
        elif init_mode == "scaled_u01":
            beta[0:3] = 1e-3 * rng.uniform(0.0, 1.0, (3, self.L))
            beta[4:7] = 1e-2 * rng.uniform(0.0, 1.0, (3, self.L))
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")
        beta[3] = np.linalg.lstsq(ops["Phi_m"], np.zeros(len(ops["z"])), rcond=1e-12)[0]
        # Fit lam_m(z) ~ 1 across interior points (z < zf) to avoid singularity at zf where Phi_lm=0
        mask = ops["z"] < self.zf - 1e-12
        beta[7] = np.linalg.lstsq(ops["Phi_lm"][mask], np.ones(int(np.sum(mask))), rcond=1e-4)[0]
        return beta.reshape(-1)


# ---------------------------------------------------------------------------
# Fast QR+SVD Levenberg-Marquardt (MATLAB lsqnonlin-compatible damping)
# ---------------------------------------------------------------------------
def _lm_qr_svd(
    res_jac_fn,
    res_fn,
    x0: np.ndarray,
    scaled: bool = False,
    max_iter: int = 60,
    ftol: float = 1e-9,
    xtol: float = 1e-9,
    gtol: float = 1e-9,
    lam0: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """QR+SVD Levenberg-Marquardt solver without squaring cond(J).

    Matches MATLAB lsqnonlin('levenberg-marquardt') with ScaleProblem='none' (scaled=False)
    or ScaleProblem='jacobian' (scaled=True), handling rank-deficient/zero columns cleanly.
    """
    x = np.array(x0, dtype=float)
    r, J = res_jac_fn(x)
    nfev = 1
    njev = 1
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(J))):
        return x, np.full_like(r, 1e6), nfev, njev, "non-finite initial residual"
    cost = 0.5 * float(np.dot(r, r))
    lam = float(lam0)

    p = len(x)
    diag_idx = np.diag_indices(p)
    for _ in range(max_iter):
        g = J.T @ r
        if np.linalg.norm(g, np.inf) < gtol:
            return x, r, nfev, njev, "gtol"

        if scaled:
            col_norms = np.linalg.norm(J, axis=0)
            d = np.where(col_norms > 1e-12, col_norms, 1.0)
            d2 = d * d
        else:
            d = None
            d2 = np.ones(p, dtype=float)

        JTJ = J.T @ J
        svd_cache = None

        accepted = False
        for _ in range(12):
            step = None
            if svd_cache is None:
                A = JTJ.copy()
                A[diag_idx] += lam * d2
                try:
                    # Fast O(p^3/3) OpenBLAS Cholesky; falls back to QR+SVD if indefinite/ill-conditioned
                    L_chol = np.linalg.cholesky(A)
                    y_tmp = np.linalg.solve(L_chol, -g)
                    step = np.linalg.solve(L_chol.T, y_tmp)
                except np.linalg.LinAlgError:
                    Js = J / d[None, :] if d is not None else J
                    Q, R_mat = np.linalg.qr(Js, mode="reduced")
                    U_r, s, Vt = np.linalg.svd(R_mat, full_matrices=False)
                    Utr = U_r.T @ (Q.T @ r)
                    svd_cache = (s, Vt, Utr)
            if svd_cache is not None:
                s, Vt, Utr = svd_cache
                step = -(Vt.T @ ((s / (s * s + lam)) * Utr))
                if d is not None:
                    step = step / d

            x_new = x + step
            r_new = res_fn(x_new)
            nfev += 1
            if not np.all(np.isfinite(r_new)):
                lam = min(lam * 10.0, 1e12)
                continue
            cost_new = 0.5 * float(np.dot(r_new, r_new))
            if cost_new < cost:
                d_cost = cost - cost_new
                x = x_new
                cost = cost_new
                r, J = res_jac_fn(x)
                njev += 1
                lam = max(lam / 10.0, 1e-12)
                accepted = True
                if d_cost <= ftol * max(cost, 1.0) or np.linalg.norm(step) <= xtol * (1.0 + np.linalg.norm(x)):
                    return x, r, nfev, njev, "converged"
                break
            else:
                lam = min(lam * 10.0, 1e12)
        if not accepted:
            return x, r, nfev, njev, "stalled"
    return x, r, nfev, njev, "max_iter"


# ---------------------------------------------------------------------------
# Result container and Solver
# ---------------------------------------------------------------------------
@dataclass
class PaperPoNNResult:
    success: bool
    status: str
    method_label: str
    solver_algo: str
    basis_type: str
    n_neurons: int
    n_points: int
    xi: np.ndarray
    z: np.ndarray
    t_days: np.ndarray
    Y14: np.ndarray
    S: np.ndarray
    delta: np.ndarray
    alpha: np.ndarray
    fuel_kg: float
    final_mass_kg: float
    residual_norm: float
    residual_rms: float
    residual_max: float
    reprop_arrival_residual: float
    energy_error: float
    switches: list[dict]
    thrust_sequence: str
    rho_reached: float
    total_nfev: int
    total_njev: int
    runtime_s: float
    level_log: list[dict]
    loss_history: list[float]
    refined: bool = False
    extra: dict = field(default_factory=dict)


def make_collocation_grid(n_points: int, grid_type: str = "cgl") -> np.ndarray:
    """Build collocation grid on [-1, 1]: 'uniform' (evenly spaced) or 'cgl' (Chebyshev-Gauss-Lobatto)."""
    if grid_type == "uniform":
        return np.linspace(-1.0, 1.0, int(n_points))
    if grid_type == "cgl":
        return -np.cos(np.pi * np.arange(int(n_points)) / float(int(n_points) - 1))
    raise ValueError(f"Unknown grid_type: {grid_type}")


def build_refined_grid(
    z_coarse: np.ndarray,
    S_coarse: np.ndarray,
    points_per_switch: int = 100,
    refine_grid_mode: str = "uniform_dense",
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Identify switch intervals from sign changes of S on z_coarse and build refined grid.

    Modes:
    - 'switch_bracket': literal Sec. 4.1 adding points_per_switch points inside each [z_k, z_{k+1}] sign change
    - 'uniform_dense': evenly spaced grid with n_new = len(z_coarse) + points_per_switch * n_switches
    - 'cgl_dense': CGL grid with n_new = len(z_coarse) + points_per_switch * n_switches
    """
    intervals: list[tuple[float, float]] = []
    extra_pts = []
    for k in range(len(S_coarse) - 1):
        s0, s1 = float(S_coarse[k]), float(S_coarse[k + 1])
        if (s0 > 0.0 and s1 < 0.0) or (s0 < 0.0 and s1 > 0.0):
            za, zb = float(z_coarse[k]), float(z_coarse[k + 1])
            intervals.append((za, zb))
            extra_pts.append(np.linspace(za, zb, points_per_switch + 2)[1:-1])

    n_sw = len(intervals)
    n_target = len(z_coarse) + points_per_switch * max(n_sw, 1)
    if refine_grid_mode == "switch_bracket":
        z_new = np.unique(np.concatenate([z_coarse] + extra_pts)) if extra_pts else z_coarse.copy()
    elif refine_grid_mode == "uniform_dense":
        z_new = np.linspace(-1.0, 1.0, n_target)
    elif refine_grid_mode == "cgl_dense":
        z_new = -np.cos(np.pi * np.arange(n_target) / float(n_target - 1))
    else:
        raise ValueError(f"Unknown refine_grid_mode: {refine_grid_mode}")
    return z_new, intervals


class PaperPoNNSolver:
    """PoNN solver implementing D'Ambrosio & Furfaro (Aerospace 2024, 11, 228) Sec. 4.1."""

    def __init__(
        self,
        n_neurons: int = 70,
        n_points: int = 140,
        rho_schedule: tuple[float, ...] = DEFAULT_RHO_SCHEDULE,
        solver_algo: str = "lm_svd",            # "lm_svd" | "lm" | "trf"
        basis_type: str = "chebyshev_shifted",  # "chebyshev" | "chebyshev_shifted" | "tanh_elm"
        grid_type: str = "cgl",                 # "uniform" | "cgl"
        init_mode: str = "scaled_u01",          # "paper_u01" | "scaled_u01"
        ftol: float = 1e-9,
        xtol: float = 1e-9,
        gtol: float = 1e-9,
        max_iter_first: int = 250,
        max_iter_cont: int = 25,
        conv_rms_tol: float = 1.5e-2,
        reprop_conv_tol: float = 5e-2,
    ):
        self.L = int(n_neurons)
        self.n = int(n_points)
        self.rho_schedule = tuple(float(r) for r in rho_schedule)
        self.solver_algo = solver_algo
        self.basis_type = basis_type
        self.grid_type = grid_type
        self.init_mode = init_mode
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_iter_first = int(max_iter_first)
        self.max_iter_cont = int(max_iter_cont)
        self.conv_rms_tol = conv_rms_tol
        self.reprop_conv_tol = reprop_conv_tol

    def _run_continuation(
        self,
        prob: ScenarioAProblem,
        model: PaperPoNNModel,
        z: np.ndarray,
        xi_init: np.ndarray,
        rho_schedule: tuple[float, ...],
        start_scaled: bool = False,
    ) -> tuple[np.ndarray, dict[int, np.ndarray], list[dict], list[float], int, int, float]:
        ops = model.operators(z)
        xi = xi_init.copy()
        snapshots: dict[int, np.ndarray] = {}
        level_log = []
        loss_history = []
        tot_nfev = 0
        tot_njev = 0
        rho_reached = float(rho_schedule[0])

        prev_diverged = False
        for k, rho in enumerate(rho_schedule):
            if k == 0 and not start_scaled:
                max_it = self.max_iter_first
            elif prev_diverged:
                max_it = min(5, self.max_iter_cont)
            else:
                max_it = self.max_iter_cont

            def fun(x, rho=rho):
                r = model.residual_only(x, ops, rho)
                if not np.all(np.isfinite(r)):
                    return np.full_like(r, 1e6)
                return r

            def res_jac(x, rho=rho):
                r, J = model.residual_and_jacobian(x, ops, rho)
                if not (np.all(np.isfinite(r)) and np.all(np.isfinite(J))):
                    return np.full_like(r, 1e6), np.zeros_like(J)
                return r, J

            if self.solver_algo == "lm_svd":
                use_scaled = start_scaled or (k > 0)
                xi_next, r_vec, nfev, njev, msg = _lm_qr_svd(
                    res_jac_fn=res_jac,
                    res_fn=fun,
                    x0=xi,
                    scaled=use_scaled,
                    max_iter=max_it,
                    ftol=self.ftol,
                    xtol=self.xtol,
                    gtol=self.gtol,
                    lam0=1e-2 if (k == 0 and not start_scaled) else 1e-4,
                )
            else:
                def jac(x, rho=rho):
                    _, J = res_jac(x, rho)
                    return J

                kw = dict(
                    fun=fun,
                    x0=xi,
                    jac=jac,
                    method=self.solver_algo,
                    ftol=self.ftol,
                    xtol=self.xtol,
                    gtol=self.gtol,
                    max_nfev=max_it,
                )
                if self.solver_algo == "trf":
                    kw["x_scale"] = "jac"
                try:
                    ls = least_squares(**kw)
                    xi_next = ls.x
                    nfev = int(ls.nfev)
                    njev = int(ls.njev) if ls.njev is not None else 0
                    r_vec = ls.fun
                    msg = str(ls.message)
                except Exception as exc:
                    xi_next = xi
                    nfev = 1
                    njev = 0
                    r_vec = fun(xi)
                    msg = f"exception: {exc}"

            tot_nfev += nfev
            tot_njev += njev
            rnorm = float(np.linalg.norm(r_vec))
            rrms = float(np.sqrt(np.mean(r_vec ** 2)))
            ev_check = model.evaluate(xi_next, ops=ops)
            if np.all(np.isfinite(xi_next)) and np.all(ev_check["m"] > 0.02):
                xi = xi_next
                rho_reached = float(rho)
            prev_diverged = bool(rrms > 0.03 and float(np.min(ev_check["m"])) < 0.08)
            snapshots[k] = xi.copy()
            loss_history.append(rrms)
            level_log.append({
                "step": k,
                "rho": float(rho),
                "residual_norm": rnorm,
                "residual_rms": rrms,
                "fuel_kg": float((prob.m0 - ev_check["m"][-1]) * prob.m_ref_kg),
                "nfev": nfev,
                "njev": njev,
                "message": msg,
            })
        return xi, snapshots, level_log, loss_history, tot_nfev, tot_njev, rho_reached

    def _package_result(
        self,
        prob: ScenarioAProblem,
        model: PaperPoNNModel,
        z: np.ndarray,
        xi: np.ndarray,
        rho_eval: float,
        level_log: list[dict],
        loss_history: list[float],
        tot_nfev: int,
        tot_njev: int,
        runtime_s: float,
        refined: bool = False,
        extra: dict | None = None,
    ) -> PaperPoNNResult:
        ops = model.operators(z)
        r_vec = model.residual_only(xi, ops, rho_eval)
        rnorm = float(np.linalg.norm(r_vec))
        rrms = float(np.sqrt(np.mean(r_vec ** 2)))
        rmax = float(np.max(np.abs(r_vec)))

        z_dense = np.linspace(-1.0, 1.0, 2001)
        ev = model.evaluate(xi, z=z_dense)
        delta_dense = np.array([throttle_paper(float(s), rho_eval) for s in ev["S"]])
        switches = detect_switches(ev["t_days"], ev["S"])
        seq = thrust_sequence_label(ev["S"], switches)

        sol_rp = propagate_control_scenario_a(prob, ev["t_tu"], delta_dense, ev["alpha"])
        if sol_rp.status == 0 and np.all(np.isfinite(sol_rp.y[:, -1])):
            xf = sol_rp.y[:, -1]
            reprop_err = float(np.linalg.norm(np.concatenate([xf[0:3] - prob.rf, xf[3:6] - prob.vf])))
        else:
            reprop_err = float("nan")

        e_err, _ = energy_balance_error_scenario_a(
            prob, ev["t_tu"], ev["r"], ev["v"], delta_dense, ev["alpha"], ev["m"]
        )
        final_mass_kg = float(ev["m"][-1] * prob.m_ref_kg)
        fuel_kg = float(prob.m0_kg - final_mass_kg)
        mass_ok = bool(np.all(ev["m"] > 0.05) and np.all(np.isfinite(ev["Y14"])))
        success = bool(
            mass_ok
            and np.isfinite(rrms)
            and rrms < self.conv_rms_tol
            and np.isfinite(reprop_err)
            and reprop_err < self.reprop_conv_tol
        )
        status = (
            f"converged (rms={rrms:.2e}, reprop={reprop_err:.2e})"
            if success
            else f"not converged (rms={rrms:.2e}, reprop={reprop_err:.2e})"
        )
        return PaperPoNNResult(
            success=success,
            status=status,
            method_label=f"PoNN ({self.basis_type}, {self.grid_type}, {self.solver_algo})",
            solver_algo=self.solver_algo,
            basis_type=self.basis_type,
            n_neurons=model.L,
            n_points=len(z),
            xi=xi,
            z=z,
            t_days=ev["t_days"],
            Y14=ev["Y14"],
            S=ev["S"],
            delta=delta_dense,
            alpha=ev["alpha"],
            fuel_kg=fuel_kg,
            final_mass_kg=final_mass_kg,
            residual_norm=rnorm,
            residual_rms=rrms,
            residual_max=rmax,
            reprop_arrival_residual=reprop_err,
            energy_error=e_err,
            switches=switches,
            thrust_sequence=seq,
            rho_reached=rho_eval,
            total_nfev=tot_nfev,
            total_njev=tot_njev,
            runtime_s=runtime_s,
            level_log=level_log,
            loss_history=loss_history,
            refined=refined,
            extra=extra or {},
        )

    def solve(
        self,
        prob: ScenarioAProblem,
        seed: int = 1000,
        xi_init: np.ndarray | None = None,
        refine: bool = True,
        refine_neurons: int = 80,
        points_per_switch: int = 100,
        refine_grid_mode: str = "uniform_dense",      # "uniform_dense" | "cgl_dense" | "switch_bracket"
        refine_init_mode: str = "pad_from_smooth",    # "pad_from_smooth" | "pad_final_rho" | "random_restart"
        smooth_rho_threshold: float = 0.1,
        skip_refine_if_coarse_diverged: bool = False,
    ) -> tuple[PaperPoNNResult, PaperPoNNResult | None]:
        """Solve Scenario A with optional Sec. 4.1 switch-interval refinement.

        Returns (coarse_result, refined_result_or_None).
        """
        t0 = time.perf_counter()
        z_coarse = make_collocation_grid(self.n, self.grid_type)
        model_c = PaperPoNNModel(prob, n_neurons=self.L, basis_type=self.basis_type, elm_seed=seed)
        ops_c = model_c.operators(z_coarse)
        xi0 = (
            model_c.initial_guess_paper(ops_c, seed=seed, init_mode=self.init_mode)
            if xi_init is None
            else np.array(xi_init, dtype=float)
        )

        xi_c, snaps_c, log_c, hist_c, nfev_c, njev_c, rho_c = self._run_continuation(
            prob, model_c, z_coarse, xi0, self.rho_schedule, start_scaled=False
        )
        t_coarse = time.perf_counter() - t0
        res_coarse = self._package_result(
            prob, model_c, z_coarse, xi_c, rho_c, log_c, hist_c, nfev_c, njev_c, t_coarse, refined=False
        )

        if not refine:
            return res_coarse, None

        if skip_refine_if_coarse_diverged and (
            res_coarse.residual_rms > 0.015
            or res_coarse.final_mass_kg < 50.0
            or len(res_coarse.switches) == 0
        ):
            return res_coarse, None

        # Sec. 4.1 Refinement: identify switch intervals from sign changes of S on z_coarse
        ev_c = model_c.evaluate(xi_c, ops=ops_c)
        z_ref, intervals = build_refined_grid(
            z_coarse, ev_c["S"], points_per_switch=points_per_switch, refine_grid_mode=refine_grid_mode
        )
        t1 = time.perf_counter()
        model_r = PaperPoNNModel(prob, n_neurons=refine_neurons, basis_type=self.basis_type, elm_seed=seed)
        ops_r = model_r.operators(z_ref)

        # Select coarse snapshot for warm-starting refinement
        smooth_idx = 0
        for idx, rho_val in enumerate(self.rho_schedule):
            if rho_val >= smooth_rho_threshold * 0.8:
                smooth_idx = idx
        xi_src = snaps_c.get(smooth_idx, xi_c) if refine_init_mode == "pad_from_smooth" else xi_c

        if refine_init_mode in ("pad_from_smooth", "pad_final_rho") and self.basis_type != "tanh_elm":
            beta_c = xi_src.reshape(8, self.L)
            beta_r0 = np.zeros((8, refine_neurons), dtype=float)
            m_copy = min(self.L, refine_neurons)
            beta_r0[:, :m_copy] = beta_c[:, :m_copy]
            xi_r0 = beta_r0.reshape(-1)
            start_scaled = True
        elif refine_init_mode in ("pad_from_smooth", "pad_final_rho"):
            ev_ref_grid = model_c.evaluate(xi_src, z=z_ref)
            xi_r0 = model_r.project_trajectory(ops_r, ev_ref_grid["Y14"])
            start_scaled = True
        else:
            xi_r0 = model_r.initial_guess_paper(ops_r, seed=seed + 10000, init_mode=self.init_mode)
            start_scaled = False

        if refine_init_mode == "pad_from_smooth":
            rho_ref_sched = self.rho_schedule[smooth_idx + 1:] or (self.rho_schedule[-1],)
        elif refine_init_mode == "pad_final_rho":
            rho_ref_sched = tuple(r for r in self.rho_schedule if r <= 1e-2) or (self.rho_schedule[-1],)
        else:
            rho_ref_sched = self.rho_schedule

        xi_r, _, log_r, hist_r, nfev_r, njev_r, rho_r = self._run_continuation(
            prob, model_r, z_ref, xi_r0, rho_ref_sched, start_scaled=start_scaled
        )
        t_ref = time.perf_counter() - t1
        res_ref = self._package_result(
            prob, model_r, z_ref, xi_r, rho_r, log_r, hist_r,
            nfev_c + nfev_r, njev_c + njev_r, t_coarse + t_ref,
            refined=True,
            extra={
                "switch_intervals_z": intervals,
                "refine_runtime_s": t_ref,
                "refine_nfev": nfev_r,
                "refine_njev": njev_r,
                "refine_init_mode": refine_init_mode,
                "refine_grid_mode": refine_grid_mode,
            },
        )
        return res_coarse, res_ref
