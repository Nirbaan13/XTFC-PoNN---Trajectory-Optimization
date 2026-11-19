"""Direct LGL pseudospectral collocation (CasADi + IPOPT).

The paper uses Dymos + IPOPT.  Dymos needs pyOptSparse built against IPOPT
for IPOPT access, which is not pip-installable in this environment, so the
same transcription (global LGL collocation, defect constraints
(D X)_k = (tf/2) f(X_k, u_k, T_k), IPOPT) is implemented directly in CasADi.

Fuel-optimal NLP (fixed tf, fixed Mars state):
    min  m0 - m_N
    s.t. D X - (tf/2) F(X, T, U) = 0           (7 x N defects)
         X_0 = [r0, v0, m0],  [r_N, v_N] = [r_M, v_M]   (hard)
         0 <= T_k <= Tmax,  |U_k| = 1
Soft terminal slack is used ONLY as a last-resort continuation step when
IPOPT fails on the hard problem; the hard problem is then re-solved from the
slack solution and the use of slack is recorded in the result.

Minimum-time NLP (free tf, free Mars arrival longitude theta on the circular
orbit) is used for the feasibility study.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp

from .problem import HelioProblem, MARS_R_CIRC_AU, tu_from_days, days_from_tu
from .helio_dynamics import make_rhs7_control, arrival_residual, energy_balance_error


# ---------------------------------------------------------------------------
# LGL utilities
# ---------------------------------------------------------------------------
def lgl_nodes_and_weights(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """LGL nodes (roots of (1 - tau^2) P'_{N-1}) and quadrature weights."""
    n = n_nodes - 1
    tau = -np.cos(np.pi * np.arange(n_nodes) / n)
    p = np.zeros((n_nodes, n + 1))
    for _ in range(100):
        tau_old = tau.copy()
        p[:, 0] = 1.0
        p[:, 1] = tau
        for k in range(1, n):
            p[:, k + 1] = ((2 * k + 1) * tau * p[:, k] - k * p[:, k - 1]) / (k + 1)
        tau = tau_old - (tau * p[:, n] - p[:, n - 1]) / (n_nodes * p[:, n])
        if np.max(np.abs(tau - tau_old)) < 1e-16:
            break
    p[:, 0] = 1.0
    p[:, 1] = tau
    for k in range(1, n):
        p[:, k + 1] = ((2 * k + 1) * tau * p[:, k] - k * p[:, k - 1]) / (k + 1)
    w = 2.0 / (n * n_nodes * p[:, n] ** 2)
    return tau, w


def lgl_differentiation_matrix(tau: np.ndarray) -> np.ndarray:
    """Spectral differentiation matrix via barycentric weights (stable for large N)."""
    n = len(tau)
    bw = barycentric_weights(tau)
    diff = tau[:, None] - tau[None, :]
    np.fill_diagonal(diff, 1.0)
    D = (bw[None, :] / bw[:, None]) / diff
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    return D


def barycentric_weights(tau: np.ndarray) -> np.ndarray:
    n = len(tau)
    diff = tau[:, None] - tau[None, :]
    np.fill_diagonal(diff, 1.0)
    # log-sum to avoid overflow
    logabs = np.sum(np.log(np.abs(diff)), axis=1)
    sign = np.prod(np.sign(diff), axis=1)
    w = sign * np.exp(-(logabs - logabs.mean()))
    return w / np.max(np.abs(w))


def barycentric_interp(tau: np.ndarray, values: np.ndarray, tau_eval: np.ndarray) -> np.ndarray:
    """Lagrange interpolation of nodal values (N x k) at tau_eval (M,)."""
    bw = barycentric_weights(tau)
    diff = tau_eval[:, None] - tau[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-15)
    diff[exact] = 1.0
    c = bw[None, :] / diff
    out = (c @ values) / c.sum(axis=1)[:, None]
    rows, cols = np.nonzero(exact)
    out[rows] = values[cols]
    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class CollocationResult:
    method: str
    success: bool
    status: str
    tau: np.ndarray
    t: np.ndarray
    X: np.ndarray           # N x 7
    T: np.ndarray           # N
    U: np.ndarray           # N x 3
    costates: np.ndarray    # N x 7 estimated from defect multipliers (covector mapping)
    fuel_canon: float
    terminal_violation: float
    ipopt_iterations: int
    runtime_s: float
    eval_counts: dict
    history: dict
    slack_used: bool = False
    slack_norm: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Seeded initial guess
# ---------------------------------------------------------------------------
def seeded_initial_guess(prob: HelioProblem, tau: np.ndarray, seed: int):
    """Random-perturbed polar-interpolation guess (documented in ASSUMPTIONS.md).

    Transfer angle = theta_f wrapped to (0, 2*pi]; radius blends from |r0| to
    |rf| with a seed-dependent shape exponent and a seed-dependent sinusoidal
    perturbation; thrust magnitude and in-plane direction are seed-random.
    """
    rng = np.random.default_rng(seed)
    s = (tau + 1.0) / 2.0
    th_f = np.arctan2(prob.rf[1], prob.rf[0]) % (2 * np.pi)
    if th_f < 1e-3:
        th_f += 2 * np.pi
    R0, Rf = np.linalg.norm(prob.r0), np.linalg.norm(prob.rf)
    p = rng.uniform(0.6, 1.6)
    amp = rng.uniform(-0.08, 0.08)
    blend = 3 * s ** 2 - 2 * s ** 3
    blend = blend ** p
    rad = R0 + (Rf - R0) * blend + amp * np.sin(np.pi * s)
    th = th_f * s
    zf = prob.rf[2]
    z = zf * s
    ds = 1e-6
    def pos(ss):
        b = (3 * ss ** 2 - 2 * ss ** 3) ** p
        rr = R0 + (Rf - R0) * b + amp * np.sin(np.pi * ss)
        return np.stack([rr * np.cos(th_f * ss), rr * np.sin(th_f * ss), zf * ss], axis=1)
    r = pos(s)
    v = (pos(np.clip(s + ds, 0, 1)) - pos(np.clip(s - ds, 0, 1))) / (
        (np.clip(s + ds, 0, 1) - np.clip(s - ds, 0, 1))[:, None] * prob.tf)
    r[0], v[0] = prob.r0, prob.v0
    r[-1], v[-1] = prob.rf, prob.vf
    frac = rng.uniform(0.3, 0.9)
    T = np.full(len(tau), frac * prob.t_max)
    m = 1.0 - frac * prob.t_max / prob.c * prob.tf * s
    ang = rng.uniform(-0.6, 0.6)
    vhat = v / np.linalg.norm(v, axis=1)[:, None]
    zhat = np.array([0.0, 0.0, 1.0])
    n_hat = np.cross(zhat, vhat)
    U = np.cos(ang) * vhat + np.sin(ang) * n_hat
    U /= np.linalg.norm(U, axis=1)[:, None]
    X = np.hstack([r, v, m[:, None]])
    return X, T, U


# ---------------------------------------------------------------------------
# NLP construction helpers
# ---------------------------------------------------------------------------
def _dyn(X, T, U, mu, k, c):
    r = X[:, 0:3]
    v = X[:, 3:6]
    m = X[:, 6]
    rn2 = ca.sum2(r ** 2)
    rn3 = rn2 * ca.sqrt(rn2)
    acc = ca.repmat(-(mu - k) / rn3, 1, 3) * r + ca.repmat(T / m, 1, 3) * U
    return ca.horzcat(v, acc, -T / c)


def _ipopt_opts(max_iter: int, print_level: int = 0, tol: float = 1e-10):
    return {
        "ipopt.max_iter": max_iter,
        "ipopt.tol": tol,
        "ipopt.constr_viol_tol": 1e-10,
        "ipopt.acceptable_iter": 0,  # disable "solved to acceptable level"
        "ipopt.print_level": print_level,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
        "print_time": False,
        "ipopt.sb": "yes",
    }


class CollocationSolver:
    """180-node global LGL collocation, fuel-optimal, hard terminal constraints."""

    def __init__(self, n_nodes: int = 180, max_iter: int = 70, max_iter_slack: int = 30,
                 allow_slack: bool = True, slack_weight: float = 1e4):
        self.n = n_nodes
        self.max_iter = max_iter
        self.max_iter_slack = max_iter_slack
        self.allow_slack = allow_slack
        self.slack_weight = slack_weight
        self.tau, self.w = lgl_nodes_and_weights(n_nodes)
        self.D = lgl_differentiation_matrix(self.tau)

    # ------------------------------------------------------------------
    def _build(self, prob: HelioProblem, slack: bool):
        N = self.n
        opti = ca.Opti()
        X = opti.variable(N, 7)
        T = opti.variable(N, 1)
        U = opti.variable(N, 3)
        F = _dyn(X, T, U, prob.mu, prob.k_srp, prob.c)
        defect = ca.mtimes(self.D, X) - (prob.tf / 2.0) * F
        g_def = ca.vec(defect) == 0
        opti.subject_to(g_def)
        x0 = np.concatenate([prob.r0, prob.v0, [prob.m0]])
        opti.subject_to(X[0, :].T == x0)
        target = np.concatenate([prob.rf, prob.vf])
        obj = prob.m0 - X[N - 1, 6]
        s = None
        if slack:
            s = opti.variable(6, 1)
            opti.subject_to(X[N - 1, 0:6].T - target == s)
            obj = obj + self.slack_weight * ca.sumsqr(s)
        else:
            opti.subject_to(X[N - 1, 0:6].T == target)
        opti.subject_to(opti.bounded(0.0, T, prob.t_max))
        opti.subject_to(ca.sum2(U ** 2) == 1.0)
        opti.subject_to(X[:, 6] >= 0.05)
        opti.subject_to(ca.sum2(X[:, 0:3] ** 2) >= 0.09)  # |r| >= 0.3 AU guard
        opti.minimize(obj)
        return opti, X, T, U, g_def, s

    def _run(self, prob, guess, slack, max_iter=None):
        opti, X, T, U, g_def, s = self._build(prob, slack)
        Xg, Tg, Ug = guess
        opti.set_initial(X, Xg)
        opti.set_initial(T, Tg)
        opti.set_initial(U, Ug)
        m_it = self.max_iter if max_iter is None else max_iter
        opti.solver("ipopt", _ipopt_opts(m_it))
        try:
            sol = opti.solve()
            ok = True
        except RuntimeError:
            sol = opti.debug
            ok = False
        st = sol.stats()
        Xv = np.array(sol.value(X))
        Tv = np.array(sol.value(T)).ravel()
        Uv = np.array(sol.value(U))
        lam = np.array(sol.value(opti.dual(g_def))).ravel().reshape(7, self.n).T  # vec is column-major
        sv = np.array(sol.value(s)).ravel() if s is not None else np.zeros(6)
        return ok, st, Xv, Tv, Uv, lam, sv

    def solve(self, prob: HelioProblem, seed: int = 1000, guess=None) -> CollocationResult:
        t_start = time.perf_counter()
        if guess is None:
            guess = seeded_initial_guess(prob, self.tau, seed)
        counts = {"nlp_f": 0, "nlp_g": 0, "nlp_grad_f": 0, "nlp_jac_g": 0, "nlp_hess_l": 0}
        hist = {"inf_pr": [], "inf_du": [], "obj": []}

        def acc(st):
            for k in counts:
                counts[k] += int(st.get(f"n_call_{k}", 0))
            it = st.get("iterations", {})
            for k in hist:
                hist[k].extend([float(x) for x in it.get(k, [])])

        ok, st, Xv, Tv, Uv, lam, sv = self._run(prob, guess, slack=False, max_iter=self.max_iter)
        acc(st)
        iters = int(st.get("iter_count", 0))
        status = st.get("return_status", "")
        slack_used = False
        slack_norm = 0.0
        if not ok and self.allow_slack:
            slack_used = True
            ok_s, st_s, Xs, Ts, Us, lam_s, sv = self._run(prob, guess, slack=True, max_iter=self.max_iter_slack)
            acc(st_s)
            iters += int(st_s.get("iter_count", 0))
            slack_norm = float(np.linalg.norm(sv))
            if slack_norm < 1e-4:
                ok, st, Xv, Tv, Uv, lam, _ = self._run(prob, (Xs, Ts, Us), slack=False, max_iter=self.max_iter)
                acc(st)
                iters += int(st.get("iter_count", 0))
                status = st.get("return_status", "") + f" (after slack continuation, slack status: {st_s.get('return_status', '')})"
            else:
                ok = False
                Xv, Tv, Uv, lam = Xs, Ts, Us, lam_s
                status = f"{st.get('return_status', '')} (slack continuation residual |s|={slack_norm:.3e}; hard problem infeasible)"
        runtime = time.perf_counter() - t_start
        target = np.concatenate([prob.rf, prob.vf])
        term_viol = float(np.linalg.norm(Xv[-1, 0:6] - target))
        # covector mapping: lam_k = nu_k / w_k (Lagrange form: lam_m + 1)
        costates = lam / self.w[:, None]
        costates[:, 6] += 1.0
        t = (self.tau + 1.0) * prob.tf / 2.0
        return CollocationResult(
            method="Pseudospectral collocation", success=bool(ok), status=status,
            tau=self.tau, t=t, X=Xv, T=Tv, U=Uv, costates=costates,
            fuel_canon=float(prob.m0 - Xv[-1, 6]), terminal_violation=term_viol,
            ipopt_iterations=iters, runtime_s=runtime, eval_counts=counts, history=hist,
            slack_used=slack_used, slack_norm=slack_norm,
        )


# ---------------------------------------------------------------------------
# Minimum-time problem (free tf, free arrival longitude on the circular orbit)
# ---------------------------------------------------------------------------
def solve_min_time(prob: HelioProblem, n_nodes: int = 180, seed: int = 1000,
                   tf_guess_days: float = 250.0, theta_guess: float = 3.0,
                   max_iter: int = 3000, guess=None):
    """Minimise tf at T <= Tmax with Mars arrival longitude free (circular Mars)."""
    t0 = time.perf_counter()
    if guess is None and n_nodes > 80:
        coarse = solve_min_time(
            prob, n_nodes=80, seed=seed,
            tf_guess_days=tf_guess_days, theta_guess=theta_guess, max_iter=max_iter
        )
        if coarse["success"]:
            tau_fine, _ = lgl_nodes_and_weights(n_nodes)
            X_fine = barycentric_interp(coarse["tau"], coarse["X"], tau_fine)
            T_fine = barycentric_interp(coarse["tau"], coarse["T"][:, None], tau_fine).ravel()
            U_fine = barycentric_interp(coarse["tau"], coarse["U"], tau_fine)
            U_fine /= np.linalg.norm(U_fine, axis=1)[:, None]
            guess = (X_fine, T_fine, U_fine, coarse["tf_TU"], coarse["theta_f_rad"])
    tau, w = lgl_nodes_and_weights(n_nodes)
    D = lgl_differentiation_matrix(tau)
    N = n_nodes
    opti = ca.Opti()
    X = opti.variable(N, 7)
    T = opti.variable(N, 1)
    U = opti.variable(N, 3)
    tf = opti.variable()
    th = opti.variable()
    F = _dyn(X, T, U, prob.mu, prob.k_srp, prob.c)
    opti.subject_to(ca.vec(ca.mtimes(D, X) - (tf / 2.0) * F) == 0)
    x0 = np.concatenate([prob.r0, prob.v0, [prob.m0]])
    opti.subject_to(X[0, :].T == x0)
    R = MARS_R_CIRC_AU
    vm = np.sqrt(prob.mu / R)
    opti.subject_to(X[N - 1, 0] == R * ca.cos(th))
    opti.subject_to(X[N - 1, 1] == R * ca.sin(th))
    opti.subject_to(X[N - 1, 2] == 0)
    opti.subject_to(X[N - 1, 3] == -vm * ca.sin(th))
    opti.subject_to(X[N - 1, 4] == vm * ca.cos(th))
    opti.subject_to(X[N - 1, 5] == 0)
    opti.subject_to(opti.bounded(0.0, T, prob.t_max))
    opti.subject_to(ca.sum2(U ** 2) == 1.0)
    opti.subject_to(X[:, 6] >= 0.05)
    opti.subject_to(ca.sum2(X[:, 0:3] ** 2) >= 0.09)
    opti.subject_to(opti.bounded(tu_from_days(30.0), tf, tu_from_days(2000.0)))
    opti.minimize(tf)
    if guess is None:
        gprob = prob.with_(tf_days=tf_guess_days, theta_f=theta_guess)
        Xg, Tg, Ug = seeded_initial_guess(gprob, tau, seed)
        Tg[:] = prob.t_max
        opti.set_initial(tf, tu_from_days(tf_guess_days))
        opti.set_initial(th, theta_guess)
    else:
        Xg, Tg, Ug, tfg, thg = guess
        opti.set_initial(tf, tfg)
        opti.set_initial(th, thg)
    opti.set_initial(X, Xg)
    opti.set_initial(T, Tg)
    opti.set_initial(U, Ug)
    opti.solver("ipopt", _ipopt_opts(max_iter))
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError:
        sol = opti.debug
        ok = False
    st = sol.stats()
    out = {
        "success": ok, "status": st.get("return_status", ""),
        "tf_TU": float(sol.value(tf)), "tf_days": days_from_tu(float(sol.value(tf))),
        "theta_f_rad": float(sol.value(th)) % (2 * np.pi),
        "X": np.array(sol.value(X)), "T": np.array(sol.value(T)).ravel(),
        "U": np.array(sol.value(U)), "tau": tau,
        "iterations": int(st.get("iter_count", 0)),
        "runtime_s": time.perf_counter() - t0,
        "n_nodes": n_nodes,
    }
    out["fuel_canon"] = float(prob.m0 - out["X"][-1, 6])
    return out


# ---------------------------------------------------------------------------
# Re-propagation of a nodal control history (cross-check)
# ---------------------------------------------------------------------------
def propagate_nodal_control(prob: HelioProblem, t_nodes: np.ndarray, T: np.ndarray, U: np.ndarray,
                            rtol: float = 1e-11, atol: float = 1e-12, tf: float | None = None):
    """Integrate the state equations with DOP853 using a piecewise-linear
    interpolation of the nodal thrust vector (clipped to Tmax)."""
    tv = np.clip(T, 0.0, prob.t_max)[:, None] * U / np.linalg.norm(U, axis=1)[:, None]

    def thrust_fn(t):
        return np.array([np.interp(t, t_nodes, tv[:, i]) for i in range(3)])

    counter = [0]
    rhs = make_rhs7_control(prob, thrust_fn, counter)
    x0 = np.concatenate([prob.r0, prob.v0, [prob.m0]])
    tf = prob.tf if tf is None else tf
    sol = solve_ivp(rhs, (0.0, tf), x0, method="DOP853", rtol=rtol, atol=atol,
                    dense_output=True, max_step=tf / 400)
    return sol
