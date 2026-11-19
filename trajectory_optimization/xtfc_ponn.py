"""X-TFC Pontryagin Neural Network (PoNN) for the 14-state fuel-optimal problem.

Architecture (Schiassi et al. 2020; D'Ambrosio & Furfaro 2024)
---------------------------------------------------------------
* One hidden layer, tanh activation, L neurons, FIXED random input weights
  and biases (extreme learning machine).  The random draw is controlled by
  the run seed.  Hidden-unit centres c_l ~ U(-1, 1) and slopes
  w_l = s_l * U(w_min, w_max) (random sign s_l); bias b_l = -w_l c_l.
* Trainable output weights beta (14 x L): one free function per
  state/costate component.
* TFC constrained expressions on tau in [-1, 1], t = (tau + 1) tf / 2:
    r_i, v_i : y = h beta + (1-tau)/2 (y0 - h(-1) beta) + (1+tau)/2 (yf - h(1) beta)
    m        : y = h beta + (m0 - h(-1) beta)
    lam_m    : y = h beta - h(1) beta           (lam_m(tf) = 0)
    lam_r, lam_v : y = h beta                   (free)
  so the departure state, Mars arrival state, m(0) and lam_m(tf) = 0 hold
  exactly for every beta.
* Loss = mean of squared residuals of all 14 state/costate ODEs at N
  uniformly spaced collocation points in [-1, 1].  The residual Jacobian
  w.r.t. beta is assembled from the analytic derivative of the tanh basis
  and the JAX automatic derivative d f / d y of the 14-state right-hand side.
* Training: Levenberg-Marquardt (damped Gauss-Newton).  LM is used instead
  of Adam because the loss is a nonlinear least-squares problem in beta;
  this is the standard X-TFC training algorithm.  One LM step = one
  recorded iteration.  The paper's 4,300 epochs is used only as the total
  iteration budget; the iterations actually used are reported.
* The throttle is smoothed with the same eps homotopy as indirect shooting.

Initialisation (disclosed in result.warm_start):
- If ``init_from`` is provided, the free functions are fitted by linear
  least squares to that trajectory and ``init_label`` is recorded.
- If ``warm_start_from_eps1`` is True (default), the smooth eps=1.0
  two-point boundary value problem is first solved by single-level
  shooting (fast, ~1 s) and projected onto the random basis as the starting
  point at eps=1.0 before the LM homotopy refines beta down to eps=0.001.
  If the eps=1.0 shooting solve fails (e.g. on an infeasible problem), it
  falls back automatically to the deterministic polar/tangential guess and
  records that fallback in ``warm_start``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp

from .problem import HelioProblem
from .helio_dynamics import arrival_residual
from .indirect_shooting import IndirectShootingSolver

jax.config.update("jax_enable_x64", True)

DEFAULT_EPS_SCHEDULE = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001)


def _f14(y, mu_k, tmax, c, eps):
    r, v, m = y[0:3], y[3:6], y[6]
    lr, lv, lm = y[7:10], y[10:13], y[13]
    r2 = jnp.dot(r, r)
    rn = jnp.sqrt(r2)
    ir3 = 1.0 / (r2 * rn)
    lvn = jnp.sqrt(jnp.dot(lv, lv))
    u = -lv / lvn
    S = (1.0 - lm) / c - lvn / m
    T = 0.5 * tmax * (1.0 - jnp.tanh(0.5 * S / eps))
    acc = -mu_k * r * ir3 + T / m * u
    rdl = jnp.dot(r, lv) / r2
    jtl = -mu_k * ir3 * (lv - 3.0 * r * rdl)
    return jnp.concatenate([v, acc, jnp.array([-T / c]), -jtl, -lr, jnp.array([-T / m ** 2 * lvn])])


_f14_batch = jax.jit(jax.vmap(_f14, in_axes=(0, None, None, None, None)))
_jf14_batch = jax.jit(jax.vmap(jax.jacfwd(_f14), in_axes=(0, None, None, None, None)))


@dataclass
class XTFCResult:
    method: str
    success: bool
    status: str
    tau: np.ndarray
    t: np.ndarray
    Y: np.ndarray            # N x 14 at collocation points
    beta: np.ndarray
    fuel_canon: float
    arrival_residual: float  # on the network's own solution (exact by construction)
    residual_rms: float      # RMS of all 14 ODE residuals (final level)
    residual_max: float
    eps_reached: float | None
    iterations: int
    loss_history: list
    level_log: list
    runtime_s: float
    n_residual_evals: int
    n_jacobian_evals: int
    n_neurons: int
    n_points: int
    warm_start: str
    extra: dict = field(default_factory=dict)
    model: "XTFCModel | None" = None


class XTFCModel:
    """Holds the random basis and constrained expressions; evaluates Y(tau)."""

    def __init__(self, prob: HelioProblem, n_neurons: int, seed: int, w_min=1.0, w_max=None):
        rng = np.random.default_rng(seed)
        self.L = n_neurons
        w_max = w_max if w_max is not None else max(10.0, 0.18 * n_neurons)
        centres = rng.uniform(-1.0, 1.0, n_neurons)
        slopes = rng.uniform(w_min, w_max, n_neurons) * rng.choice([-1.0, 1.0], n_neurons)
        self.W = slopes
        self.b = -slopes * centres
        self.prob = prob
        y0 = np.concatenate([prob.r0, prob.v0])
        yf = np.concatenate([prob.rf, prob.vf])
        self.y0, self.yf = y0, yf

    def basis(self, tau):
        z = np.tanh(np.outer(tau, self.W) + self.b)
        return z, (1.0 - z ** 2) * self.W

    def operators(self, tau):
        """A (14, N, L), Ad (14, N, L), particular parts P (N, 14), Pd (N, 14)."""
        H, Hd = self.basis(tau)
        h0, hf = self.basis(np.array([-1.0]))[0][0], self.basis(np.array([1.0]))[0][0]
        N = len(tau)
        p1, p2 = (1 - tau) / 2, (1 + tau) / 2
        A = np.zeros((14, N, self.L))
        Ad = np.zeros((14, N, self.L))
        P = np.zeros((N, 14))
        Pd = np.zeros((N, 14))
        for i in range(6):
            A[i] = H - np.outer(p1, h0) - np.outer(p2, hf)
            Ad[i] = Hd + 0.5 * h0[None, :] - 0.5 * hf[None, :]
            P[:, i] = p1 * self.y0[i] + p2 * self.yf[i]
            Pd[:, i] = -0.5 * self.y0[i] + 0.5 * self.yf[i]
        A[6] = H - h0[None, :]
        Ad[6] = Hd
        P[:, 6] = self.prob.m0
        for i in range(7, 13):
            A[i] = H
            Ad[i] = Hd
        A[13] = H - hf[None, :]
        Ad[13] = Hd
        return A, Ad, P, Pd

    def evaluate(self, beta, tau):
        A, Ad, P, Pd = self.operators(tau)
        Y = np.einsum("inl,il->ni", A, beta) + P
        dY = (np.einsum("inl,il->ni", Ad, beta) + Pd) * 2.0 / self.prob.tf
        return Y, dY


def _guess_functions(prob: HelioProblem, tau: np.ndarray):
    """Deterministic state/costate guess (independent of the run seed)."""
    s = (tau + 1) / 2
    th_f = np.arctan2(prob.rf[1], prob.rf[0]) % (2 * np.pi)
    if th_f < 1e-3:
        th_f += 2 * np.pi
    R0, Rf = np.linalg.norm(prob.r0), np.linalg.norm(prob.rf)

    def pos(ss):
        b = 3 * ss ** 2 - 2 * ss ** 3
        rr = R0 + (Rf - R0) * b
        return np.stack([rr * np.cos(th_f * ss), rr * np.sin(th_f * ss), prob.rf[2] * ss], axis=1)

    ds = 1e-6
    r = pos(s)
    v = (pos(np.minimum(s + ds, 1)) - pos(np.maximum(s - ds, 0))) / (
        (np.minimum(s + ds, 1) - np.maximum(s - ds, 0))[:, None] * prob.tf)
    m = 1.0 - 0.5 * prob.t_max / prob.c * prob.tf * s
    vhat = v / np.linalg.norm(v, axis=1)[:, None]
    lv = -(m / prob.c)[:, None] * vhat
    t = s * prob.tf
    lr = -np.gradient(lv, t, axis=0)
    lm = np.zeros_like(s)
    return np.hstack([r, v, m[:, None], lr, lv, lm[:, None]])


class XTFCPoNNSolver:
    def __init__(self, n_neurons: int = 160, n_points: int = 280, eps_schedule=DEFAULT_EPS_SCHEDULE,
                 iter_budget: int = 4300, max_iter_per_level: int = 15, level_tol: float = 1e-4,
                 final_tol: float = 1e-4, warm_start_from_eps1: bool = True, w_max: float | None = None):
        self.L = n_neurons
        self.N = n_points
        self.eps_schedule = tuple(eps_schedule)
        self.iter_budget = iter_budget
        self.max_iter_level = max_iter_per_level
        self.level_tol = level_tol
        self.final_tol = final_tol
        self.warm_start_from_eps1 = warm_start_from_eps1
        self.w_max = w_max

    def solve(self, prob: HelioProblem, seed: int = 1000, init_from: np.ndarray | None = None,
              init_label: str | None = None) -> XTFCResult:
        t_start = time.perf_counter()
        model = XTFCModel(prob, self.L, seed, w_max=self.w_max)
        tau = np.linspace(-1.0, 1.0, self.N)
        t = (tau + 1.0) * prob.tf / 2.0
        A, Ad, P, Pd = model.operators(tau)
        k = 2.0 / prob.tf
        mu_k = prob.mu - prob.k_srp

        # ---- initial beta: least-squares fit of the free functions to the guess
        if init_from is not None:
            G = init_from
            warm_desc = init_label or "caller-supplied trajectory"
        elif self.warm_start_from_eps1:
            sh1 = IndirectShootingSolver(eps_schedule=(self.eps_schedule[0],))
            rs1 = sh1.solve(prob, seed=1000)
            if rs1.success:
                sol_d = sh1.propagate(prob, rs1.lam0, self.eps_schedule[0], dense=True)
                G = sol_d.sol(t).T
                warm_desc = f"projected from eps={self.eps_schedule[0]:g} smoothed shooting extremal"
            else:
                G = _guess_functions(prob, tau)
                warm_desc = f"deterministic polar/tangential guess (eps={self.eps_schedule[0]:g} shooting did not converge)"
        else:
            G = _guess_functions(prob, tau)
            warm_desc = "none (deterministic polar/tangential guess)"

        beta = np.zeros((14, self.L))
        for i in range(14):
            beta[i] = np.linalg.lstsq(A[i], G[:, i] - P[:, i], rcond=1e-13)[0]

        n_res = [0]
        n_jac = [0]

        def residual(beta, eps):
            n_res[0] += 1
            Y = np.einsum("inl,il->ni", A, beta) + P
            dY = (np.einsum("inl,il->ni", Ad, beta) + Pd) * k
            F = np.asarray(_f14_batch(jnp.asarray(Y), mu_k, prob.t_max, prob.c, eps))
            return dY - F, Y

        def jacobian(Y, eps):
            n_jac[0] += 1
            Jf = np.asarray(_jf14_batch(jnp.asarray(Y), mu_k, prob.t_max, prob.c, eps))
            J = -np.einsum("nij,jnl->nijl", Jf, A)
            for i in range(14):
                J[:, i, i, :] += k * Ad[i]
            return J.reshape(self.N * 14, 14 * self.L)

        loss_hist: list = []
        level_log: list = []
        it_total = 0
        eps_reached = None
        status = ""
        R, Y = residual(beta, self.eps_schedule[0])
        for eps in self.eps_schedule:
            R, Y = residual(beta, eps)
            loss = float(np.mean(R ** 2))
            lam = 1e-6
            it_level = 0
            while it_level < self.max_iter_level and it_total < self.iter_budget:
                J = jacobian(Y, eps)
                r = R.reshape(-1)
                JtJ = J.T @ J
                g = J.T @ r
                diag = np.diag(JtJ).copy()
                improved = False
                d_rel = 0.0
                for _ in range(10):
                    try:
                        step = np.linalg.solve(JtJ + lam * (np.diag(diag) + 1e-12 * np.eye(len(g))), -g)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue
                    beta_try = beta + step.reshape(14, self.L)
                    R_try, Y_try = residual(beta_try, eps)
                    loss_try = float(np.mean(R_try ** 2)) if np.all(np.isfinite(R_try)) else np.inf
                    if loss_try < loss:
                        d_rel = (loss - loss_try) / max(loss, 1e-30)
                        beta, R, Y, loss = beta_try, R_try, Y_try, loss_try
                        lam = max(lam / 3.0, 1e-13)
                        improved = True
                        break
                    lam *= 4.0
                it_level += 1
                it_total += 1
                loss_hist.append(loss)
                if not improved or d_rel < 1e-3:
                    break
            rms = float(np.sqrt(loss))
            ok = rms < self.level_tol
            level_log.append({"eps": eps, "iterations": it_level, "rms_residual": rms, "ok": ok})
            if not ok:
                status = f"eps={eps:g}: RMS 14-residual {rms:.2e} > {self.level_tol:g} after {it_level} LM iterations"
                if it_total >= self.iter_budget:
                    status += " (iteration budget exhausted)"
                break
            eps_reached = eps

        rms = float(np.sqrt(np.mean(R ** 2)))
        rmax = float(np.max(np.abs(R)))
        eps_m = arrival_residual(prob, Y[-1, 0:3], Y[-1, 3:6])
        success = (eps_reached == self.eps_schedule[-1]) and rms < self.final_tol and eps_m < 1e-6
        if not status:
            status = "converged through full eps schedule" if success else "tolerance not met"
        res = XTFCResult(
            method="X-TFC PoNN", success=bool(success), status=status, tau=tau, t=t, Y=Y, beta=beta,
            fuel_canon=float(prob.m0 - Y[-1, 6]), arrival_residual=float(eps_m), residual_rms=rms,
            residual_max=rmax, eps_reached=eps_reached, iterations=it_total, loss_history=loss_hist,
            level_log=level_log, runtime_s=time.perf_counter() - t_start, n_residual_evals=n_res[0],
            n_jacobian_evals=n_jac[0], n_neurons=self.L, n_points=self.N,
            warm_start=warm_desc,
            extra={"eps_eval": eps_reached if eps_reached is not None else self.eps_schedule[0]},
        )
        res.model = model
        return res
