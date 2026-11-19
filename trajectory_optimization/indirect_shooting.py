"""Indirect shooting for the 3-D, 14-state, fuel-optimal heliocentric problem.

Unknowns : z = lambda(t0) = [lam_r0 (3), lam_v0 (3), lam_m0]           (7)
Residual : [r(tf) - r_M, v(tf) - v_M, lam_m(tf)]                        (7)
Throttle : T = Tmax / (1 + exp(S/eps)),  S = (1 - lam_m)/c - |lam_v|/m
Homotopy : eps is reduced along a fixed schedule; each level is solved with
           scipy.optimize.fsolve (MINPACK hybrd, as in the paper).  If fsolve
           does not reach the tolerance at a level, scipy.optimize.least_squares
           (method="lm", MINPACK lmdif) is tried from the same start point.
           Which solver closed each level is recorded.
Integrator: solve_ivp DOP853.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, least_squares

from .problem import HelioProblem
from .helio_dynamics import (
    make_rhs14, arrival_residual, energy_balance_error, smooth_throttle,
    switching_function, smoothed_hamiltonian, hamiltonian,
)

DEFAULT_EPS_SCHEDULE = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001)


def random_initial_costates(seed: int) -> np.ndarray:
    """Documented random costate guess: lam_r0, lam_v0 ~ U(-1, 1)^3, lam_m0 ~ U(0, 1).

    Bounds follow from the canonical scaling: with cost J = int T/c dt and
    c ~ 1, the switching condition |lam_v|/m ~ (1 - lam_m)/c gives |lam_v| = O(1),
    lam_r = -d(lam_v)/dt = O(1) per TU, and lam_m decreases monotonically to
    lam_m(tf) = 0, so lam_m0 >= 0.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), rng.uniform(0, 1, 1)])


@dataclass
class ShootingResult:
    method: str
    success: bool
    status: str
    lam0: np.ndarray
    eps_final_target: float
    eps_reached: float | None
    t: np.ndarray
    Y: np.ndarray                  # dense samples, M x 14
    fuel_canon: float
    arrival_residual: float
    lam_m_tf: float
    nonlinear_iterations: int      # propagations requested by fsolve/LM (function evaluations)
    n_propagations: int
    n_rhs: int
    n_jacobian_builds: int
    runtime_s: float
    level_log: list
    history: list                  # residual norm per propagation
    extra: dict = field(default_factory=dict)


class IndirectShootingSolver:
    def __init__(self, eps_schedule=DEFAULT_EPS_SCHEDULE, rtol: float = 1e-10, atol: float = 1e-12,
                 level_tol: float = 1e-9, maxfev_per_level: int = 150, n_dense: int = 2001):
        self.eps_schedule = tuple(eps_schedule)
        self.rtol, self.atol = rtol, atol
        self.level_tol = level_tol
        self.maxfev = maxfev_per_level
        self.n_dense = n_dense

    # ------------------------------------------------------------------
    def propagate(self, prob: HelioProblem, lam0: np.ndarray, eps: float, dense=False, counter=None):
        y0 = np.concatenate([prob.r0, prob.v0, [prob.m0], lam0])
        rhs = make_rhs14(prob, eps, counter)
        return solve_ivp(rhs, (0.0, prob.tf), y0, method="DOP853", rtol=self.rtol, atol=self.atol,
                         dense_output=dense)

    def residual(self, prob, lam0, eps, counter=None):
        try:
            sol = self.propagate(prob, lam0, eps, counter=counter)
        except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
            return np.full(7, 1e3)
        if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
            return np.full(7, 1e3)
        yf = sol.y[:, -1]
        return np.concatenate([yf[0:3] - prob.rf, yf[3:6] - prob.vf, [yf[13]]])

    # ------------------------------------------------------------------
    def solve(self, prob: HelioProblem, seed: int | None = 1000, lam0_guess: np.ndarray | None = None,
              eps_schedule=None) -> ShootingResult:
        t_start = time.perf_counter()
        sched = tuple(eps_schedule) if eps_schedule is not None else self.eps_schedule
        z = random_initial_costates(seed) if lam0_guess is None else np.array(lam0_guess, float)
        rhs_counter = [0]
        history: list = []
        n_prop = [0]
        jac_builds = 0
        level_log = []
        eps_reached = None
        ok_all = True
        status = ""

        for eps in sched:
            def F(zz, eps=eps):
                n_prop[0] += 1
                r = self.residual(prob, zz, eps, rhs_counter)
                history.append(float(np.linalg.norm(r)))
                return r

            z_start = z.copy()
            nprop0 = n_prop[0]
            zf, info, ier, msg = fsolve(F, z_start, full_output=True, xtol=1e-13, maxfev=self.maxfev)
            jac_builds += 1  # hybrd builds one forward-difference Jacobian then Broyden-updates
            res_norm = float(np.linalg.norm(info["fvec"]))
            used = "fsolve"
            if not (np.all(np.isfinite(zf)) and res_norm < self.level_tol):
                lm = least_squares(F, z_start, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                   max_nfev=self.maxfev * 2)
                res_lm = float(np.linalg.norm(lm.fun))
                jac_builds += int(lm.njev) if lm.njev is not None else 0
                if res_lm < res_norm or not np.isfinite(res_norm):
                    zf, res_norm, used = lm.x, res_lm, "lm"
                else:
                    used = "fsolve (lm fallback did not improve)"
            level_ok = bool(np.all(np.isfinite(zf)) and res_norm < self.level_tol)
            level_log.append({"eps": eps, "solver": used, "residual": res_norm, "ok": level_ok,
                              "propagations": n_prop[0] - nprop0})
            if not level_ok:
                ok_all = False
                status = f"homotopy stopped at eps={eps:g} ({used}, |res|={res_norm:.2e})"
                break
            z = zf
            eps_reached = eps

        eps_eval = eps_reached if eps_reached is not None else sched[0]
        if not ok_all and eps_reached is None:
            z_eval = zf  # report the unconverged last iterate at the first level
        else:
            z_eval = z
        sol = self.propagate(prob, z_eval, eps_eval, dense=True)
        t = np.linspace(0.0, prob.tf, self.n_dense)
        Y = sol.sol(t).T if sol.status == 0 else np.full((self.n_dense, 14), np.nan)
        yf = Y[-1]
        eps_m = arrival_residual(prob, yf[0:3], yf[3:6]) if sol.status == 0 else float("nan")
        lam_m_tf = float(yf[13])
        success = ok_all and eps_m < 1e-6 and abs(lam_m_tf) < 1e-6
        if ok_all:
            status = "converged through full eps schedule" if success else "schedule finished but tolerance not met"
        runtime = time.perf_counter() - t_start
        return ShootingResult(
            method="Indirect shooting", success=bool(success), status=status, lam0=z_eval,
            eps_final_target=sched[-1], eps_reached=eps_reached, t=t, Y=Y,
            fuel_canon=float(prob.m0 - yf[6]), arrival_residual=float(eps_m), lam_m_tf=lam_m_tf,
            nonlinear_iterations=n_prop[0], n_propagations=n_prop[0], n_rhs=rhs_counter[0],
            n_jacobian_builds=jac_builds, runtime_s=runtime, level_log=level_log, history=history,
            extra={"eps_eval": eps_eval},
        )


def shooting_controls(prob: HelioProblem, Y: np.ndarray, eps: float):
    """Thrust magnitude and direction implied by the costates on a dense grid."""
    lv = Y[:, 10:13]
    lvn = np.linalg.norm(lv, axis=1)
    S = (1.0 - Y[:, 13]) / prob.c - lvn / Y[:, 6]
    T = np.array([smooth_throttle(s, prob.t_max, eps) for s in S])
    U = -lv / lvn[:, None]
    return T, U, S
