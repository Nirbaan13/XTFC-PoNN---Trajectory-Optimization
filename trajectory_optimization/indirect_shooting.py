"""Indirect Shooting Method for Low-Thrust Heliocentric Trajectory Optimization.

Solves the two-point boundary value problem (TPBVP) formulated via Pontryagin's
Maximum Principle (PMP) using DOP853 numerical integration and Levenberg-Marquardt
least-squares root finding for initial costates.
"""

from dataclasses import dataclass, field
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .constants import (
    MU_EFF_CANON, T_MAX_CANON, C_CANON, C_EXHAUST, SCALE_T, SCALE_L, SCALE_V, SCALE_M,
    SCALE_A, DAYS_PER_TU, R_MARS_CANON, V_MARS_CANON
)
from .dynamics import (
    central_gravity, central_gravity_jacobian, get_mars_state, mars_arrival_residual,
    compute_energy_conservation_error
)


@dataclass
class TrajectorySolution:
    method_name: str
    converged: bool
    iterations: int
    runtime_seconds: float
    flight_time_days: float
    xenon_used_kg: float
    final_mass_kg: float
    arrival_residual: float
    energy_error: float

    # Time history arrays
    t_days: np.ndarray = field(default_factory=lambda: np.zeros(0))
    r_au: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    v_kms: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    m_kg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thrust_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thrust_vec: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    lambda_r: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    lambda_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    lambda_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    energy_error_hist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss_history: list[float] = field(default_factory=list)


def pmp_state_costate_ode_2d(t: float, y: np.ndarray, mu: float = MU_EFF_CANON) -> np.ndarray:
    """Evaluates the 8-state PMP heliocentric dynamics in 2D canonical units.
    
    y = [rx, ry, vx, vy, lam_rx, lam_ry, lam_vx, lam_vy]
    Control acceleration: a = -lam_v
    """
    r = y[0:2]
    v = y[2:4]
    lam_r = y[4:6]
    lam_v = y[6:8]

    rn = np.linalg.norm(r)
    g = -mu * r / (rn ** 3)

    r_hat = r / rn
    dg_dr = -(mu / (rn ** 3)) * (np.eye(2) - 3.0 * np.outer(r_hat, r_hat))

    rdot = v
    vdot = g - lam_v
    d_lam_r = -dg_dr.T @ lam_v
    d_lam_v = -lam_r

    return np.concatenate([rdot, vdot, d_lam_r, d_lam_v])


class IndirectShootingSolver:
    """Indirect Shooting solver for low-thrust Earth-Mars heliocentric transfer."""

    def __init__(
        self,
        flight_time_days: float = 248.4,
        target_mars_angle_rad: float = 3.10,
        rtol: float = 1e-10,
        atol: float = 1e-11,
        max_iter: int = 100
    ):
        self.flight_time_days = flight_time_days
        self.tf_canon = flight_time_days / DAYS_PER_TU
        self.target_mars_angle_rad = target_mars_angle_rad
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter

        # Earth departure conditions (canonical units)
        self.r0 = np.array([1.0, 0.0])
        self.v0 = np.array([0.0, 1.0])

        # Mars arrival target (canonical units)
        self.rf = np.array([
            R_MARS_CANON * np.cos(target_mars_angle_rad),
            R_MARS_CANON * np.sin(target_mars_angle_rad)
        ])
        self.vf = np.array([
            -V_MARS_CANON * np.sin(target_mars_angle_rad),
            V_MARS_CANON * np.cos(target_mars_angle_rad)
        ])

    def _get_nominal_costate_guess(self) -> np.ndarray:
        """Nominal initial costate guess for energy-optimal Earth-Mars transfer."""
        return np.array([-0.13164666, -0.05216949, -0.05830204, -0.14573942])

    def solve(self, seed: int = 1000, perturb_guess: bool = False) -> TrajectorySolution:
        """Executes indirect shooting to find initial costates that reach Mars."""
        start_time = time.perf_counter()
        rng = np.random.RandomState(seed)

        lam0_base = self._get_nominal_costate_guess()
        if perturb_guess:
            noise = rng.normal(0.0, 0.04, size=lam0_base.shape)
            lam0_init = lam0_base + noise
        else:
            lam0_init = lam0_base.copy()

        loss_hist = []

        def defect(p: np.ndarray) -> np.ndarray:
            y0 = np.concatenate([self.r0, self.v0, p])
            sol = solve_ivp(
                pmp_state_costate_ode_2d,
                (0.0, self.tf_canon),
                y0,
                method="DOP853",
                rtol=self.rtol,
                atol=self.atol
            )
            rf_calc = sol.y[0:2, -1]
            vf_calc = sol.y[2:4, -1]
            d = np.concatenate([rf_calc - self.rf, vf_calc - self.vf])
            loss_hist.append(float(np.linalg.norm(d)))
            return d

        # Solve nonlinear TPBVP using Levenberg-Marquardt
        res = least_squares(
            defect,
            lam0_init,
            method="lm",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=self.max_iter * 5
        )

        lam_opt = res.x
        n_iters = res.nfev

        # Propagate full trajectory with the found initial costates
        n_eval_points = 250
        t_eval = np.linspace(0.0, self.tf_canon, n_eval_points)
        y0_final = np.concatenate([self.r0, self.v0, lam_opt])
        final_sol = solve_ivp(
            pmp_state_costate_ode_2d,
            (0.0, self.tf_canon),
            y0_final,
            t_eval=t_eval,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol
        )

        t_days = final_sol.t * DAYS_PER_TU
        r_au = np.zeros((n_eval_points, 3))
        r_au[:, 0:2] = final_sol.y[0:2, :].T

        v_canon = np.zeros((n_eval_points, 3))
        v_canon[:, 0:2] = final_sol.y[2:4, :].T
        v_kms = v_canon * (SCALE_V / 1e3)

        lam_r = np.zeros((n_eval_points, 3))
        lam_r[:, 0:2] = final_sol.y[4:6, :].T

        lam_v = np.zeros((n_eval_points, 3))
        lam_v[:, 0:2] = final_sol.y[6:8, :].T

        # Thrust acceleration a(t) = -lam_v(t)
        a_mag_canon = np.linalg.norm(final_sol.y[6:8, :], axis=0)

        # Cumulative Delta-v
        dv_cum_canon = np.zeros(n_eval_points)
        for i in range(1, n_eval_points):
            dt = final_sol.t[i] - final_sol.t[i - 1]
            dv_cum_canon[i] = dv_cum_canon[i - 1] + 0.5 * (a_mag_canon[i] + a_mag_canon[i - 1]) * dt

        total_dv_si = dv_cum_canon[-1] * SCALE_V
        m_kg = SCALE_M * np.exp(-dv_cum_canon * SCALE_V / C_EXHAUST)
        xe_used = float(SCALE_M - m_kg[-1])
        final_m_val = float(m_kg[-1])

        # Thrust magnitude in Newtons: T = m * a
        thrust_n = (m_kg / SCALE_M) * a_mag_canon * (SCALE_M * SCALE_A)
        thrust_vec = np.zeros((n_eval_points, 3))
        for i in range(n_eval_points):
            if a_mag_canon[i] > 1e-12:
                thrust_vec[i, 0:2] = -(final_sol.y[6:8, i] / a_mag_canon[i]) * thrust_n[i]

        # Mars arrival residual
        r_final_au = r_au[-1]
        v_final_canon = v_canon[-1]
        r_target_3d = np.array([self.rf[0], self.rf[1], 0.0])
        v_target_3d = np.array([self.vf[0], self.vf[1], 0.0])
        arr_res = mars_arrival_residual(r_final_au, v_final_canon, r_target_3d, v_target_3d)

        # Energy conservation error
        energy_err, energy_err_hist = compute_energy_conservation_error(
            final_sol.t, r_au, v_canon, m_kg / SCALE_M,
            thrust_vec / (SCALE_M * SCALE_A)
        )

        converged = bool(res.success and arr_res < 1e-4)
        actual_runtime = time.perf_counter() - start_time

        return TrajectorySolution(
            method_name="Indirect Shooting",
            converged=converged,
            iterations=n_iters,
            runtime_seconds=actual_runtime,
            flight_time_days=self.flight_time_days,
            xenon_used_kg=round(xe_used, 2),
            final_mass_kg=round(final_m_val, 2),
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
            lambda_m=np.zeros(n_eval_points),
            energy_error_hist=energy_err_hist,
            loss_history=loss_hist
        )
