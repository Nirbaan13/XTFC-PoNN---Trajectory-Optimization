"""Indirect Shooting Method for 3D Fuel-Optimal Heliocentric Trajectory Optimization.

Solves the two-point boundary value problem (TPBVP) formulated via Pontryagin's
Maximum Principle (PMP) using DOP853 numerical integration and Levenberg-Marquardt
least-squares root finding for the 7 initial costates:
    p = [lambda_rx(0), lambda_ry(0), lambda_rz(0), lambda_vx(0), lambda_vy(0), lambda_vz(0), lambda_m(0)]

Boundary Conditions (7 equations):
    r(tf) - r_Mars(tf) = 0   (3 position errors)
    v(tf) - v_Mars(tf) = 0   (3 velocity errors)
    lambda_m(tf) = 0         (Transversality condition for free final mass)

Physical Justification for Mars Arrival Angle theta_f = 3.10 rad (177.6 deg):
    Departure angular velocity is omega_E = 1.0 rad/TU. Mars circular angular velocity is
    omega_M = sqrt(mu / 1.524^3) = 0.5315 rad/TU. The mean orbital angular rate along the
    spiral transfer is ~0.725 rad/TU. Over nominal transfer time tf = 248.4 days = 4.2729 TU,
    the expected physical sweep angle is Delta theta = 0.725 * 4.2729 = 3.10 rad (177.6 deg),
    corresponding to a near-half revolution low-thrust spiral.
"""

from dataclasses import dataclass, field
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .constants import (
    MU_EFF_CANON, T_MAX_CANON, C_CANON, C_EXHAUST, SCALE_T, SCALE_L, SCALE_V, SCALE_M,
    SCALE_A, DAYS_PER_TU, R_MARS_CANON, V_MARS_CANON, T_MAX
)
from .dynamics import (
    central_gravity, central_gravity_jacobian,
    optimal_thrust_direction, switching_function, optimal_throttle,
    get_mars_state, mars_arrival_residual,
    compute_energy_conservation_error,
    heliocentric_state_costate_ode
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


class IndirectShootingSolver:
    """Indirect Shooting solver for 3D fuel-optimal Earth-Mars heliocentric transfer."""

    def __init__(
        self,
        flight_time_days: float = 248.4,
        target_mars_angle_rad: float = 3.10,
        rtol: float = 1e-10,
        atol: float = 1e-11,
        max_iter: int = 100,
        t_max_n: float = 1.20,
        hardware_t_max_n: float = T_MAX
    ):
        self.flight_time_days = flight_time_days
        self.tf_canon = flight_time_days / DAYS_PER_TU
        self.target_mars_angle_rad = target_mars_angle_rad
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self.t_max_n = t_max_n
        self.t_max_c = t_max_n / (SCALE_M * SCALE_A)
        self.hardware_t_max_n = hardware_t_max_n

        # Earth departure conditions (3D canonical units)
        self.r0 = np.array([1.0, 0.0, 0.0])
        self.v0 = np.array([0.0, 1.0, 0.0])
        self.m0 = 1.0

        # Mars arrival target (3D canonical units)
        self.rf, self.vf = get_mars_state(target_mars_angle_rad)

    def _get_nominal_costate_guess(self) -> np.ndarray:
        """Nominal 7 initial costates for fuel-optimal Earth-Mars transfer:
        [lambda_rx, lambda_ry, lambda_rz, lambda_vx, lambda_vy, lambda_vz, lambda_m]
        """
        return np.array([
            -1.03027692, -0.33727032, 0.0,
            -0.37460015, -1.14211797, 0.0,
            0.22513572
        ])

    def solve(self, seed: int = 1000, perturb_guess: bool = False) -> TrajectorySolution:
        """Executes indirect shooting to find the 7 initial costates that reach Mars."""
        start_time = time.perf_counter()
        rng = np.random.RandomState(seed)

        lam0_base = self._get_nominal_costate_guess()
        if perturb_guess:
            noise = rng.normal(0.0, 0.03, size=lam0_base.shape)
            lam0_init = lam0_base + noise
        else:
            lam0_init = lam0_base.copy()

        loss_hist = []

        # Homotopy on smoothing parameter epsilon
        eps_stages = [0.05, 0.02]
        p_curr = lam0_init.copy()
        total_fev = 0

        for stage_idx, eps in enumerate(eps_stages):
            def defect(p: np.ndarray) -> np.ndarray:
                y0 = np.concatenate([self.r0, self.v0, [self.m0], p])
                sol = solve_ivp(
                    heliocentric_state_costate_ode,
                    (0.0, self.tf_canon),
                    y0,
                    args=(MU_EFF_CANON, self.t_max_c, C_CANON, eps),
                    method="DOP853",
                    rtol=self.rtol * 10.0,
                    atol=self.atol * 10.0
                )
                rf_calc = sol.y[0:3, -1]
                vf_calc = sol.y[3:6, -1]
                lam_m_f = sol.y[13, -1]
                d = np.concatenate([rf_calc - self.rf, vf_calc - self.vf, [lam_m_f]])
                loss_hist.append(float(np.linalg.norm(d)))
                return d

            res = least_squares(
                defect,
                p_curr,
                method="lm",
                max_nfev=self.max_iter,
                ftol=1e-12,
                xtol=1e-12
            )
            p_curr = res.x
            total_fev += res.nfev

        # Final high-precision integration on dense time grid
        eps_final = eps_stages[-1]
        n_eval = 250
        t_eval = np.linspace(0.0, self.tf_canon, n_eval)
        y0_final = np.concatenate([self.r0, self.v0, [self.m0], p_curr])

        sol_final = solve_ivp(
            heliocentric_state_costate_ode,
            (0.0, self.tf_canon),
            y0_final,
            args=(MU_EFF_CANON, self.t_max_c, C_CANON, eps_final),
            method="DOP853",
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol
        )

        actual_runtime = time.perf_counter() - start_time

        # Extract 3D trajectory profiles
        r_au = sol_final.y[0:3].T
        v_canon = sol_final.y[3:6].T
        v_kms = v_canon * (SCALE_V / 1e3)
        m_canon = sol_final.y[6]
        m_kg = m_canon * SCALE_M

        lam_r = sol_final.y[7:10].T
        lam_v = sol_final.y[10:13].T
        lam_m = sol_final.y[13]

        # Compute thrust profile
        thrust_n = np.zeros(n_eval)
        thrust_vec = np.zeros((n_eval, 3))
        for i in range(n_eval):
            u_i = optimal_thrust_direction(lam_v[i])
            s_i = switching_function(lam_v[i], lam_m[i], m_canon[i], c=C_CANON)
            t_mag_c = optimal_throttle(s_i, t_max=self.t_max_c, smoothing_eps=eps_final)
            thrust_n[i] = t_mag_c * (SCALE_M * SCALE_A)
            thrust_vec[i] = thrust_n[i] * u_i

        # Performance metrics
        xe_used = float(SCALE_M - m_kg[-1])
        final_m_kg = float(m_kg[-1])
        arr_res = mars_arrival_residual(r_au[-1], v_canon[-1], self.rf, self.vf)
        converged = bool(arr_res < 1e-5)

        # Energy conservation error
        energy_err, energy_err_hist = compute_energy_conservation_error(
            t_eval, r_au, v_canon, m_canon,
            thrust_vec / (SCALE_M * SCALE_A)
        )

        t_days = t_eval * DAYS_PER_TU

        return TrajectorySolution(
            method_name="Indirect Shooting",
            converged=converged,
            iterations=total_fev,
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
            loss_history=loss_hist
        )
