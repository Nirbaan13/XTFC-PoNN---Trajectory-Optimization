"""Shared evaluation metrics, cross-checks, high-precision reference, and Table 2 perturbations.

All quantities are computed directly from solver outputs with zero hardcoded
target numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    MU_SUN, AU_M, M_PARKING, M0_HELIO,
    T_MAX, P0_SRP, CR_SRP, AREA_TO_MASS,
)
from .problem import HelioProblem, kg, newton, days_from_tu
from .helio_dynamics import (
    arrival_residual, energy_balance_error, smooth_throttle,
    switching_function, hamiltonian, smoothed_hamiltonian, make_rhs14,
)
from .indirect_shooting import IndirectShootingSolver, ShootingResult, shooting_controls
from .xtfc_ponn import XTFCPoNNSolver, XTFCResult
from .pseudospectral import (
    CollocationSolver, CollocationResult,
    lgl_nodes_and_weights, lgl_differentiation_matrix, barycentric_interp,
    propagate_nodal_control,
)


# ---------------------------------------------------------------------------
# High-precision indirect shooting reference solution
# ---------------------------------------------------------------------------
def compute_high_precision_reference(prob: HelioProblem, seed: int = 1000) -> ShootingResult:
    """High-precision indirect shooting reference (DOP853 rtol=1e-13, atol=1e-14,
    homotopy down to eps=5e-4, 4001 dense sample nodes)."""
    sched = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005)
    solver = IndirectShootingSolver(
        eps_schedule=sched, rtol=1e-13, atol=1e-14, level_tol=1e-10,
        maxfev_per_level=400, n_dense=4001
    )
    return solver.solve(prob, seed=seed)


# ---------------------------------------------------------------------------
# Unified evaluation of any solver result + cross-check + reference error
# ---------------------------------------------------------------------------
def evaluate_solution(
    prob: HelioProblem,
    res: ShootingResult | XTFCResult | CollocationResult,
    ref: ShootingResult | None = None,
) -> dict:
    """Computes the 5 primary metrics, thrust/mass bounds, cross-method
    optimality residuals, and comparison against the high-precision reference."""
    t_grid = np.linspace(0.0, prob.tf, 501)
    tau180, _ = lgl_nodes_and_weights(180)
    D180 = lgl_differentiation_matrix(tau180)
    t180 = (tau180 + 1.0) * prob.tf / 2.0

    if isinstance(res, ShootingResult):
        method = res.method
        eps_eval = res.extra.get("eps_eval", 0.001)
        Y_grid = np.array([np.interp(t_grid, res.t, res.Y[:, i]) for i in range(14)]).T
        T_grid, U_grid, S_grid = shooting_controls(prob, Y_grid, eps_eval)
        Y180 = np.array([np.interp(t180, res.t, res.Y[:, i]) for i in range(14)]).T
        T180, U180, _ = shooting_controls(prob, Y180, eps_eval)
        # Re-propagate control with DOP853
        sol_p = propagate_nodal_control(prob, t_grid, T_grid, U_grid)
        xf_p = sol_p.y[:, -1]
        reprop_eps_m = arrival_residual(prob, xf_p[0:3], xf_p[3:6])
        reprop_fuel_kg = kg(prob, prob.m0 - xf_p[6])
        # 14-state ODE residual on grid
        rhs_fn = make_rhs14(prob, eps_eval)
        dY_dt = np.gradient(Y_grid, t_grid, axis=0)
        F_grid = np.array([rhs_fn(tt, yy) for tt, yy in zip(t_grid, Y_grid)])
        ode14_rms = float(np.sqrt(np.mean((dY_dt - F_grid) ** 2)))
        iters = res.nonlinear_iterations
        fevals = res.n_rhs
        jevals = res.n_jacobian_builds
        warm = "none (random costate draw)"
        slack_used = False
        slack_norm = 0.0

    elif isinstance(res, XTFCResult):
        method = res.method
        eps_eval = res.extra.get("eps_eval", 0.001)
        tau_grid = 2.0 * (t_grid / prob.tf) - 1.0
        Y_grid, dY_grid = res.model.evaluate(res.beta, tau_grid)
        T_grid, U_grid, S_grid = shooting_controls(prob, Y_grid, eps_eval)
        Y180, _ = res.model.evaluate(res.beta, tau180)
        T180, U180, _ = shooting_controls(prob, Y180, eps_eval)
        sol_p = propagate_nodal_control(prob, res.t, *shooting_controls(prob, res.Y, eps_eval)[:2])
        xf_p = sol_p.y[:, -1]
        reprop_eps_m = arrival_residual(prob, xf_p[0:3], xf_p[3:6])
        reprop_fuel_kg = kg(prob, prob.m0 - xf_p[6])
        ode14_rms = float(res.residual_rms)
        iters = res.iterations
        fevals = res.n_residual_evals
        jevals = res.n_jacobian_evals
        warm = res.warm_start
        slack_used = False
        slack_norm = 0.0

    elif isinstance(res, CollocationResult):
        method = res.method
        eps_eval = 0.001
        tau_grid = 2.0 * (t_grid / prob.tf) - 1.0
        X_grid = barycentric_interp(res.tau, res.X, tau_grid)
        Lam_grid = barycentric_interp(res.tau, res.costates, tau_grid)
        Y_grid = np.hstack([X_grid, Lam_grid])
        T_grid = np.clip(barycentric_interp(res.tau, res.T[:, None], tau_grid).ravel(), 0.0, prob.t_max)
        U_grid = barycentric_interp(res.tau, res.U, tau_grid)
        U_grid /= np.linalg.norm(U_grid, axis=1)[:, None]
        Y180 = np.hstack([
            barycentric_interp(res.tau, res.X, tau180),
            barycentric_interp(res.tau, res.costates, tau180),
        ])
        T180 = np.clip(barycentric_interp(res.tau, res.T[:, None], tau180).ravel(), 0.0, prob.t_max)
        U180 = barycentric_interp(res.tau, res.U, tau180)
        U180 /= np.linalg.norm(U180, axis=1)[:, None]
        sol_p = propagate_nodal_control(prob, res.t, res.T, res.U)
        xf_p = sol_p.y[:, -1]
        reprop_eps_m = arrival_residual(prob, xf_p[0:3], xf_p[3:6])
        reprop_fuel_kg = kg(prob, prob.m0 - xf_p[6])
        rhs_fn = make_rhs14(prob, eps_eval)
        dY_dt = np.gradient(Y_grid, t_grid, axis=0)
        F_grid = np.array([rhs_fn(tt, yy) for tt, yy in zip(t_grid, Y_grid)])
        ode14_rms = float(np.sqrt(np.mean((dY_dt - F_grid) ** 2)))
        iters = res.ipopt_iterations
        fevals = res.eval_counts.get("nlp_f", 0) + res.eval_counts.get("nlp_g", 0)
        jevals = res.eval_counts.get("nlp_jac_g", 0) + res.eval_counts.get("nlp_hess_l", 0)
        warm = "none (seeded polar interpolation)"
        slack_used = res.slack_used
        slack_norm = res.slack_norm
    else:
        raise TypeError(type(res))

    r_grid = Y_grid[:, 0:3]
    v_grid = Y_grid[:, 3:6]
    m_grid = Y_grid[:, 6]
    thrust_acc_grid = (T_grid / m_grid)[:, None] * U_grid
    eps_E, eps_E_prof = energy_balance_error(prob, t_grid, r_grid, v_grid, thrust_acc_grid)
    eps_M = arrival_residual(prob, r_grid[-1], v_grid[-1])

    # 180-node LGL state defect RMS
    r180 = Y180[:, 0:3]
    v180 = Y180[:, 3:6]
    m180 = Y180[:, 6]
    rn180 = np.linalg.norm(r180, axis=1)
    acc180 = -(prob.mu - prob.k_srp) * r180 / rn180[:, None] ** 3 + (T180 / m180)[:, None] * U180
    F180_7 = np.hstack([v180, acc180, (-T180 / prob.c)[:, None]])
    defect180 = D180 @ Y180[:, 0:7] - (prob.tf / 2.0) * F180_7
    lgl180_defect_rms = float(np.sqrt(np.mean(defect180 ** 2)))
    lgl180_defect_max = float(np.max(np.abs(defect180)))

    # Hamiltonian drift along trajectory
    H_paper = np.array([hamiltonian(prob, eps_eval, y) for y in Y_grid])
    H_smooth = np.array([smoothed_hamiltonian(prob, eps_eval, y) for y in Y_grid])
    H_paper_drift = float(np.max(np.abs(H_paper - H_paper[0])))
    H_smooth_drift = float(np.max(np.abs(H_smooth - H_smooth[0])))

    # Mass balance error: |(m0 - m(tf)) - int (T/c) dt|
    fuel_int = float(np.trapezoid(T_grid / prob.c, t_grid))
    fuel_state = float(prob.m0 - m_grid[-1])
    mass_balance_err_kg = kg(prob, abs(fuel_state - fuel_int))

    # Thrust bound check, throttle switch count, and coast-arc duration
    T_newton = newton(prob, T_grid)
    t_days = days_from_tu(t_grid)
    max_thrust_N = float(np.max(T_newton))
    min_thrust_N = float(np.min(T_newton))
    mean_thrust_N = float(np.mean(T_newton))
    thrust_bound_ok = bool(max_thrust_N <= prob.t_max_newton + 1e-6)
    n_throttle_switches, coast_duration_days = analyze_throttle_profile(
        t_days, T_newton, prob.t_max_newton
    )

    # Error against high-precision shooting reference
    ref_pos_rms_au = float("nan")
    ref_vel_rms_vu = float("nan")
    ref_fuel_diff_kg = float("nan")
    if ref is not None and ref.success:
        Y_ref = np.array([np.interp(t_grid, ref.t, ref.Y[:, i]) for i in range(14)]).T
        ref_pos_rms_au = float(np.sqrt(np.mean(np.sum((r_grid - Y_ref[:, 0:3]) ** 2, axis=1))))
        ref_vel_rms_vu = float(np.sqrt(np.mean(np.sum((v_grid - Y_ref[:, 3:6]) ** 2, axis=1))))
        ref_fuel_diff_kg = float(abs(kg(prob, fuel_state) - kg(prob, ref.fuel_canon)))

    return {
        "method": method,
        "converged": bool(res.success),
        "status": res.status,
        "tf_days": float(prob.tf_days),
        "theta_f_rad": float(prob.theta_f),
        "theta_f_deg": float(np.degrees(prob.theta_f)),
        "fuel_kg": kg(prob, fuel_state),
        "final_mass_kg": kg(prob, m_grid[-1]),
        "arrival_residual": float(eps_M),
        "reprop_arrival_residual": float(reprop_eps_m),
        "reprop_fuel_kg": float(reprop_fuel_kg),
        "energy_error": float(eps_E),
        "max_thrust_N": max_thrust_N,
        "min_thrust_N": min_thrust_N,
        "mean_thrust_N": mean_thrust_N,
        "thrust_bound_ok": thrust_bound_ok,
        "n_throttle_switches": int(n_throttle_switches),
        "coast_duration_days": float(coast_duration_days),
        "mass_balance_err_kg": mass_balance_err_kg,
        "ode14_rms": ode14_rms,
        "lgl180_defect_rms": lgl180_defect_rms,
        "lgl180_defect_max": lgl180_defect_max,
        "H_paper_drift": H_paper_drift,
        "H_smooth_drift": H_smooth_drift,
        "lam_m_tf": float(Y_grid[-1, 13]),
        "iterations": int(iters),
        "func_evals": int(fevals),
        "jac_evals": int(jevals),
        "runtime_s": float(res.runtime_s),
        "runtime_min": float(res.runtime_s / 60.0),
        "warm_start": warm,
        "slack_used": bool(slack_used),
        "slack_norm": float(slack_norm),
        "ref_pos_rms_au": ref_pos_rms_au,
        "ref_vel_rms_vu": ref_vel_rms_vu,
        "ref_fuel_diff_kg": ref_fuel_diff_kg,
        # time series for plotting
        "t_days": t_days,
        "r_au": r_grid,
        "v_vu": v_grid,
        "m_kg": m_grid * prob.m0_kg,
        "T_N": T_newton,
        "U": U_grid,
        "eps_E_profile": eps_E_prof,
    }


def analyze_throttle_profile(
    t_days: np.ndarray,
    T_newton: np.ndarray,
    t_max_newton: float,
    low_frac: float = 0.25,
    high_frac: float = 0.75,
) -> tuple[int, float]:
    """Counts hysteresis-filtered bang-bang throttle switches and coast duration (days).

    A switch is counted whenever the normalized throttle T / T_max transitions
    between the ON state (>= high_frac) and the OFF state (<= low_frac).
    Coast-arc duration is the integrated time with T < 0.5 * T_max.
    """
    if len(T_newton) == 0 or t_max_newton <= 0.0:
        return 0, 0.0
    frac = np.asarray(T_newton, dtype=float) / float(t_max_newton)
    state = None
    switches = 0
    for val in frac:
        if val >= high_frac:
            cur = 1
        elif val <= low_frac:
            cur = 0
        else:
            continue
        if state is None:
            state = cur
        elif cur != state:
            switches += 1
            state = cur
    coast_mask = (frac < 0.5).astype(float)
    coast_days = float(np.trapezoid(coast_mask, t_days))
    return int(switches), coast_days


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score confidence interval for binomial proportion k / n."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * float(np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n))
    return (max(0.0, center - half), min(1.0, center + half))



# ---------------------------------------------------------------------------
# Table 2: Perturbation Magnitudes (J2 and SRP vs thrust and central gravity)
# ---------------------------------------------------------------------------
@dataclass
class PerturbationEntry:
    case_description: str
    distance_str: str
    a_pert_m_s2: float
    a_thrust_max_m_s2: float
    ratio_pert_to_thrust: float
    ratio_pert_to_gravity: float


def compute_table_2_perturbations(final_helio_mass_kg: float = 943.0) -> list[PerturbationEntry]:
    """Computes Table 2 comparing J2 and SRP accelerations against max thrust and central gravity."""
    entries = []
    m_esc = M_PARKING
    a_t_esc = T_MAX / m_esc

    # 1. J2 at 500 km equatorial
    r1 = R_EARTH + 500.0e3
    a_g1 = MU_EARTH / (r1 ** 2)
    a_j2_1 = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH ** 2) / (r1 ** 4)
    entries.append(PerturbationEntry(
        case_description=f"J2, 500 km orbit, equatorial (m = {m_esc:,.1f} kg)",
        distance_str=f"{r1/1e3:,.0f} km",
        a_pert_m_s2=a_j2_1,
        a_thrust_max_m_s2=a_t_esc,
        ratio_pert_to_thrust=a_j2_1 / a_t_esc,
        ratio_pert_to_gravity=a_j2_1 / a_g1,
    ))

    # 2. J2 at r = 17,000 km equatorial
    r2 = 17000.0e3
    a_g2 = MU_EARTH / (r2 ** 2)
    a_j2_2 = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH ** 2) / (r2 ** 4)
    entries.append(PerturbationEntry(
        case_description=f"J2, r = 17,000 km, equatorial (m = {m_esc:,.1f} kg)",
        distance_str="17,000 km",
        a_pert_m_s2=a_j2_2,
        a_thrust_max_m_s2=a_t_esc,
        ratio_pert_to_thrust=a_j2_2 / a_t_esc,
        ratio_pert_to_gravity=a_j2_2 / a_g2,
    ))

    # 3. J2 at geostationary radius (42,164 km) equatorial
    r3 = 42164.0e3
    a_g3 = MU_EARTH / (r3 ** 2)
    a_j2_3 = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH ** 2) / (r3 ** 4)
    entries.append(PerturbationEntry(
        case_description=f"J2, geostationary radius, equatorial (m = {m_esc:,.1f} kg)",
        distance_str="42,164 km",
        a_pert_m_s2=a_j2_3,
        a_thrust_max_m_s2=a_t_esc,
        ratio_pert_to_thrust=a_j2_3 / a_t_esc,
        ratio_pert_to_gravity=a_j2_3 / a_g3,
    ))

    # 4. SRP at 1.000 AU
    r4 = 1.0 * AU_M
    a_g4 = MU_SUN / (r4 ** 2)
    a_srp_4 = P0_SRP * CR_SRP * AREA_TO_MASS
    a_t4 = T_MAX / M0_HELIO
    entries.append(PerturbationEntry(
        case_description=f"SRP, Earth distance (m = {M0_HELIO:,.0f} kg)",
        distance_str="1.000 AU",
        a_pert_m_s2=a_srp_4,
        a_thrust_max_m_s2=a_t4,
        ratio_pert_to_thrust=a_srp_4 / a_t4,
        ratio_pert_to_gravity=a_srp_4 / a_g4,
    ))

    # 5. SRP at Mars distance (1.524 AU)
    r5 = 1.524 * AU_M
    a_g5 = MU_SUN / (r5 ** 2)
    a_srp_5 = a_srp_4 / (1.524 ** 2)
    a_t5 = T_MAX / final_helio_mass_kg
    entries.append(PerturbationEntry(
        case_description=f"SRP, Mars distance (m = {final_helio_mass_kg:,.1f} kg)",
        distance_str="1.524 AU",
        a_pert_m_s2=a_srp_5,
        a_thrust_max_m_s2=a_t5,
        ratio_pert_to_thrust=a_srp_5 / a_t5,
        ratio_pert_to_gravity=a_srp_5 / a_g5,
    ))
    return entries


def format_table_2_markdown(entries: list[PerturbationEntry]) -> str:
    lines = [
        "# Table 2: Size of the J2 and SRP accelerations compared with thrust and central gravity\n",
        "| Case | Distance | Perturbing acceleration (m/s²) | Maximum thrust acceleration (m/s²) | Perturbation ÷ max thrust | Perturbation ÷ central gravity |",
        "| --- | --- | :---: | :---: | :---: | :---: |",
    ]
    for e in entries:
        r_pt = f"{e.ratio_pert_to_thrust:.1f}" if e.ratio_pert_to_thrust >= 10 else (
            f"{e.ratio_pert_to_thrust:.3f}" if e.ratio_pert_to_thrust >= 0.01 else f"{e.ratio_pert_to_thrust:.2e}"
        )
        lines.append(
            f"| {e.case_description} | {e.distance_str} | {e.a_pert_m_s2:.2e} | {e.a_thrust_max_m_s2:.2e} | {r_pt} | {e.ratio_pert_to_gravity:.2e} |"
        )
    return "\n".join(lines) + "\n"
