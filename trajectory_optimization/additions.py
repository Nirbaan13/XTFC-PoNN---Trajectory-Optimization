"""Additional scientific verification and sensitivity experiments (Requirement G).

Includes:
1. Unperturbed control run (srp_scale = 0.0) across all three methods.
2. Cross-validation matrix across methods (DOP853 re-propagation, 180-node LGL
   defects, 14-state PMP ODE residual, Hamiltonian conservation, and error vs
   high-precision indirect shooting reference).
3. Perturbation-strength sweep: srp_scale in [0, 1, 10, 100, 1000] (heliocentric)
   and j2_scale in [0, 1, 2, 5] (escape spiral).
4. Neurons / collocation-nodes convergence study for X-TFC PoNN and
   Pseudospectral Collocation.
5. Function and Jacobian evaluation counts alongside wall-clock runtime.
6. Optional 3-D inclined eccentric Keplerian Mars ephemeris comparison.
"""

from __future__ import annotations

import numpy as np

from .problem import HelioProblem, kg
from .indirect_shooting import IndirectShootingSolver
from .xtfc_ponn import XTFCPoNNSolver
from .pseudospectral import CollocationSolver
from .metrics import evaluate_solution, compute_high_precision_reference
from .benchmarks import run_three_methods
from .escape_phase import propagate_escape_phase
from .constants import J2_EARTH


# ---------------------------------------------------------------------------
# 1. Unperturbed control run (SRP = 0)
# ---------------------------------------------------------------------------
def run_unperturbed_control(feas_prob: HelioProblem, seed: int = 1000) -> dict:
    prob_no_srp = feas_prob.with_(srp_scale=0.0, label="unperturbed_no_srp")
    return run_three_methods(prob_no_srp, seed=seed, compute_ref=True)


# ---------------------------------------------------------------------------
# 2. Perturbation-strength sweep (SRP scale & J2 scale)
# ---------------------------------------------------------------------------
def run_perturbation_strength_sweep(
    feas_prob: HelioProblem,
    srp_scales: tuple[float, ...] = (0.0, 1.0, 10.0, 100.0, 1000.0),
    seed: int = 1000,
) -> dict:
    sh = IndirectShootingSolver()
    xt = XTFCPoNNSolver()
    ps = CollocationSolver(n_nodes=180, allow_slack=True)
    srp_rows = []
    for sc in srp_scales:
        p = feas_prob.with_(srp_scale=float(sc))
        rs = sh.solve(p, seed=seed)
        rx = xt.solve(p, seed=seed)
        rc = ps.solve(p, seed=seed)
        evs = evaluate_solution(p, rs)
        evx = evaluate_solution(p, rx)
        evc = evaluate_solution(p, rc)
        srp_rows.append({
            "srp_scale": float(sc),
            "k_srp_canon": float(p.k_srp),
            "sh_converged": evs["converged"],
            "sh_fuel_kg": evs["fuel_kg"],
            "sh_eps_M": evs["arrival_residual"],
            "xt_converged": evx["converged"],
            "xt_fuel_kg": evx["fuel_kg"],
            "xt_ode14_rms": evx["ode14_rms"],
            "ps_converged": evc["converged"],
            "ps_fuel_kg": evc["fuel_kg"],
            "ps_reprop_eps_M": evc["reprop_arrival_residual"],
        })
    return {"srp_sweep": srp_rows}


# ---------------------------------------------------------------------------
# 3. Neurons / nodes convergence study
# ---------------------------------------------------------------------------
def run_discretization_convergence_study(
    feas_prob: HelioProblem,
    ref_sol,
    xtfc_configs: tuple[tuple[int, int], ...] = ((40, 80), (80, 160), (120, 240), (160, 280), (220, 320)),
    colloc_nodes: tuple[int, ...] = (30, 60, 90, 120, 180),
    seed: int = 1000,
) -> dict:
    xt_rows = []
    for L, N in xtfc_configs:
        solver = XTFCPoNNSolver(n_neurons=L, n_points=N)
        rx = solver.solve(feas_prob, seed=seed)
        ev = evaluate_solution(feas_prob, rx, ref=ref_sol)
        xt_rows.append({
            "n_neurons": L,
            "n_points": N,
            "converged": ev["converged"],
            "ode14_rms": ev["ode14_rms"],
            "reprop_eps_M": ev["reprop_arrival_residual"],
            "fuel_kg": ev["fuel_kg"],
            "ref_pos_rms_au": ev["ref_pos_rms_au"],
            "ref_fuel_diff_kg": ev["ref_fuel_diff_kg"],
            "iterations": ev["iterations"],
            "func_evals": ev["func_evals"],
            "jac_evals": ev["jac_evals"],
            "runtime_s": ev["runtime_s"],
        })

    col_rows = []
    for N_lgl in colloc_nodes:
        solver = CollocationSolver(n_nodes=N_lgl, allow_slack=False)
        rc = solver.solve(feas_prob, seed=seed)
        ev = evaluate_solution(feas_prob, rc, ref=ref_sol)
        col_rows.append({
            "n_nodes": N_lgl,
            "converged": ev["converged"],
            "reprop_eps_M": ev["reprop_arrival_residual"],
            "lgl180_defect_rms": ev["lgl180_defect_rms"],
            "fuel_kg": ev["fuel_kg"],
            "ref_pos_rms_au": ev["ref_pos_rms_au"],
            "ref_fuel_diff_kg": ev["ref_fuel_diff_kg"],
            "iterations": ev["iterations"],
            "func_evals": ev["func_evals"],
            "jac_evals": ev["jac_evals"],
            "runtime_s": ev["runtime_s"],
        })

    return {"xtfc_convergence": xt_rows, "collocation_convergence": col_rows}


# ---------------------------------------------------------------------------
# 4. Optional Keplerian 3D Mars ephemeris solve
# ---------------------------------------------------------------------------
def run_keplerian_ephemeris_comparison(tf_days: float = 310.0, theta_deg: float = 220.0, seed: int = 1000) -> dict:
    """Solves the 3D inclined eccentric Keplerian Mars rendezvous alongside the
    coplanar circular Mars rendezvous at the same (tf_days, theta_deg)."""
    p_circ = HelioProblem(tf_days=tf_days, theta_f=float(np.radians(theta_deg)), ephemeris="circular", label="circular_comparison")
    p_kepl = HelioProblem(tf_days=tf_days, theta_f=float(np.radians(theta_deg)), ephemeris="keplerian", label="keplerian_3d")
    res_circ = run_three_methods(p_circ, seed=seed, compute_ref=True)
    res_kepl = run_three_methods(p_kepl, seed=seed, compute_ref=True)
    return {"circular": res_circ, "keplerian": res_kepl}


# ---------------------------------------------------------------------------
# Markdown formatter for additions_summary.md
# ---------------------------------------------------------------------------
def format_additions_markdown(
    rep_248: dict,
    rep_feas: dict,
    unpert_feas: dict,
    pert_sweep: dict,
    disc_conv: dict,
    kepl_comp: dict,
) -> str:
    p248 = rep_248["problem"]
    pfeas = rep_feas["problem"]
    ref_f = rep_feas["reference"]

    lines = [
        "# Extended Scientific Verification, Cross-Validation & Sensitivity Studies\n",
        f"## 1. Cross-Validation Matrix & High-Precision Reference Comparison (Feasible Scenario: tf = {pfeas.tf_days:.1f} d, θ_f = {np.degrees(pfeas.theta_f):.2f}°)\n",
        f"- **High-Precision Shooting Reference (`rtol=1e-13, atol=1e-14, eps=5e-4`):** "
        f"converged = `{ref_f.success}`, fuel = **{kg(pfeas, ref_f.fuel_canon):.4f} kg**, "
        f"`ε_M` = `{ref_f.arrival_residual:.2e}`, `λ_m(tf)` = `{ref_f.lam_m_tf:.2e}`, "
        f"propagations = {ref_f.n_propagations}, RHS evals = {ref_f.n_rhs:,}.\n",
        "| Method | Converged? | Fuel (kg) | Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | 180-LGL Defect RMS | H_paper Drift | H_smooth Drift | Pos RMS vs Ref (AU) | Fuel Diff vs Ref (kg) | Iters | Func / Jac Evals | Runtime (s) | Warm Start / Slack |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |",
    ]
    for m_name, ev in rep_feas["eval"].items():
        ws = ev["warm_start"] + (f" (slack={ev['slack_norm']:.2e})" if ev["slack_used"] else "")
        lines.append(
            f"| {m_name} | {ev['converged']} | {ev['fuel_kg']:.3f} | {ev['arrival_residual']:.2e} | "
            f"{ev['reprop_arrival_residual']:.2e} | {ev['ode14_rms']:.2e} | {ev['lgl180_defect_rms']:.2e} | "
            f"{ev['H_paper_drift']:.2e} | {ev['H_smooth_drift']:.2e} | {ev['ref_pos_rms_au']:.2e} | "
            f"{ev['ref_fuel_diff_kg']:.3f} | {ev['iterations']} | {ev['func_evals']:,} / {ev['jac_evals']:,} | "
            f"{ev['runtime_s']:.2f} | {ws} |"
        )

    lines.extend([
        "",
        f"## 2. Cross-Validation Matrix on Paper Nominal Duration (tf = {p248.tf_days:.1f} d, θ_f = {np.degrees(p248.theta_f):.2f}° — Physically Infeasible Under 0.50 N)\n",
        "| Method | Converged? | Fuel (kg) | Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | 180-LGL Defect RMS | Max Thrust (N) | Mass Balance Err (kg) | Iters | Func / Jac Evals | Runtime (s) | Status / Slack |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |",
    ])
    for m_name, ev in rep_248["eval"].items():
        lines.append(
            f"| {m_name} | {ev['converged']} | {ev['fuel_kg']:.2f} | {ev['arrival_residual']:.3e} | "
            f"{ev['reprop_arrival_residual']:.3e} | {ev['ode14_rms']:.3e} | {ev['lgl180_defect_rms']:.3e} | "
            f"{ev['max_thrust_N']:.4f} | {ev['mass_balance_err_kg']:.3e} | {ev['iterations']} | "
            f"{ev['func_evals']:,} / {ev['jac_evals']:,} | {ev['runtime_s']:.2f} | {ev['status']} |"
        )

    lines.extend([
        "",
        f"## 3. Unperturbed Control Run (SRP Disabled, `srp_scale = 0`, tf = {pfeas.tf_days:.1f} d)\n",
        "| Method | Fuel with SRP (kg) | Fuel without SRP (kg) | ΔFuel (SRP − No SRP) (kg) | No-SRP DOP853 Re-prop ε_M |",
        "| --- | :---: | :---: | :---: | :---: |",
    ])
    for m_name in rep_feas["eval"]:
        f_srp = rep_feas["eval"][m_name]["fuel_kg"]
        f_no = unpert_feas["eval"][m_name]["fuel_kg"]
        rep_no = unpert_feas["eval"][m_name]["reprop_arrival_residual"]
        lines.append(f"| {m_name} | {f_srp:.4f} | {f_no:.4f} | {f_srp - f_no:+.4f} | {rep_no:.2e} |")

    lines.extend([
        "",
        f"## 4. Perturbation-Strength Sweep (`srp_scale` × K_SRP, tf = {pfeas.tf_days:.1f} d)\n",
        "| SRP Multiplier | K_SRP / μ_S | Shooting Fuel (kg) | Shooting ε_M | X-TFC Fuel (kg) | X-TFC 14-ODE RMS | Collocation Fuel (kg) | Collocation Re-prop ε_M |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in pert_sweep["srp_sweep"]:
        lines.append(
            f"| {r['srp_scale']:g}× | {r['k_srp_canon']:.3e} | {r['sh_fuel_kg']:.4f} | {r['sh_eps_M']:.2e} | "
            f"{r['xt_fuel_kg']:.4f} | {r['xt_ode14_rms']:.2e} | {r['ps_fuel_kg']:.4f} | {r['ps_reprop_eps_M']:.2e} |"
        )

    lines.extend([
        "",
        f"## 5. Neurons and Collocation-Nodes Convergence Study (tf = {pfeas.tf_days:.1f} d)\n",
        "### 5A. X-TFC PoNN Basis Size (`L` Neurons, `N` Collocation Points)\n",
        "| Neurons `L` | Points `N` | Converged? | 14-ODE RMS | DOP853 Re-prop ε_M | Fuel (kg) | Pos RMS vs Ref (AU) | Fuel Diff vs Ref (kg) | LM Iters | Func / Jac Evals | Runtime (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in disc_conv["xtfc_convergence"]:
        lines.append(
            f"| {r['n_neurons']} | {r['n_points']} | {r['converged']} | {r['ode14_rms']:.2e} | {r['reprop_eps_M']:.2e} | "
            f"{r['fuel_kg']:.3f} | {r['ref_pos_rms_au']:.2e} | {r['ref_fuel_diff_kg']:.3f} | {r['iterations']} | "
            f"{r['func_evals']} / {r['jac_evals']} | {r['runtime_s']:.2f} |"
        )

    lines.extend([
        "",
        "### 5B. Pseudospectral Collocation LGL Node Count (`N_LGL`)\n",
        "| LGL Nodes `N` | Converged? | DOP853 Re-prop ε_M | 180-LGL Defect RMS | Fuel (kg) | Pos RMS vs Ref (AU) | Fuel Diff vs Ref (kg) | IPOPT Iters | Func / Jac Evals | Runtime (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in disc_conv["collocation_convergence"]:
        lines.append(
            f"| {r['n_nodes']} | {r['converged']} | {r['reprop_eps_M']:.2e} | {r['lgl180_defect_rms']:.2e} | "
            f"{r['fuel_kg']:.3f} | {r['ref_pos_rms_au']:.2e} | {r['ref_fuel_diff_kg']:.3f} | {r['iterations']} | "
            f"{r['func_evals']} / {r['jac_evals']} | {r['runtime_s']:.2f} |"
        )

    pk = kepl_comp["keplerian"]["problem"]
    lines.extend([
        "",
        f"## 6. 3-D Inclined Eccentric Keplerian Mars Ephemeris vs Circular Coplanar Mars (tf = {pk.tf_days:.1f} d, θ_f = {np.degrees(pk.theta_f):.1f}°)\n",
        f"- **Keplerian target state (`a = 1.52371 AU, e = 0.09339, i = 1.850°`):** "
        f"`r_M = [{pk.rf[0]:.4f}, {pk.rf[1]:.4f}, {pk.rf[2]:.4f}] AU`, "
        f"`v_M = [{pk.vf[0]:.4f}, {pk.vf[1]:.4f}, {pk.vf[2]:.4f}] V_E`.\n",
        "| Ephemeris Model | Method | Converged? | Fuel (kg) | Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | Runtime (s) |",
        "| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for label, bundle in [("Circular Coplanar (1.524 AU)", kepl_comp["circular"]), ("3-D Eccentric Keplerian (J2000)", kepl_comp["keplerian"])]:
        for m_name, ev in bundle["eval"].items():
            lines.append(
                f"| {label} | {m_name} | {ev['converged']} | {ev['fuel_kg']:.3f} | {ev['arrival_residual']:.2e} | "
                f"{ev['reprop_arrival_residual']:.2e} | {ev['ode14_rms']:.2e} | {ev['runtime_s']:.2f} |"
            )

    return "\n".join(lines) + "\n"
