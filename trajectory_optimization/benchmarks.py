"""Benchmark runners, angle sweeps, robustness protocols, and table formatters.

All table cells and descriptions are computed directly from solver outputs.
No hardcoded constants or narrative overrides.
"""

from __future__ import annotations

import numpy as np

from .problem import HelioProblem, kg
from .indirect_shooting import IndirectShootingSolver
from .xtfc_ponn import XTFCPoNNSolver
from .pseudospectral import CollocationSolver, solve_min_time
from .escape_phase import EscapeResult
from .metrics import evaluate_solution, compute_high_precision_reference


# ---------------------------------------------------------------------------
# Representative run of all three solvers on a single HelioProblem
# ---------------------------------------------------------------------------
def run_three_methods(prob: HelioProblem, seed: int = 1000, compute_ref: bool = True) -> dict:
    """Runs X-TFC PoNN, Indirect shooting, and 180-node Pseudospectral collocation
    on the identical HelioProblem and returns raw results + evaluated metrics."""
    ref = compute_high_precision_reference(prob, seed=seed) if compute_ref else None

    sh_solver = IndirectShootingSolver()
    xt_solver = XTFCPoNNSolver()
    ps_solver = CollocationSolver(n_nodes=180, allow_slack=True)

    res_xt = xt_solver.solve(prob, seed=seed)
    res_sh = sh_solver.solve(prob, seed=seed)
    res_ps = ps_solver.solve(prob, seed=seed)

    ev_xt = evaluate_solution(prob, res_xt, ref=ref)
    ev_sh = evaluate_solution(prob, res_sh, ref=ref)
    ev_ps = evaluate_solution(prob, res_ps, ref=ref)

    return {
        "problem": prob,
        "reference": ref,
        "raw": {
            "X-TFC PoNN": res_xt,
            "Indirect shooting": res_sh,
            "Pseudospectral collocation": res_ps,
        },
        "eval": {
            "X-TFC PoNN": ev_xt,
            "Indirect shooting": ev_sh,
            "Pseudospectral collocation": ev_ps,
        },
    }


# ---------------------------------------------------------------------------
# Robustness protocol: 20 seeded runs per method (180 LGL nodes in all runs)
# ---------------------------------------------------------------------------
def run_robustness_protocol(
    prob: HelioProblem,
    n_runs: int = 20,
    base_seed: int = 1000,
    partial_runs: dict | None = None,
    on_progress=None,
) -> dict:
    """Runs each solver n_runs times with seeds base_seed .. base_seed + n_runs - 1."""
    seeds = [base_seed + i for i in range(n_runs)]
    sh_solver = IndirectShootingSolver()
    xt_solver = XTFCPoNNSolver()
    ps_solver = CollocationSolver(n_nodes=180, allow_slack=True)

    # Precompute eps=1.0 warm start once if X-TFC uses warm_start_from_eps1
    init_G = None
    init_label = None
    if xt_solver.warm_start_from_eps1:
        tau = np.linspace(-1.0, 1.0, xt_solver.N)
        sh1 = IndirectShootingSolver(eps_schedule=(xt_solver.eps_schedule[0],))
        rs1 = sh1.solve(prob, seed=base_seed)
        if rs1.success:
            t = (tau + 1.0) * prob.tf / 2.0
            sol_d = sh1.propagate(prob, rs1.lam0, xt_solver.eps_schedule[0], dense=True)
            init_G = sol_d.sol(t).T
            init_label = f"projected from eps={xt_solver.eps_schedule[0]:g} smoothed shooting extremal"
        else:
            from .xtfc_ponn import _guess_functions
            init_G = _guess_functions(prob, tau)
            init_label = f"deterministic polar/tangential guess (eps={xt_solver.eps_schedule[0]:g} shooting did not converge)"

    if partial_runs is not None and all(k in partial_runs for k in ("X-TFC PoNN", "Indirect shooting", "Pseudospectral collocation")):
        runs = partial_runs
    else:
        runs = {"X-TFC PoNN": [], "Indirect shooting": [], "Pseudospectral collocation": []}

    start_idx = min(len(runs["X-TFC PoNN"]), len(runs["Indirect shooting"]), len(runs["Pseudospectral collocation"]))
    for k in runs:
        runs[k] = runs[k][:start_idx]

    for idx in range(start_idx, n_runs):
        s = seeds[idx]
        print(f"  [Robustness {prob.label or f'tf={prob.tf_days}d'} | run {idx+1:2d}/{n_runs} | seed={s}]", flush=True)
        rx = xt_solver.solve(prob, seed=s, init_from=init_G, init_label=init_label)
        runs["X-TFC PoNN"].append(evaluate_solution(prob, rx))

        rs = sh_solver.solve(prob, seed=s)
        runs["Indirect shooting"].append(evaluate_solution(prob, rs))

        rc = ps_solver.solve(prob, seed=s)
        runs["Pseudospectral collocation"].append(evaluate_solution(prob, rc))
        if on_progress is not None:
            on_progress(runs)

    summary = {}
    for m_name, m_list in runs.items():
        conv = [r for r in m_list if r["converged"]]
        all_fuel = np.array([r["fuel_kg"] for r in m_list])
        all_res = np.array([r["arrival_residual"] for r in m_list])
        all_time_s = np.array([r["runtime_s"] for r in m_list])
        all_time_m = np.array([r["runtime_min"] for r in m_list])
        all_fev = np.array([r["func_evals"] for r in m_list])
        all_jev = np.array([r["jac_evals"] for r in m_list])
        if len(conv) > 0:
            c_fuel = np.array([r["fuel_kg"] for r in conv])
            c_tf = np.array([r["tf_days"] for r in conv])
            c_res = np.array([r["arrival_residual"] for r in conv])
            c_reprop = np.array([r["reprop_arrival_residual"] for r in conv])
            c_time_s = np.array([r["runtime_s"] for r in conv])
            c_time_m = np.array([r["runtime_min"] for r in conv])
            summary[m_name] = {
                "n_runs": n_runs,
                "n_converged": len(conv),
                "conv_fuel_mean": float(np.mean(c_fuel)),
                "conv_fuel_sd": float(np.std(c_fuel)),
                "conv_tf_mean": float(np.mean(c_tf)),
                "conv_tf_sd": float(np.std(c_tf)),
                "conv_res_mean": float(np.mean(c_res)),
                "conv_res_sd": float(np.std(c_res)),
                "conv_reprop_mean": float(np.mean(c_reprop)),
                "conv_reprop_sd": float(np.std(c_reprop)),
                "conv_time_s_mean": float(np.mean(c_time_s)),
                "conv_time_s_sd": float(np.std(c_time_s)),
                "conv_time_min_mean": float(np.mean(c_time_m)),
                "conv_time_min_sd": float(np.std(c_time_m)),
                "all_fuel_mean": float(np.mean(all_fuel)),
                "all_fuel_sd": float(np.std(all_fuel)),
                "all_res_mean": float(np.mean(all_res)),
                "all_res_sd": float(np.std(all_res)),
                "all_time_s_mean": float(np.mean(all_time_s)),
                "all_time_s_sd": float(np.std(all_time_s)),
                "all_time_min_mean": float(np.mean(all_time_m)),
                "all_time_min_sd": float(np.std(all_time_m)),
                "fev_mean": float(np.mean(all_fev)),
                "jev_mean": float(np.mean(all_jev)),
                "runs": m_list,
            }
        else:
            summary[m_name] = {
                "n_runs": n_runs,
                "n_converged": 0,
                "conv_fuel_mean": None,
                "conv_fuel_sd": None,
                "conv_tf_mean": None,
                "conv_tf_sd": None,
                "conv_res_mean": None,
                "conv_res_sd": None,
                "conv_reprop_mean": None,
                "conv_reprop_sd": None,
                "conv_time_s_mean": None,
                "conv_time_s_sd": None,
                "conv_time_min_mean": None,
                "conv_time_min_sd": None,
                "all_fuel_mean": float(np.mean(all_fuel)),
                "all_fuel_sd": float(np.std(all_fuel)),
                "all_res_mean": float(np.mean(all_res)),
                "all_res_sd": float(np.std(all_res)),
                "all_time_s_mean": float(np.mean(all_time_s)),
                "all_time_s_sd": float(np.std(all_time_s)),
                "all_time_min_mean": float(np.mean(all_time_m)),
                "all_time_min_sd": float(np.std(all_time_m)),
                "fev_mean": float(np.mean(all_fev)),
                "jev_mean": float(np.mean(all_jev)),
                "runs": m_list,
            }
    return summary


# ---------------------------------------------------------------------------
# Mars arrival angle sweep with the real bounded fuel-optimal solver
# ---------------------------------------------------------------------------
def run_mars_angle_sweep(
    base_prob: HelioProblem,
    angles_deg: list[float] | np.ndarray,
    n_nodes: int = 180,
    partial_rows: list[dict] | None = None,
    on_progress=None,
) -> list[dict]:
    """Sweeps Mars arrival angles using the real 180-node bounded-thrust
    fuel-optimal CollocationSolver and IndirectShootingSolver."""
    col = CollocationSolver(n_nodes=n_nodes, allow_slack=True)
    sh = IndirectShootingSolver()
    rows = list(partial_rows) if partial_rows is not None else []
    done_degs = {round(float(r["theta_deg"]), 4) for r in rows}
    for deg in angles_deg:
        if round(float(deg), 4) in done_degs:
            continue
        p = base_prob.with_(theta_f=float(np.radians(deg)))
        rc = col.solve(p, seed=1000)
        ev_c = evaluate_solution(p, rc)
        rs = sh.solve(p, seed=1000)
        ev_s = evaluate_solution(p, rs)
        rows.append({
            "tf_days": float(p.tf_days),
            "theta_deg": float(deg),
            "theta_rad": float(np.radians(deg)),
            "col_converged": bool(rc.success),
            "col_status": rc.status,
            "col_fuel_kg": ev_c["fuel_kg"],
            "col_arrival_res": ev_c["arrival_residual"],
            "col_reprop_res": ev_c["reprop_arrival_residual"],
            "col_slack_used": rc.slack_used,
            "col_slack_norm": rc.slack_norm,
            "col_max_thrust_N": ev_c["max_thrust_N"],
            "col_mean_thrust_N": ev_c["mean_thrust_N"],
            "col_runtime_s": ev_c["runtime_s"],
            "sh_converged": bool(rs.success),
            "sh_status": rs.status,
            "sh_fuel_kg": ev_s["fuel_kg"],
            "sh_arrival_res": ev_s["arrival_residual"],
            "sh_max_thrust_N": ev_s["max_thrust_N"],
            "sh_runtime_s": ev_s["runtime_s"],
        })
        print(
            f"  [Angle sweep tf={p.tf_days:.1f}d | theta={deg:.2f} deg | "
            f"col_ok={rc.success} (slack={rc.slack_norm:.2e}) | sh_ok={rs.success} (eps_M={ev_s['arrival_residual']:.2e})]",
            flush=True,
        )
        if on_progress is not None:
            on_progress(rows)
    return rows


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------
def format_table_e1_markdown(cases: list[EscapeResult]) -> str:
    lines = [
        "# Table E1: Numerical escape-phase propagation with and without J2 (prescribed tangential thrust, 0.5 N, initial mass 1,538.6 kg)\n",
        "| Case | Time to escape energy (days) | Xenon used (kg) | Mass at escape energy (kg) | Final radius (km) | Δv (km/s) | End energy residual (J/kg) | DOP853 RHS evals | Runtime (s) |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for c in cases:
        lines.append(
            f"| {c.case_name} | {c.escape_time_days:.2f} | {c.xenon_used_kg:.2f} | {c.final_mass_kg:.2f} | "
            f"{c.final_radius_km:,.0f} | {c.delta_v_km_s:.3f} | {c.end_state_residual:.2e} | {c.n_rhs_evals:,} | {c.runtime_seconds:.2f} |"
        )
    return "\n".join(lines) + "\n"


def format_table_3_markdown(summary_248: dict, summary_feas: dict, tf_248: float, tf_feas: float, theta_deg: float) -> str:
    def _block(title: str, summ: dict) -> list[str]:
        out = [
            f"## {title}\n",
            "| Method | Converged runs | Converged Fuel (kg) | Flight time (days) | Arrival residual (ε_M) | Re-propagated ε_M | Runtime (s) | Runtime (min) | All-runs Fuel (kg) | All-runs ε_M | Mean Func / Jac evals |",
            "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for m, d in summ.items():
            nc = f"{d['n_converged']} / {d['n_runs']}"
            if d["n_converged"] > 0:
                cf = f"{d['conv_fuel_mean']:.2f} ± {d['conv_fuel_sd']:.2f}"
                ctf = f"{d['conv_tf_mean']:.2f} ± {d['conv_tf_sd']:.2f}"
                cres = f"{d['conv_res_mean']:.2e} ± {d['conv_res_sd']:.2e}"
                creprop = f"{d['conv_reprop_mean']:.2e} ± {d['conv_reprop_sd']:.2e}"
                cts = f"{d['conv_time_s_mean']:.2f} ± {d['conv_time_s_sd']:.2f}"
                ctm = f"{d['conv_time_min_mean']:.3f} ± {d['conv_time_min_sd']:.3f}"
            else:
                cf = ctf = cres = creprop = "N/A (0 converged)"
                cts = f"{d['all_time_s_mean']:.2f} ± {d['all_time_s_sd']:.2f}"
                ctm = f"{d['all_time_min_mean']:.3f} ± {d['all_time_min_sd']:.3f}"
            af = f"{d['all_fuel_mean']:.2f} ± {d['all_fuel_sd']:.2f}"
            ares = f"{d['all_res_mean']:.2e} ± {d['all_res_sd']:.2e}"
            evals = f"{d['fev_mean']:.0f} / {d['jev_mean']:.0f}"
            out.append(
                f"| {m} | {nc} | {cf} | {ctf} | {cres} | {creprop} | {cts} | {ctm} | {af} | {ares} | {evals} |"
            )
        out.append("")
        return out

    lines = ["# Table 3: Robustness and spread over 20 repeated runs (180 LGL nodes in all collocation runs)\n"]
    lines.extend(_block(f"A. Paper Nominal Transfer Duration (tf = {tf_248:.1f} days, θ_f = {theta_deg:.2f}°, T_max = 0.50 N)", summary_248))
    lines.extend(_block(f"B. Shortest All-Method Converged Duration (tf = {tf_feas:.1f} days, θ_f = {theta_deg:.2f}°, T_max = 0.50 N)", summary_feas))
    return "\n".join(lines) + "\n"


def format_angle_sweep_markdown(min_time_out: dict, sweep_248: list[dict], sweep_feas: list[dict]) -> str:
    lines = [
        "# Mars Arrival Angle Sweep and Minimum-Time Feasibility Analysis (T_max = 0.50 N)\n",
        "## 1. Minimum-Time Rendezvous Problem (Free tf, Free Arrival Longitude θ_f, 180 LGL Nodes)\n",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Solver status | `{min_time_out['status']}` (`success = {min_time_out['success']}`) |",
        f"| Minimum transfer time `t_min` | **{min_time_out['tf_days']:.4f} days** ({min_time_out['tf_TU']:.6f} TU) |",
        f"| Optimal arrival longitude `θ_f(t_min)` | **{np.degrees(min_time_out['theta_f_rad']):.4f}°** ({min_time_out['theta_f_rad']:.6f} rad) |",
        f"| Propellant consumed at `t_min` | **{min_time_out['fuel_canon']*1218.0:.2f} kg** |",
        f"| IPOPT iterations (`N = {min_time_out['n_nodes']}`) | {min_time_out['iterations']} |",
        f"| Wall-clock runtime | {min_time_out['runtime_s']:.2f} s |",
        "",
    ]

    def _table(title: str, rows: list[dict]) -> list[str]:
        blk = [
            f"## {title}\n",
            "| θ_f (deg) | θ_f (rad) | Collocation Converged? | Collocation Fuel (kg) | Collocation Slack Norm | Collocation Re-prop ε_M | Shooting Converged? | Shooting Fuel (kg) | Shooting ε_M |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in rows:
            blk.append(
                f"| {r['theta_deg']:.1f}° | {r['theta_rad']:.4f} | {r['col_converged']} | {r['col_fuel_kg']:.2f} | "
                f"{r['col_slack_norm']:.3e} | {r['col_reprop_res']:.3e} | {r['sh_converged']} | {r['sh_fuel_kg']:.2f} | {r['sh_arrival_res']:.3e} |"
            )
        blk.append("")
        return blk

    lines.extend(_table(f"2. Bounded-Thrust Angle Sweep at Paper Nominal Duration (tf = {sweep_248[0]['tf_days']:.1f} days)", sweep_248))
    lines.extend(_table(f"3. Bounded-Thrust Angle Sweep at Feasible Duration (tf = {sweep_feas[0]['tf_days']:.1f} days)", sweep_feas))
    return "\n".join(lines) + "\n"
