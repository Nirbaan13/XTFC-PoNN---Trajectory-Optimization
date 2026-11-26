"""Preregistration Addendum 2: Fresh Baseline Comparison on Scenario A (seeds 8000..8099).

Evaluates 4 arms on identical fresh seeds 8000..8099 (N=100, A/m0 = 0):
  1. multistart_shoot_5: Up to 5 random costate restarts per seed (sub-seeds 10*s + k, k=0..4),
     stopping at first strict success (reprop < 1e-2). Also records single_shoot (k=0 only).
  2. det_energy_shoot:   Deterministic energy-optimal shooting from documented fixed guess
     lam0_fix = [0, 0, 0, -v0/||v0||, 0.5] (5 repeated idle-machine timing trials).
  3. hybrid_rho1:        Frozen coarse PoNN stopped at rho_ext=1.0 -> exact CE costate extraction
     -> indirect shooting continuation (rho=1.0 -> 1e-5).
  4. split_domain:       Frozen coarse PoNN (full coarse run for switch times) -> 5-arc Hermite CE
     SplitDomainPoNN refinement (fixed bang-bang throttle, unknown switch times, analytic Jacobian).

Checkpoints atomically per seed to results/addendum2_baselines_8000_8099.json.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from trajectory_optimization.ponn_paper import (
    PaperPoNNModel,
    PaperPoNNSolver,
    make_collocation_grid,
)
from trajectory_optimization.scenario_a import (
    DEFAULT_RHO_SCHEDULE,
    ScenarioAProblem,
    ScenarioAShootingSolver,
)
from trajectory_optimization.split_domain_ponn import SplitDomainPoNN

OUT_PATH = Path("results/addendum2_baselines_8000_8099.json")
STOP_AT_RHO = 1e-5
SHOOT_SCHED = tuple(r for r in DEFAULT_RHO_SCHEDULE if r >= STOP_AT_RHO - 1e-15)
MAX_RESTARTS = 5


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _dist_stats(vals: list[float]) -> dict[str, float] | None:
    arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not arr:
        return None
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _extract_lam0(model: PaperPoNNModel, xi: np.ndarray) -> np.ndarray:
    ev0 = model.evaluate(xi, z=np.array([-1.0]))
    return np.concatenate([ev0["lr"][0], ev0["lv"][0], [ev0["lm"][0]]])


def _is_strict_shoot(r) -> tuple[bool, bool]:
    rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
    ar = float(r.arrival_residual) if np.isfinite(r.arrival_residual) else float("nan")
    m_f = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
    rho_r = float(r.rho_reached) if r.rho_reached is not None else float("nan")
    phys_ok = bool(m_f > 50.0 and rho_r <= 1e-4 and ar < 1e-6)
    return (
        bool(phys_ok and np.isfinite(rp) and rp < 5e-2),
        bool(phys_ok and np.isfinite(rp) and rp < 1e-2),
    )


def run_deterministic_baseline(n_trials: int = 5) -> dict[str, Any]:
    """Run deterministic energy-optimal shooting from documented fixed prograde guess."""
    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0, label="scenario_a_det_energy")
    v_hat = prob.v0 / np.linalg.norm(prob.v0)
    lam0_fix = np.concatenate([[0.0, 0.0, 0.0], -v_hat, [0.5]])
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)

    trials = []
    last_res = None
    for i in range(n_trials):
        r = shooter.solve(prob, seed=None, lam0_guess=lam0_fix, rho_schedule=SHOOT_SCHED)
        loose, strict = _is_strict_shoot(r)
        trials.append({
            "trial": i,
            "success_loose": loose,
            "success_strict": strict,
            "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
            "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
            "n_propagations": int(r.n_propagations),
            "n_jacobian_builds": int(r.n_jacobian_builds),
            "runtime_s": float(r.runtime_s),
            "status": r.status,
            "switch_times_days": [float(s["t_days"]) for s in r.switches],
        })
        last_res = r
        print(
            f"  [DET trial {i+1}/{n_trials}] strict={strict} fuel={r.fuel_kg:.4f} "
            f"reprop={r.reprop_arrival_residual:.2e} t={r.runtime_s:.2f}s n_prop={r.n_propagations}",
            flush=True,
        )

    return {
        "lam0_fix": [float(x) for x in lam0_fix],
        "description": "lam_r0=[0,0,0], lam_v0=-v0/||v0|| (prograde), lam_m0=0.5; rho=1.0 -> 1e-5",
        "n_trials": n_trials,
        "success_strict": bool(trials[0]["success_strict"]),
        "fuel_kg": trials[0]["fuel_kg"],
        "reprop_arrival_residual": trials[0]["reprop_arrival_residual"],
        "n_propagations": trials[0]["n_propagations"],
        "switch_times_days": trials[0]["switch_times_days"],
        "runtime_stats": _dist_stats([t["runtime_s"] for t in trials]),
        "trials": trials,
    }


_SD_CACHE: dict[tuple[float, ...], SplitDomainPoNN] = {}


def _get_sd(prob: ScenarioAProblem, arcs: list[float]) -> SplitDomainPoNN:
    key = tuple(arcs)
    if key not in _SD_CACHE:
        _SD_CACHE[key] = SplitDomainPoNN(prob, thrust_arcs=arcs, n_neurons_arc=18, n_points_arc=36)
    return _SD_CACHE[key]


def _run_seed(args_tuple: tuple[int, float, list[float]]) -> tuple[int, dict[str, Any]]:
    seed, fuel_ref, sw_ref_days = args_tuple
    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0, label="scenario_a_add2")
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)

    # 1. Multi-start random shooting (up to 5 restarts; attempt 0 is single_shoot)
    attempts = []
    cum_t = 0.0
    cum_nfev = 0
    cum_njev = 0
    ms_strict = False
    ms_loose = False
    first_strict_attempt = None
    best_fuel = None
    best_reprop = None

    for k in range(MAX_RESTARTS):
        sub_seed = 10 * seed + k
        r = shooter.solve(prob, seed=sub_seed, rho_schedule=SHOOT_SCHED)
        loose, strict = _is_strict_shoot(r)
        cum_t += float(r.runtime_s)
        cum_nfev += int(r.n_propagations)
        cum_njev += int(r.n_jacobian_builds)
        att = {
            "attempt": k + 1,
            "sub_seed": sub_seed,
            "success_loose": loose,
            "success_strict": strict,
            "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
            "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
            "n_propagations": int(r.n_propagations),
            "n_jacobian_builds": int(r.n_jacobian_builds),
            "runtime_s": float(r.runtime_s),
        }
        attempts.append(att)
        if loose:
            ms_loose = True
        if strict:
            ms_strict = True
            first_strict_attempt = k + 1
            best_fuel = att["fuel_kg"]
            best_reprop = att["reprop_arrival_residual"]
            break

    single_shoot = {
        "seed": seed,
        "sub_seed": attempts[0]["sub_seed"],
        "success_loose": attempts[0]["success_loose"],
        "success_strict": attempts[0]["success_strict"],
        "fuel_kg": attempts[0]["fuel_kg"],
        "reprop_arrival_residual": attempts[0]["reprop_arrival_residual"],
        "n_propagations": attempts[0]["n_propagations"],
        "n_jacobian_builds": attempts[0]["n_jacobian_builds"],
        "runtime_s": attempts[0]["runtime_s"],
    }

    multistart_shoot_5 = {
        "seed": seed,
        "success_loose": ms_loose,
        "success_strict": ms_strict,
        "n_attempts": len(attempts),
        "first_strict_attempt": first_strict_attempt,
        "fuel_kg": best_fuel,
        "reprop_arrival_residual": best_reprop,
        "total_propagations": cum_nfev,
        "total_jacobian_builds": cum_njev,
        "total_runtime_s": cum_t,
        "attempts": attempts,
    }

    # 2. Hybrid (PoNN rho_ext = 1.0 -> indirect shooting continuation)
    z_c = make_collocation_grid(140, "cgl")
    model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=seed)
    ops_c = model_c.operators(z_c)
    xi0 = model_c.initial_guess_paper(ops_c, seed=seed, init_mode="scaled_u01")
    ponn_solver = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    t0_p1 = time.perf_counter()
    xi_rho1, _, _, _, nf_p1, nj_p1, _ = ponn_solver._run_continuation(
        prob, model_c, z_c, xi0, (1.0,), start_scaled=False
    )
    lam0_rho1 = _extract_lam0(model_c, xi_rho1)
    t_ponn_rho1 = time.perf_counter() - t0_p1

    r_hyb = shooter.solve(prob, seed=None, lam0_guess=lam0_rho1, rho_schedule=SHOOT_SCHED)
    hyb_loose, hyb_strict = _is_strict_shoot(r_hyb)
    hybrid_rho1 = {
        "seed": seed,
        "success_loose": hyb_loose,
        "success_strict": hyb_strict,
        "fuel_kg": float(r_hyb.fuel_kg) if np.isfinite(r_hyb.fuel_kg) else None,
        "reprop_arrival_residual": float(r_hyb.reprop_arrival_residual) if np.isfinite(r_hyb.reprop_arrival_residual) else None,
        "ponn_nfev": int(nf_p1),
        "ponn_njev": int(nj_p1),
        "ponn_runtime_s": float(t_ponn_rho1),
        "shoot_nfev": int(r_hyb.n_propagations),
        "shoot_njev": int(r_hyb.n_jacobian_builds),
        "shoot_runtime_s": float(r_hyb.runtime_s),
        "total_runtime_s": float(t_ponn_rho1 + r_hyb.runtime_s),
    }

    # 3. Split-domain pipeline (full coarse PoNN -> 5-arc SplitDomainPoNN)
    t0_sd_p = time.perf_counter()
    rc, _ = ponn_solver.solve(prob, seed=seed, refine=False)
    t_sd_ponn = time.perf_counter() - t0_sd_p

    sw_days = [float(sw["t_days"]) for sw in rc.switches]
    if len(sw_days) != 4 or not np.isfinite(rc.residual_rms) or rc.residual_rms > 0.1:
        split_domain = {
            "seed": seed,
            "success_loose": False,
            "success_strict": False,
            "passed_tier3_criteria": False,
            "status": f"ponn_coarse_switch_mismatch (n_sw={len(sw_days)})",
            "ponn_n_switches": len(sw_days),
            "fuel_kg": None,
            "fuel_err_kg": None,
            "max_switch_err_days": None,
            "reprop_arrival_residual": None,
            "ponn_runtime_s": float(t_sd_ponn),
            "split_runtime_s": 0.0,
            "total_runtime_s": float(t_sd_ponn),
        }
    else:
        def _eval_ponn(t_tu_arr: np.ndarray) -> np.ndarray:
            z_arr = 2.0 * np.asarray(t_tu_arr, dtype=float) / prob.tf - 1.0
            return model_c.evaluate(rc.xi, z=z_arr)["Y14"]

        init_on = 1.0 if float(rc.S[0]) >= 0.0 else 0.0
        arcs = [init_on if (k % 2 == 0) else (1.0 - init_on) for k in range(len(sw_days) + 1)]
        sd = _get_sd(prob, arcs)
        theta0 = sd.project_trajectory([t / prob.t_ref_days for t in sw_days], _eval_ponn)
        res_sd = sd.solve(theta0, max_iter=80)
        fuel_err = abs(res_sd.fuel_kg - fuel_ref) if np.isfinite(res_sd.fuel_kg) else float("inf")
        sw_err = float(np.max(np.abs(np.asarray(res_sd.switch_times_days) - np.asarray(sw_ref_days))))
        rp_sd = float(res_sd.reprop_arrival_residual) if np.isfinite(res_sd.reprop_arrival_residual) else float("nan")
        sd_loose = bool(np.isfinite(rp_sd) and rp_sd < 5e-2)
        sd_strict = bool(np.isfinite(rp_sd) and rp_sd < 1e-2)
        sd_tier3 = bool(sd_strict and fuel_err <= 0.05 and sw_err <= 0.5)
        split_domain = {
            "seed": seed,
            "success_loose": sd_loose,
            "success_strict": sd_strict,
            "passed_tier3_criteria": sd_tier3,
            "status": res_sd.status,
            "ponn_n_switches": len(sw_days),
            "fuel_kg": float(res_sd.fuel_kg) if np.isfinite(res_sd.fuel_kg) else None,
            "fuel_err_kg": float(fuel_err) if np.isfinite(fuel_err) else None,
            "switch_times_days": res_sd.switch_times_days,
            "max_switch_err_days": float(sw_err) if np.isfinite(sw_err) else None,
            "residual_rms": float(res_sd.residual_rms) if np.isfinite(res_sd.residual_rms) else None,
            "reprop_arrival_residual": rp_sd if np.isfinite(rp_sd) else None,
            "nfev": int(res_sd.nfev),
            "njev": int(res_sd.njev),
            "ponn_runtime_s": float(t_sd_ponn),
            "split_runtime_s": float(res_sd.runtime_s),
            "total_runtime_s": float(t_sd_ponn + res_sd.runtime_s),
        }

    return seed, {
        "seed": seed,
        "single_shoot": single_shoot,
        "multistart_shoot_5": multistart_shoot_5,
        "hybrid_rho1": hybrid_rho1,
        "split_domain": split_domain,
    }


def _exact_mcnemar(a_ok: list[bool], b_ok: list[bool]) -> dict[str, Any]:
    n = len(a_ok)
    n11 = sum(1 for a, b in zip(a_ok, b_ok) if a and b)
    n10 = sum(1 for a, b in zip(a_ok, b_ok) if a and not b)  # A only
    n01 = sum(1 for a, b in zip(a_ok, b_ok) if not a and b)  # B only
    n00 = sum(1 for a, b in zip(a_ok, b_ok) if not a and not b)
    disc = n10 + n01
    p_exact = float(binomtest(min(n10, n01), disc, 0.5, alternative="two-sided").pvalue) if disc > 0 else 1.0
    delta_pp = 100.0 * (sum(a_ok) - sum(b_ok)) / n if n > 0 else 0.0
    return {
        "n": n,
        "a_strict": int(sum(a_ok)),
        "b_strict": int(sum(b_ok)),
        "delta_pp": float(delta_pp),
        "n11": n11,
        "a_only": n10,
        "b_only": n01,
        "n00": n00,
        "p_raw": p_exact,
    }


def _paired_wilcoxon(t_a: list[float], t_b: list[float]) -> dict[str, Any]:
    diffs = np.asarray(t_a, dtype=float) - np.asarray(t_b, dtype=float)
    n = int(len(diffs))
    if n == 0 or np.all(diffs == 0.0):
        return {"n_paired": n, "median_diff_s": 0.0, "stat": 0.0, "p_raw": 1.0}
    res = wilcoxon(diffs, alternative="two-sided")
    return {
        "n_paired": n,
        "median_a_s": float(np.median(t_a)),
        "median_b_s": float(np.median(t_b)),
        "median_diff_s": float(np.median(diffs)),
        "stat": float(res.statistic),
        "p_raw": float(res.pvalue),
    }


def compute_summary(data: dict[str, Any]) -> dict[str, Any]:
    seeds_dict = data["seeds"]
    keys = sorted(seeds_dict.keys(), key=lambda x: int(x))
    n = len(keys)
    if n == 0:
        return {"n_seeds": 0}

    rows = [seeds_dict[k] for k in keys]
    det = data["det_energy_shoot"]
    t_det_med = float(det["runtime_stats"]["median"])

    arms_summary = {}
    for arm_name, time_key in [
        ("single_shoot", "runtime_s"),
        ("multistart_shoot_5", "total_runtime_s"),
        ("hybrid_rho1", "total_runtime_s"),
        ("split_domain", "total_runtime_s"),
    ]:
        ok_s = [bool(r[arm_name]["success_strict"]) for r in rows]
        ok_l = [bool(r[arm_name]["success_loose"]) for r in rows]
        ks, kl = sum(ok_s), sum(ok_l)
        t_all = [float(r[arm_name][time_key]) for r in rows]
        t_succ = [float(r[arm_name][time_key]) for r in rows if r[arm_name]["success_strict"]]
        t_fail = [float(r[arm_name][time_key]) for r in rows if not r[arm_name]["success_strict"]]
        reprops = [r[arm_name]["reprop_arrival_residual"] for r in rows if r[arm_name]["success_strict"]]
        fuels = [r[arm_name]["fuel_kg"] for r in rows if r[arm_name]["success_strict"]]
        entry: dict[str, Any] = {
            "n_seeds": n,
            "n_loose": kl,
            "rate_loose": kl / n,
            "ci_loose": list(wilson_ci(kl, n)),
            "n_strict": ks,
            "rate_strict": ks / n,
            "ci_strict": list(wilson_ci(ks, n)),
            "time_all_s": _dist_stats(t_all),
            "time_strict_s": _dist_stats(t_succ),
            "time_fail_s": _dist_stats(t_fail),
            "reprop_strict": _dist_stats(reprops),
            "fuel_strict_kg": _dist_stats(fuels),
        }
        if arm_name == "multistart_shoot_5":
            atts_all = [r[arm_name]["n_attempts"] for r in rows]
            atts_succ = [r[arm_name]["first_strict_attempt"] for r in rows if r[arm_name]["success_strict"]]
            entry["attempts_all"] = _dist_stats(atts_all)
            entry["attempts_strict"] = _dist_stats(atts_succ)
        if arm_name == "hybrid_rho1":
            entry["ponn_time_s"] = _dist_stats([r[arm_name]["ponn_runtime_s"] for r in rows])
            entry["shoot_time_s"] = _dist_stats([r[arm_name]["shoot_runtime_s"] for r in rows])
        if arm_name == "split_domain":
            entry["ponn_time_s"] = _dist_stats([r[arm_name]["ponn_runtime_s"] for r in rows])
            entry["split_time_s"] = _dist_stats([r[arm_name]["split_runtime_s"] for r in rows if r[arm_name]["success_strict"]])
            k_t3 = sum(1 for r in rows if r[arm_name].get("passed_tier3_criteria"))
            entry["n_tier3_full"] = k_t3
            entry["ci_tier3_full"] = list(wilson_ci(k_t3, n))
        arms_summary[arm_name] = entry

    # M = 10 preregistered hypothesis tests with Holm-Bonferroni correction
    ok_ss = [bool(r["single_shoot"]["success_strict"]) for r in rows]
    ok_ms = [bool(r["multistart_shoot_5"]["success_strict"]) for r in rows]
    ok_hy = [bool(r["hybrid_rho1"]["success_strict"]) for r in rows]
    ok_sd = [bool(r["split_domain"]["success_strict"]) for r in rows]

    tests: list[dict[str, Any]] = []

    def add_mc(tid: str, desc: str, a_ok, b_ok, rule: str):
        mc = _exact_mcnemar(a_ok, b_ok)
        tests.append({"id": tid, "type": "mcnemar", "comparison": desc, "rule": rule, **mc})

    add_mc("M1", "hybrid_rho1 vs single_shoot", ok_hy, ok_ss, "delta>=15pp & p_holm<0.05")
    add_mc("M2", "split_domain vs single_shoot", ok_sd, ok_ss, "delta>=15pp & p_holm<0.05")
    add_mc("M3", "multistart_shoot_5 vs single_shoot", ok_ms, ok_ss, "delta>=15pp & p_holm<0.05")
    add_mc("M4", "hybrid_rho1 vs multistart_shoot_5", ok_hy, ok_ms, "p_holm<0.05")
    add_mc("M5", "split_domain vs multistart_shoot_5", ok_sd, ok_ms, "p_holm<0.05")
    add_mc("M6", "hybrid_rho1 vs split_domain", ok_hy, ok_sd, "p_holm<0.05")

    def add_wx(tid: str, desc: str, arm_a: str, arm_b: str, rule: str):
        ta = [r[arm_a]["total_runtime_s"] for r in rows if r[arm_a]["success_strict"] and r[arm_b]["success_strict"]]
        tb = [r[arm_b]["total_runtime_s"] for r in rows if r[arm_a]["success_strict"] and r[arm_b]["success_strict"]]
        wx = _paired_wilcoxon(ta, tb)
        tests.append({"id": tid, "type": "wilcoxon_paired", "comparison": desc, "rule": rule, **wx})

    add_wx("T1", "hybrid_rho1 vs multistart_shoot_5 (time)", "hybrid_rho1", "multistart_shoot_5", "med_diff<0 & p_holm<0.05")
    add_wx("T2", "split_domain vs multistart_shoot_5 (time)", "split_domain", "multistart_shoot_5", "med_diff<0 & p_holm<0.05")
    add_wx("T3", "hybrid_rho1 vs split_domain (time)", "hybrid_rho1", "split_domain", "p_holm<0.05")

    # T4: one-sample Wilcoxon of hybrid_rho1 strict times vs deterministic median
    t_hy_strict = [r["hybrid_rho1"]["total_runtime_s"] for r in rows if r["hybrid_rho1"]["success_strict"]]
    wx4 = _paired_wilcoxon(t_hy_strict, [t_det_med] * len(t_hy_strict))
    tests.append({
        "id": "T4", "type": "wilcoxon_onesample",
        "comparison": "hybrid_rho1 vs det_energy_shoot (time)",
        "rule": "med_diff<0 & p_holm<0.05",
        **wx4,
    })

    # Holm-Bonferroni correction across all M = 10 tests
    M = len(tests)
    order = sorted(range(M), key=lambda i: tests[i]["p_raw"])
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, tests[idx]["p_raw"] * (M - rank))
        running_max = max(running_max, adj)
        tests[idx]["p_holm"] = float(running_max)

    for t in tests:
        ph = t["p_holm"]
        r = t["rule"]
        if r == "delta>=15pp & p_holm<0.05":
            sup = (t["delta_pp"] >= 15.0) and (ph < 0.05)
        elif r == "med_diff<0 & p_holm<0.05":
            sup = (t["median_diff_s"] < 0.0) and (ph < 0.05)
        else:
            sup = ph < 0.05
        t["verdict"] = "SUPPORTED" if sup else "CONTRADICTED"

    return {"n_seeds": n, "arms": arms_summary, "holm_tests": tests}


def _save(data: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=8000)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=440.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        prob = ScenarioAProblem()
        sh_ref = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO).solve(prob, seed=1000)
        print("[Addendum 2] Running deterministic energy-optimal baseline (5 timing trials)...", flush=True)
        det = run_deterministic_baseline(n_trials=5)
        data = {
            "meta": {
                "description": "Preregistration Addendum 2 baseline comparison on Scenario A (seeds 8000..8099)",
                "prereg_addendum2_commit": "24cd1df",
                "seed_start": args.seed_start,
                "n_seeds_target": args.n_seeds,
                "max_restarts": MAX_RESTARTS,
                "ref_fuel_kg": float(sh_ref.fuel_kg),
                "ref_switch_times_days": [float(s["t_days"]) for s in sh_ref.switches],
            },
            "det_energy_shoot": det,
            "seeds": {},
            "summary": {},
        }
        _save(data)

    ref_fuel = float(data["meta"]["ref_fuel_kg"])
    ref_sw = list(data["meta"]["ref_switch_times_days"])
    target_seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    pending = [s for s in target_seeds if str(s) not in data["seeds"]]
    print(f"[Addendum 2] Done: {len(data['seeds'])}/{len(target_seeds)}, Pending: {len(pending)}", flush=True)

    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i in range(0, len(pending), args.workers):
                if time.perf_counter() - t_start > args.max_wall_s:
                    print(f"[Addendum 2] Wall-time budget ({args.max_wall_s:.0f}s) reached; checkpoint saved.", flush=True)
                    break
                batch = [(s, ref_fuel, ref_sw) for s in pending[i : i + args.workers]]
                results = pool.map(_run_seed, batch)
                for s, rec in results:
                    data["seeds"][str(s)] = rec
                    print(
                        f"  [s={s}] ss={int(rec['single_shoot']['success_strict'])} "
                        f"ms5={int(rec['multistart_shoot_5']['success_strict'])}(att={rec['multistart_shoot_5']['n_attempts']},t={rec['multistart_shoot_5']['total_runtime_s']:.1f}s) "
                        f"hyb={int(rec['hybrid_rho1']['success_strict'])}(t={rec['hybrid_rho1']['total_runtime_s']:.1f}s) "
                        f"sd={int(rec['split_domain']['success_strict'])}(t={rec['split_domain']['total_runtime_s']:.1f}s)",
                        flush=True,
                    )
                data["summary"] = compute_summary(data)
                _save(data)

    data["summary"] = compute_summary(data)
    _save(data)
    print(f"[Addendum 2] Total completed: {len(data['seeds'])}/{len(target_seeds)}", flush=True)


if __name__ == "__main__":
    main()
