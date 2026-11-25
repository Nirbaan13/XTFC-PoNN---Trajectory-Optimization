"""Tier 1.2: Extra Physical Cases Benchmark (7 cases x 50 fresh seeds = 350 seeds x 4 methods).

Cases (all pre-verified t_min <= 0.95 * tf in PREREG_ADDENDUM.md):
  1. scen_a_m0_800   : Scenario A, m0 = 800 kg   (seeds 5000..5049)
  2. scen_a_m0_1200  : Scenario A, m0 = 1200 kg  (seeds 5100..5149)
  3. scen_a_tmax_0.4 : Scenario A, Tmax = 0.4 N  (seeds 5200..5249)
  4. scen_a_tmax_0.7 : Scenario A, Tmax = 0.7 N  (seeds 5300..5349)
  5. scen_b_tf_300   : Scenario B (theta_f=210 deg), tf = 300 d (seeds 5400..5449)
  6. scen_b_tf_330   : Scenario B (theta_f=210 deg), tf = 330 d (seeds 5500..5549)
  7. scen_b_tf_360   : Scenario B (theta_f=210 deg), tf = 360 d (seeds 5600..5649)

Methods per seed (frozen settings, zero retuning):
  - ponn (scaled_u01, L=70 CGL -> L=80 uniform_dense)
  - shooting (random 7-D costates, stop_at_rho=1e-5)
  - hybrid (PoNN coarse up to k_ext=1, rho=0.2976 -> shooting; always uses best-effort PoNN costates)
  - collocation (180-node LGL, max_iter=500, tol=1e-7)

Checkpoints per seed to results/tier1_extra_cases_5000.json.
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from scripts.run_phase_d import build_scenario_b_as_scenario_a
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
from trajectory_optimization.scenario_a_collocation import ScenarioACollocationSolver

OUT_PATH = Path("results/tier1_extra_cases_5000.json")
K_EXT = 1
STOP_AT_RHO = 1e-5
RHO_SCHED_HYB = tuple(r for r in DEFAULT_RHO_SCHEDULE[K_EXT:] if r >= STOP_AT_RHO - 1e-15)

EXTRA_CASES: list[tuple[str, int, float]] = [
    ("scen_a_m0_800", 5000, 280.884),
    ("scen_a_m0_1200", 5100, 326.882),
    ("scen_a_tmax_0.4", 5200, 330.919),
    ("scen_a_tmax_0.7", 5300, 265.979),
    ("scen_b_tf_300", 5400, 272.881),
    ("scen_b_tf_330", 5700, 293.082),
    ("scen_b_tf_360", 5800, 308.375),
]


def build_case_problem(case_id: str) -> ScenarioAProblem:
    if case_id == "scen_a_m0_800":
        return ScenarioAProblem(m0_kg=800.0, label=case_id)
    if case_id == "scen_a_m0_1200":
        return ScenarioAProblem(m0_kg=1200.0, label=case_id)
    if case_id == "scen_a_tmax_0.4":
        return ScenarioAProblem(t_max_N=0.4, label=case_id)
    if case_id == "scen_a_tmax_0.7":
        return ScenarioAProblem(t_max_N=0.7, label=case_id)
    if case_id == "scen_b_tf_300":
        return build_scenario_b_as_scenario_a(tf_days=300.0, theta_f_deg=210.0)
    if case_id == "scen_b_tf_330":
        return build_scenario_b_as_scenario_a(tf_days=330.0, theta_f_deg=235.0)
    if case_id == "scen_b_tf_360":
        return build_scenario_b_as_scenario_a(tf_days=360.0, theta_f_deg=255.0)
    raise ValueError(f"Unknown case_id: {case_id}")


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


_COL_SOLVER_CACHE: dict[int, ScenarioACollocationSolver] = {}


def _get_col_solver(n_nodes: int = 180) -> ScenarioACollocationSolver:
    if n_nodes not in _COL_SOLVER_CACHE:
        _COL_SOLVER_CACHE[n_nodes] = ScenarioACollocationSolver(n_nodes=n_nodes, max_iter=500, tol=1e-7)
    return _COL_SOLVER_CACHE[n_nodes]


def _run_seed_case(args_tuple: tuple[str, int]) -> tuple[str, int, dict[str, Any]]:
    case_id, seed = args_tuple
    prob = build_case_problem(case_id)

    # 1. Indirect Shooting (random costates)
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)
    r_sh = shooter.solve(prob, seed=seed)
    rp_sh = float(r_sh.reprop_arrival_residual) if np.isfinite(r_sh.reprop_arrival_residual) else float("nan")
    ar_sh = float(r_sh.arrival_residual) if np.isfinite(r_sh.arrival_residual) else float("nan")
    mf_sh = float(r_sh.final_mass_kg) if np.isfinite(r_sh.final_mass_kg) else 0.0
    rr_sh = float(r_sh.rho_reached) if r_sh.rho_reached is not None else float("nan")
    phys_sh = bool(mf_sh > 50.0 and rr_sh <= 1e-4 and ar_sh < 1e-6)
    rec_sh = {
        "success_loose": bool(phys_sh and np.isfinite(rp_sh) and rp_sh < 5e-2),
        "success_strict": bool(phys_sh and np.isfinite(rp_sh) and rp_sh < 1e-2),
        "status": r_sh.status,
        "fuel_kg": float(r_sh.fuel_kg) if np.isfinite(r_sh.fuel_kg) else None,
        "reprop_arrival_residual": rp_sh if np.isfinite(rp_sh) else None,
        "n_switches": len(r_sh.switches),
        "switch_times_days": [float(s["t_days"]) for s in r_sh.switches],
        "thrust_sequence": r_sh.thrust_sequence,
        "nfev": int(r_sh.n_propagations),
        "njev": int(r_sh.n_jacobian_builds),
        "runtime_s": float(r_sh.runtime_s),
    }

    # 2. Hybrid (frozen PoNN coarse k_ext=1 -> Shooting continuation, best-effort costates)
    t_p0 = time.perf_counter()
    z_c = make_collocation_grid(140, "cgl")
    model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=seed)
    ops_c = model_c.operators(z_c)
    xi0 = model_c.initial_guess_paper(ops_c, seed=seed, init_mode="scaled_u01")
    ponn_solver = PaperPoNNSolver()
    xi_c, _, _, _, nfev_p, njev_p, _ = ponn_solver._run_continuation(
        prob, model_c, z_c, xi0, DEFAULT_RHO_SCHEDULE[:K_EXT + 1], start_scaled=False
    )
    t_ponn_hyb = time.perf_counter() - t_p0
    ev0 = model_c.evaluate(xi_c, z=np.array([-1.0]))
    lam0_hyb = np.concatenate([ev0["lr"][0], ev0["lv"][0], [ev0["lm"][0]]])
    if not np.all(np.isfinite(lam0_hyb)):
        lam0_hyb = np.zeros(7)

    r_hy = shooter.solve(prob, seed=seed, lam0_guess=lam0_hyb, rho_schedule=RHO_SCHED_HYB)
    rp_hy = float(r_hy.reprop_arrival_residual) if np.isfinite(r_hy.reprop_arrival_residual) else float("nan")
    ar_hy = float(r_hy.arrival_residual) if np.isfinite(r_hy.arrival_residual) else float("nan")
    mf_hy = float(r_hy.final_mass_kg) if np.isfinite(r_hy.final_mass_kg) else 0.0
    rr_hy = float(r_hy.rho_reached) if r_hy.rho_reached is not None else float("nan")
    phys_hy = bool(mf_hy > 50.0 and rr_hy <= 1e-4 and ar_hy < 1e-6)
    rec_hy = {
        "success_loose": bool(phys_hy and np.isfinite(rp_hy) and rp_hy < 5e-2),
        "success_strict": bool(phys_hy and np.isfinite(rp_hy) and rp_hy < 1e-2),
        "status": r_hy.status,
        "fuel_kg": float(r_hy.fuel_kg) if np.isfinite(r_hy.fuel_kg) else None,
        "reprop_arrival_residual": rp_hy if np.isfinite(rp_hy) else None,
        "n_switches": len(r_hy.switches),
        "switch_times_days": [float(s["t_days"]) for s in r_hy.switches],
        "thrust_sequence": r_hy.thrust_sequence,
        "ponn_nfev": int(nfev_p),
        "ponn_njev": int(njev_p),
        "shoot_nfev": int(r_hy.n_propagations),
        "shoot_njev": int(r_hy.n_jacobian_builds),
        "ponn_runtime_s": float(t_ponn_hyb),
        "shoot_runtime_s": float(r_hy.runtime_s),
        "runtime_s": float(t_ponn_hyb + r_hy.runtime_s),
    }

    # 3. Standalone Frozen PoNN (scaled_u01, L=70 CGL -> L=80 uniform_dense)
    ponn_full = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    rc, rr = ponn_full.solve(
        prob, seed=seed, refine=True, refine_neurons=80,
        points_per_switch=100, refine_grid_mode="uniform_dense",
        skip_refine_if_coarse_diverged=True,
    )
    r_use = rr if rr is not None else rc
    rp_p = float(r_use.reprop_arrival_residual) if np.isfinite(r_use.reprop_arrival_residual) else float("nan")
    rms_p = float(r_use.residual_rms) if np.isfinite(r_use.residual_rms) else float("nan")
    mf_p = float(r_use.final_mass_kg) if np.isfinite(r_use.final_mass_kg) else 0.0
    phys_p = bool(mf_p > 50.0 and np.isfinite(rms_p) and rms_p < 1.5e-2)
    rec_ponn = {
        "success_loose": bool(phys_p and np.isfinite(rp_p) and rp_p < 5e-2),
        "success_strict": bool(phys_p and np.isfinite(rp_p) and rp_p < 1e-2),
        "status": r_use.status,
        "refined": bool(rr is not None),
        "fuel_kg": float(r_use.fuel_kg) if np.isfinite(r_use.fuel_kg) else None,
        "residual_rms": rms_p if np.isfinite(rms_p) else None,
        "reprop_arrival_residual": rp_p if np.isfinite(rp_p) else None,
        "n_switches": len(r_use.switches),
        "switch_times_days": [float(s["t_days"]) for s in r_use.switches],
        "thrust_sequence": r_use.thrust_sequence,
        "nfev": int(rc.total_nfev + (rr.total_nfev if rr is not None else 0)),
        "njev": int(rc.total_njev + (rr.total_njev if rr is not None else 0)),
        "runtime_s": float(rc.runtime_s + (rr.runtime_s if rr is not None else 0.0)),
    }

    # 4. 180-Node LGL Collocation (cached NLP graph per worker)
    col_solver = _get_col_solver(180)
    r_col = col_solver.solve(prob, seed=seed)
    rp_c = float(r_col.reprop_arrival_residual) if np.isfinite(r_col.reprop_arrival_residual) else float("nan")
    te_c = float(r_col.terminal_error) if np.isfinite(r_col.terminal_error) else float("nan")
    mf_c = float(r_col.final_mass_kg) if np.isfinite(r_col.final_mass_kg) else 0.0
    phys_c = bool(r_col.success and mf_c > 50.0 and te_c < 1e-4)
    rec_col = {
        "success_loose": bool(phys_c and np.isfinite(rp_c) and rp_c < 5e-2),
        "success_strict": bool(phys_c and np.isfinite(rp_c) and rp_c < 1e-2),
        "status": r_col.status,
        "fuel_kg": float(r_col.fuel_kg) if np.isfinite(r_col.fuel_kg) else None,
        "reprop_arrival_residual": rp_c if np.isfinite(rp_c) else None,
        "n_switches": len(r_col.switches),
        "switch_times_days": [float(s["t_days"]) for s in r_col.switches],
        "thrust_sequence": r_col.thrust_sequence,
        "iterations": int(r_col.iterations),
        "nfev": int(r_col.n_func_evals),
        "njev": int(r_col.n_jac_evals),
        "runtime_s": float(r_col.runtime_s),
    }

    return case_id, seed, {
        "seed": seed,
        "shooting": rec_sh,
        "hybrid": rec_hy,
        "ponn": rec_ponn,
        "collocation": rec_col,
    }


def summarize_case(case_dict: dict) -> dict:
    seeds = case_dict["seeds"]
    keys = sorted(seeds.keys(), key=int)
    n = len(keys)
    out: dict[str, Any] = {"n_seeds": n, "methods": {}}

    for m in ("ponn", "shooting", "hybrid", "collocation"):
        recs = [seeds[k][m] for k in keys]
        ns = sum(1 for r in recs if r["success_strict"])
        nl = sum(1 for r in recs if r["success_loose"])
        cis = wilson_ci(ns, n)
        cil = wilson_ci(nl, n)
        s_recs = [r for r in recs if r["success_strict"]]
        l_recs = [r for r in recs if r["success_loose"]]
        sw_counts = Counter(r["n_switches"] for r in (s_recs if s_recs else l_recs))
        seqs = Counter(r["thrust_sequence"] for r in (s_recs if s_recs else l_recs))
        out["methods"][m] = {
            "n_strict": ns,
            "rate_strict": ns / n if n else 0.0,
            "ci_strict": [cis[0], cis[1]],
            "n_loose": nl,
            "rate_loose": nl / n if n else 0.0,
            "ci_loose": [cil[0], cil[1]],
            "fuel_kg_strict": _dist_stats([r["fuel_kg"] for r in s_recs]),
            "fuel_kg_loose": _dist_stats([r["fuel_kg"] for r in l_recs]),
            "reprop_all": _dist_stats([r["reprop_arrival_residual"] for r in recs]),
            "runtime_s": _dist_stats([r["runtime_s"] for r in recs]),
            "switch_count_mode": sw_counts.most_common(1)[0][0] if sw_counts else None,
            "switch_counts_hist": dict(sw_counts),
            "thrust_sequence_mode": seqs.most_common(1)[0][0] if seqs else None,
        }

    # Exact paired McNemar Hybrid vs Shooting (strict)
    n11 = n10 = n01 = n00 = 0
    for k in keys:
        h = seeds[k]["hybrid"]["success_strict"]
        s = seeds[k]["shooting"]["success_strict"]
        if h and s:
            n11 += 1
        elif h and not s:
            n10 += 1
        elif not h and s:
            n01 += 1
        else:
            n00 += 1
    disc = n10 + n01
    p_exact = float(binomtest(min(n10, n01), disc, 0.5).pvalue) if disc > 0 else 1.0
    delta_pp = 100.0 * (out["methods"]["hybrid"]["rate_strict"] - out["methods"]["shooting"]["rate_strict"])
    out["mcnemar_hybrid_vs_shooting"] = {
        "n11": n11,
        "n10_hyb_only": n10,
        "n01_shoot_only": n01,
        "n00": n00,
        "delta_strict_pp": delta_pp,
        "p_exact_raw": p_exact,
    }
    return out


def _save(data: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUT_PATH)


def _load() -> dict:
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            return json.load(f)
    cases = {}
    for cid, s0, tmin in EXTRA_CASES:
        p = build_case_problem(cid)
        cases[cid] = {
            "meta": {
                "case_id": cid,
                "seed_start": s0,
                "tf_days": float(p.tf_days),
                "t_min_days": float(tmin),
                "t_min_ratio": float(tmin / p.tf_days),
                "m0_kg": float(p.m0_kg),
                "t_max_N": float(p.t_max_N),
                "isp_s": float(p.isp_s),
                "srp_mode": p.srp_mode,
                "srp_area_to_mass": float(p.srp_area_to_mass),
            },
            "seeds": {},
            "summary": {},
        }
    return {
        "meta": {
            "description": "Tier 1.2 Extra Physical Cases (7 cases x 50 fresh seeds each)",
            "prereg_addendum_commit": "38e2908",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=420.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    data = _load()

    todo: list[tuple[str, int]] = []
    for cid, s0, _ in EXTRA_CASES:
        done_keys = data["cases"][cid]["seeds"]
        for i in range(args.n_seeds):
            s = s0 + i
            if str(s) not in done_keys:
                todo.append((cid, s))

    total_target = len(EXTRA_CASES) * args.n_seeds
    total_done = total_target - len(todo)
    print(f"[Tier 1.2 Extra Cases] {total_done}/{total_target} done, {len(todo)} remaining", flush=True)

    if todo:
        with mp.Pool(processes=args.workers) as pool:
            for cid, seed, rec in pool.imap_unordered(_run_seed_case, todo):
                data["cases"][cid]["seeds"][str(seed)] = rec
                data["cases"][cid]["summary"] = summarize_case(data["cases"][cid])
                _save(data)
                total_done += 1
                print(
                    f"  [{cid} s={seed}] sh={int(rec['shooting']['success_strict'])} "
                    f"hyb={int(rec['hybrid']['success_strict'])} ponnL={int(rec['ponn']['success_loose'])} "
                    f"colS={int(rec['collocation']['success_strict'])} sw={rec['hybrid']['n_switches']} "
                    f"| {total_done}/{total_target} ({time.perf_counter()-t_start:.1f}s)",
                    flush=True,
                )
                if time.perf_counter() - t_start > args.max_wall_s:
                    print("[checkpoint] wall limit reached, terminating pool cleanly", flush=True)
                    pool.terminate()
                    break

    for cid, _, _ in EXTRA_CASES:
        data["cases"][cid]["summary"] = summarize_case(data["cases"][cid])
    _save(data)
    print(f"[Tier 1.2 Extra Cases] Saved {total_done}/{total_target} to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
