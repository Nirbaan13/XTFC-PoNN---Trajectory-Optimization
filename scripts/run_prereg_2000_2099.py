"""Preregistered benchmark runner for held-out seeds 2000..2099.

Evaluates:
  1. Paired SRP A/m in {0.0, 0.5, 1.0} m^2/kg:
     - Indirect Shooting (random 7-D costates, stop_at_rho=1e-5)
     - Hybrid (PoNN coarse up to k_ext=1, rho=0.2976 -> indirect shooting)
  2. Standalone PoNN on Scenario A (A/m=0.0):
     - Scaled U(0,1) init (L=70 coarse CGL -> L=80 refined uniform_dense)
     - Exact paper U(0,1) init (L=70 coarse CGL -> L=80 refined uniform_dense)
  3. Standalone 180-Node LGL Collocation on Scenario A (A/m=0.0)

All evaluations use frozen settings committed in PREREG.md (commit 3c87713).
Checkpoints per seed to results/prereg_fresh_2000_2099.json.
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
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, chi2

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

OUT_PATH = Path("results/prereg_fresh_2000_2099.json")
BACKUP_PATH = Path("results/prereg_fresh_2000_2099_backup.json")
K_EXT = 1
RHO_EXT = float(DEFAULT_RHO_SCHEDULE[K_EXT])  # ~0.297635
STOP_AT_RHO = 1e-5
RHO_SCHEDULE_SHOOTING = tuple(r for r in DEFAULT_RHO_SCHEDULE if r >= STOP_AT_RHO - 1e-15)
RHO_SCHEDULE_HYBRID_SHOOT = tuple(r for r in DEFAULT_RHO_SCHEDULE[K_EXT:] if r >= STOP_AT_RHO - 1e-15)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _stats(vals: list[float]) -> dict[str, float] | None:
    arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not arr:
        return None
    a = np.asarray(arr, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _save(data: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUT_PATH)
    try:
        shutil.copy2(OUT_PATH, BACKUP_PATH)
    except Exception:
        pass


def _load() -> dict:
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            return json.load(f)
    return {
        "meta": {
            "description": "Preregistered held-out Scenario A benchmark and Hybrid evaluation on fresh seeds 2000..2099",
            "prereg_commit": "3c87713",
            "k_ext": K_EXT,
            "rho_ext": RHO_EXT,
            "stop_at_rho": STOP_AT_RHO,
            "threshold_loose": 0.05,
            "threshold_strict": 0.01,
        },
        "paired_srp_0.0": {"seeds": {}, "summary": {}},
        "paired_srp_0.5": {"seeds": {}, "summary": {}},
        "paired_srp_1.0": {"seeds": {}, "summary": {}},
        "ponn_scaled_am0.0": {"seeds": {}, "summary": {}},
        "ponn_paper_u01_am0.0": {"seeds": {}, "summary": {}},
        "collocation_am0.0": {"seeds": {}, "summary": {}},
    }


# ---------------------------------------------------------------------------
# Worker Functions for Evaluation
# ---------------------------------------------------------------------------

def run_single_shooting(prob: ScenarioAProblem, seed: int) -> dict:
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)
    r = shooter.solve(prob, seed=seed)
    rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
    ar = float(r.arrival_residual) if np.isfinite(r.arrival_residual) else float("nan")
    m_f = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
    rho_r = float(r.rho_reached) if r.rho_reached is not None else float("nan")
    phys_ok = bool(m_f > 50.0 and rho_r <= 1e-4 and ar < 1e-6)
    succ_loose = bool(phys_ok and np.isfinite(rp) and rp < 5e-2)
    succ_strict = bool(phys_ok and np.isfinite(rp) and rp < 1e-2)

    return {
        "seed": seed,
        "success": bool(r.success),
        "success_loose": succ_loose,
        "success_strict": succ_strict,
        "status": r.status,
        "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
        "final_mass_kg": m_f,
        "arrival_residual": ar,
        "reprop_arrival_residual": rp,
        "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
        "n_switches": len(r.switches),
        "switch_times_days": [float(s["t_days"]) for s in r.switches],
        "thrust_sequence": r.thrust_sequence,
        "rho_reached": rho_r,
        "nfev": int(r.n_propagations),
        "njev": int(r.n_jacobian_builds),
        "runtime_s": float(r.runtime_s),
    }


def run_single_hybrid(prob: ScenarioAProblem, seed: int) -> dict:
    # 1. PoNN stage up to k_ext
    t_start = time.perf_counter()
    z_c = make_collocation_grid(140, "cgl")
    model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=seed)
    ops_c = model_c.operators(z_c)
    xi0 = model_c.initial_guess_paper(ops_c, seed=seed, init_mode="scaled_u01")
    ponn = PaperPoNNSolver()
    sched_ponn = DEFAULT_RHO_SCHEDULE[:K_EXT + 1]
    xi_c, snaps_c, log_c, hist_c, nfev_c, njev_c, rho_c = ponn._run_continuation(
        prob, model_c, z_c, xi0, sched_ponn, start_scaled=False
    )
    t_ponn = time.perf_counter() - t_start

    # Costate extraction at z0 = -1 (t = t0)
    ev0 = model_c.evaluate(xi_c, z=np.array([-1.0]))
    lam0_hyb = np.concatenate([ev0["lr"][0], ev0["lv"][0], [ev0["lm"][0]]])

    # 2. Shooting stage continuing schedule from rho_ext
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)
    t_shoot_start = time.perf_counter()
    r = shooter.solve(prob, seed=seed, lam0_guess=lam0_hyb, rho_schedule=RHO_SCHEDULE_HYBRID_SHOOT)
    t_shoot = time.perf_counter() - t_shoot_start
    t_total = t_ponn + t_shoot

    rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
    ar = float(r.arrival_residual) if np.isfinite(r.arrival_residual) else float("nan")
    m_f = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
    rho_r = float(r.rho_reached) if r.rho_reached is not None else float("nan")
    phys_ok = bool(m_f > 50.0 and rho_r <= 1e-4 and ar < 1e-6)
    succ_loose = bool(phys_ok and np.isfinite(rp) and rp < 5e-2)
    succ_strict = bool(phys_ok and np.isfinite(rp) and rp < 1e-2)

    return {
        "seed": seed,
        "success": bool(r.success),
        "success_loose": succ_loose,
        "success_strict": succ_strict,
        "status": r.status,
        "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
        "final_mass_kg": m_f,
        "arrival_residual": ar,
        "reprop_arrival_residual": rp,
        "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
        "n_switches": len(r.switches),
        "switch_times_days": [float(s["t_days"]) for s in r.switches],
        "thrust_sequence": r.thrust_sequence,
        "rho_reached": rho_r,
        "runtime_ponn_s": float(t_ponn),
        "runtime_shoot_s": float(t_shoot),
        "runtime_total_s": float(t_total),
        "runtime_s": float(t_total),  # honest total runtime
        "ponn_nfev": int(nfev_c),
        "ponn_njev": int(njev_c),
        "shoot_propagations": int(r.n_propagations),
        "shoot_jac_builds": int(r.n_jacobian_builds),
        "total_evaluations": int(nfev_c + r.n_propagations),
    }


def run_paired_srp_worker(args: tuple[str, float, int]) -> tuple[int, dict, dict]:
    srp_mode, a_over_m, seed = args
    prob = ScenarioAProblem(srp_mode=srp_mode, srp_area_to_mass=a_over_m)
    sh_res = run_single_shooting(prob, seed)
    hyb_res = run_single_hybrid(prob, seed)
    return seed, sh_res, hyb_res


def run_ponn_worker(args: tuple[str, int]) -> tuple[int, dict]:
    init_mode, seed = args
    prob = ScenarioAProblem()
    max_iter_first = 100 if init_mode == "scaled_u01" else 250
    solver = PaperPoNNSolver(
        init_mode=init_mode,
        max_iter_first=max_iter_first,
        max_iter_cont=25,
        reprop_conv_tol=5e-2,
    )
    rc, rr = solver.solve(
        prob,
        seed=seed,
        refine=True,
        refine_neurons=80,
        refine_grid_mode="uniform_dense",
        refine_init_mode="pad_from_smooth",
        skip_refine_if_coarse_diverged=True,
    )

    def pack(r):
        if r is None:
            return None
        rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
        rms = float(r.residual_rms) if np.isfinite(r.residual_rms) else float("nan")
        m_f = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
        phys_ok = bool(m_f > 50.0 and rms < 1.5e-2)
        return {
            "success": bool(r.success),
            "success_loose": bool(phys_ok and np.isfinite(rp) and rp < 5e-2),
            "success_strict": bool(phys_ok and np.isfinite(rp) and rp < 1e-2),
            "status": r.status,
            "fuel_kg": float(r.fuel_kg),
            "final_mass_kg": m_f,
            "residual_rms": rms,
            "residual_norm": float(r.residual_norm),
            "residual_max": float(r.residual_max),
            "reprop_arrival_residual": rp,
            "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
            "n_switches": len(r.switches),
            "switch_times_days": [float(s["t_days"]) for s in r.switches],
            "thrust_sequence": r.thrust_sequence,
            "rho_reached": float(r.rho_reached),
            "total_nfev": int(r.total_nfev),
            "total_njev": int(r.total_njev),
            "runtime_s": float(r.runtime_s),
        }

    c_pack = pack(rc)
    if rr is not None:
        r_pack = pack(rr)
        if r_pack is not None:
            r_pack["refinement_skipped"] = False
    else:
        r_pack = dict(c_pack) if c_pack is not None else None
        if r_pack is not None:
            r_pack["refinement_skipped"] = True

    return seed, {"coarse": c_pack, "refined": r_pack}


def run_collocation_worker(seed: int) -> tuple[int, dict]:
    prob = ScenarioAProblem()
    solver = ScenarioACollocationSolver(n_nodes=180, max_iter=500, tol=1e-7)
    r = solver.solve(prob, seed=seed)
    rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
    te = float(r.terminal_error) if np.isfinite(r.terminal_error) else float("nan")
    m_f = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
    phys_ok = bool(r.success and m_f > 50.0 and te < 1e-4)
    return seed, {
        "seed": seed,
        "success": bool(r.success),
        "success_loose": bool(phys_ok and np.isfinite(rp) and rp < 5e-2),
        "success_strict": bool(phys_ok and np.isfinite(rp) and rp < 1e-2),
        "status": r.status,
        "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
        "final_mass_kg": m_f,
        "terminal_error": te,
        "reprop_arrival_residual": rp,
        "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
        "n_switches": len(r.switches),
        "switch_times_days": [float(s["t_days"]) for s in r.switches],
        "thrust_sequence": r.thrust_sequence,
        "iterations": int(r.iterations),
        "nfev": int(r.n_func_evals),
        "njev": int(r.n_jac_evals),
        "runtime_s": float(r.runtime_s),
    }


# ---------------------------------------------------------------------------
# Summaries & Statistical Tests
# ---------------------------------------------------------------------------

def compute_paired_srp_summary(arm_data: dict) -> dict:
    seeds_dict = arm_data["seeds"]
    n = len(seeds_dict)
    if n == 0:
        return {}
    sh_list = [v["shooting"] for v in seeds_dict.values()]
    hyb_list = [v["hybrid"] for v in seeds_dict.values()]

    sh_loose = sum(1 for v in sh_list if v["success_loose"])
    sh_strict = sum(1 for v in sh_list if v["success_strict"])
    hyb_loose = sum(1 for v in hyb_list if v["success_loose"])
    hyb_strict = sum(1 for v in hyb_list if v["success_strict"])

    # McNemar contingency table on strict convergence
    # n11: both succeed, n00: both fail, n10: hyb ok & shoot fail, n01: shoot ok & hyb fail
    n11 = sum(1 for s, h in zip(sh_list, hyb_list) if h["success_strict"] and s["success_strict"])
    n00 = sum(1 for s, h in zip(sh_list, hyb_list) if (not h["success_strict"]) and (not s["success_strict"]))
    n10 = sum(1 for s, h in zip(sh_list, hyb_list) if h["success_strict"] and (not s["success_strict"]))
    n01 = sum(1 for s, h in zip(sh_list, hyb_list) if (not h["success_strict"]) and s["success_strict"])

    disc_total = n10 + n01
    if disc_total > 0:
        p_mcnemar = float(binomtest(min(n10, n01), disc_total, 0.5, alternative="two-sided").pvalue)
        chi2_stat = float((abs(n10 - n01) - 1.0) ** 2 / disc_total)
        p_chi2 = float(chi2.sf(chi2_stat, df=1))
    else:
        p_mcnemar = 1.0
        chi2_stat = 0.0
        p_chi2 = 1.0

    delta_strict_pp = 100.0 * (hyb_strict - sh_strict) / n
    delta_loose_pp = 100.0 * (hyb_loose - sh_loose) / n
    supported = bool(delta_strict_pp >= 15.0 and p_mcnemar < 0.05)

    return {
        "n_total": n,
        "shooting": {
            "n_loose": sh_loose,
            "rate_loose": sh_loose / n,
            "ci95_loose": list(wilson_ci(sh_loose, n)),
            "n_strict": sh_strict,
            "rate_strict": sh_strict / n,
            "ci95_strict": list(wilson_ci(sh_strict, n)),
            "fuel_kg": _stats([v["fuel_kg"] for v in sh_list if v["success_strict"]]),
            "reprop_residual": _stats([v["reprop_arrival_residual"] for v in sh_list if v["success_strict"]]),
            "runtime_s": _stats([v["runtime_s"] for v in sh_list]),
            "nfev": _stats([v["nfev"] for v in sh_list]),
            "njev": _stats([v["njev"] for v in sh_list]),
        },
        "hybrid": {
            "n_loose": hyb_loose,
            "rate_loose": hyb_loose / n,
            "ci95_loose": list(wilson_ci(hyb_loose, n)),
            "n_strict": hyb_strict,
            "rate_strict": hyb_strict / n,
            "ci95_strict": list(wilson_ci(hyb_strict, n)),
            "fuel_kg": _stats([v["fuel_kg"] for v in hyb_list if v["success_strict"]]),
            "reprop_residual": _stats([v["reprop_arrival_residual"] for v in hyb_list if v["success_strict"]]),
            "runtime_ponn_s": _stats([v["runtime_ponn_s"] for v in hyb_list]),
            "runtime_shoot_s": _stats([v["runtime_shoot_s"] for v in hyb_list]),
            "runtime_total_s": _stats([v["runtime_total_s"] for v in hyb_list]),
            "ponn_nfev": _stats([v["ponn_nfev"] for v in hyb_list]),
            "shoot_propagations": _stats([v["shoot_propagations"] for v in hyb_list]),
            "total_evaluations": _stats([v["total_evaluations"] for v in hyb_list]),
        },
        "comparison": {
            "delta_strict_pp": delta_strict_pp,
            "delta_loose_pp": delta_loose_pp,
            "concordant_both_ok_n11": n11,
            "concordant_both_fail_n00": n00,
            "discordant_hyb_only_n10": n10,
            "discordant_shoot_only_n01": n01,
            "mcnemar_exact_p": p_mcnemar,
            "mcnemar_chi2": chi2_stat,
            "mcnemar_chi2_p": p_chi2,
            "prereg_verdict": "SUPPORTED" if supported else "CONTRADICTED",
        },
    }


def compute_ponn_summary(arm_data: dict, stage: str = "refined") -> dict:
    seeds_dict = arm_data["seeds"]
    n = len(seeds_dict)
    if n == 0:
        return {}
    recs = [v[stage] for v in seeds_dict.values() if v.get(stage) is not None]
    n_stage = len(recs)
    n_loose = sum(1 for v in recs if v["success_loose"])
    n_strict = sum(1 for v in recs if v["success_strict"])
    n_rms = sum(1 for v in recs if v["residual_rms"] < 1.5e-2 and v["final_mass_kg"] > 50.0)

    sw_dist: dict[str, int] = {}
    for v in recs:
        if v["success_loose"]:
            k = str(v["n_switches"])
            sw_dist[k] = sw_dist.get(k, 0) + 1

    return {
        "n_total": n_stage,
        "n_loose": n_loose,
        "rate_loose": n_loose / n_stage,
        "ci95_loose": list(wilson_ci(n_loose, n_stage)),
        "n_strict": n_strict,
        "rate_strict": n_strict / n_stage,
        "ci95_strict": list(wilson_ci(n_strict, n_stage)),
        "n_rms_only": n_rms,
        "rate_rms_only": n_rms / n_stage,
        "ci95_rms_only": list(wilson_ci(n_rms, n_stage)),
        "fuel_kg": _stats([v["fuel_kg"] for v in recs if v["success_loose"]]),
        "residual_rms": _stats([v["residual_rms"] for v in recs if v["success_loose"]]),
        "reprop_residual": _stats([v["reprop_arrival_residual"] for v in recs if v["success_loose"]]),
        "energy_error": _stats([v["energy_error"] for v in recs if v["success_loose"]]),
        "runtime_s": _stats([v["runtime_s"] for v in recs]),
        "nfev": _stats([v["total_nfev"] for v in recs]),
        "njev": _stats([v["total_njev"] for v in recs]),
        "switch_distribution": sw_dist,
    }


def compute_collocation_summary(arm_data: dict) -> dict:
    recs = list(arm_data["seeds"].values())
    n = len(recs)
    if n == 0:
        return {}
    n_loose = sum(1 for v in recs if v["success_loose"])
    n_strict = sum(1 for v in recs if v["success_strict"])
    sw_dist: dict[str, int] = {}
    for v in recs:
        if v["success_loose"]:
            k = str(v["n_switches"])
            sw_dist[k] = sw_dist.get(k, 0) + 1

    return {
        "n_total": n,
        "n_loose": n_loose,
        "rate_loose": n_loose / n,
        "ci95_loose": list(wilson_ci(n_loose, n)),
        "n_strict": n_strict,
        "rate_strict": n_strict / n,
        "ci95_strict": list(wilson_ci(n_strict, n)),
        "fuel_kg": _stats([v["fuel_kg"] for v in recs if v["success_loose"]]),
        "reprop_residual": _stats([v["reprop_arrival_residual"] for v in recs if v["success_loose"]]),
        "energy_error": _stats([v["energy_error"] for v in recs if v["success_loose"]]),
        "runtime_s": _stats([v["runtime_s"] for v in recs]),
        "iterations": _stats([v["iterations"] for v in recs]),
        "nfev": _stats([v["nfev"] for v in recs]),
        "njev": _stats([v["njev"] for v in recs]),
        "switch_distribution": sw_dist,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preregistered Held-Out Benchmark Runner (Seeds 2000..2099)")
    parser.add_argument("--arm", type=str, default="all",
                        choices=["all", "paired_srp_0.0", "paired_srp_0.5", "paired_srp_1.0",
                                 "ponn_scaled", "ponn_paper_u01", "collocation"],
                        help="Specific arm to run, or 'all'")
    parser.add_argument("--seeds-start", type=int, default=2000)
    parser.add_argument("--seeds-count", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=3600.0)
    args = parser.parse_args()

    t_global_start = time.perf_counter()
    data = _load()
    all_seeds = [args.seeds_start + i for i in range(args.seeds_count)]

    # --- 1. Paired SRP Arms ---
    paired_cases = [
        ("paired_srp_0.0", "none", 0.0),
        ("paired_srp_0.5", "radial", 0.5),
        ("paired_srp_1.0", "radial", 1.0),
    ]

    for arm_name, srp_mode, a_over_m in paired_cases:
        if args.arm not in ("all", arm_name):
            continue
        print(f"\n=======================================================")
        print(f"Running Arm: {arm_name} (SRP: {srp_mode}, A/m={a_over_m})")
        print(f"=======================================================")
        arm_data = data[arm_name]
        pending_seeds = [s for s in all_seeds if str(s) not in arm_data["seeds"]]
        print(f"  Total seeds: {len(all_seeds)}, Done: {len(arm_data['seeds'])}, Pending: {len(pending_seeds)}")

        if pending_seeds:
            pool_args = [(srp_mode, a_over_m, s) for s in pending_seeds]
            # Use imap_unordered for incremental checkpointing
            with mp.Pool(processes=args.n_workers) as pool:
                for seed, sh_res, hyb_res in pool.imap_unordered(run_paired_srp_worker, pool_args):
                    arm_data["seeds"][str(seed)] = {
                        "shooting": sh_res,
                        "hybrid": hyb_res,
                    }
                    arm_data["summary"] = compute_paired_srp_summary(arm_data)
                    _save(data)
                    c_done = len(arm_data["seeds"])
                    sh_ok = sum(1 for v in arm_data["seeds"].values() if v["shooting"]["success_strict"])
                    hyb_ok = sum(1 for v in arm_data["seeds"].values() if v["hybrid"]["success_strict"])
                    print(f"  [{arm_name}] {c_done:3d}/{len(all_seeds)} | seed {seed} | Shoot Strict: {sh_ok}/{c_done} | Hyb Strict: {hyb_ok}/{c_done} (+{100*(hyb_ok-sh_ok)/c_done:+.1f} pp)")
                    if time.perf_counter() - t_global_start > args.max_wall_s:
                        print(f"  [wall-timeout] Reached {args.max_wall_s}s limit.")
                        return

        arm_data["summary"] = compute_paired_srp_summary(arm_data)
        _save(data)
        summ = arm_data["summary"]
        if summ:
            comp = summ["comparison"]
            print(f"  --> {arm_name} Complete: Shooting {summ['shooting']['n_strict']}/{summ['n_total']} vs Hybrid {summ['hybrid']['n_strict']}/{summ['n_total']} (Delta: {comp['delta_strict_pp']:+.1f} pp, McNemar p={comp['mcnemar_exact_p']:.2e}, Verdict: {comp['prereg_verdict']})")

    # --- 2. PoNN Scaled Init Arm ---
    if args.arm in ("all", "ponn_scaled"):
        arm_name = "ponn_scaled_am0.0"
        print(f"\n=======================================================")
        print(f"Running Arm: {arm_name} (PoNN Scaled U(0,1), L=70->80)")
        print(f"=======================================================")
        arm_data = data[arm_name]
        pending_seeds = [s for s in all_seeds if str(s) not in arm_data["seeds"]]
        print(f"  Total seeds: {len(all_seeds)}, Done: {len(arm_data['seeds'])}, Pending: {len(pending_seeds)}")

        if pending_seeds:
            pool_args = [("scaled_u01", s) for s in pending_seeds]
            with mp.Pool(processes=args.n_workers) as pool:
                for seed, res in pool.imap_unordered(run_ponn_worker, pool_args):
                    arm_data["seeds"][str(seed)] = res
                    arm_data["summary"] = {
                        "coarse": compute_ponn_summary(arm_data, "coarse"),
                        "refined": compute_ponn_summary(arm_data, "refined"),
                    }
                    _save(data)
                    c_done = len(arm_data["seeds"])
                    loose_ok = sum(1 for v in arm_data["seeds"].values() if v.get("refined") and v["refined"].get("success_loose"))
                    strict_ok = sum(1 for v in arm_data["seeds"].values() if v.get("refined") and v["refined"].get("success_strict"))
                    print(f"  [{arm_name}] {c_done:3d}/{len(all_seeds)} | seed {seed} | Refined Loose: {loose_ok}/{c_done} | Strict: {strict_ok}/{c_done}")
                    if time.perf_counter() - t_global_start > args.max_wall_s:
                        print(f"  [wall-timeout] Reached {args.max_wall_s}s limit.")
                        return

        arm_data["summary"] = {
            "coarse": compute_ponn_summary(arm_data, "coarse"),
            "refined": compute_ponn_summary(arm_data, "refined"),
        }
        _save(data)
        ref_s = arm_data["summary"]["refined"]
        print(f"  --> {arm_name} Complete: Refined Loose {ref_s['n_loose']}/{ref_s['n_total']} ({100*ref_s['rate_loose']:.1f}%), Strict {ref_s['n_strict']}/{ref_s['n_total']} ({100*ref_s['rate_strict']:.1f}%)")

    # --- 3. PoNN Exact Paper U(0,1) Arm ---
    if args.arm in ("all", "ponn_paper_u01"):
        arm_name = "ponn_paper_u01_am0.0"
        print(f"\n=======================================================")
        print(f"Running Arm: {arm_name} (PoNN Exact Paper U(0,1), L=70->80)")
        print(f"=======================================================")
        arm_data = data[arm_name]
        pending_seeds = [s for s in all_seeds if str(s) not in arm_data["seeds"]]
        print(f"  Total seeds: {len(all_seeds)}, Done: {len(arm_data['seeds'])}, Pending: {len(pending_seeds)}")

        if pending_seeds:
            pool_args = [("paper_u01", s) for s in pending_seeds]
            with mp.Pool(processes=args.n_workers) as pool:
                for seed, res in pool.imap_unordered(run_ponn_worker, pool_args):
                    arm_data["seeds"][str(seed)] = res
                    arm_data["summary"] = {
                        "coarse": compute_ponn_summary(arm_data, "coarse"),
                        "refined": compute_ponn_summary(arm_data, "refined"),
                    }
                    _save(data)
                    c_done = len(arm_data["seeds"])
                    loose_ok = sum(1 for v in arm_data["seeds"].values() if v.get("refined") and v["refined"].get("success_loose"))
                    strict_ok = sum(1 for v in arm_data["seeds"].values() if v.get("refined") and v["refined"].get("success_strict"))
                    print(f"  [{arm_name}] {c_done:3d}/{len(all_seeds)} | seed {seed} | Refined Loose: {loose_ok}/{c_done} | Strict: {strict_ok}/{c_done}")
                    if time.perf_counter() - t_global_start > args.max_wall_s:
                        print(f"  [wall-timeout] Reached {args.max_wall_s}s limit.")
                        return

        arm_data["summary"] = {
            "coarse": compute_ponn_summary(arm_data, "coarse"),
            "refined": compute_ponn_summary(arm_data, "refined"),
        }
        _save(data)
        ref_s = arm_data["summary"]["refined"]
        print(f"  --> {arm_name} Complete: Refined Loose {ref_s['n_loose']}/{ref_s['n_total']} ({100*ref_s['rate_loose']:.1f}%), Strict {ref_s['n_strict']}/{ref_s['n_total']} ({100*ref_s['rate_strict']:.1f}%)")

    # --- 4. 180-Node LGL Collocation Arm ---
    if args.arm in ("all", "collocation"):
        arm_name = "collocation_am0.0"
        print(f"\n=======================================================")
        print(f"Running Arm: {arm_name} (180-Node LGL Collocation, CasADi+IPOPT)")
        print(f"=======================================================")
        arm_data = data[arm_name]
        pending_seeds = [s for s in all_seeds if str(s) not in arm_data["seeds"]]
        print(f"  Total seeds: {len(all_seeds)}, Done: {len(arm_data['seeds'])}, Pending: {len(pending_seeds)}")

        if pending_seeds:
            with mp.Pool(processes=args.n_workers) as pool:
                for seed, res in pool.imap_unordered(run_collocation_worker, pending_seeds):
                    arm_data["seeds"][str(seed)] = res
                    arm_data["summary"] = compute_collocation_summary(arm_data)
                    _save(data)
                    c_done = len(arm_data["seeds"])
                    loose_ok = sum(1 for v in arm_data["seeds"].values() if v["success_loose"])
                    strict_ok = sum(1 for v in arm_data["seeds"].values() if v["success_strict"])
                    print(f"  [{arm_name}] {c_done:3d}/{len(all_seeds)} | seed {seed} | Loose: {loose_ok}/{c_done} | Strict: {strict_ok}/{c_done}")
                    if time.perf_counter() - t_global_start > args.max_wall_s:
                        print(f"  [wall-timeout] Reached {args.max_wall_s}s limit.")
                        return

        arm_data["summary"] = compute_collocation_summary(arm_data)
        _save(data)
        col_s = arm_data["summary"]
        print(f"  --> {arm_name} Complete: Loose {col_s['n_loose']}/{col_s['n_total']} ({100*col_s['rate_loose']:.1f}%), Strict {col_s['n_strict']}/{col_s['n_total']} ({100*col_s['rate_strict']:.1f}%)")

    print(f"\nAll requested benchmark arms finished in {time.perf_counter() - t_global_start:.1f}s.")
    print(f"Results saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
