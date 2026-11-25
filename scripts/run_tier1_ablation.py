"""Tier 1.1: Preregistered Extraction-Level Ablation (Scenario A, fresh seeds 4000..4099).

Evaluates 7 paired arms on identical seeds 4000..4099 (N=100):
  - random_shooting: Indirect Shooting from random 7-D costates (stop_at_rho=1e-5)
  - hybrid_rho_1e0:      PoNN coarse to rho=1.0      -> Shooting continuation
  - hybrid_rho_2.98e-1:  PoNN coarse to rho=2.976e-1 -> Shooting continuation
  - hybrid_rho_1e-1:     PoNN coarse to rho=1e-1     -> Shooting continuation
  - hybrid_rho_1e-2:     PoNN coarse to rho=1e-2     -> Shooting continuation
  - hybrid_rho_1e-3:     PoNN coarse to rho=1e-3     -> Shooting continuation
  - hybrid_rho_final:    PoNN coarse to rho=1e-10    -> Shooting at terminal 1.833e-5

Exact costate extraction formulas at z0 = -1 (t = t0):
  lam_v(t0) = sigma(-1)^T beta_lv
  lam_r(t0) = -b^2 sigma'(-1)^T beta_lv
  lam_m(t0) = (sigma(-1) - sigma(1))^T beta_lm

Checkpoints per seed to results/tier1_extraction_ablation_4000_4099.json.
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

OUT_PATH = Path("results/tier1_extraction_ablation_4000_4099.json")
STOP_AT_RHO = 1e-5
SHOOT_FINAL_RHO = float([r for r in DEFAULT_RHO_SCHEDULE if r >= STOP_AT_RHO - 1e-15][-1])  # 1.832981e-5

EXTRACTION_LEVELS: list[tuple[str, float, tuple[float, ...]]] = [
    ("hybrid_rho_1e0", 1.0, tuple(r for r in DEFAULT_RHO_SCHEDULE if r >= STOP_AT_RHO - 1e-15)),
    ("hybrid_rho_2.98e-1", float(DEFAULT_RHO_SCHEDULE[1]), tuple(r for r in DEFAULT_RHO_SCHEDULE[1:] if r >= STOP_AT_RHO - 1e-15)),
    ("hybrid_rho_1e-1", 1e-1, (1e-1,) + tuple(r for r in DEFAULT_RHO_SCHEDULE if r < 1e-1 - 1e-15 and r >= STOP_AT_RHO - 1e-15)),
    ("hybrid_rho_1e-2", 1e-2, (1e-2,) + tuple(r for r in DEFAULT_RHO_SCHEDULE if r < 1e-2 - 1e-15 and r >= STOP_AT_RHO - 1e-15)),
    ("hybrid_rho_1e-3", 1e-3, (1e-3,) + tuple(r for r in DEFAULT_RHO_SCHEDULE if r < 1e-3 - 1e-15 and r >= STOP_AT_RHO - 1e-15)),
    ("hybrid_rho_final", float(DEFAULT_RHO_SCHEDULE[-1]), (SHOOT_FINAL_RHO,)),
]


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
        "q05": float(np.percentile(a, 5)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "q95": float(np.percentile(a, 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _extract_lam0(model: PaperPoNNModel, xi: np.ndarray) -> np.ndarray:
    """Exact costate extraction at z0 = -1 (t = t0)."""
    ev0 = model.evaluate(xi, z=np.array([-1.0]))
    return np.concatenate([ev0["lr"][0], ev0["lv"][0], [ev0["lm"][0]]])


def _pack_shoot(r, seed: int) -> dict[str, Any]:
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
        "arrival_residual": ar if np.isfinite(ar) else None,
        "reprop_arrival_residual": rp if np.isfinite(rp) else None,
        "n_switches": len(r.switches),
        "switch_times_days": [float(s["t_days"]) for s in r.switches],
        "thrust_sequence": r.thrust_sequence,
        "rho_reached": rho_r if np.isfinite(rho_r) else None,
        "shoot_nfev": int(r.n_propagations),
        "shoot_njev": int(r.n_jacobian_builds),
        "shoot_runtime_s": float(r.runtime_s),
    }


def run_seed_ablation(seed: int) -> dict[str, Any]:
    """Run all 7 paired arms for a single seed."""
    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0, label="scenario_a_ablation")
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)

    # 1. Baseline: Random-costate Indirect Shooting
    r_rand = shooter.solve(prob, seed=seed)
    out: dict[str, Any] = {"seed": seed, "random_shooting": _pack_shoot(r_rand, seed)}
    out["random_shooting"]["ponn_nfev"] = 0
    out["random_shooting"]["ponn_njev"] = 0
    out["random_shooting"]["ponn_runtime_s"] = 0.0
    out["random_shooting"]["total_runtime_s"] = out["random_shooting"]["shoot_runtime_s"]

    # 2. Nested PoNN coarse pass (L=70, n=140 CGL, scaled_u01) recording exact snapshots
    z_c = make_collocation_grid(140, "cgl")
    model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=seed)
    ops_c = model_c.operators(z_c)
    xi = model_c.initial_guess_paper(ops_c, seed=seed, init_mode="scaled_u01")
    ponn = PaperPoNNSolver()

    extracted: dict[str, dict[str, Any]] = {}
    cum_t = 0.0
    cum_nfev = 0
    cum_njev = 0

    for k, rho in enumerate(DEFAULT_RHO_SCHEDULE):
        t0 = time.perf_counter()
        xi, _, _, _, nfev_k, njev_k, _ = ponn._run_continuation(
            prob, model_c, z_c, xi, (rho,), start_scaled=(k > 0)
        )
        dt = time.perf_counter() - t0
        cum_t += dt
        cum_nfev += nfev_k
        cum_njev += njev_k

        if k == 0:
            extracted["hybrid_rho_1e0"] = {
                "lam0": _extract_lam0(model_c, xi),
                "ponn_runtime_s": cum_t,
                "ponn_nfev": cum_nfev,
                "ponn_njev": cum_njev,
            }
        elif k == 1:
            extracted["hybrid_rho_2.98e-1"] = {
                "lam0": _extract_lam0(model_c, xi),
                "ponn_runtime_s": cum_t,
                "ponn_nfev": cum_nfev,
                "ponn_njev": cum_njev,
            }
            # Branch 1 step to exact rho = 1e-1
            tb = time.perf_counter()
            xi_b, _, _, _, nf_b, nj_b, _ = ponn._run_continuation(
                prob, model_c, z_c, xi.copy(), (1e-1,), start_scaled=True
            )
            dtb = time.perf_counter() - tb
            extracted["hybrid_rho_1e-1"] = {
                "lam0": _extract_lam0(model_c, xi_b),
                "ponn_runtime_s": cum_t + dtb,
                "ponn_nfev": cum_nfev + nf_b,
                "ponn_njev": cum_njev + nj_b,
            }
        elif k == 3:
            # Branch 1 step to exact rho = 1e-2
            tb = time.perf_counter()
            xi_b, _, _, _, nf_b, nj_b, _ = ponn._run_continuation(
                prob, model_c, z_c, xi.copy(), (1e-2,), start_scaled=True
            )
            dtb = time.perf_counter() - tb
            extracted["hybrid_rho_1e-2"] = {
                "lam0": _extract_lam0(model_c, xi_b),
                "ponn_runtime_s": cum_t + dtb,
                "ponn_nfev": cum_nfev + nf_b,
                "ponn_njev": cum_njev + nj_b,
            }
        elif k == 5:
            # Branch 1 step to exact rho = 1e-3
            tb = time.perf_counter()
            xi_b, _, _, _, nf_b, nj_b, _ = ponn._run_continuation(
                prob, model_c, z_c, xi.copy(), (1e-3,), start_scaled=True
            )
            dtb = time.perf_counter() - tb
            extracted["hybrid_rho_1e-3"] = {
                "lam0": _extract_lam0(model_c, xi_b),
                "ponn_runtime_s": cum_t + dtb,
                "ponn_nfev": cum_nfev + nf_b,
                "ponn_njev": cum_njev + nj_b,
            }
        elif k == len(DEFAULT_RHO_SCHEDULE) - 1:
            extracted["hybrid_rho_final"] = {
                "lam0": _extract_lam0(model_c, xi),
                "ponn_runtime_s": cum_t,
                "ponn_nfev": cum_nfev,
                "ponn_njev": cum_njev,
            }

    # 3. Run Indirect Shooting continuation from each extracted costate
    for arm_name, rho_ext, shoot_sched in EXTRACTION_LEVELS:
        ext = extracted[arm_name]
        r_hyb = shooter.solve(prob, seed=seed, lam0_guess=ext["lam0"], rho_schedule=shoot_sched)
        rec = _pack_shoot(r_hyb, seed)
        rec["rho_ext"] = float(rho_ext)
        rec["ponn_nfev"] = int(ext["ponn_nfev"])
        rec["ponn_njev"] = int(ext["ponn_njev"])
        rec["ponn_runtime_s"] = float(ext["ponn_runtime_s"])
        rec["total_runtime_s"] = float(ext["ponn_runtime_s"] + rec["shoot_runtime_s"])
        out[arm_name] = rec

    return out


def compute_summary(data: dict) -> dict:
    seeds_dict = data["seeds"]
    seed_keys = sorted(seeds_dict.keys(), key=int)
    n = len(seed_keys)
    arms = ["random_shooting"] + [a for a, _, _ in EXTRACTION_LEVELS]
    summary: dict[str, Any] = {"n_seeds": n, "arms": {}, "mcnemar_vs_random": {}}

    for arm in arms:
        recs = [seeds_dict[k][arm] for k in seed_keys]
        n_strict = sum(1 for r in recs if r["success_strict"])
        n_loose = sum(1 for r in recs if r["success_loose"])
        ci_s = wilson_ci(n_strict, n)
        ci_l = wilson_ci(n_loose, n)
        succ_recs = [r for r in recs if r["success_strict"]]
        fail_recs = [r for r in recs if not r["success_strict"]]
        summary["arms"][arm] = {
            "n_strict": n_strict,
            "rate_strict": n_strict / n if n else 0.0,
            "ci_strict": [ci_s[0], ci_s[1]],
            "n_loose": n_loose,
            "rate_loose": n_loose / n if n else 0.0,
            "ci_loose": [ci_l[0], ci_l[1]],
            "fuel_kg_strict": _dist_stats([r["fuel_kg"] for r in succ_recs]),
            "reprop_strict": _dist_stats([r["reprop_arrival_residual"] for r in succ_recs]),
            "ponn_runtime_s": _dist_stats([r["ponn_runtime_s"] for r in recs]),
            "shoot_runtime_s": _dist_stats([r["shoot_runtime_s"] for r in recs]),
            "total_runtime_s": _dist_stats([r["total_runtime_s"] for r in recs]),
            "total_runtime_s_success": _dist_stats([r["total_runtime_s"] for r in succ_recs]),
            "total_runtime_s_failure": _dist_stats([r["total_runtime_s"] for r in fail_recs]),
            "ponn_nfev": _dist_stats([r["ponn_nfev"] for r in recs]),
            "ponn_njev": _dist_stats([r["ponn_njev"] for r in recs]),
            "shoot_nfev": _dist_stats([r["shoot_nfev"] for r in recs]),
            "shoot_njev": _dist_stats([r["shoot_njev"] for r in recs]),
        }

    # Exact paired McNemar vs random_shooting
    for arm, rho_ext, _ in EXTRACTION_LEVELS:
        n11 = n10 = n01 = n00 = 0
        for k in seed_keys:
            h = seeds_dict[k][arm]["success_strict"]
            s = seeds_dict[k]["random_shooting"]["success_strict"]
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
        chi2_stat = float(((abs(n10 - n01) - 1.0) ** 2) / disc) if disc > 0 else 0.0
        delta_pp = 100.0 * (summary["arms"][arm]["rate_strict"] - summary["arms"]["random_shooting"]["rate_strict"])
        summary["mcnemar_vs_random"][arm] = {
            "rho_ext": float(rho_ext),
            "n11": n11,
            "n10_hyb_only": n10,
            "n01_shoot_only": n01,
            "n00": n00,
            "delta_strict_pp": delta_pp,
            "p_exact_raw": p_exact,
            "chi2_cc": chi2_stat,
        }

    return summary


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
    return {
        "meta": {
            "description": "Tier 1.1 Extraction-Level Ablation on Scenario A (fresh seeds 4000..4099)",
            "prereg_addendum_commit": "38e2908",
            "seed_start": 4000,
            "seed_end": 4099,
            "stop_at_rho": STOP_AT_RHO,
            "threshold_strict": 1e-2,
            "threshold_loose": 5e-2,
        },
        "seeds": {},
        "summary": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=420.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    data = _load()
    target_seeds = [4000 + i for i in range(args.n_seeds)]
    todo = [s for s in target_seeds if str(s) not in data["seeds"]]
    print(f"[Tier 1.1 Ablation] {len(data['seeds'])}/{args.n_seeds} done, {len(todo)} remaining", flush=True)

    if todo:
        if args.workers <= 1:
            for s in todo:
                if time.perf_counter() - t_start > args.max_wall_s:
                    print("[checkpoint] wall limit reached", flush=True)
                    break
                res = run_seed_ablation(s)
                data["seeds"][str(s)] = res
                data["summary"] = compute_summary(data)
                _save(data)
                r_s = res["random_shooting"]["success_strict"]
                r_1 = res["hybrid_rho_2.98e-1"]["success_strict"]
                print(f"  seed={s} | rand={int(r_s)} hyb(0.298)={int(r_1)} | done={len(data['seeds'])}/{args.n_seeds} ({time.perf_counter()-t_start:.1f}s)", flush=True)
        else:
            with mp.Pool(processes=args.workers) as pool:
                for res in pool.imap_unordered(run_seed_ablation, todo):
                    s = res["seed"]
                    data["seeds"][str(s)] = res
                    data["summary"] = compute_summary(data)
                    _save(data)
                    r_s = res["random_shooting"]["success_strict"]
                    r_1 = res["hybrid_rho_2.98e-1"]["success_strict"]
                    print(f"  seed={s} | rand={int(r_s)} hyb(0.298)={int(r_1)} | done={len(data['seeds'])}/{args.n_seeds} ({time.perf_counter()-t_start:.1f}s)", flush=True)
                    if time.perf_counter() - t_start > args.max_wall_s:
                        print("[checkpoint] wall limit reached, terminating pool cleanly", flush=True)
                        pool.terminate()
                        break

    data["summary"] = compute_summary(data)
    _save(data)
    print(f"[Tier 1.1 Ablation] Saved {len(data['seeds'])}/{args.n_seeds} seeds to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
