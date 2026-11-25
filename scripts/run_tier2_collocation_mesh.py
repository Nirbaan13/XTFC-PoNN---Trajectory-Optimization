"""Tier 2.8: Collocation Mesh Study (Scenario A, fresh seeds 6000..6029, N=30 per mesh).

Evaluates:
  - Single-mesh LGL collocation at N in {60, 120, 180, 240}
  - Two-stage mesh refinement: N=120 -> N=240 warm-started by interpolating the N=120 solution
    onto the N=240 LGL nodes.

Reports strict/loose convergence with 95% Wilson CIs, discretization error (e_reprop and
|fuel - fuel_shoot|), and wall-clock runtime. Checkpoints to results/tier2_collocation_mesh_6000_6029.json.
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

import casadi as ca
import numpy as np

from trajectory_optimization.pseudospectral import lgl_nodes_and_weights
from trajectory_optimization.scenario_a import (
    ScenarioAProblem,
    ScenarioAShootingSolver,
    energy_balance_error_scenario_a,
    propagate_control_scenario_a,
)
from trajectory_optimization.scenario_a_collocation import (
    ScenarioACollocationSolver,
    _detect_throttle_switches,
)

OUT_PATH = Path("results/tier2_collocation_mesh_6000_6029.json")
MESHES = ["N60", "N120", "N180", "N240", "N120_to_N240_refined"]


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


def _pack_col(r, fuel_ref: float, extra_rt: float = 0.0) -> dict[str, Any]:
    rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
    te = float(r.terminal_error) if np.isfinite(r.terminal_error) else float("nan")
    mf = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
    phys_ok = bool(r.success and mf > 50.0 and te < 1e-4)
    f_kg = float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None
    return {
        "success_loose": bool(phys_ok and np.isfinite(rp) and rp < 5e-2),
        "success_strict": bool(phys_ok and np.isfinite(rp) and rp < 1e-2),
        "status": r.status,
        "fuel_kg": f_kg,
        "fuel_err_vs_shoot_kg": abs(f_kg - fuel_ref) if f_kg is not None else None,
        "terminal_error": te if np.isfinite(te) else None,
        "reprop_arrival_residual": rp if np.isfinite(rp) else None,
        "n_switches": len(r.switches),
        "iterations": int(r.iterations),
        "nfev": int(r.n_func_evals),
        "njev": int(r.n_jac_evals),
        "runtime_s": float(r.runtime_s + extra_rt),
    }


def solve_warmstarted_240(prob: ScenarioAProblem, r120, fuel_ref: float) -> dict[str, Any]:
    """Warm-start N=240 LGL collocation by interpolating the N=120 solution onto 240 LGL nodes."""
    solver240 = ScenarioACollocationSolver(n_nodes=240, max_iter=500, tol=1e-7)
    nlp, lbx, ubx, lbg, ubg = solver240._build_nlp(prob)
    tau240, _ = lgl_nodes_and_weights(240)
    t240 = 0.5 * (tau240 + 1.0) * prob.tf
    t120 = r120.t_tu

    R_g = np.column_stack([np.interp(t240, t120, r120.r[:, j]) for j in range(3)])
    V_g = np.column_stack([np.interp(t240, t120, r120.v[:, j]) for j in range(3)])
    M_g = np.interp(t240, t120, r120.m)
    D_g = np.clip(np.interp(t240, t120, r120.delta), 0.0, 1.0)
    A_g = np.column_stack([np.interp(t240, t120, r120.alpha[:, j]) for j in range(3)])
    A_g /= np.maximum(np.linalg.norm(A_g, axis=1, keepdims=True), 1e-12)
    x0 = np.concatenate([
        R_g.ravel(order="F"),
        V_g.ravel(order="F"),
        M_g,
        D_g,
        A_g.ravel(order="F"),
    ])

    t0 = time.perf_counter()
    res = nlp(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    rt2 = time.perf_counter() - t0
    stats = nlp.stats()
    status = str(stats.get("return_status", "unknown"))
    ok = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

    x_sol = np.asarray(res["x"], dtype=float).ravel()
    N = 240
    R = x_sol[0:3 * N].reshape((N, 3), order="F")
    V = x_sol[3 * N:6 * N].reshape((N, 3), order="F")
    M = x_sol[6 * N:7 * N]
    Dlt = x_sol[7 * N:8 * N]
    A = x_sol[8 * N:11 * N].reshape((N, 3), order="F")
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

    term_err = float(np.linalg.norm(np.concatenate([R[-1] - prob.rf, V[-1] - prob.vf])))
    prop = propagate_control_scenario_a(prob, t240, Dlt, A)
    if prop.success and prop.y.shape[1] > 0:
        rp = float(np.linalg.norm(np.concatenate([prop.y[0:3, -1] - prob.rf, prop.y[3:6, -1] - prob.vf])))
    else:
        rp = float("nan")
    mf_kg = float(M[-1] * prob.m_ref_kg)
    f_kg = float((prob.m0 - M[-1]) * prob.m_ref_kg)
    sw, _ = _detect_throttle_switches(t240 * prob.t_ref_days, Dlt)
    phys_ok = bool(ok and mf_kg > 50.0 and term_err < 1e-4)
    return {
        "success_loose": bool(phys_ok and np.isfinite(rp) and rp < 5e-2),
        "success_strict": bool(phys_ok and np.isfinite(rp) and rp < 1e-2),
        "status": status,
        "fuel_kg": f_kg if np.isfinite(f_kg) else None,
        "fuel_err_vs_shoot_kg": abs(f_kg - fuel_ref) if np.isfinite(f_kg) else None,
        "terminal_error": term_err if np.isfinite(term_err) else None,
        "reprop_arrival_residual": rp if np.isfinite(rp) else None,
        "n_switches": len(sw),
        "iterations": int(stats.get("iter_count", 0)),
        "nfev": int(stats.get("n_call_nlp_f", 0)),
        "njev": int(stats.get("n_call_nlp_jac_g", 0)),
        "stage1_runtime_s": float(r120.runtime_s),
        "stage2_runtime_s": float(rt2),
        "runtime_s": float(r120.runtime_s + rt2),
    }


_COL_CACHE: dict[int, ScenarioACollocationSolver] = {}


def _get_col(N: int) -> ScenarioACollocationSolver:
    if N not in _COL_CACHE:
        _COL_CACHE[N] = ScenarioACollocationSolver(n_nodes=N, max_iter=500, tol=1e-7)
    return _COL_CACHE[N]


def _run_seed_all_meshes(args_tuple: tuple[int, float]) -> tuple[int, dict[str, Any]]:
    seed, fuel_ref = args_tuple
    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0, label="tier2_colloc")
    out: dict[str, Any] = {}
    r120 = None
    for N in (60, 120, 180, 240):
        sol = _get_col(N)
        r = sol.solve(prob, seed=seed)
        out[f"N{N}"] = _pack_col(r, fuel_ref)
        if N == 120:
            r120 = r
    out["N120_to_N240_refined"] = solve_warmstarted_240(prob, r120, fuel_ref)
    return seed, out


def summarize(data: dict) -> dict:
    seeds = data["seeds"]
    keys = sorted(seeds.keys(), key=int)
    n = len(keys)
    out: dict[str, Any] = {"n_seeds": n, "meshes": {}}
    for m in MESHES:
        recs = [seeds[k][m] for k in keys]
        ns = sum(1 for r in recs if r["success_strict"])
        nl = sum(1 for r in recs if r["success_loose"])
        s_recs = [r for r in recs if r["success_strict"]]
        l_recs = [r for r in recs if r["success_loose"]]
        out["meshes"][m] = {
            "n_strict": ns,
            "rate_strict": ns / n if n else 0.0,
            "ci_strict": list(wilson_ci(ns, n)),
            "n_loose": nl,
            "rate_loose": nl / n if n else 0.0,
            "ci_loose": list(wilson_ci(nl, n)),
            "reprop_all": _dist_stats([r["reprop_arrival_residual"] for r in recs]),
            "reprop_loose": _dist_stats([r["reprop_arrival_residual"] for r in l_recs]),
            "fuel_err_loose": _dist_stats([r["fuel_err_vs_shoot_kg"] for r in l_recs]),
            "fuel_err_strict": _dist_stats([r["fuel_err_vs_shoot_kg"] for r in s_recs]),
            "runtime_s": _dist_stats([r["runtime_s"] for r in recs]),
        }
    return out


def _save(data: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, OUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=520.0)
    parser.add_argument("--max-batch", type=int, default=2)
    args = parser.parse_args()

    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text())
        fuel_ref = float(data["meta"]["shooting_ref_fuel_kg"])
    else:
        sh = ScenarioAShootingSolver(stop_at_rho=1e-5).solve(ScenarioAProblem(), seed=1000)
        fuel_ref = float(sh.fuel_kg)
        data = {
            "meta": {
                "description": "Tier 2.8 Collocation Mesh Study (N in {60, 120, 180, 240} + N120->N240 refinement, seeds 6000..6029)",
                "prereg_addendum_commit": "38e2908",
                "shooting_ref_fuel_kg": fuel_ref,
            },
            "seeds": {},
            "summary": {},
        }

    t_start = time.perf_counter()
    todo = [(6000 + i, fuel_ref) for i in range(args.n_seeds) if str(6000 + i) not in data["seeds"]]
    if args.max_batch > 0:
        todo = todo[:args.max_batch]
    print(f"[Tier 2.8 Colloc Mesh] {len(data['seeds'])}/{args.n_seeds} done, {len(todo)} remaining", flush=True)

    if todo:
        with mp.Pool(processes=args.workers) as pool:
            for s, rec in pool.imap_unordered(_run_seed_all_meshes, todo):
                data["seeds"][str(s)] = rec
                data["summary"] = summarize(data)
                _save(data)
                print(
                    f"  seed={s} | N60={int(rec['N60']['success_strict'])} N120={int(rec['N120']['success_strict'])} "
                    f"N180={int(rec['N180']['success_strict'])} N240={int(rec['N240']['success_strict'])} "
                    f"ref={int(rec['N120_to_N240_refined']['success_strict'])} | {len(data['seeds'])}/{args.n_seeds} ({time.perf_counter()-t_start:.1f}s)",
                    flush=True,
                )
                if time.perf_counter() - t_start > args.max_wall_s:
                    print("[checkpoint] wall limit reached, terminating pool cleanly", flush=True)
                    pool.terminate()
                    break

    data["summary"] = summarize(data)
    _save(data)
    print(f"[Tier 2.8 Colloc Mesh] Saved {len(data['seeds'])}/{args.n_seeds} to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
