"""Phase B runner: 100-seed 3-Method Benchmark on Scenario A.

Benchmarks on identical ScenarioAProblem (Table 1, Sun gravity only):
  1. Validated PoNN (frozen Phase A2 settings — linked/copied from Phase A3 results)
  2. Indirect Shooting (100 random 7-D initial costates, 20-step rho continuation)
  3. 180-Node LGL Collocation (100 random initial trajectory guesses, CasADi + IPOPT)

Checkpoints per seed to results/scenario_a_phase_b.json and backs up to scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from trajectory_optimization.scenario_a import (
    ScenarioAProblem,
    ScenarioAShootingSolver,
)
from trajectory_optimization.scenario_a_collocation import ScenarioACollocationSolver

OUT_PATH = Path("results/scenario_a_phase_b.json")
A3_PATH = Path("results/scenario_a_phase_a3.json")
SCRATCH_BACKUP = Path(
    "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/"
    "2f3a08db-5952-4e2e-99ea-99a0ed91d155/scratch/scenario_a_phase_b.json"
)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _save(data: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUT_PATH)
    try:
        SCRATCH_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_PATH, SCRATCH_BACKUP)
    except Exception:
        pass


def _load() -> dict:
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            return json.load(f)
    return {
        "shooting": {},
        "collocation": {},
        "ponn_scaled_u01_ref": {},
        "ponn_paper_u01_ref": {},
        "summary": {},
    }


def _stats(vals: list[float]) -> dict | None:
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


def _summarize(records: dict, conv_key: str = "success") -> dict:
    vals = list(records.values())
    n = len(vals)
    if n == 0:
        return {"n": 0}
    conv = [v for v in vals if v.get(conv_key)]
    conv_1e2 = [
        v for v in vals
        if v.get(conv_key) and v.get("reprop_arrival_residual") is not None and v["reprop_arrival_residual"] < 1e-2
    ]
    sw_dist: dict[str, int] = {}
    for v in conv:
        k = str(v.get("n_switches", 0))
        sw_dist[k] = sw_dist.get(k, 0) + 1
    return {
        "n_total": n,
        "n_converged": len(conv),
        "rate": len(conv) / n,
        "ci95": list(wilson_ci(len(conv), n)),
        "n_converged_reprop_1e2": len(conv_1e2),
        "rate_reprop_1e2": len(conv_1e2) / n,
        "ci95_reprop_1e2": list(wilson_ci(len(conv_1e2), n)),
        "fuel_kg": _stats([v["fuel_kg"] for v in conv]),
        "reprop_arrival_residual": _stats([v["reprop_arrival_residual"] for v in conv]),
        "energy_error": _stats([v["energy_error"] for v in conv]),
        "runtime_s_all": _stats([v["runtime_s"] for v in vals]),
        "runtime_s_conv": _stats([v["runtime_s"] for v in conv]),
        "nfev_all": _stats([v["nfev"] for v in vals]),
        "njev_all": _stats([v["njev"] for v in vals]),
        "switch_count_distribution_conv": sw_dist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--max-wall-s", type=float, default=480.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    prob = ScenarioAProblem()
    data = _load()

    # Sync frozen PoNN results from Phase A3 if available
    if A3_PATH.exists():
        with A3_PATH.open() as f:
            a3 = json.load(f)
        for sk, rec in a3.get("seeds_scaled_u01", {}).items():
            rr = rec["refined"]
            data["ponn_scaled_u01_ref"][sk] = {
                "seed": int(sk),
                "success": bool(rr["success_relaxed_5e2"]),
                "fuel_kg": rr["fuel_kg"],
                "reprop_arrival_residual": rr["reprop_arrival_residual"],
                "energy_error": rr["energy_error"],
                "n_switches": rr["n_switches"],
                "switch_times_days": rr["switch_times_days"],
                "thrust_sequence": rr["thrust_sequence"],
                "nfev": rr["total_nfev"],
                "njev": rr["total_njev"],
                "runtime_s": rr["runtime_s"],
                "residual_rms": rr["residual_rms"],
            }
        for sk, rec in a3.get("seeds_paper_u01", {}).items():
            rr = rec["refined"]
            data["ponn_paper_u01_ref"][sk] = {
                "seed": int(sk),
                "success": bool(rr["success_relaxed_5e2"]),
                "fuel_kg": rr["fuel_kg"],
                "reprop_arrival_residual": rr["reprop_arrival_residual"],
                "energy_error": rr["energy_error"],
                "n_switches": rr["n_switches"],
                "switch_times_days": rr["switch_times_days"],
                "thrust_sequence": rr["thrust_sequence"],
                "nfev": rr["total_nfev"],
                "njev": rr["total_njev"],
                "runtime_s": rr["runtime_s"],
                "residual_rms": rr["residual_rms"],
            }
        _save(data)

    # 1. Indirect Shooting — 100 random seeds
    print(f"=== Phase B: Indirect Shooting ({len(data['shooting'])}/{args.n_seeds} done) ===")
    shooter = ScenarioAShootingSolver(stop_at_rho=1e-5)
    for i in range(args.n_seeds):
        if time.perf_counter() - t_start > args.max_wall_s:
            print(f"[checkpoint] wall limit reached in shooting at {len(data['shooting'])}/{args.n_seeds}")
            break
        seed = 1000 + i
        sk = str(seed)
        if sk in data["shooting"]:
            continue
        r = shooter.solve(prob, seed=seed)
        data["shooting"][sk] = {
            "seed": seed,
            "success": bool(r.success),
            "status": r.status,
            "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
            "arrival_residual": float(r.arrival_residual) if np.isfinite(r.arrival_residual) else None,
            "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
            "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
            "n_switches": len(r.switches),
            "switch_times_days": [float(s["t_days"]) for s in r.switches],
            "thrust_sequence": r.thrust_sequence,
            "rho_reached": float(r.rho_reached) if r.rho_reached is not None else None,
            "nfev": int(r.n_propagations),
            "njev": int(r.n_jacobian_builds),
            "runtime_s": float(r.runtime_s),
        }
        if (i + 1) % 5 == 0 or i == 0:
            n_ok = sum(1 for v in data["shooting"].values() if v["success"])
            print(f"  [shooting] {len(data['shooting']):3d}/{args.n_seeds} | s={seed} ok={r.success} fuel={r.fuel_kg:7.2f} rp={r.reprop_arrival_residual:.2e} {r.runtime_s:.2f}s | cum_ok={n_ok}/{len(data['shooting'])}")
        _save(data)

    # 2. 180-Node LGL Collocation — 100 random seeds
    print(f"=== Phase B: 180-Node LGL Collocation ({len(data['collocation'])}/{args.n_seeds} done) ===")
    col_solver = ScenarioACollocationSolver(n_nodes=180, max_iter=500, tol=1e-7)
    for i in range(args.n_seeds):
        if time.perf_counter() - t_start > args.max_wall_s:
            print(f"[checkpoint] wall limit reached in collocation at {len(data['collocation'])}/{args.n_seeds}")
            break
        seed = 1000 + i
        sk = str(seed)
        if sk in data["collocation"]:
            continue
        r = col_solver.solve(prob, seed=seed)
        data["collocation"][sk] = {
            "seed": seed,
            "success": bool(r.success),
            "status": r.status,
            "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
            "terminal_error": float(r.terminal_error) if np.isfinite(r.terminal_error) else None,
            "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
            "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
            "n_switches": len(r.switches),
            "switch_times_days": [float(s["t_days"]) for s in r.switches],
            "thrust_sequence": r.thrust_sequence,
            "iterations": int(r.iterations),
            "nfev": int(r.n_func_evals),
            "njev": int(r.n_jac_evals),
            "runtime_s": float(r.runtime_s),
        }
        if (i + 1) % 5 == 0 or i == 0:
            n_ok = sum(1 for v in data["collocation"].values() if v["success"])
            print(f"  [colloc  ] {len(data['collocation']):3d}/{args.n_seeds} | s={seed} ok={r.success} fuel={r.fuel_kg:7.2f} rp={r.reprop_arrival_residual:.2e} it={r.iterations} {r.runtime_s:.1f}s | cum_ok={n_ok}/{len(data['collocation'])}")
        _save(data)

    data["summary"] = {
        "shooting": _summarize(data["shooting"]),
        "collocation": _summarize(data["collocation"]),
        "ponn_scaled_u01_ref": _summarize(data["ponn_scaled_u01_ref"]),
        "ponn_paper_u01_ref": _summarize(data["ponn_paper_u01_ref"]),
    }
    _save(data)
    print("=== PHASE B SUMMARY ===")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
