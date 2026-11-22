"""Phase A3 runner: Detailed Scenario A reproduction + 100-seed random-init study.

Saves all raw per-seed metrics and single-seed time-series to:
    results/scenario_a_phase_a3.json

Checkpoints after every seed and gracefully exits before wall-time limit so it
can be resumed cleanly across multiple command chunks.
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

from trajectory_optimization.ponn_paper import PaperPoNNSolver
from trajectory_optimization.scenario_a import (
    DEFAULT_RHO_SCHEDULE,
    ScenarioAProblem,
    ScenarioAShootingSolver,
)
from trajectory_optimization.scenario_a_collocation import ScenarioACollocationSolver

OUT_PATH = Path("results/scenario_a_phase_a3.json")
SCRATCH_BACKUP = Path(
    "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/"
    "2f3a08db-5952-4e2e-99ea-99a0ed91d155/scratch/scenario_a_phase_a3.json"
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
        "meta": {
            "paper": "D'Ambrosio & Furfaro, Aerospace 2024, 11(3), 228",
            "paper_checks_only": {
                "shooting_fuel_kg": 396.06,
                "ponn_refined_fuel_kg": 396.85,
                "switches_approx_days": [47, 66, 148, 292],
                "sequence": "ON-OFF-ON-OFF-ON",
            },
        },
        "single_seed": {},
        "seeds_scaled_u01": {},
        "seeds_paper_u01": {},
    }


def _pack_ponn_summary(res) -> dict:
    return {
        "success": bool(res.success),
        "success_strict_1e2": bool(
            res.residual_rms < 1.5e-2 and np.isfinite(res.reprop_arrival_residual) and res.reprop_arrival_residual < 1e-2
        ),
        "success_relaxed_5e2": bool(
            res.residual_rms < 1.5e-2 and np.isfinite(res.reprop_arrival_residual) and res.reprop_arrival_residual < 5e-2
        ),
        "status": res.status,
        "n_neurons": int(res.n_neurons),
        "n_points": int(res.n_points),
        "fuel_kg": float(res.fuel_kg),
        "final_mass_kg": float(res.final_mass_kg),
        "residual_norm": float(res.residual_norm),
        "residual_rms": float(res.residual_rms),
        "residual_max": float(res.residual_max),
        "reprop_arrival_residual": float(res.reprop_arrival_residual) if np.isfinite(res.reprop_arrival_residual) else None,
        "energy_error": float(res.energy_error) if np.isfinite(res.energy_error) else None,
        "n_switches": int(len(res.switches)),
        "switch_times_days": [float(s["t_days"]) for s in res.switches],
        "thrust_sequence": res.thrust_sequence,
        "rho_reached": float(res.rho_reached),
        "total_nfev": int(res.total_nfev),
        "total_njev": int(res.total_njev),
        "runtime_s": float(res.runtime_s),
        "refined": bool(res.refined),
    }


def _pack_ponn_full(res, stride: int = 5) -> dict:
    d = _pack_ponn_summary(res)
    idx = np.arange(0, len(res.t_days), stride)
    if idx[-1] != len(res.t_days) - 1:
        idx = np.append(idx, len(res.t_days) - 1)
    d["series"] = {
        "t_days": [float(x) for x in res.t_days[idx]],
        "r": [[float(c) for c in row] for row in res.Y14[idx, 0:3]],
        "v": [[float(c) for c in row] for row in res.Y14[idx, 3:6]],
        "m_kg": [float(x * 1000.0) for x in res.Y14[idx, 6]],
        "lm": [float(x) for x in res.Y14[idx, 13]],
        "S": [float(x) for x in res.S[idx]],
        "delta": [float(x) for x in res.delta[idx]],
        "alpha": [[float(c) for c in row] for row in res.alpha[idx]],
    }
    d["level_log"] = res.level_log
    d["loss_history"] = [float(x) for x in res.loss_history]
    return d


def _summarize_seed_suite(records: dict, key: str = "refined") -> dict:
    vals = list(records.values())
    n = len(vals)
    if n == 0:
        return {"n": 0}
    sub = [v[key] for v in vals if v.get(key) is not None]
    n_sub = len(sub)
    conv_5e2 = [r for r in sub if r["success_relaxed_5e2"]]
    conv_1e2 = [r for r in sub if r["success_strict_1e2"]]
    conv_rms = [r for r in sub if r["residual_rms"] < 1.5e-2 and r["final_mass_kg"] > 50.0]
    conv_4sw = [r for r in conv_5e2 if r["n_switches"] == 4 and r["thrust_sequence"] == "ON-OFF-ON-OFF-ON"]

    def stats(arr):
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

    sw_counts: dict[str, int] = {}
    for r in sub:
        k_sw = str(r["n_switches"])
        sw_counts[k_sw] = sw_counts.get(k_sw, 0) + 1

    ci_5e2 = wilson_ci(len(conv_5e2), n)
    ci_1e2 = wilson_ci(len(conv_1e2), n)
    ci_rms = wilson_ci(len(conv_rms), n)
    return {
        "n_total": n,
        "n_evaluated": n_sub,
        "converged_reprop_5e2": len(conv_5e2),
        "rate_reprop_5e2": len(conv_5e2) / n,
        "ci95_reprop_5e2": list(ci_5e2),
        "converged_reprop_1e2": len(conv_1e2),
        "rate_reprop_1e2": len(conv_1e2) / n,
        "ci95_reprop_1e2": list(ci_1e2),
        "converged_rms_only": len(conv_rms),
        "rate_rms_only": len(conv_rms) / n,
        "ci95_rms_only": list(ci_rms),
        "converged_exact_4sw_seq": len(conv_4sw),
        "fuel_kg_converged_5e2": stats([r["fuel_kg"] for r in conv_5e2]),
        "reprop_converged_5e2": stats([r["reprop_arrival_residual"] for r in conv_5e2 if r["reprop_arrival_residual"] is not None]),
        "energy_err_converged_5e2": stats([r["energy_error"] for r in conv_5e2 if r["energy_error"] is not None]),
        "runtime_s_all": stats([r["runtime_s"] for r in sub]),
        "nfev_all": stats([r["total_nfev"] for r in sub]),
        "njev_all": stats([r["total_njev"] for r in sub]),
        "switch_count_distribution": sw_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--max-wall-s", type=float, default=480.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    prob = ScenarioAProblem()
    data = _load()

    # 1. Single-seed detailed comparisons (seed=1000)
    ss = data["single_seed"]
    if "shooting" not in ss:
        print("=== Single-seed: Phase A1 Indirect Shooting ===")
        sh = ScenarioAShootingSolver(stop_at_rho=1e-5).solve(prob, seed=1000)
        idx = np.linspace(0, len(sh.t_days) - 1, 401, dtype=int)
        ss["shooting"] = {
            "success": bool(sh.success),
            "fuel_kg": float(sh.fuel_kg),
            "final_mass_kg": float(sh.final_mass_kg),
            "arrival_residual": float(sh.arrival_residual),
            "reprop_arrival_residual": float(sh.reprop_arrival_residual),
            "energy_error": float(sh.energy_error),
            "n_switches": len(sh.switches),
            "switch_times_days": [float(s["t_days"]) for s in sh.switches],
            "thrust_sequence": sh.thrust_sequence,
            "lam0": [float(x) for x in sh.lam0],
            "nfev": int(sh.n_propagations),
            "njev": int(sh.n_jacobian_builds),
            "runtime_s": float(sh.runtime_s),
            "series": {
                "t_days": [float(x) for x in sh.t_days[idx]],
                "r": [[float(c) for c in row] for row in sh.Y[idx, 0:3]],
                "v": [[float(c) for c in row] for row in sh.Y[idx, 3:6]],
                "m_kg": [float(x * 1000.0) for x in sh.Y[idx, 6]],
                "lm": [float(x) for x in sh.Y[idx, 13]],
                "S": [float(x) for x in sh.S[idx]],
                "delta": [float(x) for x in sh.delta[idx]],
                "alpha": [[float(c) for c in row] for row in sh.alpha[idx]],
            },
        }
        print(f"  Shooting: fuel={sh.fuel_kg:.4f} kg, sw={[round(s['t_days'],2) for s in sh.switches]}, reprop={sh.reprop_arrival_residual:.2e}, {sh.runtime_s:.2f}s")
        _save(data)

    if "collocation" not in ss:
        print("=== Single-seed: 180-Node LGL Collocation ===")
        col = ScenarioACollocationSolver(n_nodes=180).solve(prob, seed=None)
        ss["collocation"] = {
            "success": bool(col.success),
            "fuel_kg": float(col.fuel_kg),
            "final_mass_kg": float(col.final_mass_kg),
            "terminal_error": float(col.terminal_error),
            "reprop_arrival_residual": float(col.reprop_arrival_residual),
            "energy_error": float(col.energy_error),
            "n_switches": len(col.switches),
            "switch_times_days": [float(s["t_days"]) for s in col.switches],
            "thrust_sequence": col.thrust_sequence,
            "iterations": int(col.iterations),
            "nfev": int(col.n_func_evals),
            "njev": int(col.n_jac_evals),
            "runtime_s": float(col.runtime_s),
            "series": {
                "t_days": [float(x) for x in col.t_days],
                "r": [[float(c) for c in row] for row in col.r],
                "v": [[float(c) for c in row] for row in col.v],
                "m_kg": [float(x * 1000.0) for x in col.m],
                "delta": [float(x) for x in col.delta],
                "alpha": [[float(c) for c in row] for row in col.alpha],
            },
        }
        print(f"  Collocation: fuel={col.fuel_kg:.4f} kg, sw={[round(s['t_days'],2) for s in col.switches]}, reprop={col.reprop_arrival_residual:.2e}, {col.runtime_s:.2f}s")
        _save(data)

    if "ponn_refined_uniform" not in ss:
        print("=== Single-seed: PoNN Coarse (L=70, n=140 CGL) & Refined (L=80, n=540 uniform) ===")
        ponn = PaperPoNNSolver(
            n_neurons=70, n_points=140, solver_algo="lm_svd",
            basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        )
        rc, rr_u = ponn.solve(prob, seed=1000, refine=True, refine_neurons=80, points_per_switch=100, refine_grid_mode="uniform_dense")
        ss["ponn_coarse"] = _pack_ponn_full(rc)
        ss["ponn_refined_uniform"] = _pack_ponn_full(rr_u)
        print(f"  PoNN Coarse         : fuel={rc.fuel_kg:.4f} kg, sw={[round(s['t_days'],2) for s in rc.switches]}, rms={rc.residual_rms:.2e}, reprop={rc.reprop_arrival_residual:.2e}, {rc.runtime_s:.2f}s")
        print(f"  PoNN Refined (uni)  : fuel={rr_u.fuel_kg:.4f} kg, sw={[round(s['t_days'],2) for s in rr_u.switches]}, rms={rr_u.residual_rms:.2e}, reprop={rr_u.reprop_arrival_residual:.2e}, {rr_u.runtime_s:.2f}s")
        _save(data)

    if "ponn_refined_cgl" not in ss:
        print("=== Single-seed: PoNN Refined (L=80, n=540 CGL & literal switch_bracket) ===")
        ponn = PaperPoNNSolver(
            n_neurons=70, n_points=140, solver_algo="lm_svd",
            basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        )
        _, rr_c = ponn.solve(prob, seed=1000, refine=True, refine_neurons=80, points_per_switch=100, refine_grid_mode="cgl_dense")
        _, rr_b = ponn.solve(prob, seed=1000, refine=True, refine_neurons=80, points_per_switch=100, refine_grid_mode="switch_bracket")
        ss["ponn_refined_cgl"] = _pack_ponn_full(rr_c)
        ss["ponn_refined_bracket"] = _pack_ponn_full(rr_b)
        print(f"  PoNN Refined (cgl)  : fuel={rr_c.fuel_kg:.4f} kg, sw={[round(s['t_days'],2) for s in rr_c.switches]}, rms={rr_c.residual_rms:.2e}, reprop={rr_c.reprop_arrival_residual:.2e}, {rr_c.runtime_s:.2f}s")
        print(f"  PoNN Refined (brkt) : fuel={rr_b.fuel_kg:.4f} kg, sw={[round(s['t_days'],2) for s in rr_b.switches]}, rms={rr_b.residual_rms:.2e}, reprop={rr_b.reprop_arrival_residual:.2e}, {rr_b.runtime_s:.2f}s")
        _save(data)

    if "ponn_literal_uni140" not in ss:
        print("=== Single-seed: PoNN Literal (L=70, n=140 uniform, chebyshev, paper_u01) ===")
        ponn_lit = PaperPoNNSolver(
            n_neurons=70, n_points=140, solver_algo="lm_svd",
            basis_type="chebyshev", grid_type="uniform", init_mode="paper_u01",
        )
        rl, _ = ponn_lit.solve(prob, seed=1000, refine=False)
        ss["ponn_literal_uni140"] = _pack_ponn_full(rl)
        print(f"  PoNN Literal uni140 : fuel={rl.fuel_kg:.4f} kg, sw={len(rl.switches)}, rms={rl.residual_rms:.2e}, reprop={rl.reprop_arrival_residual:.2e}, {rl.runtime_s:.2f}s")
        _save(data)

    # 2. 100-seed Suite A: exact unscaled paper_u01 (CGL n=140 -> refined uniform_dense n=540 if step 0 converges)
    print(f"=== 100-seed Suite 1: Exact paper U(0,1) init ({len(data['seeds_paper_u01'])}/{args.n_seeds} done) ===")
    ponn_u01 = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="paper_u01",
        max_iter_first=250, max_iter_cont=25,
    )
    for idx_s in range(args.n_seeds):
        if time.perf_counter() - t_start > args.max_wall_s:
            print(f"[checkpoint] wall limit reached during paper_u01 at {len(data['seeds_paper_u01'])}/{args.n_seeds}")
            break
        seed = 1000 + idx_s
        skey = str(seed)
        if skey in data["seeds_paper_u01"]:
            continue
        rc, rr = ponn_u01.solve(
            prob, seed=seed, refine=True, refine_neurons=80,
            points_per_switch=100, refine_grid_mode="uniform_dense",
            skip_refine_if_coarse_diverged=True,
        )
        if rr is not None:
            ref_pack = _pack_ponn_summary(rr)
            ref_pack["refinement_skipped"] = False
        else:
            ref_pack = _pack_ponn_summary(rc)
            ref_pack["refinement_skipped"] = True
        data["seeds_paper_u01"][skey] = {
            "seed": seed,
            "coarse": _pack_ponn_summary(rc),
            "refined": ref_pack,
        }
        if (idx_s + 1) % 5 == 0 or idx_s == 0:
            n_ok = sum(1 for v in data["seeds_paper_u01"].values() if v["refined"]["success_relaxed_5e2"])
            rp_s = ref_pack["reprop_arrival_residual"]
            rp_str = f"{rp_s:.2e}" if rp_s is not None else "nan"
            print(f"  [paper_u01] {len(data['seeds_paper_u01']):3d}/{args.n_seeds} | seed={seed} ok={ref_pack['success_relaxed_5e2']} fuel={ref_pack['fuel_kg']:7.2f} reprop={rp_str} {ref_pack['runtime_s']:.1f}s | cum_ok={n_ok}/{len(data['seeds_paper_u01'])}")
        _save(data)

    # 3. 100-seed Suite B: scaled_u01 (CGL n=140 -> refined uniform_dense n=540)
    print(f"=== 100-seed Suite 2: Scaled U(0,1) init ({len(data['seeds_scaled_u01'])}/{args.n_seeds} done) ===")
    ponn_sc = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    for idx_s in range(args.n_seeds):
        if time.perf_counter() - t_start > args.max_wall_s:
            print(f"[checkpoint] wall limit reached during scaled_u01 at {len(data['seeds_scaled_u01'])}/{args.n_seeds}")
            break
        seed = 1000 + idx_s
        skey = str(seed)
        if skey in data["seeds_scaled_u01"]:
            continue
        rc, rr = ponn_sc.solve(
            prob, seed=seed, refine=True, refine_neurons=80,
            points_per_switch=100, refine_grid_mode="uniform_dense",
        )
        data["seeds_scaled_u01"][skey] = {
            "seed": seed,
            "coarse": _pack_ponn_summary(rc),
            "refined": _pack_ponn_summary(rr),
        }
        if (idx_s + 1) % 5 == 0 or idx_s == 0:
            n_ok = sum(1 for v in data["seeds_scaled_u01"].values() if v["refined"]["success_relaxed_5e2"])
            rp_s = rr.reprop_arrival_residual
            print(f"  [scaled_u01] {len(data['seeds_scaled_u01']):3d}/{args.n_seeds} | seed={seed} ok={rr.success} fuel={rr.fuel_kg:7.2f} reprop={rp_s:.2e} {rr.runtime_s:.1f}s | cum_ok={n_ok}/{len(data['seeds_scaled_u01'])}")
        _save(data)

    data["summary"] = {
        "paper_u01_coarse": _summarize_seed_suite(data["seeds_paper_u01"], "coarse"),
        "paper_u01_refined": _summarize_seed_suite(data["seeds_paper_u01"], "refined"),
        "scaled_u01_coarse": _summarize_seed_suite(data["seeds_scaled_u01"], "coarse"),
        "scaled_u01_refined": _summarize_seed_suite(data["seeds_scaled_u01"], "refined"),
    }
    _save(data)
    print("=== PHASE A3 SUMMARY ===")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
