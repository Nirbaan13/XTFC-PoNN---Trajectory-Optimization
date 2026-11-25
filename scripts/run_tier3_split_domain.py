"""Tier 3.9: Split-Domain PoNN Refinement with Mandatory 5-Seed Go/No-Go Gate.

GATE (5 seeds 7000..7004):
  Project the exact shooting solution onto the split-domain multi-arc basis as initial guess.
  Must recover fuel within 0.05 kg of shooting, all switches within 0.5 d, and e_reprop < 1e-2
  on ALL 5/5 gate seeds. If any seed fails, STOP immediately and do NOT continue to 100 seeds.

FULL RUN (only if GATE PASSES, fresh seeds 7000..7099, N=100):
  Initialize from standalone PoNN coarse switch times and trajectory, solve split-domain PoNN,
  and test preregistered rule: >= 90/100 seeds meeting fuel within 0.05 kg, switches within 0.5 d,
  and e_reprop < 1e-2.

Checkpoints to results/tier3_split_domain_7000.json.
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
from scipy.integrate import solve_ivp

from trajectory_optimization.ponn_paper import PaperPoNNModel, PaperPoNNSolver, make_collocation_grid
from trajectory_optimization.scenario_a import (
    DEFAULT_RHO_SCHEDULE,
    ScenarioAProblem,
    ScenarioAShootingSolver,
    make_scenario_a_rhs14,
)
from trajectory_optimization.split_domain_ponn import SplitDomainPoNN

OUT_PATH = Path("results/tier3_split_domain_7000.json")


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def get_shooting_reference() -> dict[str, Any]:
    prob = ScenarioAProblem()
    sh = ScenarioAShootingSolver(stop_at_rho=1e-5).solve(prob, seed=1000)
    y0 = np.concatenate([prob.r0, prob.v0, [prob.m0], sh.lam0])
    sol = solve_ivp(
        make_scenario_a_rhs14(prob, float(sh.rho_reached)),
        (0.0, prob.tf), y0, method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True,
    )
    return {
        "prob": prob,
        "fuel_kg": float(sh.fuel_kg),
        "switch_times_days": [float(s["t_days"]) for s in sh.switches],
        "switch_times_tu": [float(s["t_days"]) / prob.t_ref_days for s in sh.switches],
        "thrust_arcs": [1.0, 0.0, 1.0, 0.0, 1.0],
        "sol": sol,
    }


def run_gate(ref: dict[str, Any]) -> dict[str, Any]:
    """Mandatory 5-seed Go/No-Go Gate (seeds 7000..7004) with projected exact shooting initial guess."""
    prob = ref["prob"]
    sd = SplitDomainPoNN(prob, thrust_arcs=ref["thrust_arcs"], n_neurons_arc=18, n_points_arc=36)
    theta_proj = sd.project_trajectory(ref["switch_times_tu"], lambda t_arr: ref["sol"].sol(t_arr).T)

    gate_seeds = {}
    all_pass = True
    for s in range(7000, 7005):
        rng = np.random.default_rng(s)
        # Tiny 1e-4 relative perturbation on projected shooting solution per gate seed
        theta0 = theta_proj.copy()
        theta0[sd.n_beta + 6 * sd.K:] += rng.uniform(-1e-3, 1e-3, size=sd.K)
        res = sd.solve(theta0, max_iter=60)
        fuel_err = abs(res.fuel_kg - ref["fuel_kg"]) if np.isfinite(res.fuel_kg) else float("inf")
        sw_err = float(np.max(np.abs(np.asarray(res.switch_times_days) - np.asarray(ref["switch_times_days"]))))
        passed = bool(fuel_err <= 0.05 and sw_err <= 0.5 and np.isfinite(res.reprop_arrival_residual) and res.reprop_arrival_residual < 1e-2)
        if not passed:
            all_pass = False
        gate_seeds[str(s)] = {
            "seed": s,
            "passed": passed,
            "status": res.status,
            "fuel_kg": res.fuel_kg,
            "fuel_err_kg": fuel_err,
            "switch_times_days": res.switch_times_days,
            "max_switch_err_days": sw_err,
            "residual_rms": res.residual_rms,
            "reprop_arrival_residual": res.reprop_arrival_residual,
            "nfev": res.nfev,
            "njev": res.njev,
            "runtime_s": res.runtime_s,
        }
        print(
            f"  [GATE s={s}] pass={passed} fuel={res.fuel_kg:.4f} (err={fuel_err:.4f} kg) "
            f"sw_err={sw_err:.4f} d reprop={res.reprop_arrival_residual:.2e} rms={res.residual_rms:.2e} ({res.runtime_s:.1f}s)",
            flush=True,
        )

    return {"passed_all_5": all_pass, "n_passed": sum(1 for v in gate_seeds.values() if v["passed"]), "seeds": gate_seeds}


_SD_CACHE: dict[tuple[float, ...], SplitDomainPoNN] = {}


def _get_sd(prob: ScenarioAProblem, arcs: list[float]) -> SplitDomainPoNN:
    key = tuple(arcs)
    if key not in _SD_CACHE:
        _SD_CACHE[key] = SplitDomainPoNN(prob, thrust_arcs=arcs, n_neurons_arc=18, n_points_arc=36)
    return _SD_CACHE[key]


def _run_full_seed(args_tuple: tuple[int, float, list[float]]) -> tuple[int, dict[str, Any]]:
    seed, fuel_ref, sw_ref_days = args_tuple
    prob = ScenarioAProblem()

    # 1. Run frozen PoNN coarse (L=70, n=140 CGL, scaled_u01) to obtain switch times & trajectory
    t0 = time.perf_counter()
    ponn = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    rc, _ = ponn.solve(prob, seed=seed, refine=False)
    t_ponn = time.perf_counter() - t0

    # Extract switch times and initial arc structure from coarse PoNN
    sw_days = [float(sw["t_days"]) for sw in rc.switches]
    if len(sw_days) != 4 or not np.isfinite(rc.residual_rms) or rc.residual_rms > 0.1:
        return seed, {
            "seed": seed,
            "passed_prereg": False,
            "status": f"ponn_coarse_switch_mismatch (n_sw={len(sw_days)})",
            "ponn_n_switches": len(sw_days),
            "fuel_kg": None,
            "fuel_err_kg": None,
            "max_switch_err_days": None,
            "reprop_arrival_residual": None,
            "ponn_runtime_s": float(t_ponn),
            "split_runtime_s": 0.0,
            "total_runtime_s": float(t_ponn),
        }

    # Build continuous evaluator from coarse PoNN Chebyshev expansion
    model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=seed)

    def _eval_ponn(t_tu_arr: np.ndarray) -> np.ndarray:
        z_arr = 2.0 * np.asarray(t_tu_arr, dtype=float) / prob.tf - 1.0
        return model_c.evaluate(rc.xi, z=z_arr)["Y14"]

    # Determine arc throttle sequence from coarse PoNN
    init_on = 1.0 if float(rc.S[0]) >= 0.0 else 0.0
    arcs = [init_on if (k % 2 == 0) else (1.0 - init_on) for k in range(len(sw_days) + 1)]
    sd = _get_sd(prob, arcs)
    theta0 = sd.project_trajectory([t / prob.t_ref_days for t in sw_days], _eval_ponn)
    res = sd.solve(theta0, max_iter=80)

    fuel_err = abs(res.fuel_kg - fuel_ref) if np.isfinite(res.fuel_kg) else float("inf")
    sw_err = float(np.max(np.abs(np.asarray(res.switch_times_days) - np.asarray(sw_ref_days))))
    passed = bool(
        fuel_err <= 0.05
        and sw_err <= 0.5
        and np.isfinite(res.reprop_arrival_residual)
        and res.reprop_arrival_residual < 1e-2
    )
    return seed, {
        "seed": seed,
        "passed_prereg": passed,
        "status": res.status,
        "ponn_n_switches": len(sw_days),
        "fuel_kg": res.fuel_kg if np.isfinite(res.fuel_kg) else None,
        "fuel_err_kg": fuel_err if np.isfinite(fuel_err) else None,
        "switch_times_days": res.switch_times_days,
        "max_switch_err_days": sw_err if np.isfinite(sw_err) else None,
        "residual_rms": res.residual_rms if np.isfinite(res.residual_rms) else None,
        "reprop_arrival_residual": res.reprop_arrival_residual if np.isfinite(res.reprop_arrival_residual) else None,
        "nfev": res.nfev,
        "njev": res.njev,
        "ponn_runtime_s": float(t_ponn),
        "split_runtime_s": float(res.runtime_s),
        "total_runtime_s": float(t_ponn + res.runtime_s),
    }


def _save(data: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, OUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=420.0)
    args = parser.parse_args()

    ref = get_shooting_reference()
    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text())
    else:
        data = {
            "meta": {
                "description": "Tier 3.9 Split-Domain PoNN Refinement with 5-seed Go/No-Go Gate (fresh seeds 7000..7099)",
                "prereg_addendum_commit": "38e2908",
                "shooting_ref_fuel_kg": ref["fuel_kg"],
                "shooting_ref_switches_days": ref["switch_times_days"],
            },
            "gate": None,
            "full_100": {},
            "summary": {},
        }

    if data["gate"] is None:
        print("=== Tier 3.9 Mandatory 5-Seed Go/No-Go Gate (seeds 7000..7004) ===", flush=True)
        data["gate"] = run_gate(ref)
        _save(data)

    if not data["gate"]["passed_all_5"]:
        print("[Tier 3.9 GATE FAILED] Stopping immediately per PREREG_ADDENDUM.md; NOT running 100 seeds.", flush=True)
        data["summary"] = {"gate_passed": False, "verdict": "CONTRADICTED (gate failed)"}
        _save(data)
        return

    print("[Tier 3.9 GATE PASSED 5/5] Running full 100 fresh seeds (7000..7099) from PoNN coarse switch times...", flush=True)
    t_start = time.perf_counter()
    todo = [
        (7000 + i, ref["fuel_kg"], ref["switch_times_days"])
        for i in range(args.n_seeds)
        if str(7000 + i) not in data["full_100"]
    ]
    print(f"[Tier 3.9 Full] {len(data['full_100'])}/{args.n_seeds} done, {len(todo)} remaining", flush=True)

    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for s, rec in pool.imap_unordered(_run_full_seed, todo):
                data["full_100"][str(s)] = rec
                _save(data)
                print(
                    f"  [T3 s={s}] pass={int(rec['passed_prereg'])} fuel_err={rec['fuel_err_kg']} "
                    f"reprop={rec['reprop_arrival_residual']} tot={rec['total_runtime_s']:.1f}s "
                    f"({len(data['full_100'])}/{args.n_seeds})",
                    flush=True,
                )
                if time.perf_counter() - t_start > args.max_wall_s:
                    print("[checkpoint] wall limit reached, terminating pool cleanly", flush=True)
                    pool.terminate()
                    break

    n_done = len(data["full_100"])
    n_pass = sum(1 for r in data["full_100"].values() if r["passed_prereg"])
    ci = wilson_ci(n_pass, n_done)
    verdict = "SUPPORTED" if (n_done == 100 and n_pass >= 90) else ("CONTRADICTED" if n_done == 100 else "IN_PROGRESS")
    data["summary"] = {
        "gate_passed": True,
        "n_seeds": n_done,
        "n_passed_prereg": n_pass,
        "rate_passed_prereg": n_pass / n_done if n_done else 0.0,
        "ci_passed_prereg": [ci[0], ci[1]],
        "verdict": verdict,
    }
    _save(data)
    print(f"[Tier 3.9] Saved {n_done}/{args.n_seeds} (passed={n_pass}/{n_done}, verdict={verdict})", flush=True)


if __name__ == "__main__":
    main()
