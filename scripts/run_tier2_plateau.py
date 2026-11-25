"""Tier 2.7: PoNN Observed-Plateau Study (Scenario A, fresh seeds 6000..6029, N=30 per cell).

Sweeps:
  1. (L, n) grid: L in {40, 60, 80, 120, 160} x n in {140, 280, 540, 1080} (CGL nodes, default tol=1e-9).
     Note: (L=160, n=140) is underdetermined (8n < 8L) and recorded explicitly.
  2. Uniform vs CGL nodes at representative (L=80, n in {140, 280, 540, 1080}).
  3. Tightened least-squares tolerances (ftol=xtol=gtol=1e-12 vs default 1e-9) at L=80, n=540 (CGL & uniform).
  4. Smoothed (rho=1e-10) vs hard-switched (rho=0) control in DOP853 re-propagation for every converged solve.
  5. Error decomposition: control-discretization error (re-propagating exact shooting control sampled on
     the PoNN grid) vs total open-loop re-propagation error vs state-approximation RMS error.
  6. Exact Jacobian condition numbers at terminal rho:
     - kappa_2(J_s): 2-norm condition number of column-scaled Jacobian J_s = J diag(||J_{:,j}||_2)^{-1}
       (the matrix factorized inside lm_svd)
     - kappa_2(J): unscaled 2-norm condition number of J.

Checkpoints per (cell, seed) to results/tier2_plateau_6000_6029.json.
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

from trajectory_optimization.ponn_paper import (
    PaperPoNNModel,
    PaperPoNNSolver,
    make_collocation_grid,
)
from trajectory_optimization.scenario_a import (
    DEFAULT_RHO_SCHEDULE,
    ScenarioAProblem,
    ScenarioAShootingSolver,
    make_scenario_a_rhs14,
    propagate_control_scenario_a,
    throttle_paper,
)

OUT_PATH = Path("results/tier2_plateau_6000_6029.json")

# Sweep cells: (cell_id, L, n, grid_type, tol)
CELLS: list[tuple[str, int, int, str, float]] = []
for _L in (40, 60, 80, 120, 160):
    for _n in (140, 280, 540, 1080):
        CELLS.append((f"cgl_L{_L}_n{_n}_t1e-9", _L, _n, "cgl", 1e-9))
# Uniform vs CGL comparison at L=80 across n
for _n in (140, 280, 540, 1080):
    CELLS.append((f"uni_L80_n{_n}_t1e-9", 80, _n, "uniform", 1e-9))
# Tightened tolerances (1e-12) at L=80, n=540 (both CGL and uniform)
CELLS.append(("cgl_L80_n540_t1e-12", 80, 540, "cgl", 1e-12))
CELLS.append(("uni_L80_n540_t1e-12", 80, 540, "uniform", 1e-12))


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


def compute_control_discretization_baseline() -> dict[str, Any]:
    """Error decomposition baseline: re-propagate exact shooting control sampled on each (n, grid_type)."""
    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0)
    sh = ScenarioAShootingSolver(stop_at_rho=1e-5).solve(prob, seed=1000)
    y0 = np.concatenate([prob.r0, prob.v0, [prob.m0], sh.lam0])
    sol = solve_ivp(
        make_scenario_a_rhs14(prob, float(sh.rho_reached)),
        (0.0, prob.tf),
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
    )
    out: dict[str, Any] = {
        "shooting_fuel_kg": float(sh.fuel_kg),
        "shooting_reprop_err": float(sh.reprop_arrival_residual),
        "grids": {},
    }
    for grid_type in ("cgl", "uniform"):
        for n in (140, 280, 540, 1080):
            z = make_collocation_grid(n, grid_type)
            t_tu = 0.5 * (z + 1.0) * prob.tf
            Y = sol.sol(t_tu).T
            lv = Y[:, 10:13]
            lvn = np.linalg.norm(lv, axis=1)
            alpha = -lv / np.maximum(lvn, 1e-15)[:, None]
            S = prob.c * lvn / Y[:, 6] + Y[:, 13] - 1.0
            delta_smooth = np.array([throttle_paper(float(s), float(sh.rho_reached)) for s in S])
            delta_hard = np.array([throttle_paper(float(s), 0.0) for s in S])
            err_smooth = float(propagate_control_scenario_a(prob, t_tu, delta_smooth, alpha)["arrival_residual"])
            err_hard = float(propagate_control_scenario_a(prob, t_tu, delta_hard, alpha)["arrival_residual"])
            out["grids"][f"{grid_type}_n{n}"] = {
                "ctrl_disc_err_smooth": err_smooth,
                "ctrl_disc_err_hard": err_hard,
            }
    return out


def _run_cell_seed(args_tuple: tuple[str, int, int, str, float, int]) -> tuple[str, int, dict[str, Any]]:
    cell_id, L, n, grid_type, tol, seed = args_tuple
    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0, label=cell_id)
    solver = PaperPoNNSolver(
        n_neurons=L,
        n_points=n,
        solver_algo="lm_svd",
        basis_type="chebyshev_shifted",
        grid_type=grid_type,
        init_mode="scaled_u01",
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_iter_first=100,
        max_iter_cont=25,
    )
    t0 = time.perf_counter()
    rc, _ = solver.solve(prob, seed=seed, refine=False)
    rt = time.perf_counter() - t0

    # Evaluate smoothed (rho_reached) vs hard-switched (rho=0) re-propagation + state-approx error
    model = PaperPoNNModel(prob, n_neurons=L, basis_type="chebyshev_shifted", elm_seed=seed)
    z = make_collocation_grid(n, grid_type)
    ops = model.operators(z)
    ev = model.evaluate(rc.xi, ops=ops)
    t_tu = ops["t_tu"]
    S = ev["S"]
    alpha = ev["alpha"]
    delta_smooth = np.array([throttle_paper(float(s), float(rc.rho_reached)) for s in S])
    delta_hard = np.array([throttle_paper(float(s), 0.0) for s in S])

    prop_smooth = propagate_control_scenario_a(prob, t_tu, delta_smooth, alpha)
    prop_hard = propagate_control_scenario_a(prob, t_tu, delta_hard, alpha)
    err_smooth = float(prop_smooth["arrival_residual"]) if np.isfinite(prop_smooth["arrival_residual"]) else float("nan")
    err_hard = float(prop_hard["arrival_residual"]) if np.isfinite(prop_hard["arrival_residual"]) else float("nan")

    # State-approximation RMS error: ||[r_ponn, v_ponn] - [r_reprop, v_reprop]|| on the collocation grid
    state_err_rms = float("nan")
    if "r" in prop_smooth and "v" in prop_smooth and len(prop_smooth["r"]) == len(t_tu):
        dr = ev["r"] - prop_smooth["r"]
        dv = ev["v"] - prop_smooth["v"]
        state_err_rms = float(np.sqrt(np.mean(np.sum(dr ** 2 + dv ** 2, axis=1))))

    # Condition numbers of terminal Jacobian J and column-scaled J_s (compute on first 5 seeds per cell to save SVD time on huge n=1080)
    cond_scaled = None
    cond_unscaled = None
    if seed < 6005 and np.isfinite(rc.residual_rms):
        try:
            _, J = model.residual_and_jacobian(rc.xi, ops, rho=float(rc.rho_reached))
            col_norms = np.linalg.norm(J, axis=0)
            col_norms = np.where(col_norms > 0.0, col_norms, 1.0)
            Js = J / col_norms[None, :]
            s_scaled = np.linalg.svd(Js, compute_uv=False)
            s_unscaled = np.linalg.svd(J, compute_uv=False)
            cond_scaled = float(s_scaled[0] / max(s_scaled[-1], 1e-300))
            cond_unscaled = float(s_unscaled[0] / max(s_unscaled[-1], 1e-300))
        except Exception:
            pass

    rms = float(rc.residual_rms) if np.isfinite(rc.residual_rms) else float("nan")
    mf = float(rc.final_mass_kg) if np.isfinite(rc.final_mass_kg) else 0.0
    phys_ok = bool(mf > 50.0 and np.isfinite(rms) and rms < 1.5e-2)
    return cell_id, seed, {
        "seed": seed,
        "success_loose": bool(phys_ok and np.isfinite(err_smooth) and err_smooth < 5e-2),
        "success_strict": bool(phys_ok and np.isfinite(err_smooth) and err_smooth < 1e-2),
        "status": rc.status,
        "fuel_kg": float(rc.fuel_kg) if np.isfinite(rc.fuel_kg) else None,
        "residual_rms": rms if np.isfinite(rms) else None,
        "reprop_err_smooth": err_smooth if np.isfinite(err_smooth) else None,
        "reprop_err_hard": err_hard if np.isfinite(err_hard) else None,
        "state_approx_rms": state_err_rms if np.isfinite(state_err_rms) else None,
        "cond_J_scaled": cond_scaled,
        "cond_J_unscaled": cond_unscaled,
        "nfev": int(rc.total_nfev),
        "njev": int(rc.total_njev),
        "runtime_s": float(rt),
    }


def summarize_cell(cell_obj: dict) -> dict:
    seeds = cell_obj["seeds"]
    keys = sorted(seeds.keys(), key=int)
    n = len(keys)
    recs = [seeds[k] for k in keys]
    nl = sum(1 for r in recs if r["success_loose"])
    ns = sum(1 for r in recs if r["success_strict"])
    l_recs = [r for r in recs if r["success_loose"]]
    return {
        "n_seeds": n,
        "n_loose": nl,
        "rate_loose": nl / n if n else 0.0,
        "ci_loose": list(wilson_ci(nl, n)),
        "n_strict": ns,
        "rate_strict": ns / n if n else 0.0,
        "ci_strict": list(wilson_ci(ns, n)),
        "reprop_smooth_loose": _dist_stats([r["reprop_err_smooth"] for r in l_recs]),
        "reprop_hard_loose": _dist_stats([r["reprop_err_hard"] for r in l_recs]),
        "reprop_smooth_all": _dist_stats([r["reprop_err_smooth"] for r in recs]),
        "state_approx_rms_loose": _dist_stats([r["state_approx_rms"] for r in l_recs]),
        "residual_rms_loose": _dist_stats([r["residual_rms"] for r in l_recs]),
        "fuel_kg_loose": _dist_stats([r["fuel_kg"] for r in l_recs]),
        "cond_J_scaled": _dist_stats([r["cond_J_scaled"] for r in recs if r.get("cond_J_scaled") is not None]),
        "cond_J_unscaled": _dist_stats([r["cond_J_unscaled"] for r in recs if r.get("cond_J_unscaled") is not None]),
        "runtime_s": _dist_stats([r["runtime_s"] for r in recs]),
    }


def _save(data: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, OUT_PATH)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    cells = {}
    for cid, L, n, g, tol in CELLS:
        cells[cid] = {
            "meta": {"cell_id": cid, "L": L, "n": n, "grid_type": g, "tol": tol, "underdetermined": bool(n < L)},
            "seeds": {},
            "summary": {},
        }
    return {
        "meta": {
            "description": "Tier 2.7 PoNN Observed-Plateau Study (seeds 6000..6029, N=30 per cell)",
            "prereg_addendum_commit": "38e2908",
            "cond_definition": "kappa_2(J_s) = sigma_max(J_s)/sigma_min(J_s) where J_s = J diag(||J_{:,j}||_2)^{-1} at terminal rho; plus unscaled kappa_2(J).",
        },
        "control_discretization_baseline": compute_control_discretization_baseline(),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=450.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    data = _load()
    todo = []
    for cid, L, n, g, tol in CELLS:
        done = data["cells"][cid]["seeds"]
        for i in range(args.n_seeds):
            s = 6000 + i
            if str(s) not in done:
                todo.append((cid, L, n, g, tol, s))

    total_target = len(CELLS) * args.n_seeds
    total_done = total_target - len(todo)
    print(f"[Tier 2.7 Plateau] {total_done}/{total_target} done, {len(todo)} remaining", flush=True)

    if todo:
        with mp.Pool(processes=args.workers) as pool:
            for cid, s, rec in pool.imap_unordered(_run_cell_seed, todo):
                data["cells"][cid]["seeds"][str(s)] = rec
                data["cells"][cid]["summary"] = summarize_cell(data["cells"][cid])
                _save(data)
                total_done += 1
                if total_done % 5 == 0 or total_done == total_target:
                    print(
                        f"  [{cid} s={s}] L={int(rec['success_loose'])} rp={rec['reprop_err_smooth']} "
                        f"| {total_done}/{total_target} ({time.perf_counter()-t_start:.1f}s)",
                        flush=True,
                    )
                if time.perf_counter() - t_start > args.max_wall_s:
                    print("[checkpoint] wall limit reached, terminating pool cleanly", flush=True)
                    pool.terminate()
                    break

    for cid, _, _, _, _ in CELLS:
        data["cells"][cid]["summary"] = summarize_cell(data["cells"][cid])
    _save(data)
    print(f"[Tier 2.7 Plateau] Saved {total_done}/{total_target} to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
