"""Phase 1 (Global Feasibility Scan & Multi-tf 20-Run Protocol) and
Phase 2 (100-Seed Random-Initialization Basin Study & Hybrid X-TFC -> Shooting Test).

Every number is computed directly from live solver executions with checkpoint
resumability and zero hardcoded results.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .problem import HelioProblem
from .indirect_shooting import IndirectShootingSolver
from .xtfc_ponn import XTFCPoNNSolver, _guess_functions
from .pseudospectral import (
    CollocationSolver,
    solve_min_time,
    solve_min_time_fixed_theta,
)
from .metrics import evaluate_solution, wilson_ci

# ---------------------------------------------------------------------------
# Compact scalar extraction (avoid bloating JSON/cache with large time series)
# ---------------------------------------------------------------------------
SCALAR_KEYS = (
    "method",
    "converged",
    "status",
    "tf_days",
    "theta_f_rad",
    "theta_f_deg",
    "fuel_kg",
    "final_mass_kg",
    "arrival_residual",
    "reprop_arrival_residual",
    "reprop_fuel_kg",
    "energy_error",
    "max_thrust_N",
    "min_thrust_N",
    "mean_thrust_N",
    "thrust_bound_ok",
    "n_throttle_switches",
    "coast_duration_days",
    "mass_balance_err_kg",
    "ode14_rms",
    "lgl180_defect_rms",
    "lgl180_defect_max",
    "H_paper_drift",
    "H_smooth_drift",
    "lam_m_tf",
    "iterations",
    "func_evals",
    "jac_evals",
    "runtime_s",
    "runtime_min",
    "warm_start",
    "slack_used",
    "slack_norm",
)


def _strip_timeseries(ev: dict, seed: int, extra: dict | None = None) -> dict:
    out = {"seed": int(seed)}
    for k in SCALAR_KEYS:
        if k in ev:
            v = ev[k]
            if isinstance(v, (np.floating, float)):
                out[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                out[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                out[k] = bool(v)
            else:
                out[k] = v
    if extra:
        out.update(extra)
    return out


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )


# ---------------------------------------------------------------------------
# Phase 1.1: Global Feasibility Scan at T_max = 0.50 N
# ---------------------------------------------------------------------------
DEFAULT_SCAN_ANGLES_DEG = [float(a) for a in range(60, 301, 10)]
DEFAULT_GUESS_MODES = ("direct", "spiral_0rev", "spiral_inward", "spiral_1rev")
DEFAULT_FREETHETA_STARTS = (
    (180.0, 100.0),
    (210.0, 120.0),
    (230.0, 150.0),
    (255.0, 192.5),
    (300.0, 240.0),
)


def _scan_one_angle_worker(args: tuple[float, int, tuple[str, ...], int]) -> dict:
    deg, n_nodes, guess_modes, max_iter = args
    prob = HelioProblem(tf_days=260.0, theta_f=float(np.radians(deg)))
    attempts = []
    for mode in guess_modes:
        r = solve_min_time_fixed_theta(
            prob,
            theta_f_rad=float(np.radians(deg)),
            n_nodes=n_nodes,
            guess_mode=mode,
            max_iter=max_iter,
        )
        attempts.append(r)
    conv = [a for a in attempts if a["success"]]
    if conv:
        best = min(conv, key=lambda a: a["tf_days"])
        return {
            "theta_f_deg": float(deg),
            "n_attempts": len(attempts),
            "n_converged": len(conv),
            "t_min_days": float(best["tf_days"]),
            "fuel_kg": float(best["fuel_kg"]),
            "best_guess_mode": best["guess_mode"],
            "reprop_arrival_residual": float(best["reprop_arrival_residual"]),
            "attempts": attempts,
        }
    return {
        "theta_f_deg": float(deg),
        "n_attempts": len(attempts),
        "n_converged": 0,
        "t_min_days": None,
        "fuel_kg": None,
        "best_guess_mode": "none",
        "reprop_arrival_residual": None,
        "attempts": attempts,
    }


def _multistart_free_theta_worker(args: tuple[float, float, int]) -> dict:
    tf_g, th_deg, n_nodes = args
    prob = HelioProblem(tf_days=tf_g, theta_f=float(np.radians(th_deg)))
    out = solve_min_time(
        prob,
        n_nodes=n_nodes,
        seed=1000,
        tf_guess_days=tf_g,
        theta_guess=float(np.radians(th_deg)),
        max_iter=2000,
    )
    return {
        "start_tf_days": float(tf_g),
        "start_theta_deg": float(th_deg),
        "success": bool(out["success"]),
        "status": str(out["status"]),
        "tf_days": float(out["tf_days"]),
        "tf_TU": float(out["tf_TU"]),
        "theta_f_deg": float(np.degrees(out["theta_f_rad"])),
        "theta_f_rad": float(out["theta_f_rad"]),
        "fuel_kg": float(out["fuel_canon"] * prob.m0_kg),
        "iterations": int(out["iterations"]),
        "runtime_s": float(out["runtime_s"]),
        "n_nodes": int(out["n_nodes"]),
    }


def run_global_feasibility_scan(
    angles_deg: list[float] | None = None,
    guess_modes: tuple[str, ...] = DEFAULT_GUESS_MODES,
    n_nodes_fixed: int = 80,
    n_nodes_free: int = 180,
    max_iter_fixed: int = 250,
    free_starts: tuple[tuple[float, float], ...] = DEFAULT_FREETHETA_STARTS,
    cache_path: Path | None = None,
    n_workers: int = 2,
) -> dict:
    """Phase 1.1: Global min-time feasibility scan across theta_f in [60, 300] deg
    plus multi-start free-theta_f 180-node solves."""
    angles_deg = list(DEFAULT_SCAN_ANGLES_DEG if angles_deg is None else angles_deg)

    cache = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = pickle.loads(cache_path.read_bytes())
        except Exception:
            cache = {}

    fixed_map: dict[float, dict] = cache.get("phase1_1_fixed", {})
    free_list: list[dict] = cache.get("phase1_1_free", [])

    def _save():
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache["phase1_1_fixed"] = fixed_map
            cache["phase1_1_free"] = free_list
            cache_path.write_bytes(pickle.dumps(cache))

    todo_angles = [deg for deg in angles_deg if float(deg) not in fixed_map]
    print(
        f"  [Phase 1.1] Scanning fixed theta_f in [{min(angles_deg):.0f}, {max(angles_deg):.0f}] deg "
        f"({len(guess_modes)} guesses/angle, N={n_nodes_fixed}; {len(fixed_map)}/{len(angles_deg)} cached)...",
        flush=True,
    )
    if todo_angles:
        tasks = [(deg, n_nodes_fixed, guess_modes, max_iter_fixed) for deg in todo_angles]
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx, initializer=_worker_init
            ) as ex:
                for row in ex.map(_scan_one_angle_worker, tasks):
                    fixed_map[row["theta_f_deg"]] = row
                    _save()
                    tmin_s = f"{row['t_min_days']:7.3f} d" if row["t_min_days"] is not None else "INFEASIBLE / NON-CONV"
                    print(
                        f"    theta_f = {row['theta_f_deg']:5.1f} deg | conv {row['n_converged']}/{row['n_attempts']} | "
                        f"t_min = {tmin_s} ({row['best_guess_mode']})",
                        flush=True,
                    )
        else:
            for t in tasks:
                row = _scan_one_angle_worker(t)
                fixed_map[row["theta_f_deg"]] = row
                _save()
                tmin_s = f"{row['t_min_days']:7.3f} d" if row["t_min_days"] is not None else "INFEASIBLE / NON-CONV"
                print(
                    f"    theta_f = {row['theta_f_deg']:5.1f} deg | conv {row['n_converged']}/{row['n_attempts']} | "
                    f"t_min = {tmin_s} ({row['best_guess_mode']})",
                    flush=True,
                )

    fixed_rows = [fixed_map[float(deg)] for deg in angles_deg]

    done_free = {(r["start_tf_days"], r["start_theta_deg"]) for r in free_list}
    todo_free = [(tf_g, th_g, n_nodes_free) for (tf_g, th_g) in free_starts if (float(tf_g), float(th_g)) not in done_free]
    print(
        f"  [Phase 1.1] Multi-start free-theta_f (N={n_nodes_free}) solves "
        f"({len(free_list)}/{len(free_starts)} cached)...",
        flush=True,
    )
    if todo_free:
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx, initializer=_worker_init
            ) as ex:
                for r in ex.map(_multistart_free_theta_worker, todo_free):
                    free_list.append(r)
                    _save()
                    print(
                        f"    start=({r['start_tf_days']:.0f} d, {r['start_theta_deg']:.1f} deg) -> "
                        f"ok={r['success']} | t_min={r['tf_days']:.4f} d | theta_f={r['theta_f_deg']:.3f} deg | "
                        f"fuel={r['fuel_kg']:.2f} kg | iters={r['iterations']} | t={r['runtime_s']:.2f}s",
                        flush=True,
                    )
        else:
            for t in todo_free:
                r = _multistart_free_theta_worker(t)
                free_list.append(r)
                _save()
                print(
                    f"    start=({r['start_tf_days']:.0f} d, {r['start_theta_deg']:.1f} deg) -> "
                    f"ok={r['success']} | t_min={r['tf_days']:.4f} d | theta_f={r['theta_f_deg']:.3f} deg | "
                    f"fuel={r['fuel_kg']:.2f} kg | iters={r['iterations']} | t={r['runtime_s']:.2f}s",
                    flush=True,
                )

    free_order = {(float(a), float(b)): i for i, (a, b) in enumerate(free_starts)}
    free_list_sorted = sorted(free_list, key=lambda r: free_order.get((r["start_tf_days"], r["start_theta_deg"]), 999))

    conv_fixed = [r for r in fixed_rows if r["t_min_days"] is not None]
    conv_free = [r for r in free_list_sorted if r["success"]]
    best_fixed = min(conv_fixed, key=lambda r: r["t_min_days"]) if conv_fixed else None
    best_free = min(conv_free, key=lambda r: r["tf_days"]) if conv_free else None

    candidates = []
    if best_fixed is not None:
        candidates.append((best_fixed["t_min_days"], best_fixed["theta_f_deg"], "fixed_scan"))
    if best_free is not None:
        candidates.append((best_free["tf_days"], best_free["theta_f_deg"], "free_theta_180"))
    global_t_min, global_theta_deg, global_src = min(candidates, key=lambda x: x[0])

    # Check explicitly whether earlier 260.8333 d result was a local minimum
    earlier_t_min = 260.8333
    was_local_min = bool(global_t_min < earlier_t_min - 0.05)

    return {
        "fixed_rows": fixed_rows,
        "free_multistart": free_list_sorted,
        "best_fixed": best_fixed,
        "best_free": best_free,
        "global_t_min_days": float(global_t_min),
        "global_theta_f_deg": float(global_theta_deg),
        "global_source": global_src,
        "earlier_t_min_days": earlier_t_min,
        "earlier_was_local_minimum": was_local_min,
    }


# ---------------------------------------------------------------------------
# Phase 1.2: Select fuel-minimal theta_f*(tf) at each tf
# ---------------------------------------------------------------------------
def _eval_angle_fuel_worker(args: tuple[float, float]) -> dict:
    tf_days, deg = args
    prob = HelioProblem(tf_days=tf_days, theta_f=float(np.radians(deg)))
    sh = IndirectShootingSolver()
    best_ev = None
    for s in (1000, 1001, 1002):
        rs = sh.solve(prob, seed=s)
        ev_s = evaluate_solution(prob, rs)
        best_ev = ev_s
        if ev_s["converged"]:
            break
    return {
        "tf_days": float(tf_days),
        "theta_f_deg": float(deg),
        "sh_converged": bool(best_ev["converged"]),
        "sh_fuel_kg": float(best_ev["fuel_kg"]),
        "sh_reprop_err": float(best_ev["reprop_arrival_residual"]),
    }


def select_fuel_minimal_angles(
    scan_data: dict,
    tf_list: tuple[float, ...] = (247.9, 280.0, 300.0, 330.0, 360.0),
    cache_path: Path | None = None,
    n_workers: int = 2,
) -> dict[float, dict]:
    """Selects the fuel-minimal Mars arrival longitude theta_f*(tf) for each tf
    directly from computed shooting + 180-node collocation checks over the feasible band."""
    cache = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = pickle.loads(cache_path.read_bytes())
        except Exception:
            cache = {}

    selected: dict[float, dict] = cache.get("phase1_2_angles", {})

    def _save():
        if cache_path is not None:
            cache["phase1_2_angles"] = selected
            cache_path.write_bytes(pickle.dumps(cache))

    global_theta_deg = float(scan_data["global_theta_f_deg"])
    global_t_min = float(scan_data["global_t_min_days"])
    fixed_rows = scan_data["fixed_rows"]

    for tf_days in tf_list:
        tf_key = float(tf_days)
        if tf_key in selected:
            s = selected[tf_key]
            print(
                f"  [Phase 1.2 cached] tf={tf_key:.1f} d -> theta_f*={s['theta_f_deg']:.2f} deg "
                f"(fuel={s.get('best_fuel_kg')})",
                flush=True,
            )
            continue

        if tf_key < global_t_min:
            selected[tf_key] = {
                "tf_days": tf_key,
                "theta_f_deg": global_theta_deg,
                "theta_f_rad": float(np.radians(global_theta_deg)),
                "selection_reason": f"tf < t_min ({global_t_min:.4f} d); evaluated at global min-time angle theta_f*(t_min)",
                "best_fuel_kg": None,
                "col180_fuel_kg": None,
                "col180_converged": False,
                "band_deg": [global_theta_deg, global_theta_deg],
                "sweep_rows": [],
            }
            _save()
            print(
                f"  [Phase 1.2] tf={tf_key:.1f} d < t_min ({global_t_min:.2f} d) -> using theta_f*={global_theta_deg:.2f} deg",
                flush=True,
            )
            continue

        # Determine feasible longitude band from Phase 1.1 scan
        feas_angles = [
            r["theta_f_deg"]
            for r in fixed_rows
            if r["t_min_days"] is not None and r["t_min_days"] <= tf_key + 15.0
        ]
        lo = max(190.0, min(feas_angles) - 5.0) if feas_angles else 190.0
        hi = min(300.0, max(feas_angles) + 5.0) if feas_angles else 280.0

        # Stage 1: coarse 10 deg sweep across feasible band
        coarse_grid = [round(float(a), 2) for a in np.arange(lo, hi + 0.01, 10.0)]
        tasks_c = [(tf_key, deg) for deg in coarse_grid]
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx, initializer=_worker_init
            ) as ex:
                rows_c = list(ex.map(_eval_angle_fuel_worker, tasks_c))
        else:
            rows_c = [_eval_angle_fuel_worker(t) for t in tasks_c]

        conv_c = [r for r in rows_c if r["sh_converged"]]
        if conv_c:
            best_c = min(conv_c, key=lambda r: r["sh_fuel_kg"])["theta_f_deg"]
            fine_lo = max(lo, best_c - 15.0)
            fine_hi = min(hi, best_c + 15.0)
        else:
            fine_lo, fine_hi = lo, hi

        # Stage 2: fine 2.5 deg sweep around coarse minimum
        seen = {r["theta_f_deg"] for r in rows_c}
        fine_grid = [round(float(a), 2) for a in np.arange(fine_lo, fine_hi + 0.01, 2.5) if round(float(a), 2) not in seen]
        tasks_f = [(tf_key, deg) for deg in fine_grid]
        if tasks_f:
            if n_workers > 1:
                ctx = mp.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers, mp_context=ctx, initializer=_worker_init
                ) as ex:
                    rows_f = list(ex.map(_eval_angle_fuel_worker, tasks_f))
            else:
                rows_f = [_eval_angle_fuel_worker(t) for t in tasks_f]
        else:
            rows_f = []

        rows = sorted(rows_c + rows_f, key=lambda r: r["theta_f_deg"])
        conv_rows = [r for r in rows if r["sh_converged"]]

        # Pick true fuel-minimal angle from converged shooting sweep, then verify with 180-node collocation
        col = CollocationSolver(n_nodes=180, allow_slack=True)
        col_checks = []
        if conv_rows:
            best_s = min(conv_rows, key=lambda r: r["sh_fuel_kg"])
            best_deg = float(best_s["theta_f_deg"])
            best_fuel = float(best_s["sh_fuel_kg"])
            col_fuel = None
            col_ok = False
            p_best = HelioProblem(tf_days=tf_key, theta_f=float(np.radians(best_deg)))
            for s_chk in (1000, 1001, 1002):
                rc = col.solve(p_best, seed=s_chk)
                evc = evaluate_solution(p_best, rc)
                col_checks.append({
                    "theta_f_deg": best_deg,
                    "seed": s_chk,
                    "col_converged": bool(evc["converged"]),
                    "col_fuel_kg": float(evc["fuel_kg"]),
                })
                if evc["converged"]:
                    col_ok = True
                    col_fuel = float(evc["fuel_kg"])
                    break
        else:
            best_deg = global_theta_deg
            best_fuel = None
            col_fuel = None
            col_ok = False

        selected[tf_key] = {
            "tf_days": tf_key,
            "theta_f_deg": best_deg,
            "theta_f_rad": float(np.radians(best_deg)),
            "selection_reason": f"fuel-minimal angle in [{lo:.1f}, {hi:.1f}] deg (coarse 10 deg -> fine 2.5 deg shooting sweep, verified with 180-node LGL)",
            "best_fuel_kg": best_fuel,
            "col180_fuel_kg": col_fuel,
            "col180_converged": col_ok,
            "band_deg": [float(lo), float(hi)],
            "sweep_rows": rows,
            "col_checks": col_checks,
        }
        _save()
        print(
            f"  [Phase 1.2] tf={tf_key:.1f} d | band=[{lo:.1f}, {hi:.1f}] deg | "
            f"theta_f*={best_deg:.2f} deg | sh_fuel={best_fuel:.2f} kg | col180_ok={col_ok}"
            + (f" ({col_fuel:.2f} kg)" if col_fuel is not None else ""),
            flush=True,
        )

    return selected


# ---------------------------------------------------------------------------
# Phase 1.3–1.4: Multi-tf 20-Run Robustness Protocol (4 solver configs)
# ---------------------------------------------------------------------------
METHOD_KEYS_P1 = (
    "X-TFC PoNN (warm)",
    "X-TFC PoNN (no warm start)",
    "Indirect shooting",
    "Pseudospectral collocation",
)


def _phase1_seed_worker(args: tuple[float, float, int, np.ndarray | None, str | None]) -> tuple[int, dict]:
    tf_days, theta_rad, seed, init_G, init_label = args
    prob = HelioProblem(tf_days=tf_days, theta_f=theta_rad, label=f"tf_{tf_days:.1f}d")

    xt_warm = XTFCPoNNSolver(warm_start_from_eps1=True)
    rx_w = xt_warm.solve(prob, seed=seed, init_from=init_G, init_label=init_label)
    ev_xw = _strip_timeseries(evaluate_solution(prob, rx_w), seed)

    xt_cold = XTFCPoNNSolver(warm_start_from_eps1=False)
    rx_c = xt_cold.solve(prob, seed=seed)
    ev_xc = _strip_timeseries(evaluate_solution(prob, rx_c), seed)

    sh = IndirectShootingSolver()
    rs = sh.solve(prob, seed=seed)
    ev_sh = _strip_timeseries(evaluate_solution(prob, rs), seed)

    ps = CollocationSolver(n_nodes=180, allow_slack=True)
    rc = ps.solve(prob, seed=seed)
    ev_ps = _strip_timeseries(evaluate_solution(prob, rc), seed)

    return seed, {
        "X-TFC PoNN (warm)": ev_xw,
        "X-TFC PoNN (no warm start)": ev_xc,
        "Indirect shooting": ev_sh,
        "Pseudospectral collocation": ev_ps,
    }


def _summarize_method_runs(runs: list[dict]) -> dict:
    n = len(runs)
    conv = [r for r in runs if r["converged"]]
    k = len(conv)
    ci_lo, ci_hi = wilson_ci(k, n)

    def _stats(arr: list[float]):
        a = np.asarray(arr, dtype=float)
        if len(a) == 0:
            return {"mean": None, "sd": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(a)),
            "sd": float(np.std(a)),
            "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    out = {
        "n_runs": n,
        "n_converged": k,
        "conv_rate": float(k / n) if n > 0 else 0.0,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "all_runtime_s": _stats([r["runtime_s"] for r in runs]),
        "all_func_evals": _stats([r["func_evals"] for r in runs]),
        "all_jac_evals": _stats([r["jac_evals"] for r in runs]),
        "all_fuel_kg": _stats([r["fuel_kg"] for r in runs]),
        "all_reprop_err": _stats([r["reprop_arrival_residual"] for r in runs]),
        "runs": runs,
    }
    if k > 0:
        out.update({
            "conv_fuel_kg": _stats([r["fuel_kg"] for r in conv]),
            "conv_switches": _stats([r["n_throttle_switches"] for r in conv]),
            "conv_coast_days": _stats([r["coast_duration_days"] for r in conv]),
            "conv_reprop_err": _stats([r["reprop_arrival_residual"] for r in conv]),
            "conv_arrival_err": _stats([r["arrival_residual"] for r in conv]),
            "conv_energy_err": _stats([r["energy_error"] for r in conv]),
            "conv_runtime_s": _stats([r["runtime_s"] for r in conv]),
            "conv_func_evals": _stats([r["func_evals"] for r in conv]),
            "conv_jac_evals": _stats([r["jac_evals"] for r in conv]),
            "conv_iterations": _stats([r["iterations"] for r in conv]),
        })
    else:
        for key in (
            "conv_fuel_kg", "conv_switches", "conv_coast_days", "conv_reprop_err",
            "conv_arrival_err", "conv_energy_err", "conv_runtime_s",
            "conv_func_evals", "conv_jac_evals", "conv_iterations",
        ):
            out[key] = None
    return out


def run_multi_tf_robustness(
    selected_angles: dict[float, dict],
    tf_list: tuple[float, ...] = (247.9, 280.0, 300.0, 330.0, 360.0),
    n_runs: int = 20,
    base_seed: int = 1000,
    cache_path: Path | None = None,
    n_workers: int = 2,
) -> dict[float, dict]:
    """Runs the 20-run protocol for all 4 method configurations across tf_list."""
    cache = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = pickle.loads(cache_path.read_bytes())
        except Exception:
            cache = {}

    rob_store: dict[float, dict[int, dict]] = cache.get("phase1_rob_seeds", {})

    def _save():
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache["phase1_rob_seeds"] = rob_store
            cache_path.write_bytes(pickle.dumps(cache))

    seeds = [base_seed + i for i in range(n_runs)]
    results_by_tf: dict[float, dict] = {}

    for tf_days in tf_list:
        tf_key = float(tf_days)
        ang_info = selected_angles[tf_key]
        theta_rad = float(ang_info["theta_f_rad"])
        theta_deg = float(ang_info["theta_f_deg"])
        prob = HelioProblem(tf_days=tf_key, theta_f=theta_rad, label=f"tf_{tf_key:.1f}d")

        seed_map = rob_store.setdefault(tf_key, {})
        todo_seeds = [s for s in seeds if s not in seed_map]

        if not todo_seeds:
            print(f"  [robustness tf={tf_key:.1f} d loaded from cache ({n_runs}/{n_runs})]", flush=True)
        else:
            # Precompute eps=1.0 warm-start trajectory once for X-TFC (warm)
            xt_tmp = XTFCPoNNSolver(warm_start_from_eps1=True)
            tau = np.linspace(-1.0, 1.0, xt_tmp.N)
            sh1 = IndirectShootingSolver(eps_schedule=(xt_tmp.eps_schedule[0],))
            rs1 = sh1.solve(prob, seed=base_seed)
            if rs1.success:
                t_tau = (tau + 1.0) * prob.tf / 2.0
                sol_d = sh1.propagate(prob, rs1.lam0, xt_tmp.eps_schedule[0], dense=True)
                init_G = sol_d.sol(t_tau).T
                init_label = f"projected from eps={xt_tmp.eps_schedule[0]:g} smoothed shooting extremal"
            else:
                init_G = _guess_functions(prob, tau)
                init_label = f"deterministic polar/tangential guess (eps={xt_tmp.eps_schedule[0]:g} shooting did not converge)"

            tasks = [(tf_key, theta_rad, s, init_G, init_label) for s in todo_seeds]
            if n_workers > 1:
                ctx = mp.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers, mp_context=ctx, initializer=_worker_init
                ) as ex:
                    for s, res_dict in ex.map(_phase1_seed_worker, tasks):
                        seed_map[s] = res_dict
                        _save()
                        done_n = len(seed_map)
                        xw = res_dict["X-TFC PoNN (warm)"]
                        xc = res_dict["X-TFC PoNN (no warm start)"]
                        sh_r = res_dict["Indirect shooting"]
                        ps_r = res_dict["Pseudospectral collocation"]
                        print(
                            f"  [tf={tf_key:5.1f}d, th={theta_deg:5.1f}deg | {done_n:2d}/{n_runs} | s={s}] "
                            f"XTw={int(xw['converged'])} XTc={int(xc['converged'])} "
                            f"SH={int(sh_r['converged'])}({sh_r['fuel_kg']:.1f}kg) "
                            f"PS={int(ps_r['converged'])}({ps_r['fuel_kg']:.1f}kg)",
                            flush=True,
                        )
            else:
                for t in tasks:
                    s, res_dict = _phase1_seed_worker(t)
                    seed_map[s] = res_dict
                    _save()
                    done_n = len(seed_map)
                    xw = res_dict["X-TFC PoNN (warm)"]
                    xc = res_dict["X-TFC PoNN (no warm start)"]
                    sh_r = res_dict["Indirect shooting"]
                    ps_r = res_dict["Pseudospectral collocation"]
                    print(
                        f"  [tf={tf_key:5.1f}d, th={theta_deg:5.1f}deg | {done_n:2d}/{n_runs} | s={s}] "
                        f"XTw={int(xw['converged'])} XTc={int(xc['converged'])} "
                        f"SH={int(sh_r['converged'])}({sh_r['fuel_kg']:.1f}kg) "
                        f"PS={int(ps_r['converged'])}({ps_r['fuel_kg']:.1f}kg)",
                        flush=True,
                    )

        methods_summary = {}
        for m in METHOD_KEYS_P1:
            m_runs = [seed_map[s][m] for s in seeds]
            methods_summary[m] = _summarize_method_runs(m_runs)

        results_by_tf[tf_key] = {
            "tf_days": tf_key,
            "theta_f_deg": theta_deg,
            "theta_f_rad": theta_rad,
            "methods": methods_summary,
        }

    return results_by_tf


# ---------------------------------------------------------------------------
# Phase 2: 100-Seed Random-Initialization Basin Study (2A) + Hybrid Test (2B)
# ---------------------------------------------------------------------------
METHOD_KEYS_P2 = (
    "X-TFC PoNN (no warm start)",
    "Indirect shooting (random costates)",
    "Pseudospectral collocation (random guess)",
    "Hybrid (X-TFC -> Indirect shooting)",
)


def _phase2_seed_worker(args: tuple[float, float, int]) -> tuple[int, dict]:
    tf_days, theta_rad, seed = args
    prob = HelioProblem(tf_days=tf_days, theta_f=theta_rad, label=f"phase2_tf{tf_days:.1f}d")

    # 1. X-TFC without warm start
    xt_cold = XTFCPoNNSolver(warm_start_from_eps1=False)
    rx_c = xt_cold.solve(prob, seed=seed)
    ev_xc = _strip_timeseries(evaluate_solution(prob, rx_c), seed)

    # 2. Indirect shooting with random costates
    sh = IndirectShootingSolver()
    rs_rand = sh.solve(prob, seed=seed)
    ev_sh_rand = _strip_timeseries(
        evaluate_solution(prob, rs_rand),
        seed,
        extra={"n_propagations": int(rs_rand.n_propagations)},
    )

    # 3. 180-node LGL collocation with random polar guess
    ps = CollocationSolver(n_nodes=180, allow_slack=True)
    rc = ps.solve(prob, seed=seed)
    ev_ps = _strip_timeseries(evaluate_solution(prob, rc), seed)

    # 4. Hybrid: use X-TFC (converged or best-effort) initial costates lam(t0) for shooting
    lam0_xtfc = np.array(rx_c.Y[0, 7:14], dtype=float)
    rs_hyb = sh.solve(prob, seed=None, lam0_guess=lam0_xtfc)
    ev_hyb = _strip_timeseries(
        evaluate_solution(prob, rs_hyb),
        seed,
        extra={
            "n_propagations": int(rs_hyb.n_propagations),
            "xtfc_converged": bool(rx_c.success),
            "xtfc_runtime_s": float(rx_c.runtime_s),
            "total_hybrid_runtime_s": float(rx_c.runtime_s + rs_hyb.runtime_s),
            "lam0_xtfc_norm": float(np.linalg.norm(lam0_xtfc)),
        },
    )

    return seed, {
        "X-TFC PoNN (no warm start)": ev_xc,
        "Indirect shooting (random costates)": ev_sh_rand,
        "Pseudospectral collocation (random guess)": ev_ps,
        "Hybrid (X-TFC -> Indirect shooting)": ev_hyb,
    }


def run_phase2_basin_and_hybrid(
    tf_days: float,
    theta_f_deg: float,
    n_seeds: int = 100,
    base_seed: int = 2000,
    cache_path: Path | None = None,
    n_workers: int = 2,
) -> dict:
    """Phase 2A (100-seed random-initialization basin study) and Phase 2B
    (Hybrid X-TFC -> Indirect shooting costate initialization on the same 100 seeds)."""
    theta_rad = float(np.radians(theta_f_deg))
    cache = {}
    if cache_path is not None and cache_path.exists():
        try:
            cache = pickle.loads(cache_path.read_bytes())
        except Exception:
            cache = {}

    seed_map: dict[int, dict] = cache.get("phase2_seeds", {})

    def _save():
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache["phase2_seeds"] = seed_map
            cache_path.write_bytes(pickle.dumps(cache))

    seeds = [base_seed + i for i in range(n_seeds)]
    todo_seeds = [s for s in seeds if s not in seed_map]
    print(
        f"  [Phase 2] tf={tf_days:.1f} d, theta_f={theta_f_deg:.2f} deg | "
        f"{len(seed_map)}/{n_seeds} cached, {len(todo_seeds)} remaining...",
        flush=True,
    )

    def _log_progress(s: int):
        done = len(seed_map)
        if done == 1 or done % 5 == 0 or done == n_seeds:
            c_xt = sum(1 for v in seed_map.values() if v["X-TFC PoNN (no warm start)"]["converged"])
            c_sh = sum(1 for v in seed_map.values() if v["Indirect shooting (random costates)"]["converged"])
            c_ps = sum(1 for v in seed_map.values() if v["Pseudospectral collocation (random guess)"]["converged"])
            c_hy = sum(1 for v in seed_map.values() if v["Hybrid (X-TFC -> Indirect shooting)"]["converged"])
            print(
                f"  [Phase 2 | {done:3d}/{n_seeds} | s={s}] "
                f"XT_cold={c_xt}/{done} | SH_rand={c_sh}/{done} | PS180={c_ps}/{done} | Hybrid={c_hy}/{done}",
                flush=True,
            )

    if todo_seeds:
        tasks = [(float(tf_days), theta_rad, s) for s in todo_seeds]
        if n_workers > 1:
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx, initializer=_worker_init
            ) as ex:
                for s, res_dict in ex.map(_phase2_seed_worker, tasks):
                    seed_map[s] = res_dict
                    _save()
                    _log_progress(s)
        else:
            for t in tasks:
                s, res_dict = _phase2_seed_worker(t)
                seed_map[s] = res_dict
                _save()
                _log_progress(s)

    methods_summary = {}
    for m in METHOD_KEYS_P2:
        m_runs = [seed_map[s][m] for s in seeds]
        methods_summary[m] = _summarize_method_runs(m_runs)

    # Paired seed-by-seed comparison: Hybrid vs Indirect shooting (random costates)
    sh_runs = [seed_map[s]["Indirect shooting (random costates)"] for s in seeds]
    hy_runs = [seed_map[s]["Hybrid (X-TFC -> Indirect shooting)"] for s in seeds]
    both_conv = [
        (r_sh, r_hy) for r_sh, r_hy in zip(sh_runs, hy_runs) if r_sh["converged"] and r_hy["converged"]
    ]
    hy_only = sum(1 for r_sh, r_hy in zip(sh_runs, hy_runs) if (not r_sh["converged"]) and r_hy["converged"])
    sh_only = sum(1 for r_sh, r_hy in zip(sh_runs, hy_runs) if r_sh["converged"] and (not r_hy["converged"]))
    both_fail = sum(1 for r_sh, r_hy in zip(sh_runs, hy_runs) if (not r_sh["converged"]) and (not r_hy["converged"]))

    hy_conv = [r for r in hy_runs if r["converged"]]
    sh_conv = [r for r in sh_runs if r["converged"]]

    paired_comparison = {
        "n_seeds": n_seeds,
        "sh_conv": len(sh_conv),
        "hy_conv": len(hy_conv),
        "both_conv": len(both_conv),
        "hy_only_conv": hy_only,
        "sh_only_conv": sh_only,
        "both_fail": both_fail,
        "sh_conv_mean_iters": float(np.mean([r["iterations"] for r in sh_conv])) if sh_conv else None,
        "hy_conv_mean_iters": float(np.mean([r["iterations"] for r in hy_conv])) if hy_conv else None,
        "paired_sh_mean_iters": float(np.mean([a["iterations"] for a, _ in both_conv])) if both_conv else None,
        "paired_hy_mean_iters": float(np.mean([b["iterations"] for _, b in both_conv])) if both_conv else None,
        "paired_sh_mean_rhs": float(np.mean([a["func_evals"] for a, _ in both_conv])) if both_conv else None,
        "paired_hy_mean_rhs": float(np.mean([b["func_evals"] for _, b in both_conv])) if both_conv else None,
        "paired_sh_mean_runtime_s": float(np.mean([a["runtime_s"] for a, _ in both_conv])) if both_conv else None,
        "paired_hy_shoot_runtime_s": float(np.mean([b["runtime_s"] for _, b in both_conv])) if both_conv else None,
        "paired_hy_total_runtime_s": float(np.mean([b["total_hybrid_runtime_s"] for _, b in both_conv])) if both_conv else None,
        "improves_convergence": bool(len(hy_conv) > len(sh_conv)),
        "reduces_shooting_iters": bool(
            both_conv and np.mean([b["iterations"] for _, b in both_conv]) < np.mean([a["iterations"] for a, _ in both_conv])
        ),
    }

    return {
        "tf_days": float(tf_days),
        "theta_f_deg": float(theta_f_deg),
        "theta_f_rad": theta_rad,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "methods": methods_summary,
        "paired_hybrid_vs_random": paired_comparison,
    }


# ---------------------------------------------------------------------------
# Terminal summary printers
# ---------------------------------------------------------------------------
def print_phase1_1_summary(scan_data: dict) -> None:
    print("\n=== PHASE 1.1: GLOBAL FEASIBILITY SCAN SUMMARY (T_max = 0.50 N) ===")
    print("theta_f (deg) |  Conv |  t_min (d) |  Fuel (kg) |  Best Guess Mode")
    print("--------------------------------------------------------------------")
    for r in scan_data["fixed_rows"]:
        t_s = f"{r['t_min_days']:10.3f}" if r["t_min_days"] is not None else "       N/A"
        f_s = f"{r['fuel_kg']:10.2f}" if r["fuel_kg"] is not None else "       N/A"
        print(
            f"{r['theta_f_deg']:13.1f} |  {r['n_converged']}/{r['n_attempts']}  | {t_s} | {f_s} | {r['best_guess_mode']:>16s}"
        )
    print("Multi-start free-theta_f (N=180) results:")
    for r in scan_data["free_multistart"]:
        print(
            f"  start=({r['start_tf_days']:.0f} d, {r['start_theta_deg']:.1f} deg) -> "
            f"ok={r['success']} | t_min={r['tf_days']:.4f} d | theta_f={r['theta_f_deg']:.3f} deg | "
            f"fuel={r['fuel_kg']:.2f} kg | iters={r['iterations']} | t={r['runtime_s']:.2f}s"
        )
    print(
        f"Global minimum t_min found: {scan_data['global_t_min_days']:.4f} days at "
        f"theta_f = {scan_data['global_theta_f_deg']:.3f} deg | "
        f"Earlier {scan_data['earlier_t_min_days']:.2f} d result was local minimum? {scan_data['earlier_was_local_minimum']}",
        flush=True,
    )


def print_phase1_robustness_summary(rob_by_tf: dict[float, dict]) -> None:
    print("\n=== PHASE 1.2–1.4: MULTI-TF 20-RUN ROBUSTNESS SUMMARY ===")
    for tf_key in sorted(rob_by_tf.keys()):
        entry = rob_by_tf[tf_key]
        print(f"\n--- tf = {entry['tf_days']:.1f} d, fuel-minimal theta_f = {entry['theta_f_deg']:.2f} deg ---")
        for m in METHOD_KEYS_P1:
            d = entry["methods"][m]
            ci = f"[{d['ci95_low']:.2f},{d['ci95_high']:.2f}]"
            if d["n_converged"] > 0:
                print(
                    f"  {m:<27s} : {d['n_converged']:2d}/{d['n_runs']} (95% CI {ci}) | "
                    f"fuel={d['conv_fuel_kg']['mean']:.2f}±{d['conv_fuel_kg']['sd']:.2f} kg | "
                    f"sw={d['conv_switches']['mean']:.1f} | coast={d['conv_coast_days']['mean']:.1f} d | "
                    f"reprop_eM={d['conv_reprop_err']['mean']:.2e} | eE={d['conv_energy_err']['mean']:.2e} | "
                    f"t={d['conv_runtime_s']['mean']:.2f}s | "
                    f"fev/jev={d['conv_func_evals']['mean']:.0f}/{d['conv_jac_evals']['mean']:.0f}"
                )
            else:
                print(
                    f"  {m:<27s} : {d['n_converged']:2d}/{d['n_runs']} (95% CI {ci}) | "
                    f"all_t={d['all_runtime_s']['mean']:.2f}s | "
                    f"all_fev/jev={d['all_func_evals']['mean']:.0f}/{d['all_jac_evals']['mean']:.0f}"
                )
    print("", flush=True)


def print_phase2_summary(p2: dict) -> None:
    print(
        f"\n=== PHASE 2A & 2B: 100-SEED BASIN & HYBRID STUDY "
        f"(tf = {p2['tf_days']:.1f} d, theta_f = {p2['theta_f_deg']:.2f} deg) ==="
    )
    for m in METHOD_KEYS_P2:
        d = p2["methods"][m]
        ci = f"[{d['ci95_low']:.2f}, {d['ci95_high']:.2f}]"
        if d["n_converged"] > 0:
            f = d["conv_fuel_kg"]
            print(
                f"  {m:<42s} : {d['n_converged']:3d}/{d['n_runs']} (95% CI {ci}) | "
                f"fuel={f['mean']:.2f}±{f['sd']:.2f} kg (med={f['median']:.2f}, IQR=[{f['q25']:.2f},{f['q75']:.2f}], "
                f"range=[{f['min']:.2f},{f['max']:.2f}]) | "
                f"iters={d['conv_iterations']['mean']:.1f} | t={d['conv_runtime_s']['mean']:.2f}s"
            )
        else:
            print(
                f"  {m:<42s} : {d['n_converged']:3d}/{d['n_runs']} (95% CI {ci}) | "
                f"fuel=N/A | all_t={d['all_runtime_s']['mean']:.2f}s"
            )
    pc = p2["paired_hybrid_vs_random"]
    print("\n  Paired Hybrid vs Random-Costate Indirect Shooting (same 100 seeds):")
    print(
        f"    Convergence: Hybrid {pc['hy_conv']}/{pc['n_seeds']} vs Random {pc['sh_conv']}/{pc['n_seeds']} "
        f"(both={pc['both_conv']}, hybrid_only={pc['hy_only_conv']}, random_only={pc['sh_only_conv']}, both_fail={pc['both_fail']})"
    )
    if pc["paired_sh_mean_iters"] is not None:
        print(
            f"    On {pc['both_conv']} jointly converged seeds: "
            f"shooting propagations {pc['paired_sh_mean_iters']:.1f} (random) -> {pc['paired_hy_mean_iters']:.1f} (hybrid); "
            f"RHS evals {pc['paired_sh_mean_rhs']:.0f} -> {pc['paired_hy_mean_rhs']:.0f}; "
            f"shooting time {pc['paired_sh_mean_runtime_s']:.2f}s -> {pc['paired_hy_shoot_runtime_s']:.2f}s "
            f"(end-to-end incl. X-TFC: {pc['paired_hy_total_runtime_s']:.2f}s)"
        )
    print(
        f"    Improves convergence? {pc['improves_convergence']} | Reduces shooting iterations? {pc['reduces_shooting_iters']}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Markdown table formatters
# ---------------------------------------------------------------------------
def format_phase1_scan_markdown(scan_data: dict) -> str:
    lines = [
        "# Phase 1.1: Global Minimum-Time Feasibility Scan at $T_{\\max} = 0.50\\text{ N}$\n",
        f"Fixed-longitude minimum-time NLP solves ($N = 80$ LGL nodes, 4 initial guesses per angle: "
        f"`direct`, `spiral_0rev`, `spiral_inward`, `spiral_1rev`) across $\\theta_f \\in [60^\\circ, 300^\\circ]$ "
        f"(step $10^\\circ$), supplemented by 5 multi-start free-$\\theta_f$ $N = 180$ LGL minimum-time solves.\n",
        "## 1. Fixed-Longitude Minimum-Time Scan ($\\theta_f \\in [60^\\circ, 300^\\circ]$)\n",
        "| $\\theta_f$ (deg) | Converged Starts | $t_{\\min}(\\theta_f)$ (days) | Propellant at $t_{\\min}$ (kg) | Best Guess Mode | DOP853 Re-prop $\\varepsilon_M$ |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in scan_data["fixed_rows"]:
        t_s = f"{r['t_min_days']:.3f}" if r["t_min_days"] is not None else "N/A (0/4)"
        f_s = f"{r['fuel_kg']:.2f}" if r["fuel_kg"] is not None else "N/A"
        e_s = f"{r['reprop_arrival_residual']:.2e}" if r["reprop_arrival_residual"] is not None else "N/A"
        lines.append(
            f"| {r['theta_f_deg']:.1f}° | {r['n_converged']} / {r['n_attempts']} | {t_s} | {f_s} | `{r['best_guess_mode']}` | {e_s} |"
        )

    lines.extend([
        "",
        "## 2. Multi-Start Free-$\\theta_f$ Minimum-Time Solves ($N = 180$ LGL Nodes)\n",
        "| Initial Guess $(t_f, \\theta_f)$ | Converged? | $t_{\\min}$ (days) | $\\theta_f^*$ (deg) | Fuel (kg) | IPOPT Iters | Runtime (s) | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |",
    ])
    for r in scan_data["free_multistart"]:
        lines.append(
            f"| ({r['start_tf_days']:.0f} d, {r['start_theta_deg']:.1f}°) | {r['success']} | "
            f"{r['tf_days']:.4f} | {r['theta_f_deg']:.3f}° | {r['fuel_kg']:.2f} | "
            f"{r['iterations']} | {r['runtime_s']:.2f} | `{r['status']}` |"
        )

    local_str = "yes" if scan_data["earlier_was_local_minimum"] else "no"
    low_ang_t = [
        r["t_min_days"] for r in scan_data["fixed_rows"]
        if r["theta_f_deg"] <= 180.0 and r["t_min_days"] is not None
    ]
    hi_ang_t = [
        r["t_min_days"] for r in scan_data["fixed_rows"]
        if r["theta_f_deg"] >= 200.0 and r["t_min_days"] is not None
    ]
    low_desc = (
        f"Low-angle targets ($60^\\circ\\text{{--}}180^\\circ$) that converged required "
        f"$t_{{\\min}} \\in [{min(low_ang_t):.1f}, {max(low_ang_t):.1f}]\\text{{ d}}$"
        if low_ang_t else "No low-angle target in $60^\\circ\\text{--}180^\\circ$ converged"
    )
    hi_desc = (
        f"while targets in $[200^\\circ, 300^\\circ]$ achieved $t_{{\\min}} \\ge {min(hi_ang_t):.1f}\\text{{ d}}$ on the $10^\\circ$ grid"
        if hi_ang_t else "while no target in $[200^\\circ, 300^\\circ]$ converged"
    )
    lines.extend([
        "",
        "## 3. Global Scan Assessment\n",
        f"- **Shortest transfer time found across all tested starts:** **{scan_data['global_t_min_days']:.4f} days** "
        f"at $\\theta_f^* = {scan_data['global_theta_f_deg']:.3f}^\\circ$ (source: `{scan_data['global_source']}`).",
        f"- **Was the earlier {scan_data['earlier_t_min_days']:.2f} d result a local minimum?** **{local_str}** — "
        f"within the tested 0-rev direct/spiral and 1-rev spiral trajectory families across $\\theta_f \\in [60^\\circ, 300^\\circ]$ "
        f"and all {len(scan_data['free_multistart'])} free-$\\theta_f$ multi-start solves, no tested longitude or initialization achieved a "
        f"feasible transfer below {scan_data['global_t_min_days']:.2f} days. {low_desc}, {hi_desc}, with the "
        f"free-$\\theta_f$ optimum at $\\theta_f^* = {scan_data['global_theta_f_deg']:.3f}^\\circ$, "
        f"$t_{{\\min}} = {scan_data['global_t_min_days']:.4f}\\text{{ d}}$.",
        "",
    ])
    return "\n".join(lines) + "\n"


def format_phase1_multi_tf_markdown(selected_angles: dict[float, dict], rob_by_tf: dict[float, dict]) -> str:
    lines = [
        "# Phase 1.2–1.4: Multi-$t_f$ 20-Run Robustness Protocol (Fuel-Minimal $\\theta_f^*(t_f)$, All Methods + Cold-Start X-TFC)\n",
        "All four solver configurations are evaluated on the identical 3-D 14-state `HelioProblem` at each transfer duration "
        "$t_f \\in \\{247.9, 280.0, 300.0, 330.0, 360.0\\}\\text{ days}$ over 20 seeds (`1000..1019`). "
        "Re-propagated arrival error $\\varepsilon_M$ is computed by integrating each solver's control profile forward with `DOP853` "
        "(never reporting the TFC boundary-embedded 0 as arrival accuracy).\n",
    ]
    for tf_key in sorted(rob_by_tf.keys()):
        entry = rob_by_tf[tf_key]
        lines.extend([
            f"## $t_f = {entry['tf_days']:.1f}$ days, Fuel-Minimal $\\theta_f^* = {entry['theta_f_deg']:.2f}^\\circ$ ($T_{{\\max}} = 0.50\\text{{ N}}$)\n",
            "| Method | Converged (95% Wilson CI) | Fuel (kg) | Throttle Switches | Coast Arc (days) | Re-prop $\\varepsilon_M$ | Energy Err $\\varepsilon_E$ | Runtime (s) | Mean Func / Jac Evals |",
            "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for m in METHOD_KEYS_P1:
            d = entry["methods"][m]
            ci_s = f"{d['n_converged']}/{d['n_runs']} ([{d['ci95_low']:.2f}, {d['ci95_high']:.2f}])"
            if d["n_converged"] > 0:
                fuel_s = f"{d['conv_fuel_kg']['mean']:.2f} ± {d['conv_fuel_kg']['sd']:.2f}"
                sw_s = f"{d['conv_switches']['mean']:.1f} ± {d['conv_switches']['sd']:.1f}"
                coast_s = f"{d['conv_coast_days']['mean']:.1f} ± {d['conv_coast_days']['sd']:.1f}"
                rep_s = f"{d['conv_reprop_err']['mean']:.2e}"
                ee_s = f"{d['conv_energy_err']['mean']:.2e}"
                rt_s = f"{d['conv_runtime_s']['mean']:.2f} ± {d['conv_runtime_s']['sd']:.2f}"
                ev_s = f"{d['conv_func_evals']['mean']:,.0f} / {d['conv_jac_evals']['mean']:,.0f}"
            else:
                fuel_s = sw_s = coast_s = rep_s = ee_s = "N/A (0 conv)"
                rt_s = f"{d['all_runtime_s']['mean']:.2f} ± {d['all_runtime_s']['sd']:.2f} (all)"
                ev_s = f"{d['all_func_evals']['mean']:,.0f} / {d['all_jac_evals']['mean']:,.0f} (all)"
            lines.append(f"| {m} | {ci_s} | {fuel_s} | {sw_s} | {coast_s} | {rep_s} | {ee_s} | {rt_s} | {ev_s} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def format_phase2_markdown(p2: dict) -> str:
    pc = p2["paired_hybrid_vs_random"]
    lines = [
        f"# Phase 2: 100-Seed Random-Initialization Basin Study & Hybrid X-TFC $\\to$ Shooting Test ($t_f = {p2['tf_days']:.1f}$ d, $\\theta_f^* = {p2['theta_f_deg']:.2f}^\\circ$)\n",
        f"Evaluated on {p2['n_seeds']} independent seeds (`{p2['base_seed']}..{p2['base_seed'] + p2['n_seeds'] - 1}`) on the identical "
        f"3-D 14-state `HelioProblem` ($T_{{\\max}} = 0.50\\text{{ N}}$, $I_{{\\text{{sp}}}} = 3,100\\text{{ s}}$, SRP enabled).\n",
        "## Phase 2A: Convergence Rates (95% Wilson CI) & Final Fuel Distribution\n",
        "| Method | Converged (95% Wilson CI) | Fuel Mean ± SD (kg) | Fuel Median [Q25, Q75] (kg) | Fuel [Min, Max] (kg) | Re-prop $\\varepsilon_M$ | Mean Iters | Mean Runtime (s) | Mean Func / Jac Evals |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in METHOD_KEYS_P2:
        d = p2["methods"][m]
        ci_s = f"{d['n_converged']}/{d['n_runs']} ([{d['ci95_low']:.2f}, {d['ci95_high']:.2f}])"
        if d["n_converged"] > 0:
            f = d["conv_fuel_kg"]
            f_ms = f"{f['mean']:.2f} ± {f['sd']:.2f}"
            f_iqr = f"{f['median']:.2f} [{f['q25']:.2f}, {f['q75']:.2f}]"
            f_rng = f"[{f['min']:.2f}, {f['max']:.2f}]"
            rep_s = f"{d['conv_reprop_err']['mean']:.2e}"
            it_s = f"{d['conv_iterations']['mean']:.1f}"
            rt_s = f"{d['conv_runtime_s']['mean']:.2f}"
            ev_s = f"{d['conv_func_evals']['mean']:,.0f} / {d['conv_jac_evals']['mean']:,.0f}"
        else:
            f_ms = f_iqr = f_rng = rep_s = it_s = "N/A (0 conv)"
            rt_s = f"{d['all_runtime_s']['mean']:.2f} (all)"
            ev_s = f"{d['all_func_evals']['mean']:,.0f} / {d['all_jac_evals']['mean']:,.0f} (all)"
        lines.append(f"| {m} | {ci_s} | {f_ms} | {f_iqr} | {f_rng} | {rep_s} | {it_s} | {rt_s} | {ev_s} |")

    lines.extend([
        "",
        "## Phase 2B: Paired Comparison — Hybrid (Cold-Start X-TFC $\\to$ Shooting) vs Random-Costate Shooting\n",
        "| Metric (Same 100 Seeds) | Random-Costate Shooting | Hybrid (X-TFC $\\to$ Shooting) |",
        "| --- | :---: | :---: |",
        f"| Converged runs (95% Wilson CI) | {p2['methods']['Indirect shooting (random costates)']['n_converged']}/{pc['n_seeds']} ([{p2['methods']['Indirect shooting (random costates)']['ci95_low']:.2f}, {p2['methods']['Indirect shooting (random costates)']['ci95_high']:.2f}]) | {p2['methods']['Hybrid (X-TFC -> Indirect shooting)']['n_converged']}/{pc['n_seeds']} ([{p2['methods']['Hybrid (X-TFC -> Indirect shooting)']['ci95_low']:.2f}, {p2['methods']['Hybrid (X-TFC -> Indirect shooting)']['ci95_high']:.2f}]) |",
        f"| Seed-by-seed contingency | Both conv: {pc['both_conv']} | Hybrid-only conv: {pc['hy_only_conv']}, Random-only conv: {pc['sh_only_conv']}, Both fail: {pc['both_fail']} |",
    ])
    if pc["paired_sh_mean_iters"] is not None:
        lines.extend([
            f"| Mean shooting propagations (on {pc['both_conv']} jointly converged seeds) | {pc['paired_sh_mean_iters']:.1f} | {pc['paired_hy_mean_iters']:.1f} |",
            f"| Mean DOP853 RHS evaluations (jointly converged seeds) | {pc['paired_sh_mean_rhs']:,.0f} | {pc['paired_hy_mean_rhs']:,.0f} |",
            f"| Mean shooting-stage wall-clock runtime (s) | {pc['paired_sh_mean_runtime_s']:.2f} | {pc['paired_hy_shoot_runtime_s']:.2f} |",
            f"| Mean end-to-end wall-clock runtime incl. X-TFC pre-solve (s) | {pc['paired_sh_mean_runtime_s']:.2f} | {pc['paired_hy_total_runtime_s']:.2f} |",
        ])

    verdict_parts = []
    if pc["improves_convergence"]:
        verdict_parts.append(
            f"Initializing indirect shooting with the cold-start X-TFC initial costate vector improved convergence from "
            f"{pc['sh_conv']}/{pc['n_seeds']} to {pc['hy_conv']}/{pc['n_seeds']} (recovering {pc['hy_only_conv']} seeds where "
            f"random costates failed, with {pc['sh_only_conv']} regressions)."
        )
    else:
        verdict_parts.append(
            f"Initializing indirect shooting with the cold-start X-TFC initial costate vector did not improve convergence "
            f"({pc['hy_conv']}/{pc['n_seeds']} vs {pc['sh_conv']}/{pc['n_seeds']} for random costates)."
        )
    if pc["paired_sh_mean_iters"] is not None:
        if pc["reduces_shooting_iters"]:
            verdict_parts.append(
                f"On the {pc['both_conv']} jointly converged seeds, shooting propagations dropped from "
                f"{pc['paired_sh_mean_iters']:.1f} to {pc['paired_hy_mean_iters']:.1f} "
                f"(shooting stage runtime {pc['paired_sh_mean_runtime_s']:.2f} s vs {pc['paired_hy_shoot_runtime_s']:.2f} s), "
                f"though total end-to-end runtime including the X-TFC pre-solve was {pc['paired_hy_total_runtime_s']:.2f} s."
            )
        else:
            verdict_parts.append(
                f"On the {pc['both_conv']} jointly converged seeds, shooting propagations did not decrease "
                f"({pc['paired_sh_mean_iters']:.1f} random vs {pc['paired_hy_mean_iters']:.1f} hybrid), and end-to-end "
                f"runtime including X-TFC was {pc['paired_hy_total_runtime_s']:.2f} s vs {pc['paired_sh_mean_runtime_s']:.2f} s."
            )

    lines.extend(["", "**Assessment:** " + " ".join(verdict_parts), ""])
    return "\n".join(lines) + "\n"


def save_json_clean(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
