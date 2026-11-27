"""Addendum 3 — TASK 2: Fair Equal-Smoothing Accuracy Comparison (fresh seeds 9100..9199, N=100).

Compares on Scenario A (A/m0 = 0):
  1. shoot_rho_1e8:      Indirect shooting continued to rho = 1.27e-8 (stop_at_rho=1e-8).
                         Evaluates BOTH uniform 2001-pt grid reprop AND exact event-aware hard-switch reprop.
  2. shoot_hard_polish:  Starting from converged shoot_rho_1e8 costates, solves exact hard-switched
                         bang-bang BVP (delta = 1_{S>=0}, event-triggered DOP853 at S(t)=0) via fsolve.
  3. split_domain:       Frozen coarse PoNN -> 5-arc Hermite CE SplitDomainPoNN (exact delta in {0, 1}).

Checkpoints atomically per seed to results/addendum3_task2_accuracy_9100_9199.json.
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
from scipy.optimize import fsolve
from scipy.stats import binomtest, wilcoxon

from trajectory_optimization.ponn_paper import PaperPoNNModel, PaperPoNNSolver
from trajectory_optimization.scenario_a import ScenarioAProblem, ScenarioAShootingSolver
from trajectory_optimization.split_domain_ponn import SplitDomainPoNN

OUT_PATH = Path("results/addendum3_task2_accuracy_9100_9199.json")


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


def propagate_hard_bang_bang(prob: ScenarioAProblem, lam0: np.ndarray) -> tuple[np.ndarray, list[float], bool]:
    """Event-aware exact hard-switch (delta in {0, 1}) propagation of 14-state PMP ODEs."""
    y = np.concatenate([prob.r0, prob.v0, [prob.m0], lam0])
    t = 0.0
    sw_times: list[float] = []
    mu = float(prob.mu - (prob.k_srp if prob.srp_mode == "radial" else 0.0))

    def S_val(yy: np.ndarray) -> float:
        lvn = float(np.linalg.norm(yy[10:13]))
        return float(prob.c * lvn / max(float(yy[6]), 1e-9) + float(yy[13]) - 1.0)

    for _ in range(16):
        delta_const = 1.0 if S_val(y) >= 0.0 else 0.0

        def rhs(tt: float, yy: np.ndarray, d: float = delta_const) -> np.ndarray:
            r = yy[0:3]
            v = yy[3:6]
            m = max(float(yy[6]), 1e-9)
            lr = yy[7:10]
            lv = yy[10:13]
            rn = float(np.linalg.norm(r))
            rn3 = rn * rn * rn
            rn5 = rn3 * rn * rn
            lvn = max(float(np.linalg.norm(lv)), 1e-12)
            alpha = -lv / lvn
            dr = v
            dv = -mu * r / rn3 + (prob.t_max * d / m) * alpha
            dm = -prob.t_max * d / prob.c
            dlr = mu * (lv / rn3 - 3.0 * float(np.dot(r, lv)) * r / rn5)
            dlv = -lr
            dlm = -prob.t_max * d * lvn / (m * m)
            return np.concatenate([dr, dv, [dm], dlr, dlv, [dlm]])

        def ev(tt: float, yy: np.ndarray) -> float:
            return S_val(yy)

        ev.terminal = True
        ev.direction = 0

        sol = solve_ivp(rhs, (t, prob.tf), y, method="DOP853", rtol=1e-11, atol=1e-13, events=ev)
        if not sol.success or sol.y.shape[1] == 0:
            return np.full(14, np.nan), sw_times, False
        y = sol.y[:, -1]
        t = float(sol.t[-1])
        if t >= prob.tf - 1e-12:
            return y, sw_times, True
        sw_times.append(t)
        # Nudge 1e-9 TU across zero crossing so next arc picks opposite sign cleanly
        t_next = min(t + 1e-9, prob.tf)
        sol2 = solve_ivp(rhs, (t, t_next), y, method="DOP853", rtol=1e-11, atol=1e-13)
        if not sol2.success or sol2.y.shape[1] == 0:
            return np.full(14, np.nan), sw_times, False
        y = sol2.y[:, -1]
        t = float(sol2.t[-1])
        if t >= prob.tf - 1e-12:
            return y, sw_times, True
    return y, sw_times, False


_SD_CACHE: dict[tuple[float, ...], SplitDomainPoNN] = {}


def _get_sd(prob: ScenarioAProblem, arcs: list[float]) -> SplitDomainPoNN:
    key = tuple(arcs)
    if key not in _SD_CACHE:
        _SD_CACHE[key] = SplitDomainPoNN(prob, thrust_arcs=arcs, n_neurons_arc=18, n_points_arc=36)
    return _SD_CACHE[key]


def _run_seed(seed: int) -> tuple[int, dict[str, Any]]:
    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0, label="scenario_a_task2_acc")

    # 1. Shooting continued to rho = 1e-8 (1.27e-8)
    shooter8 = ScenarioAShootingSolver(stop_at_rho=1e-8)
    r8 = shooter8.solve(prob, seed=seed)
    rp_grid = float(r8.reprop_arrival_residual) if np.isfinite(r8.reprop_arrival_residual) else float("nan")
    ar8 = float(r8.arrival_residual) if np.isfinite(r8.arrival_residual) else float("nan")
    mf8 = float(r8.final_mass_kg) if np.isfinite(r8.final_mass_kg) else 0.0
    rho8 = float(r8.rho_reached) if r8.rho_reached is not None else float("nan")
    ok8_phys = bool(mf8 > 50.0 and np.isfinite(rho8) and rho8 <= 1e-7 and np.isfinite(ar8) and ar8 < 1e-6)

    # Exact event-aware arc-by-arc hard-switch re-propagation of shoot_rho_1e8 costates
    rp8_exact = float("nan")
    if ok8_phys:
        yf_e, sw_e, ok_e = propagate_hard_bang_bang(prob, r8.lam0)
        if ok_e and np.all(np.isfinite(yf_e)):
            rp8_exact = float(np.linalg.norm(np.concatenate([yf_e[0:3] - prob.rf, yf_e[3:6] - prob.vf])))

    shoot_rho_1e8 = {
        "seed": seed,
        "success_loose": bool(ok8_phys and np.isfinite(rp_grid) and rp_grid < 5e-2),
        "success_strict": bool(ok8_phys and np.isfinite(rp_grid) and rp_grid < 1e-2),
        "fuel_kg": float(r8.fuel_kg) if np.isfinite(r8.fuel_kg) else None,
        "rho_reached": rho8 if np.isfinite(rho8) else None,
        "arrival_residual": ar8 if np.isfinite(ar8) else None,
        "reprop_grid_2001": rp_grid if np.isfinite(rp_grid) else None,
        "reprop_exact_arcs": rp8_exact if np.isfinite(rp8_exact) else None,
        "n_propagations": int(r8.n_propagations),
        "runtime_s": float(r8.runtime_s),
    }

    # 2. Hard-switch event polish starting from shoot_rho_1e8
    t0_pol = time.perf_counter()
    pol_ok = False
    pol_err = float("nan")
    pol_fuel = float("nan")
    pol_sw_days: list[float] = []
    if ok8_phys:
        def F_hard(lam: np.ndarray) -> np.ndarray:
            yf, _, ok_f = propagate_hard_bang_bang(prob, lam)
            if not ok_f or not np.all(np.isfinite(yf)):
                return np.full(7, 1e3)
            return np.concatenate([yf[0:3] - prob.rf, yf[3:6] - prob.vf, [yf[13]]])

        try:
            lam_pol, info, ier, _ = fsolve(F_hard, r8.lam0, full_output=True, xtol=1e-12, maxfev=120)
            yf_p, sw_p, ok_p = propagate_hard_bang_bang(prob, lam_pol)
            if ok_p and np.all(np.isfinite(yf_p)):
                pol_err = float(np.linalg.norm(np.concatenate([yf_p[0:3] - prob.rf, yf_p[3:6] - prob.vf])))
                mf_p = float(yf_p[6] * prob.m_ref_kg)
                pol_fuel = float(prob.m0_kg - mf_p)
                pol_sw_days = [float(x * prob.t_ref_days) for x in sw_p]
                pol_ok = bool(mf_p > 50.0 and pol_err < 1e-2 and abs(float(yf_p[13])) < 1e-6)
        except Exception:
            pass
    dt_pol = time.perf_counter() - t0_pol

    shoot_hard_polish = {
        "seed": seed,
        "success_strict": bool(pol_ok),
        "fuel_kg": pol_fuel if np.isfinite(pol_fuel) else None,
        "reprop_exact_arcs": pol_err if np.isfinite(pol_err) else None,
        "switch_times_days": pol_sw_days,
        "polish_runtime_s": float(dt_pol),
        "total_runtime_s": float(r8.runtime_s + dt_pol),
    }

    # 3. Split-domain PoNN pipeline (coarse PoNN -> 5-arc SplitDomainPoNN)
    t0_p = time.perf_counter()
    ponn_solver = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    rc, _ = ponn_solver.solve(prob, seed=seed, refine=False)
    t_ponn = time.perf_counter() - t0_p

    sw_days = [float(sw["t_days"]) for sw in rc.switches]
    if len(sw_days) != 4 or not np.isfinite(rc.residual_rms) or rc.residual_rms > 0.1:
        split_domain = {
            "seed": seed,
            "success_loose": False,
            "success_strict": False,
            "status": f"ponn_coarse_switch_mismatch (n_sw={len(sw_days)})",
            "fuel_kg": None,
            "reprop_exact_arcs": None,
            "ponn_runtime_s": float(t_ponn),
            "split_runtime_s": 0.0,
            "total_runtime_s": float(t_ponn),
        }
    else:
        model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=seed)

        def _eval_ponn(t_tu_arr: np.ndarray) -> np.ndarray:
            z_arr = 2.0 * np.asarray(t_tu_arr, dtype=float) / prob.tf - 1.0
            return model_c.evaluate(rc.xi, z=z_arr)["Y14"]

        init_on = 1.0 if float(rc.S[0]) >= 0.0 else 0.0
        arcs = [init_on if (k % 2 == 0) else (1.0 - init_on) for k in range(len(sw_days) + 1)]
        sd = _get_sd(prob, arcs)
        theta0 = sd.project_trajectory([t / prob.t_ref_days for t in sw_days], _eval_ponn)
        res_sd = sd.solve(theta0, max_iter=80)
        rp_sd = float(res_sd.reprop_arrival_residual) if np.isfinite(res_sd.reprop_arrival_residual) else float("nan")
        split_domain = {
            "seed": seed,
            "success_loose": bool(np.isfinite(rp_sd) and rp_sd < 5e-2),
            "success_strict": bool(np.isfinite(rp_sd) and rp_sd < 1e-2),
            "status": res_sd.status,
            "fuel_kg": float(res_sd.fuel_kg) if np.isfinite(res_sd.fuel_kg) else None,
            "switch_times_days": res_sd.switch_times_days,
            "residual_rms": float(res_sd.residual_rms) if np.isfinite(res_sd.residual_rms) else None,
            "reprop_exact_arcs": rp_sd if np.isfinite(rp_sd) else None,
            "ponn_runtime_s": float(t_ponn),
            "split_runtime_s": float(res_sd.runtime_s),
            "total_runtime_s": float(t_ponn + res_sd.runtime_s),
        }

    return seed, {
        "seed": seed,
        "shoot_rho_1e8": shoot_rho_1e8,
        "shoot_hard_polish": shoot_hard_polish,
        "split_domain": split_domain,
    }


def compute_summary(data: dict[str, Any]) -> dict[str, Any]:
    seeds = data["seeds"]
    keys = sorted(seeds.keys(), key=int)
    n = len(keys)
    if n == 0:
        return {"n_seeds": 0}
    rows = [seeds[k] for k in keys]

    def _arm_s(arm: str, time_k: str, err_k: str) -> dict[str, Any]:
        ok = [bool(r[arm]["success_strict"]) for r in rows]
        ks = sum(ok)
        return {
            "n_seeds": n,
            "n_strict": ks,
            "rate_strict": ks / n,
            "ci_strict": list(wilson_ci(ks, n)),
            "reprop_exact_strict": _dist_stats([r[arm][err_k] for r in rows if r[arm]["success_strict"]]),
            "fuel_strict_kg": _dist_stats([r[arm]["fuel_kg"] for r in rows if r[arm]["success_strict"]]),
            "time_all_s": _dist_stats([r[arm][time_k] for r in rows]),
            "time_strict_s": _dist_stats([r[arm][time_k] for r in rows if r[arm]["success_strict"]]),
        }

    arms = {
        "shoot_rho_1e8": {
            **_arm_s("shoot_rho_1e8", "runtime_s", "reprop_exact_arcs"),
            "reprop_grid_2001_strict": _dist_stats(
                [r["shoot_rho_1e8"]["reprop_grid_2001"] for r in rows if r["shoot_rho_1e8"]["success_strict"]]
            ),
        },
        "shoot_hard_polish": _arm_s("shoot_hard_polish", "total_runtime_s", "reprop_exact_arcs"),
        "split_domain": _arm_s("split_domain", "total_runtime_s", "reprop_exact_arcs"),
    }

    # Raw tests A1..A4 (Holm correction across all M=40 tests happens jointly in generator)
    ok_sh8 = [bool(r["shoot_rho_1e8"]["success_strict"]) for r in rows]
    ok_pol = [bool(r["shoot_hard_polish"]["success_strict"]) for r in rows]
    ok_sd = [bool(r["split_domain"]["success_strict"]) for r in rows]
    a_only = sum(1 for a, b in zip(ok_sd, ok_sh8) if a and not b)
    b_only = sum(1 for a, b in zip(ok_sd, ok_sh8) if not a and b)
    p_a1 = float(binomtest(min(a_only, b_only), a_only + b_only, 0.5, alternative="two-sided").pvalue) if (a_only + b_only) > 0 else 1.0

    def _wx(vals_a: list[float], vals_b: list[float]) -> dict[str, Any]:
        if not vals_a:
            return {"n_paired": 0, "median_diff": 0.0, "p_raw": 1.0}
        diffs = np.asarray(vals_a, dtype=float) - np.asarray(vals_b, dtype=float)
        if np.all(diffs == 0.0):
            return {"n_paired": len(diffs), "median_diff": 0.0, "p_raw": 1.0}
        w = wilcoxon(diffs, alternative="two-sided")
        return {
            "n_paired": int(len(diffs)),
            "median_a": float(np.median(vals_a)),
            "median_b": float(np.median(vals_b)),
            "median_diff": float(np.median(diffs)),
            "p_raw": float(w.pvalue),
        }

    e_sd_8 = [r["split_domain"]["reprop_exact_arcs"] for r in rows if r["split_domain"]["success_strict"] and r["shoot_rho_1e8"]["success_strict"]]
    e_sh8 = [r["shoot_rho_1e8"]["reprop_exact_arcs"] for r in rows if r["split_domain"]["success_strict"] and r["shoot_rho_1e8"]["success_strict"]]

    e_sd_p = [r["split_domain"]["reprop_exact_arcs"] for r in rows if r["split_domain"]["success_strict"] and r["shoot_hard_polish"]["success_strict"]]
    e_pol = [r["shoot_hard_polish"]["reprop_exact_arcs"] for r in rows if r["split_domain"]["success_strict"] and r["shoot_hard_polish"]["success_strict"]]

    t_sd_p = [r["split_domain"]["total_runtime_s"] for r in rows if r["split_domain"]["success_strict"] and r["shoot_hard_polish"]["success_strict"]]
    t_pol = [r["shoot_hard_polish"]["total_runtime_s"] for r in rows if r["split_domain"]["success_strict"] and r["shoot_hard_polish"]["success_strict"]]

    tests = {
        "A1": {
            "comparison": "split_domain vs shoot_rho_1e8 (strict rate)",
            "type": "mcnemar",
            "delta_pp": 100.0 * (sum(ok_sd) - sum(ok_sh8)) / n,
            "a_only": a_only,
            "b_only": b_only,
            "p_raw": p_a1,
        },
        "A2": {"comparison": "split_domain vs shoot_rho_1e8 (reprop_exact_arcs)", "type": "wilcoxon", **_wx(e_sd_8, e_sh8)},
        "A3": {"comparison": "split_domain vs shoot_hard_polish (reprop_exact_arcs)", "type": "wilcoxon", **_wx(e_sd_p, e_pol)},
        "A4": {"comparison": "split_domain vs shoot_hard_polish (total_runtime_s)", "type": "wilcoxon", **_wx(t_sd_p, t_pol)},
    }
    return {"n_seeds": n, "arms": arms, "tests": tests}


def _save(data: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=9100)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=380.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        data = {
            "meta": {
                "description": "Addendum 3 Task 2: Fair equal-smoothing accuracy comparison (seeds 9100..9199)",
                "prereg_addendum3_commit": "792b304",
                "seed_start": args.seed_start,
                "n_seeds_target": args.n_seeds,
            },
            "seeds": {},
            "summary": {},
        }
        _save(data)

    target = list(range(args.seed_start, args.seed_start + args.n_seeds))
    pending = [s for s in target if str(s) not in data["seeds"]]
    print(f"[Addendum 3 Task 2] Done: {len(data['seeds'])}/{len(target)}, Pending: {len(pending)}", flush=True)

    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i in range(0, len(pending), args.workers):
                if time.perf_counter() - t_start > args.max_wall_s:
                    print(f"[Addendum 3 Task 2] Wall-time budget ({args.max_wall_s:.0f}s) reached; checkpoint saved.", flush=True)
                    break
                batch = pending[i : i + args.workers]
                for s, rec in pool.map(_run_seed, batch):
                    data["seeds"][str(s)] = rec
                    print(
                        f"  [s={s}] sh8={int(rec['shoot_rho_1e8']['success_strict'])}(e={rec['shoot_rho_1e8']['reprop_exact_arcs']},t={rec['shoot_rho_1e8']['runtime_s']:.1f}s) "
                        f"pol={int(rec['shoot_hard_polish']['success_strict'])}(e={rec['shoot_hard_polish']['reprop_exact_arcs']},t={rec['shoot_hard_polish']['total_runtime_s']:.1f}s) "
                        f"sd={int(rec['split_domain']['success_strict'])}(e={rec['split_domain']['reprop_exact_arcs']},t={rec['split_domain']['total_runtime_s']:.1f}s)",
                        flush=True,
                    )
                data["summary"] = compute_summary(data)
                _save(data)

    data["summary"] = compute_summary(data)
    _save(data)
    print(f"[Addendum 3 Task 2] Total completed: {len(data['seeds'])}/{len(target)}", flush=True)


if __name__ == "__main__":
    main()
