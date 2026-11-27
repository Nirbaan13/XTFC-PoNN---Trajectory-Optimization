"""Addendum 3 — TASK 1: Cross-Case Baseline Comparison Across 9 Cases (50 fresh seeds each, 9200..9649).

Cases (9 total, 50 fresh seeds each = 450 seeds):
  1. scen_a_m0_800   (seeds 9200..9249): Scenario A, m0 = 800 kg
  2. scen_a_m0_1200  (seeds 9250..9299): Scenario A, m0 = 1200 kg
  3. scen_a_tmax_0.4 (seeds 9300..9349): Scenario A, Tmax = 0.4 N
  4. scen_a_tmax_0.7 (seeds 9350..9399): Scenario A, Tmax = 0.7 N
  5. scen_b_tf_300   (seeds 9400..9449): Scenario B (theta_f=210 deg), tf = 300 d
  6. scen_b_tf_330   (seeds 9450..9499): Scenario B (theta_f=235 deg), tf = 330 d
  7. scen_b_tf_360   (seeds 9500..9549): Scenario B (theta_f=255 deg), tf = 360 d
  8. scen_a_srp_0.5  (seeds 9550..9599): Scenario A, radial SRP A/m0 = 0.5 m^2/kg
  9. scen_a_srp_1.0  (seeds 9600..9649): Scenario A, radial SRP A/m0 = 1.0 m^2/kg

Five arms per case:
  (a) single_shoot:        k=0 of multistart (sub-seed 10*s + 0)
  (b) multistart_shoot_5:  up to 5 random restarts (sub-seeds 10*s + k, k=0..4), stops at first strict success
  (c) det_energy_shoot:    fixed prograde guess lam0_fix = [0,0,0, -v0/||v0||, 0.5] (5 timing trials per case)
  (d) hybrid_rho1:         frozen coarse PoNN stopped at rho_ext=1.0 -> exact CE costates -> shooting
  (e) split_domain:        frozen coarse PoNN (full coarse) -> multi-arc SplitDomainPoNN refinement

Checkpoints atomically to results/addendum3_task1_cross_case_9200_9649.json.
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
from scipy.stats import binomtest, wilcoxon

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
from trajectory_optimization.split_domain_ponn import SplitDomainPoNN

OUT_PATH = Path("results/addendum3_task1_cross_case_9200_9649.json")
STOP_AT_RHO = 1e-5
SHOOT_SCHED = tuple(r for r in DEFAULT_RHO_SCHEDULE if r >= STOP_AT_RHO - 1e-15)
MAX_RESTARTS = 5

CASES: list[tuple[str, int]] = [
    ("scen_a_m0_800", 9200),
    ("scen_a_m0_1200", 9250),
    ("scen_a_tmax_0.4", 9300),
    ("scen_a_tmax_0.7", 9350),
    ("scen_b_tf_300", 9400),
    ("scen_b_tf_330", 9450),
    ("scen_b_tf_360", 9500),
    ("scen_a_srp_0.5", 9550),
    ("scen_a_srp_1.0", 9600),
]


def build_problem(case_id: str) -> ScenarioAProblem:
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
    if case_id == "scen_a_srp_0.5":
        return ScenarioAProblem(srp_mode="radial", srp_area_to_mass=0.5, label=case_id)
    if case_id == "scen_a_srp_1.0":
        return ScenarioAProblem(srp_mode="radial", srp_area_to_mass=1.0, label=case_id)
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


def _extract_lam0(model: PaperPoNNModel, xi: np.ndarray) -> np.ndarray:
    ev0 = model.evaluate(xi, z=np.array([-1.0]))
    return np.concatenate([ev0["lr"][0], ev0["lv"][0], [ev0["lm"][0]]])


def _is_ok_shoot(r) -> tuple[bool, bool]:
    rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
    ar = float(r.arrival_residual) if np.isfinite(r.arrival_residual) else float("nan")
    mf = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
    rho_r = float(r.rho_reached) if r.rho_reached is not None else float("nan")
    phys_ok = bool(mf > 50.0 and rho_r <= 1e-4 and ar < 1e-6)
    return (
        bool(phys_ok and np.isfinite(rp) and rp < 5e-2),
        bool(phys_ok and np.isfinite(rp) and rp < 1e-2),
    )


def run_det_for_case(case_id: str, n_trials: int = 5) -> dict[str, Any]:
    prob = build_problem(case_id)
    v_hat = prob.v0 / np.linalg.norm(prob.v0)
    lam0_fix = np.concatenate([[0.0, 0.0, 0.0], -v_hat, [0.5]])
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)
    trials = []
    for i in range(n_trials):
        r = shooter.solve(prob, seed=None, lam0_guess=lam0_fix, rho_schedule=SHOOT_SCHED)
        loose, strict = _is_ok_shoot(r)
        trials.append({
            "trial": i,
            "success_loose": loose,
            "success_strict": strict,
            "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
            "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
            "n_switches": len(r.switches),
            "n_propagations": int(r.n_propagations),
            "runtime_s": float(r.runtime_s),
            "status": r.status,
        })
    return {
        "case_id": case_id,
        "lam0_fix": [float(x) for x in lam0_fix],
        "success_loose": bool(trials[0]["success_loose"]),
        "success_strict": bool(trials[0]["success_strict"]),
        "fuel_kg": trials[0]["fuel_kg"],
        "reprop_arrival_residual": trials[0]["reprop_arrival_residual"],
        "n_switches": trials[0]["n_switches"],
        "runtime_stats": _dist_stats([t["runtime_s"] for t in trials]),
        "trials": trials,
    }


_SD_CACHE: dict[tuple[str, tuple[float, ...]], SplitDomainPoNN] = {}


def _get_sd(case_id: str, prob: ScenarioAProblem, arcs: list[float]) -> SplitDomainPoNN:
    key = (case_id, tuple(arcs))
    if key not in _SD_CACHE:
        _SD_CACHE[key] = SplitDomainPoNN(prob, thrust_arcs=arcs, n_neurons_arc=18, n_points_arc=36)
    return _SD_CACHE[key]


def _run_case_seed(args_tuple: tuple[str, int]) -> tuple[str, int, dict[str, Any]]:
    case_id, seed = args_tuple
    prob = build_problem(case_id)
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)

    # 1. Multi-start random shooting (up to 5 restarts; k=0 is single_shoot)
    attempts = []
    cum_t = 0.0
    cum_nfev = 0
    ms_strict = False
    ms_loose = False
    first_att = None
    best_fuel = None
    best_rp = None
    best_nsw = None

    for k in range(MAX_RESTARTS):
        sub_seed = 10 * seed + k
        r = shooter.solve(prob, seed=sub_seed, rho_schedule=SHOOT_SCHED)
        loose, strict = _is_ok_shoot(r)
        cum_t += float(r.runtime_s)
        cum_nfev += int(r.n_propagations)
        att = {
            "attempt": k + 1,
            "sub_seed": sub_seed,
            "success_loose": loose,
            "success_strict": strict,
            "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
            "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
            "n_switches": len(r.switches),
            "n_propagations": int(r.n_propagations),
            "runtime_s": float(r.runtime_s),
        }
        attempts.append(att)
        if loose:
            ms_loose = True
        if strict:
            ms_strict = True
            first_att = k + 1
            best_fuel = att["fuel_kg"]
            best_rp = att["reprop_arrival_residual"]
            best_nsw = att["n_switches"]
            break

    single_shoot = {
        "seed": seed,
        "success_loose": attempts[0]["success_loose"],
        "success_strict": attempts[0]["success_strict"],
        "fuel_kg": attempts[0]["fuel_kg"],
        "reprop_arrival_residual": attempts[0]["reprop_arrival_residual"],
        "n_switches": attempts[0]["n_switches"],
        "runtime_s": attempts[0]["runtime_s"],
    }
    multistart_shoot_5 = {
        "seed": seed,
        "success_loose": ms_loose,
        "success_strict": ms_strict,
        "n_attempts": len(attempts),
        "first_strict_attempt": first_att,
        "fuel_kg": best_fuel,
        "reprop_arrival_residual": best_rp,
        "n_switches": best_nsw,
        "total_propagations": cum_nfev,
        "total_runtime_s": cum_t,
    }

    # 2. Hybrid PoNN(rho_ext=1.0) -> shooting
    z_c = make_collocation_grid(140, "cgl")
    model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=seed)
    ops_c = model_c.operators(z_c)
    xi0 = model_c.initial_guess_paper(ops_c, seed=seed, init_mode="scaled_u01")
    ponn_solver = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    t0_p1 = time.perf_counter()
    xi_rho1, _, _, _, nf_p1, nj_p1, _ = ponn_solver._run_continuation(
        prob, model_c, z_c, xi0, (1.0,), start_scaled=False
    )
    lam0_rho1 = _extract_lam0(model_c, xi_rho1)
    t_ponn_rho1 = time.perf_counter() - t0_p1

    r_hyb = shooter.solve(prob, seed=None, lam0_guess=lam0_rho1, rho_schedule=SHOOT_SCHED)
    hyb_loose, hyb_strict = _is_ok_shoot(r_hyb)
    hybrid_rho1 = {
        "seed": seed,
        "success_loose": hyb_loose,
        "success_strict": hyb_strict,
        "fuel_kg": float(r_hyb.fuel_kg) if np.isfinite(r_hyb.fuel_kg) else None,
        "reprop_arrival_residual": float(r_hyb.reprop_arrival_residual) if np.isfinite(r_hyb.reprop_arrival_residual) else None,
        "n_switches": len(r_hyb.switches),
        "ponn_runtime_s": float(t_ponn_rho1),
        "shoot_runtime_s": float(r_hyb.runtime_s),
        "total_runtime_s": float(t_ponn_rho1 + r_hyb.runtime_s),
    }

    # 3. Split-domain PoNN pipeline (coarse PoNN -> multi-arc SplitDomainPoNN)
    t0_sd_p = time.perf_counter()
    rc, _ = ponn_solver.solve(prob, seed=seed, refine=False)
    t_sd_ponn = time.perf_counter() - t0_sd_p

    sw_days = sorted(float(sw["t_days"]) for sw in rc.switches)
    # Guard: require 1..6 valid interior switches with min arc length >= 2 days and reasonable coarse RMS
    tf_days = float(prob.tf * prob.t_ref_days)
    t_nodes_d = [0.0] + sw_days + [tf_days]
    min_arc_d = min(t_nodes_d[i + 1] - t_nodes_d[i] for i in range(len(t_nodes_d) - 1)) if sw_days else 0.0

    if not (1 <= len(sw_days) <= 6) or min_arc_d < 2.0 or not np.isfinite(rc.residual_rms) or rc.residual_rms > 0.15:
        split_domain = {
            "seed": seed,
            "success_loose": False,
            "success_strict": False,
            "status": f"coarse_ponn_unusable (n_sw={len(sw_days)}, rms={rc.residual_rms:.2e})",
            "ponn_n_switches": len(sw_days),
            "fuel_kg": None,
            "reprop_arrival_residual": None,
            "ponn_runtime_s": float(t_sd_ponn),
            "split_runtime_s": 0.0,
            "total_runtime_s": float(t_sd_ponn),
        }
    else:
        def _eval_ponn(t_tu_arr: np.ndarray) -> np.ndarray:
            z_arr = 2.0 * np.asarray(t_tu_arr, dtype=float) / prob.tf - 1.0
            return model_c.evaluate(rc.xi, z=z_arr)["Y14"]

        init_on = 1.0 if float(rc.S[0]) >= 0.0 else 0.0
        arcs = [init_on if (k % 2 == 0) else (1.0 - init_on) for k in range(len(sw_days) + 1)]
        try:
            sd = _get_sd(case_id, prob, arcs)
            theta0 = sd.project_trajectory([t / prob.t_ref_days for t in sw_days], _eval_ponn)
            res_sd = sd.solve(theta0, max_iter=80)
            rp_sd = float(res_sd.reprop_arrival_residual) if np.isfinite(res_sd.reprop_arrival_residual) else float("nan")
            ok_phys = bool(np.isfinite(res_sd.final_mass_kg) and res_sd.final_mass_kg > 50.0 and np.isfinite(res_sd.residual_rms) and res_sd.residual_rms < 1e-2)
            split_domain = {
                "seed": seed,
                "success_loose": bool(ok_phys and np.isfinite(rp_sd) and rp_sd < 5e-2),
                "success_strict": bool(ok_phys and np.isfinite(rp_sd) and rp_sd < 1e-2),
                "status": res_sd.status,
                "ponn_n_switches": len(sw_days),
                "fuel_kg": float(res_sd.fuel_kg) if np.isfinite(res_sd.fuel_kg) else None,
                "switch_times_days": res_sd.switch_times_days,
                "residual_rms": float(res_sd.residual_rms) if np.isfinite(res_sd.residual_rms) else None,
                "reprop_arrival_residual": rp_sd if np.isfinite(rp_sd) else None,
                "ponn_runtime_s": float(t_sd_ponn),
                "split_runtime_s": float(res_sd.runtime_s),
                "total_runtime_s": float(t_sd_ponn + res_sd.runtime_s),
            }
        except Exception as exc:
            split_domain = {
                "seed": seed,
                "success_loose": False,
                "success_strict": False,
                "status": f"split_exception: {exc}",
                "ponn_n_switches": len(sw_days),
                "fuel_kg": None,
                "reprop_arrival_residual": None,
                "ponn_runtime_s": float(t_sd_ponn),
                "split_runtime_s": 0.0,
                "total_runtime_s": float(t_sd_ponn),
            }

    return case_id, seed, {
        "seed": seed,
        "single_shoot": single_shoot,
        "multistart_shoot_5": multistart_shoot_5,
        "hybrid_rho1": hybrid_rho1,
        "split_domain": split_domain,
    }


def _summarize_case(case_data: dict[str, Any]) -> dict[str, Any]:
    seeds = case_data["seeds"]
    keys = sorted(seeds.keys(), key=int)
    n = len(keys)
    if n == 0:
        return {"n_seeds": 0}
    rows = [seeds[k] for k in keys]

    arms = {}
    for arm, tk in [
        ("single_shoot", "runtime_s"),
        ("multistart_shoot_5", "total_runtime_s"),
        ("hybrid_rho1", "total_runtime_s"),
        ("split_domain", "total_runtime_s"),
    ]:
        ok_s = [bool(r[arm]["success_strict"]) for r in rows]
        ok_l = [bool(r[arm]["success_loose"]) for r in rows]
        ks, kl = sum(ok_s), sum(ok_l)
        arms[arm] = {
            "n_seeds": n,
            "n_loose": kl,
            "rate_loose": kl / n,
            "ci_loose": list(wilson_ci(kl, n)),
            "n_strict": ks,
            "rate_strict": ks / n,
            "ci_strict": list(wilson_ci(ks, n)),
            "time_all_s": _dist_stats([r[arm][tk] for r in rows]),
            "time_strict_s": _dist_stats([r[arm][tk] for r in rows if r[arm]["success_strict"]]),
            "fuel_strict_kg": _dist_stats([r[arm]["fuel_kg"] for r in rows if r[arm]["success_strict"]]),
        }

    ok_ms = [bool(r["multistart_shoot_5"]["success_strict"]) for r in rows]
    ok_hy = [bool(r["hybrid_rho1"]["success_strict"]) for r in rows]
    ok_sd = [bool(r["split_domain"]["success_strict"]) for r in rows]

    def _mc(a_ok, b_ok):
        ao = sum(1 for a, b in zip(a_ok, b_ok) if a and not b)
        bo = sum(1 for a, b in zip(a_ok, b_ok) if not a and b)
        p = float(binomtest(min(ao, bo), ao + bo, 0.5, alternative="two-sided").pvalue) if (ao + bo) > 0 else 1.0
        return {"delta_pp": 100.0 * (sum(a_ok) - sum(b_ok)) / n, "a_only": ao, "b_only": bo, "p_raw": p}

    def _wx(arm_a: str, arm_b: str):
        ta = [r[arm_a]["total_runtime_s"] for r in rows if r[arm_a]["success_strict"] and r[arm_b]["success_strict"]]
        tb = [r[arm_b]["total_runtime_s"] for r in rows if r[arm_a]["success_strict"] and r[arm_b]["success_strict"]]
        if not ta:
            return {"n_paired": 0, "median_diff_s": 0.0, "p_raw": 1.0}
        diffs = np.asarray(ta, dtype=float) - np.asarray(tb, dtype=float)
        if np.all(diffs == 0.0):
            return {"n_paired": len(diffs), "median_diff_s": 0.0, "p_raw": 1.0}
        w = wilcoxon(diffs, alternative="two-sided")
        return {
            "n_paired": int(len(diffs)),
            "median_a_s": float(np.median(ta)),
            "median_b_s": float(np.median(tb)),
            "median_diff_s": float(np.median(diffs)),
            "p_raw": float(w.pvalue),
        }

    return {
        "n_seeds": n,
        "arms": arms,
        "tests_vs_multistart": {
            "hyb_rate": _mc(ok_hy, ok_ms),
            "sd_rate": _mc(ok_sd, ok_ms),
            "hyb_time": _wx("hybrid_rho1", "multistart_shoot_5"),
            "sd_time": _wx("split_domain", "multistart_shoot_5"),
        },
    }


def _save(data: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-wall-s", type=float, default=360.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        data = {
            "meta": {
                "description": "Addendum 3 Task 1: Cross-case baseline comparison (9 cases x 50 fresh seeds, 9200..9649)",
                "prereg_addendum3_commit": "792b304",
                "n_seeds_per_case": args.n_seeds,
                "max_restarts": MAX_RESTARTS,
            },
            "cases": {},
        }
        _save(data)

    # Ensure every case entry + deterministic baseline exists
    for cid, s0 in CASES:
        if cid not in data["cases"]:
            print(f"[Task 1] Running deterministic energy-opt baseline for {cid} (5 trials)...", flush=True)
            det = run_det_for_case(cid, n_trials=5)
            print(
                f"  [DET {cid}] strict={det['success_strict']} fuel={det['fuel_kg']} "
                f"sw={det['n_switches']} med_t={det['runtime_stats']['median']:.2f}s",
                flush=True,
            )
            data["cases"][cid] = {"seed_start": s0, "det_energy_shoot": det, "seeds": {}, "summary": {}}
            _save(data)
            if time.perf_counter() - t_start > args.max_wall_s:
                print(f"[Task 1] Wall-time budget reached after deterministic init.", flush=True)
                return

    # Collect pending (case_id, seed) tasks across all 9 cases
    pending: list[tuple[str, int]] = []
    for cid, s0 in CASES:
        done = data["cases"][cid]["seeds"]
        for s in range(s0, s0 + args.n_seeds):
            if str(s) not in done:
                pending.append((cid, s))

    total_target = len(CASES) * args.n_seeds
    total_done = total_target - len(pending)
    print(f"[Task 1] Done: {total_done}/{total_target}, Pending: {len(pending)}", flush=True)

    if pending:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=args.workers)
        try:
            for cid, s, rec in pool.imap_unordered(_run_case_seed, pending):
                data["cases"][cid]["seeds"][str(s)] = rec
                data["cases"][cid]["summary"] = _summarize_case(data["cases"][cid])
                print(
                    f"  [{cid} s={s}] ss={int(rec['single_shoot']['success_strict'])} "
                    f"ms5={int(rec['multistart_shoot_5']['success_strict'])}(att={rec['multistart_shoot_5']['n_attempts']},t={rec['multistart_shoot_5']['total_runtime_s']:.1f}s) "
                    f"hyb={int(rec['hybrid_rho1']['success_strict'])}(t={rec['hybrid_rho1']['total_runtime_s']:.1f}s) "
                    f"sd={int(rec['split_domain']['success_strict'])}(t={rec['split_domain']['total_runtime_s']:.1f}s)",
                    flush=True,
                )
                _save(data)
                if time.perf_counter() - t_start > args.max_wall_s:
                    print(f"[Task 1] Wall-time budget ({args.max_wall_s:.0f}s) reached; checkpoint saved.", flush=True)
                    pool.terminate()
                    break
        finally:
            pool.close()
            pool.join()

    for cid, _ in CASES:
        data["cases"][cid]["summary"] = _summarize_case(data["cases"][cid])
    _save(data)
    done_now = sum(len(data["cases"][cid]["seeds"]) for cid, _ in CASES)
    print(f"[Task 1] Total completed: {done_now}/{total_target}", flush=True)


if __name__ == "__main__":
    main()
