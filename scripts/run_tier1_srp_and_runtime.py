"""Tier 1.3 & 1.4: SRP Finite-Difference Audit (1000 random states), Resolution Check,
and Single-Process Idle-Machine Runtime Audit (fresh seeds 4100..4129, N=30).

Outputs:
  - results/tier1_srp_audit.json
  - results/tier1_runtime_single_process.json
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import casadi
import jax
import matplotlib
import numpy as np
import pytest
import scipy

from trajectory_optimization.ponn_paper import (
    PaperPoNNModel,
    PaperPoNNSolver,
    make_collocation_grid,
)
from trajectory_optimization.scenario_a import (
    DEFAULT_RHO_SCHEDULE,
    ScenarioAProblem,
    ScenarioAShootingSolver,
    hamiltonian_paper,
    make_scenario_a_rhs14,
    srp_accel_and_jacobian,
)
from trajectory_optimization.scenario_a_collocation import ScenarioACollocationSolver

SRP_OUT = Path("results/tier1_srp_audit.json")
RT_OUT = Path("results/tier1_runtime_single_process.json")

K_EXT = 1
STOP_AT_RHO = 1e-5
RHO_SCHED_HYB = tuple(r for r in DEFAULT_RHO_SCHEDULE[K_EXT:] if r >= STOP_AT_RHO - 1e-15)


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


def collect_hardware_software_meta() -> dict[str, Any]:
    cpu_model = platform.processor()
    try:
        out = subprocess.check_output("lscpu", shell=True, text=True)
        for line in out.splitlines():
            if "Model name:" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_gb = round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
                    break
    except Exception:
        pass
    return {
        "os": platform.platform(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": mem_gb,
        "python_version": sys.version.split()[0],
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "casadi": casadi.__version__,
            "jax": jax.__version__,
            "pytest": pytest.__version__,
        },
        "env_threads": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "single_process_idle": True,
    }


def run_srp_fd_and_resolution_audit(n_states: int = 1000, seed: int = 4900) -> dict[str, Any]:
    """Central FD check of PMP costate ODE terms at N>=1000 random states + SRP resolution check."""
    rng = np.random.default_rng(seed)
    eps = 1e-6
    rho_test = 1e-2
    modes_ams = [
        ("radial", 0.02),
        ("radial", 0.10),
        ("radial", 0.50),
        ("radial", 1.00),
        ("attitude_flat_plate", 0.02),
        ("attitude_flat_plate", 0.10),
        ("attitude_flat_plate", 0.50),
        ("attitude_flat_plate", 1.00),
    ]

    # Sample N random physical states in heliocentric canonical units:
    # r in [0.7, 2.0] AU, v in [-1.5, 1.5], m in [0.3, 1.2], lr, lv in [-2, 2], lm in [-1, 2]
    radii = rng.uniform(0.7, 2.0, size=n_states)
    dirs = rng.normal(size=(n_states, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r_all = dirs * radii[:, None]
    v_all = rng.uniform(-1.5, 1.5, size=(n_states, 3))
    m_all = rng.uniform(0.3, 1.2, size=(n_states, 1))
    lr_all = rng.uniform(-2.0, 2.0, size=(n_states, 3))
    lv_all = rng.uniform(-2.0, 2.0, size=(n_states, 3))
    # Avoid degenerate ||lv|| ~ 0
    lv_norm = np.linalg.norm(lv_all, axis=1, keepdims=True)
    lv_all = np.where(lv_norm < 0.05, lv_all + 0.2, lv_all)
    lm_all = rng.uniform(-1.0, 2.0, size=(n_states, 1))
    Y_all = np.hstack([r_all, v_all, m_all, lr_all, lv_all, lm_all])

    fd_results = []
    for mode, am in modes_ams:
        prob = ScenarioAProblem(srp_mode=mode, srp_area_to_mass=am)
        rhs = make_scenario_a_rhs14(prob, rho_test)
        max_rel_lr = 0.0
        max_rel_lv = 0.0
        max_rel_lm = 0.0
        max_abs_lr = 0.0
        for i in range(n_states):
            y = Y_all[i]
            dy = rhs(0.0, y)
            lr_dot_an = dy[7:10]
            lv_dot_an = dy[10:13]
            lm_dot_an = dy[13]

            # Central FD of exact PMP partial derivatives -dH/dr, -dH/dv, -dH/dm
            # (holding control (delta, alpha) fixed per PMP Eqs. 39-41)
            r = y[0:3]; v = y[3:6]; m = y[6]; lr = y[7:10]; lv = y[10:13]; lm = y[13]
            lvn = float(np.linalg.norm(lv))
            alpha = -lv / lvn
            S = prob.c * lvn / m + lm - 1.0
            delta = 0.5 * (1.0 + np.tanh(S / rho_test))

            def _H_fixed_ctrl(r_in, v_in, m_in):
                rn_in = float(np.linalg.norm(r_in))
                ap_in, _ = srp_accel_and_jacobian(prob, r_in, alpha)
                acc_in = -prob.mu * r_in / (rn_in ** 3) + ap_in + (prob.t_max * delta / m_in) * alpha
                return float(prob.t_max * delta / prob.c + np.dot(lr, v_in) + np.dot(lv, acc_in) - lm * prob.t_max * delta / prob.c)

            lr_dot_fd = np.zeros(3)
            for j in range(3):
                dr = np.zeros(3); dr[j] = eps
                lr_dot_fd[j] = -(_H_fixed_ctrl(r + dr, v, m) - _H_fixed_ctrl(r - dr, v, m)) / (2.0 * eps)

            lv_dot_fd = np.zeros(3)
            for j in range(3):
                dv = np.zeros(3); dv[j] = eps
                lv_dot_fd[j] = -(_H_fixed_ctrl(r, v + dv, m) - _H_fixed_ctrl(r, v - dv, m)) / (2.0 * eps)

            lm_dot_fd = -(_H_fixed_ctrl(r, v, m + eps) - _H_fixed_ctrl(r, v, m - eps)) / (2.0 * eps)

            err_lr = float(np.linalg.norm(lr_dot_an - lr_dot_fd))
            rel_lr = err_lr / max(float(np.linalg.norm(lr_dot_an)), 1.0)
            rel_lv = float(np.linalg.norm(lv_dot_an - lv_dot_fd)) / max(float(np.linalg.norm(lv_dot_an)), 1.0)
            err_lm = abs(lm_dot_an - lm_dot_fd)
            rel_lm = err_lm / max(abs(lm_dot_an), 1.0)
            if rel_lr > max_rel_lr:
                max_rel_lr = rel_lr
            if err_lr > max_abs_lr:
                max_abs_lr = err_lr
            if rel_lv > max_rel_lv:
                max_rel_lv = rel_lv
            if rel_lm > max_rel_lm:
                max_rel_lm = rel_lm

        fd_results.append({
            "srp_mode": mode,
            "area_to_mass_m2_kg": am,
            "k_srp_nd": float(prob.k_srp),
            "n_states": n_states,
            "max_rel_err_lam_r_dot": max_rel_lr,
            "max_abs_err_lam_r_dot": max_abs_lr,
            "max_rel_err_lam_v_dot": max_rel_lv,
            "max_rel_err_lam_m_dot": max_rel_lm,
            "pass_1e-7": bool(max(max_rel_lr, max_rel_lv, max_rel_lm) < 1e-7),
        })

    # Resolution check from stored results/scenario_a_phase_c.json
    resolution_rows = []
    phase_c_path = Path("results/scenario_a_phase_c.json")
    if phase_c_path.exists():
        pc = json.loads(phase_c_path.read_text())
        cases = pc["cases"]

        def _med_fuel(case_obj, method_key):
            vals = []
            for sk, rec in case_obj.get(method_key, {}).items():
                if rec.get("success") and rec.get("fuel_kg") is not None:
                    vals.append(float(rec["fuel_kg"]))
                elif method_key == "ponn" and isinstance(rec.get("refined"), dict):
                    rr = rec["refined"]
                    if rr.get("success") and rr.get("fuel_kg") is not None:
                        vals.append(float(rr["fuel_kg"]))
            return float(np.median(vals)) if vals else None

        base_sh = _med_fuel(cases["none_0.00"], "shooting")
        base_ponn = _med_fuel(cases["none_0.00"], "ponn")
        base_col = _med_fuel(cases["none_0.00"], "collocation")

        for cid, cobj in cases.items():
            if cid == "none_0.00":
                continue
            f_sh = _med_fuel(cobj, "shooting")
            f_ponn = _med_fuel(cobj, "ponn")
            f_col = _med_fuel(cobj, "collocation")
            dm_sh = (f_sh - base_sh) if (f_sh is not None and base_sh is not None) else None
            dm_ponn = (f_ponn - base_ponn) if (f_ponn is not None and base_ponn is not None) else None
            dm_col = (f_col - base_col) if (f_col is not None and base_col is not None) else None
            err_ponn_vs_sh = abs(f_ponn - f_sh) if (f_ponn is not None and f_sh is not None) else None
            err_col_vs_sh = abs(f_col - f_sh) if (f_col is not None and f_sh is not None) else None
            # Resolution ratio: |Delta m_f (shooting reference)| / |m_method - m_shoot|
            res_ponn = (abs(dm_sh) / max(err_ponn_vs_sh, 1e-12)) if (dm_sh is not None and err_ponn_vs_sh is not None) else None
            res_col = (abs(dm_sh) / max(err_col_vs_sh, 1e-12)) if (dm_sh is not None and err_col_vs_sh is not None) else None
            resolution_rows.append({
                "case_id": cid,
                "srp_mode": cobj["srp_mode"],
                "area_to_mass_m2_kg": cobj["area_to_mass_m2_kg"],
                "fuel_shoot_kg": f_sh,
                "fuel_ponn_kg": f_ponn,
                "fuel_colloc_kg": f_col,
                "delta_fuel_shoot_kg": dm_sh,
                "delta_fuel_ponn_kg": dm_ponn,
                "delta_fuel_colloc_kg": dm_col,
                "err_ponn_vs_shoot_kg": err_ponn_vs_sh,
                "err_colloc_vs_shoot_kg": err_col_vs_sh,
                "resolution_ratio_ponn": res_ponn,
                "resolution_ratio_colloc": res_col,
            })

    out = {
        "meta": {
            "description": "Tier 1.3 SRP flat-plate documentation, 1000-random-state FD audit, and fuel-change resolution check",
            "n_fd_states": n_states,
            "fd_eps": eps,
            "fd_rho": rho_test,
            "flat_plate_model": {
                "frame": "Heliocentric J2000 inertial Cartesian frame",
                "attitude_coupling": "Prescribed thrust-aligned normal n_hat = alpha* = -lam_v / ||lam_v|| (NOT independently optimized as a separate attitude control)",
                "eclipse_model": "None (continuous solar illumination; no planetary shadowing)",
                "area_projection_factor": "phi(r, alpha) = 0.5 + 0.5 * (r_hat . alpha)^2",
                "mass_scaling": "Constant initial area-to-mass ratio A/m0 (cannonball scaling; does not vary with m(t))",
                "acceleration_formula": "a_srp(r, alpha) = k_srp * phi(r, alpha) * r / ||r||^3, where k_srp = P0 * C_R * (A/m0) / a_ref",
            },
        },
        "fd_checks": fd_results,
        "resolution_check": resolution_rows,
    }
    SRP_OUT.parent.mkdir(parents=True, exist_ok=True)
    SRP_OUT.write_text(json.dumps(out, indent=2))
    print(f"[Tier 1.3 SRP Audit] Saved to {SRP_OUT}", flush=True)
    return out


def run_single_process_runtime_audit(n_seeds: int = 30, max_wall_s: float = 420.0) -> dict[str, Any]:
    """Strictly single-process idle-machine runtime audit on fresh seeds 4100..4129."""
    if RT_OUT.exists():
        data = json.loads(RT_OUT.read_text())
    else:
        data = {
            "meta": {
                "description": "Tier 1.4 Single-Process Idle-Machine Runtime Audit (Scenario A, fresh seeds 4100..4129)",
                "prereg_addendum_commit": "38e2908",
                "seed_start": 4100,
                "seed_end": 4100 + n_seeds - 1,
                "hardware_software": collect_hardware_software_meta(),
            },
            "seeds": {},
            "summary": {},
        }

    prob = ScenarioAProblem(srp_mode="none", srp_area_to_mass=0.0, label="scenario_a_runtime_sp")
    shooter = ScenarioAShootingSolver(stop_at_rho=STOP_AT_RHO)
    ponn_full = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    col_solver = ScenarioACollocationSolver(n_nodes=180, max_iter=500, tol=1e-7)

    # One-time JAX JIT warmup (recorded separately in meta so seed 4100 is not biased by compilation)
    t_jit0 = time.perf_counter()
    z_w = make_collocation_grid(140, "cgl")
    m_w = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=0)
    ops_w = m_w.operators(z_w)
    xi_w = m_w.initial_guess_paper(ops_w, seed=0, init_mode="scaled_u01")
    _ = m_w.residual_and_jacobian(xi_w, ops_w, rho=1.0)
    data["meta"]["jax_jit_warmup_s"] = float(time.perf_counter() - t_jit0)

    t_start = time.perf_counter()
    for i in range(n_seeds):
        s = 4100 + i
        sk = str(s)
        if sk in data["seeds"]:
            continue
        if time.perf_counter() - t_start > max_wall_s:
            print(f"[Tier 1.4 Runtime] wall limit reached at {len(data['seeds'])}/{n_seeds}", flush=True)
            break

        # 1. Shooting
        t0 = time.perf_counter()
        r_sh = shooter.solve(prob, seed=s)
        t_sh = time.perf_counter() - t0
        rp_sh = float(r_sh.reprop_arrival_residual) if np.isfinite(r_sh.reprop_arrival_residual) else float("nan")
        phys_sh = bool(r_sh.final_mass_kg > 50.0 and (r_sh.rho_reached or 1.0) <= 1e-4 and r_sh.arrival_residual < 1e-6)
        rec_sh = {
            "success_strict": bool(phys_sh and np.isfinite(rp_sh) and rp_sh < 1e-2),
            "success_loose": bool(phys_sh and np.isfinite(rp_sh) and rp_sh < 5e-2),
            "runtime_s": float(t_sh),
            "nfev": int(r_sh.n_propagations),
            "njev": int(r_sh.n_jacobian_builds),
        }

        # 2. Hybrid (PoNN stage + Shooting stage separately)
        t0 = time.perf_counter()
        z_c = make_collocation_grid(140, "cgl")
        model_c = PaperPoNNModel(prob, n_neurons=70, basis_type="chebyshev_shifted", elm_seed=s)
        ops_c = model_c.operators(z_c)
        xi0 = model_c.initial_guess_paper(ops_c, seed=s, init_mode="scaled_u01")
        ponn_s = PaperPoNNSolver()
        xi_c, _, _, _, nf_p, nj_p, _ = ponn_s._run_continuation(
            prob, model_c, z_c, xi0, DEFAULT_RHO_SCHEDULE[:K_EXT + 1], start_scaled=False
        )
        t_hyb_ponn = time.perf_counter() - t0
        ev0 = model_c.evaluate(xi_c, z=np.array([-1.0]))
        lam0 = np.concatenate([ev0["lr"][0], ev0["lv"][0], [ev0["lm"][0]]])
        t1 = time.perf_counter()
        r_hy = shooter.solve(prob, seed=s, lam0_guess=lam0, rho_schedule=RHO_SCHED_HYB)
        t_hyb_shoot = time.perf_counter() - t1
        rp_hy = float(r_hy.reprop_arrival_residual) if np.isfinite(r_hy.reprop_arrival_residual) else float("nan")
        phys_hy = bool(r_hy.final_mass_kg > 50.0 and (r_hy.rho_reached or 1.0) <= 1e-4 and r_hy.arrival_residual < 1e-6)
        rec_hy = {
            "success_strict": bool(phys_hy and np.isfinite(rp_hy) and rp_hy < 1e-2),
            "success_loose": bool(phys_hy and np.isfinite(rp_hy) and rp_hy < 5e-2),
            "ponn_runtime_s": float(t_hyb_ponn),
            "shoot_runtime_s": float(t_hyb_shoot),
            "runtime_s": float(t_hyb_ponn + t_hyb_shoot),
            "ponn_nfev": int(nf_p),
            "ponn_njev": int(nj_p),
            "shoot_nfev": int(r_hy.n_propagations),
            "shoot_njev": int(r_hy.n_jacobian_builds),
        }

        # 3. Standalone PoNN (coarse + refined)
        t0 = time.perf_counter()
        rc, rr = ponn_full.solve(
            prob, seed=s, refine=True, refine_neurons=80,
            points_per_switch=100, refine_grid_mode="uniform_dense",
            skip_refine_if_coarse_diverged=True,
        )
        t_ponn = time.perf_counter() - t0
        ru = rr if rr is not None else rc
        rp_p = float(ru.reprop_arrival_residual) if np.isfinite(ru.reprop_arrival_residual) else float("nan")
        phys_p = bool(ru.final_mass_kg > 50.0 and ru.residual_rms < 1.5e-2)
        rec_ponn = {
            "success_strict": bool(phys_p and np.isfinite(rp_p) and rp_p < 1e-2),
            "success_loose": bool(phys_p and np.isfinite(rp_p) and rp_p < 5e-2),
            "coarse_runtime_s": float(rc.runtime_s),
            "refine_runtime_s": float(rr.runtime_s) if rr is not None else 0.0,
            "runtime_s": float(t_ponn),
            "nfev": int(rc.total_nfev + (rr.total_nfev if rr is not None else 0)),
            "njev": int(rc.total_njev + (rr.total_njev if rr is not None else 0)),
        }

        # 4. 180-Node LGL Collocation
        t0 = time.perf_counter()
        r_col = col_solver.solve(prob, seed=s)
        t_col = time.perf_counter() - t0
        rp_c = float(r_col.reprop_arrival_residual) if np.isfinite(r_col.reprop_arrival_residual) else float("nan")
        phys_c = bool(r_col.success and r_col.final_mass_kg > 50.0 and r_col.terminal_error < 1e-4)
        rec_col = {
            "success_strict": bool(phys_c and np.isfinite(rp_c) and rp_c < 1e-2),
            "success_loose": bool(phys_c and np.isfinite(rp_c) and rp_c < 5e-2),
            "runtime_s": float(t_col),
            "nfev": int(r_col.n_func_evals),
            "njev": int(r_col.n_jac_evals),
            "iterations": int(r_col.iterations),
        }

        data["seeds"][sk] = {
            "seed": s,
            "shooting": rec_sh,
            "hybrid": rec_hy,
            "ponn": rec_ponn,
            "collocation": rec_col,
        }
        _summarize_runtime(data)
        tmp = RT_OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, RT_OUT)
        print(
            f"  [SP runtime s={s}] sh={t_sh:.1f}s hyb={t_hyb_ponn:.1f}+{t_hyb_shoot:.1f}s "
            f"ponn={t_ponn:.1f}s col={t_col:.1f}s ({len(data['seeds'])}/{n_seeds})",
            flush=True,
        )

    _summarize_runtime(data)
    tmp = RT_OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, RT_OUT)
    print(f"[Tier 1.4 Runtime] Saved {len(data['seeds'])}/{n_seeds} seeds to {RT_OUT}", flush=True)
    return data


def _summarize_runtime(data: dict) -> None:
    keys = sorted(data["seeds"].keys(), key=int)
    summary: dict[str, Any] = {"n_seeds": len(keys)}
    for m in ("shooting", "hybrid", "ponn", "collocation"):
        recs = [data["seeds"][k][m] for k in keys]
        succ_s = [r for r in recs if r["success_strict"]]
        fail_s = [r for r in recs if not r["success_strict"]]
        succ_l = [r for r in recs if r["success_loose"]]
        fail_l = [r for r in recs if not r["success_loose"]]
        s_entry: dict[str, Any] = {
            "n_strict": len(succ_s),
            "n_loose": len(succ_l),
            "runtime_all": _dist_stats([r["runtime_s"] for r in recs]),
            "runtime_strict_success": _dist_stats([r["runtime_s"] for r in succ_s]),
            "runtime_strict_failure": _dist_stats([r["runtime_s"] for r in fail_s]),
            "runtime_loose_success": _dist_stats([r["runtime_s"] for r in succ_l]),
            "runtime_loose_failure": _dist_stats([r["runtime_s"] for r in fail_l]),
        }
        if m == "hybrid":
            s_entry["ponn_stage_runtime_all"] = _dist_stats([r["ponn_runtime_s"] for r in recs])
            s_entry["shoot_stage_runtime_all"] = _dist_stats([r["shoot_runtime_s"] for r in recs])
            s_entry["ponn_nfev_all"] = _dist_stats([r["ponn_nfev"] for r in recs])
            s_entry["ponn_njev_all"] = _dist_stats([r["ponn_njev"] for r in recs])
            s_entry["shoot_nfev_all"] = _dist_stats([r["shoot_nfev"] for r in recs])
            s_entry["shoot_njev_all"] = _dist_stats([r["shoot_njev"] for r in recs])
        else:
            s_entry["nfev_all"] = _dist_stats([r["nfev"] for r in recs])
            s_entry["njev_all"] = _dist_stats([r["njev"] for r in recs])
        summary[m] = s_entry
    data["summary"] = summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srp-only", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--max-wall-s", type=float, default=420.0)
    args = parser.parse_args()

    if not args.runtime_only:
        run_srp_fd_and_resolution_audit(n_states=1000)
    if not args.srp_only:
        run_single_process_runtime_audit(n_seeds=args.n_seeds, max_wall_s=args.max_wall_s)


if __name__ == "__main__":
    main()
