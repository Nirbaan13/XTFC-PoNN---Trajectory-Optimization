"""Phase C runner: Radial & Attitude-Dependent SRP Perturbation Study on Scenario A.

Sweeps:
  - Radial SRP (P0=4.56e-6 N/m^2, C_R=1.2): A/m in {0.0, 0.02, 0.1, 0.25, 0.5, 1.0} m^2/kg
  - Attitude flat-plate SRP: A/m in {0.1, 0.5, 1.0} m^2/kg

Runs all three methods (frozen Phase A2 PoNN, Indirect Shooting, 180-node LGL Collocation)
across 5 seeds per case and checkpoints to results/scenario_a_phase_c.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from trajectory_optimization.ponn_paper import PaperPoNNSolver
from trajectory_optimization.scenario_a import ScenarioAProblem, ScenarioAShootingSolver
from trajectory_optimization.scenario_a_collocation import ScenarioACollocationSolver

OUT_PATH = Path("results/scenario_a_phase_c.json")
SCRATCH_BACKUP = Path(
    "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/"
    "2f3a08db-5952-4e2e-99ea-99a0ed91d155/scratch/scenario_a_phase_c.json"
)

CASES = [
    ("none_0.00", "none", 0.0),
    ("radial_0.02", "radial", 0.02),
    ("radial_0.10", "radial", 0.10),
    ("radial_0.25", "radial", 0.25),
    ("radial_0.50", "radial", 0.50),
    ("radial_1.00", "radial", 1.00),
    ("attitude_0.10", "attitude_flat_plate", 0.10),
    ("attitude_0.50", "attitude_flat_plate", 0.50),
    ("attitude_1.00", "attitude_flat_plate", 1.00),
]


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
    return {"cases": {}, "summary": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--colloc-seeds", type=int, default=3)
    parser.add_argument("--max-wall-s", type=float, default=500.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    data = _load()
    seeds = [1000 + i for i in range(args.n_seeds)]

    ponn = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    shooter = ScenarioAShootingSolver(stop_at_rho=1e-5)
    col_solver = ScenarioACollocationSolver(n_nodes=180, max_iter=400, tol=1e-7)

    for label, mode, a_over_m in CASES:
        prob = ScenarioAProblem(srp_mode=mode, srp_area_to_mass=a_over_m, label=label)
        c = data["cases"].setdefault(label, {
            "srp_mode": mode,
            "area_to_mass_m2_kg": a_over_m,
            "k_srp_nd": float(prob.k_srp),
            "ponn": {},
            "shooting": {},
            "collocation": {},
        })
        print(f"=== Phase C Case: {label} (mode={mode}, A/m={a_over_m}, k_srp={prob.k_srp:.3e}) ===")
        for idx_s, s in enumerate(seeds):
            if time.perf_counter() - t_start > args.max_wall_s:
                print(f"[checkpoint] wall limit reached in {label}")
                _save(data)
                return
            sk = str(s)
            if sk not in c["shooting"]:
                r = shooter.solve(prob, seed=s)
                c["shooting"][sk] = {
                    "success": bool(r.success),
                    "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
                    "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
                    "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
                    "n_switches": len(r.switches),
                    "switch_times_days": [float(x["t_days"]) for x in r.switches],
                    "thrust_sequence": r.thrust_sequence,
                    "runtime_s": float(r.runtime_s),
                }
                print(f"  [shoot  s={s}] ok={r.success} fuel={r.fuel_kg:.3f} sw={[round(x['t_days'],2) for x in r.switches]} {r.runtime_s:.1f}s")
                _save(data)

            if sk not in c["ponn"]:
                rc, rr = ponn.solve(
                    prob, seed=s, refine=True, refine_neurons=80,
                    points_per_switch=100, refine_grid_mode="uniform_dense",
                    skip_refine_if_coarse_diverged=True,
                )
                rr_eff = rr if rr is not None else rc
                c["ponn"][sk] = {
                    "success": bool(rr_eff.success),
                    "coarse_fuel_kg": float(rc.fuel_kg),
                    "fuel_kg": float(rr_eff.fuel_kg),
                    "residual_rms": float(rr_eff.residual_rms),
                    "reprop_arrival_residual": float(rr_eff.reprop_arrival_residual) if np.isfinite(rr_eff.reprop_arrival_residual) else None,
                    "energy_error": float(rr_eff.energy_error) if np.isfinite(rr_eff.energy_error) else None,
                    "n_switches": len(rr_eff.switches),
                    "switch_times_days": [float(x["t_days"]) for x in rr_eff.switches],
                    "thrust_sequence": rr_eff.thrust_sequence,
                    "runtime_s": float(rr_eff.runtime_s),
                }
                print(f"  [ponn   s={s}] ok={rr_eff.success} fuel={rr_eff.fuel_kg:.3f} sw={[round(x['t_days'],2) for x in rr_eff.switches]} rp={rr_eff.reprop_arrival_residual:.2e} {rr_eff.runtime_s:.1f}s")
                _save(data)

            if idx_s < args.colloc_seeds and sk not in c["collocation"]:
                # Use smooth guess on seed 1000 and random perturbations on 1001..
                col_seed = None if s == 1000 else s
                r = col_solver.solve(prob, seed=col_seed)
                c["collocation"][sk] = {
                    "success": bool(r.success),
                    "fuel_kg": float(r.fuel_kg) if np.isfinite(r.fuel_kg) else None,
                    "reprop_arrival_residual": float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else None,
                    "energy_error": float(r.energy_error) if np.isfinite(r.energy_error) else None,
                    "n_switches": len(r.switches),
                    "switch_times_days": [float(x["t_days"]) for x in r.switches],
                    "thrust_sequence": r.thrust_sequence,
                    "runtime_s": float(r.runtime_s),
                }
                print(f"  [colloc s={s}] ok={r.success} fuel={r.fuel_kg:.3f} sw={[round(x['t_days'],2) for x in r.switches]} rp={r.reprop_arrival_residual:.2e} {r.runtime_s:.1f}s")
                _save(data)

    # Summarize across cases
    summary = {}
    for label, _, _ in CASES:
        c = data["cases"].get(label, {})
        row = {"srp_mode": c.get("srp_mode"), "area_to_mass_m2_kg": c.get("area_to_mass_m2_kg"), "k_srp_nd": c.get("k_srp_nd")}
        for mname in ("shooting", "ponn", "collocation"):
            recs = list(c.get(mname, {}).values())
            conv = [r for r in recs if r.get("success")]
            fuels = [r["fuel_kg"] for r in conv if r.get("fuel_kg") is not None]
            sw0 = [r["switch_times_days"] for r in conv if len(r.get("switch_times_days", [])) == 4]
            row[mname] = {
                "n_conv": len(conv),
                "n_total": len(recs),
                "fuel_mean_kg": float(np.mean(fuels)) if fuels else None,
                "fuel_std_kg": float(np.std(fuels, ddof=1)) if len(fuels) > 1 else (0.0 if fuels else None),
                "fuel_min_kg": float(np.min(fuels)) if fuels else None,
                "switches_mean_days": [float(np.mean([sw[k] for sw in sw0])) for k in range(4)] if sw0 else None,
            }
        summary[label] = row
    data["summary"] = summary
    _save(data)
    print("=== PHASE C COMPLETE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
