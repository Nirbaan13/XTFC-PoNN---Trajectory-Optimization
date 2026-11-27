"""TASK 5: Fair tanh-ELM ablation with deterministic spanning weights on Scenario A (A4(b) seeds 1000..1009).

Evaluates deterministic spanning tanh features:
  w_q = w,  b_q = w_q * (-1 + 2*(q-1)/(L-1)),  q = 1..L
  output weights beta ~ U(-0.5, 0.5)
for w = 5.2 (CGL and uniform) and a full w = 1..10 sweep on CGL (L=70->80, n=140->540),
on the exact same 10 seeds (1000..1009) used in Phase A4(b), comparing convergence,
open-loop DOP853 error, fuel, and runtime against Chebyshev and random-draw tanh ELM.

Checkpoints atomically to results/task5_tanh_spanning_ablation.json.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

from trajectory_optimization.ponn_paper import PaperPoNNSolver
from trajectory_optimization.scenario_a import ScenarioAProblem

OUT_PATH = Path("results/task5_tanh_spanning_ablation.json")
SEEDS = list(range(1000, 1010))

CONFIGS: list[tuple[str, float, str, str]] = [
    ("tanh_spanning_w5.2_cgl_u05", 5.2, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w5.2_uni_u05", 5.2, "uniform", "u_neg05_pos05"),
    ("tanh_spanning_w5.2_cgl_scaled", 5.2, "cgl", "scaled_u01"),
    ("tanh_spanning_w1.0_cgl_u05", 1.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w2.0_cgl_u05", 2.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w3.0_cgl_u05", 3.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w4.0_cgl_u05", 4.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w5.0_cgl_u05", 5.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w6.0_cgl_u05", 6.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w7.0_cgl_u05", 7.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w8.0_cgl_u05", 8.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w9.0_cgl_u05", 9.0, "cgl", "u_neg05_pos05"),
    ("tanh_spanning_w10.0_cgl_u05", 10.0, "cgl", "u_neg05_pos05"),
]


def _pack(r, seed: int) -> dict[str, Any]:
    rp = float(r.reprop_arrival_residual) if np.isfinite(r.reprop_arrival_residual) else float("nan")
    rms = float(r.residual_rms) if np.isfinite(r.residual_rms) else float("nan")
    fk = float(r.fuel_kg) if np.isfinite(r.fuel_kg) else float("nan")
    mf = float(r.final_mass_kg) if np.isfinite(r.final_mass_kg) else 0.0
    phys = bool(mf > 50.0 and np.isfinite(rp) and np.isfinite(rms))
    return {
        "seed": seed,
        "success_loose": bool(phys and rp < 5e-2),
        "success_strict": bool(phys and rp < 1e-2),
        "fuel_kg": fk if np.isfinite(fk) else None,
        "final_mass_kg": mf,
        "residual_rms": rms if np.isfinite(rms) else None,
        "reprop_arrival_residual": rp if np.isfinite(rp) else None,
        "n_switches": len(r.switches),
        "switch_times_days": [float(sw["t_days"]) for sw in r.switches],
        "runtime_s": float(r.runtime_s),
        "nfev": int(r.total_nfev),
        "njev": int(r.total_njev),
    }


def _run_one(label: str, w_val: float, gtype: str, imode: str, seed: int) -> dict[str, Any]:
    prob = ScenarioAProblem()
    solver = PaperPoNNSolver(
        n_neurons=70,
        n_points=140,
        solver_algo="lm_svd",
        basis_type="tanh_spanning",
        grid_type=gtype,
        init_mode=imode,
        spanning_w=w_val,
        max_iter_first=100,
    )
    rc, rr = solver.solve(
        prob,
        seed=seed,
        refine=True,
        refine_neurons=80,
        points_per_switch=100,
        refine_grid_mode="uniform_dense",
        skip_refine_if_coarse_diverged=True,
    )
    rr_eff = rr if rr is not None else rc
    total_rt = float(rc.runtime_s + (rr.runtime_s if rr is not None else 0.0))
    ref_pack = _pack(rr_eff, seed)
    ref_pack["total_runtime_s"] = total_rt
    return {"coarse": _pack(rc, seed), "refined": ref_pack}


def _summarize(recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = [v["refined"] for v in recs.values()]
    n = len(vals)
    if n == 0:
        return {"n_seeds": 0}
    n_loose = sum(1 for v in vals if v["success_loose"])
    n_strict = sum(1 for v in vals if v["success_strict"])
    rps_all = [v["reprop_arrival_residual"] for v in vals if v["reprop_arrival_residual"] is not None]
    rps_ok = [v["reprop_arrival_residual"] for v in vals if v["success_loose"] and v["reprop_arrival_residual"] is not None]
    fuels_ok = [v["fuel_kg"] for v in vals if v["success_loose"] and v["fuel_kg"] is not None]
    rmss_all = [v["residual_rms"] for v in vals if v["residual_rms"] is not None]
    rts = [v["total_runtime_s"] for v in vals]
    return {
        "n_seeds": n,
        "n_loose": n_loose,
        "rate_loose": n_loose / n,
        "n_strict": n_strict,
        "rate_strict": n_strict / n,
        "median_fuel_loose_kg": float(np.median(fuels_ok)) if fuels_ok else None,
        "median_reprop_loose": float(np.median(rps_ok)) if rps_ok else None,
        "median_reprop_all": float(np.median(rps_all)) if rps_all else None,
        "median_rms_all": float(np.median(rmss_all)) if rmss_all else None,
        "median_total_runtime_s": float(np.median(rts)) if rts else None,
    }


def _save(data: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wall-s", type=float, default=440.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    if OUT_PATH.exists():
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        # ensure new configs exist
        for lbl, w_val, gtype, imode in CONFIGS:
            data["configs"].setdefault(lbl, {"w": w_val, "grid_type": gtype, "init_mode": imode, "seeds": {}, "summary": {}})
    else:
        data = {
            "meta": {
                "description": "Task 5: Fair deterministic spanning tanh-ELM ablation on Scenario A (A4(b) seeds 1000..1009)",
                "seeds": SEEDS,
                "formula": "w_q = w, b_q = w * (-1 + 2*(q-1)/(L-1)), q=1..L; beta ~ U(-0.5, 0.5) (plus scaled_u01 check at w=5.2)",
            },
            "configs": {
                lbl: {"w": w_val, "grid_type": gtype, "init_mode": imode, "seeds": {}, "summary": {}}
                for lbl, w_val, gtype, imode in CONFIGS
            },
        }
        _save(data)

    pending = []
    for lbl, w_val, gtype, imode in CONFIGS:
        done = data["configs"][lbl]["seeds"]
        for s in SEEDS:
            if str(s) not in done:
                pending.append((lbl, w_val, gtype, imode, s))

    total_target = len(CONFIGS) * len(SEEDS)
    print(f"[Task 5] Done: {total_target - len(pending)}/{total_target}, Pending: {len(pending)}", flush=True)

    for lbl, w_val, gtype, imode, s in pending:
        if time.perf_counter() - t_start > args.max_wall_s:
            print(f"[Task 5] Wall-time budget ({args.max_wall_s:.0f}s) reached; checkpoint saved.", flush=True)
            break
        rec = _run_one(lbl, w_val, gtype, imode, s)
        data["configs"][lbl]["seeds"][str(s)] = rec
        data["configs"][lbl]["summary"] = _summarize(data["configs"][lbl]["seeds"])
        rf = rec["refined"]
        print(
            f"  [{lbl:28s} s={s}] loose={int(rf['success_loose'])} strict={int(rf['success_strict'])} "
            f"fuel={rf['fuel_kg']:.2f} rp={rf['reprop_arrival_residual']:.2e} sw={rf['n_switches']} t={rf['total_runtime_s']:.1f}s"
            if rf["fuel_kg"] is not None and rf["reprop_arrival_residual"] is not None
            else f"  [{lbl:28s} s={s}] loose={int(rf['success_loose'])} strict={int(rf['success_strict'])} t={rf['total_runtime_s']:.1f}s",
            flush=True,
        )
        _save(data)

    for lbl, _, _, _ in CONFIGS:
        data["configs"][lbl]["summary"] = _summarize(data["configs"][lbl]["seeds"])
    _save(data)
    done_now = sum(len(data["configs"][lbl]["seeds"]) for lbl, _, _, _ in CONFIGS)
    print(f"[Task 5] Total completed: {done_now}/{total_target}", flush=True)


if __name__ == "__main__":
    main()
