"""Phase D runner: Secondary Scenario B (1218 kg, 0.5 N, Isp 3100 s) with FROZEN Phase A2 PoNN.

Evaluates the validated Phase A2 PaperPoNNSolver (NO retuning!) on the Scenario B
Earth-Mars transfer (tf = 300 d, fuel-minimal theta_f = 225 deg, m0 = 1218 kg,
Tmax = 0.5 N, Isp = 3100 s, SRP A/m = 0.02 m^2/kg) across 20 seeds, plus a hybrid
PoNN -> Indirect Shooting warm-start comparison, preserving all existing Scenario B
Phase 1/Phase 2 results intact.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from trajectory_optimization.constants import AU_M, C_EXHAUST, DAYS_PER_TU, G0, ISP, MU_SUN, SCALE_V
from trajectory_optimization.ponn_paper import PaperPoNNSolver
from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.scenario_a import (
    CR_SRP_DEFAULT,
    P0_SRP_N_M2,
    ScenarioAProblem,
    ScenarioAShootingSolver,
)
from trajectory_optimization.scenario_a_collocation import ScenarioACollocationSolver

OUT_PATH = Path("results/scenario_b_phase_d.json")
SCRATCH_BACKUP = Path(
    "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/"
    "2f3a08db-5952-4e2e-99ea-99a0ed91d155/scratch/scenario_b_phase_d.json"
)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_scenario_b_as_scenario_a(tf_days: float = 300.0, theta_f_deg: float = 225.0) -> ScenarioAProblem:
    """Express Scenario B (HelioProblem: 1218 kg, 0.5 N, Isp 3100 s, SRP) in exact ScenarioAProblem form."""
    hp = HelioProblem(tf_days=tf_days, theta_f=math.radians(theta_f_deg))
    r_ref_km = AU_M / 1e3
    v_ref_km_s = SCALE_V / 1e3
    a_ref_m_s2 = AU_M / ((DAYS_PER_TU * 86400.0) ** 2)
    # Match HelioProblem's exact canonical k_srp
    a_over_m = float(hp.k_srp * a_ref_m_s2 / (P0_SRP_N_M2 * CR_SRP_DEFAULT))
    return ScenarioAProblem(
        r0_km=hp.r0 * r_ref_km,
        v0_km_s=hp.v0 * v_ref_km_s,
        rf_km=hp.rf * r_ref_km,
        vf_km_s=hp.vf * v_ref_km_s,
        m0_kg=hp.m0_kg,
        tf_days=tf_days,
        t_max_N=hp.t_max_newton,
        isp_s=ISP,
        g0=G0,
        mu_km3_s2=MU_SUN / 1e9,
        r_ref_km=r_ref_km,
        t_ref_days=DAYS_PER_TU,
        m_ref_kg=hp.m0_kg,
        srp_area_to_mass=a_over_m,
        srp_cr=CR_SRP_DEFAULT,
        srp_p0=P0_SRP_N_M2,
        srp_mode="radial" if hp.k_srp > 0.0 else "none",
        label=f"scenario_b_tf{tf_days:.0f}_th{theta_f_deg:.0f}",
    )


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


def main():
    prob_b = build_scenario_b_as_scenario_a(tf_days=300.0, theta_f_deg=225.0)
    data = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {
        "meta": {
            "tf_days": 300.0,
            "theta_f_deg": 225.0,
            "m0_kg": prob_b.m0_kg,
            "t_max_N": prob_b.t_max_N,
            "isp_s": prob_b.isp_s,
            "srp_area_to_mass": prob_b.srp_area_to_mass,
            "frozen_from": "Phase A2 PaperPoNNSolver (L=70->80, n=140 CGL->540 uniform, lm_svd, 20-step rho)",
        },
        "ponn_frozen": {},
        "shooting_random": {},
        "shooting_hybrid_from_ponn": {},
        "collocation_ref": None,
    }

    # Reference collocation on Scenario B
    if data["collocation_ref"] is None:
        col = ScenarioACollocationSolver(n_nodes=180, max_iter=500, tol=1e-7).solve(prob_b, seed=None)
        data["collocation_ref"] = {
            "success": bool(col.success),
            "fuel_kg": float(col.fuel_kg),
            "reprop_arrival_residual": float(col.reprop_arrival_residual),
            "energy_error": float(col.energy_error),
            "n_switches": len(col.switches),
            "switch_times_days": [float(s["t_days"]) for s in col.switches],
            "thrust_sequence": col.thrust_sequence,
            "runtime_s": float(col.runtime_s),
        }
        print(f"[Scenario B Colloc ref] ok={col.success} fuel={col.fuel_kg:.3f} kg sw={[round(s['t_days'],2) for s in col.switches]} rp={col.reprop_arrival_residual:.2e}")
        _save(data)

    # Frozen Phase A2 PoNN (ZERO retuning!) on 20 seeds + hybrid PoNN -> Shooting
    ponn = PaperPoNNSolver(
        n_neurons=70, n_points=140, solver_algo="lm_svd",
        basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        max_iter_first=100, max_iter_cont=25,
    )
    shooter = ScenarioAShootingSolver(stop_at_rho=1e-5)

    for i in range(20):
        seed = 2000 + i
        sk = str(seed)
        if sk not in data["ponn_frozen"]:
            rc, rr = ponn.solve(
                prob_b, seed=seed, refine=True, refine_neurons=80,
                points_per_switch=100, refine_grid_mode="uniform_dense",
                skip_refine_if_coarse_diverged=True,
            )
            rr_eff = rr if rr is not None else rc
            # Extract initial costates lam0 = [lr(0), lv(0), lm(0)] from PoNN for hybrid test
            lam0_ponn = rr_eff.Y14[0, 7:14].copy()
            data["ponn_frozen"][sk] = {
                "success": bool(rr_eff.success),
                "coarse_fuel_kg": float(rc.fuel_kg),
                "fuel_kg": float(rr_eff.fuel_kg),
                "residual_rms": float(rr_eff.residual_rms),
                "reprop_arrival_residual": float(rr_eff.reprop_arrival_residual) if np.isfinite(rr_eff.reprop_arrival_residual) else None,
                "energy_error": float(rr_eff.energy_error) if np.isfinite(rr_eff.energy_error) else None,
                "n_switches": len(rr_eff.switches),
                "switch_times_days": [float(s["t_days"]) for s in rr_eff.switches],
                "thrust_sequence": rr_eff.thrust_sequence,
                "lam0_ponn": [float(x) for x in lam0_ponn],
                "runtime_s": float(rr_eff.runtime_s),
            }
            print(f"  [ScenB PoNN   s={seed}] ok={rr_eff.success} fuel={rr_eff.fuel_kg:.2f} rp={rr_eff.reprop_arrival_residual:.2e} {rr_eff.runtime_s:.1f}s")
            _save(data)

        if sk not in data["shooting_random"]:
            r_sh = shooter.solve(prob_b, seed=seed)
            data["shooting_random"][sk] = {
                "success": bool(r_sh.success),
                "fuel_kg": float(r_sh.fuel_kg) if np.isfinite(r_sh.fuel_kg) else None,
                "reprop_arrival_residual": float(r_sh.reprop_arrival_residual) if np.isfinite(r_sh.reprop_arrival_residual) else None,
                "n_switches": len(r_sh.switches),
                "nfev": int(r_sh.n_propagations),
                "runtime_s": float(r_sh.runtime_s),
            }
            _save(data)

        if sk not in data["shooting_hybrid_from_ponn"]:
            lam0_init = np.array(data["ponn_frozen"][sk]["lam0_ponn"], dtype=float)
            r_hy = shooter.solve(prob_b, seed=seed, lam0_guess=lam0_init)
            data["shooting_hybrid_from_ponn"][sk] = {
                "success": bool(r_hy.success),
                "fuel_kg": float(r_hy.fuel_kg) if np.isfinite(r_hy.fuel_kg) else None,
                "reprop_arrival_residual": float(r_hy.reprop_arrival_residual) if np.isfinite(r_hy.reprop_arrival_residual) else None,
                "n_switches": len(r_hy.switches),
                "nfev": int(r_hy.n_propagations),
                "runtime_s": float(r_hy.runtime_s),
            }
            print(f"  [ScenB Shoot  s={seed}] rand_ok={data['shooting_random'][sk]['success']} hyb_ok={r_hy.success} (hyb_fuel={r_hy.fuel_kg:.2f}, nfev={r_hy.n_propagations})")
            _save(data)

    # Summary
    def summ(d):
        vals = list(d.values())
        n = len(vals)
        conv = [v for v in vals if v["success"]]
        fuels = [v["fuel_kg"] for v in conv if v.get("fuel_kg") is not None]
        return {
            "n_conv": len(conv),
            "n_total": n,
            "rate": len(conv) / n if n else 0.0,
            "ci95": list(wilson_ci(len(conv), n)),
            "fuel_mean_kg": float(np.mean(fuels)) if fuels else None,
            "fuel_std_kg": float(np.std(fuels, ddof=1)) if len(fuels) > 1 else (0.0 if fuels else None),
            "runtime_mean_s": float(np.mean([v["runtime_s"] for v in vals])) if vals else None,
        }

    data["summary"] = {
        "ponn_frozen": summ(data["ponn_frozen"]),
        "shooting_random": summ(data["shooting_random"]),
        "shooting_hybrid_from_ponn": summ(data["shooting_hybrid_from_ponn"]),
    }
    _save(data)
    print("=== PHASE D SUMMARY ===")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
