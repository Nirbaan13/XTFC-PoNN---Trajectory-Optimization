"""Phase A4 runner: Ablations on Scenario A (D'Ambrosio & Furfaro 2024).

Runs and checkpoints to results/scenario_a_phase_a4.json:
  (a) 14-residual formulation (unreduced r, v, m, lr, lv, lm) vs Paper's 8-residual CE
  (b) Chebyshev NN (chebyshev_shifted & literal chebyshev) vs random-weight tanh ELM
  (c) Continuation schedule & step count: N_steps in {5, 10, 20, 40}, rho_end in {1e-3, 1e-6, 1e-10}
  (d) Discretization grid (uniform_140 vs cgl_140) & optimizer (lm_svd vs scipy lm vs scipy trf)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trajectory_optimization.ponn_paper import (
    PaperPoNNModel,
    PaperPoNNSolver,
    _lm_qr_svd,
    chebyshev_basis,
    make_collocation_grid,
)
from trajectory_optimization.scenario_a import (
    DEFAULT_RHO_SCHEDULE,
    ScenarioAProblem,
    detect_switches,
    energy_balance_error_scenario_a,
    propagate_control_scenario_a,
    throttle_paper,
    thrust_sequence_label,
)

jax.config.update("jax_enable_x64", True)

OUT_PATH = Path("results/scenario_a_phase_a4.json")
SCRATCH_BACKUP = Path(
    "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/"
    "2f3a08db-5952-4e2e-99ea-99a0ed91d155/scratch/scenario_a_phase_a4.json"
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


def _load() -> dict:
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            return json.load(f)
    return {
        "ablation_a_14res_vs_8res": {},
        "ablation_b_basis": {},
        "ablation_c_rho_schedule": {},
        "ablation_d_grid_and_solver": {},
    }


# ---------------------------------------------------------------------------
# 14-residual unreduced formulation on Scenario A (for Ablation A4(a))
# ---------------------------------------------------------------------------
def _rhs14_point(y, mu, tmax, c, rho):
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    lr = y[7:10]
    lv = y[10:13]
    lm = y[13]
    r2 = jnp.dot(r, r)
    rn = jnp.sqrt(r2)
    ir3 = 1.0 / (r2 * rn)
    lvn = jnp.sqrt(jnp.dot(lv, lv) + 1e-30)
    alpha = -lv / lvn
    S = c * lvn / m + lm - 1.0
    delta = 0.5 * (1.0 + jnp.tanh(S / rho))
    rdot = v
    vdot = -mu * r * ir3 + (tmax * delta / m) * alpha
    mdot = -tmax * delta / c
    rdl = jnp.dot(r, lv) / r2
    lrdot = mu * ir3 * (lv - 3.0 * r * rdl)
    lvdot = -lr
    lmdot = -(tmax * delta / (m * m)) * lvn
    return jnp.concatenate([rdot, vdot, jnp.array([mdot]), lrdot, lvdot, jnp.array([lmdot])])


_RHS14_BATCH = jax.jit(jax.vmap(_rhs14_point, in_axes=(0, None, None, None, None)))
_JRHS14_BATCH = jax.jit(jax.vmap(jax.jacfwd(_rhs14_point), in_axes=(0, None, None, None, None)))


def solve_14res_scenario_a(
    prob: ScenarioAProblem,
    seed: int,
    n_neurons: int = 50,
    n_points: int = 120,
    rho_schedule: tuple[float, ...] = DEFAULT_RHO_SCHEDULE,
) -> dict:
    """Solve Scenario A with unreduced 14-state, 14-residual X-TFC CEs (r(0),r(tf),v(0),v(tf),m(0),lm(tf))."""
    t0 = time.perf_counter()
    L = n_neurons
    z = make_collocation_grid(n_points, "cgl")
    b2 = 2.0 / prob.tf
    # Use shifted Chebyshev T_2..T_{L+1} for 2-point CEs (r, v), T_1..T_L for 1-point (m, lm), T_0..T_{L-1} for lr, lv
    H_all, dH_all, _ = chebyshev_basis(z, L + 2)
    h_ends, _, _ = chebyshev_basis(np.array([-1.0, 1.0]), L + 2)

    s_lin = 0.5 * (z + 1.0)
    om1 = (1.0 - s_lin)[:, None]
    om2 = s_lin[:, None]
    dom1 = np.full((len(z), 1), -0.5)
    dom2 = np.full((len(z), 1), 0.5)

    # 2-point CEs for r and v
    H2, dH2 = H_all[:, 2:2 + L], dH_all[:, 2:2 + L]
    h20, h2f = h_ends[0, 2:2 + L], h_ends[1, 2:2 + L]
    Phi_2pt = H2 - om1 * h20[None, :] - om2 * h2f[None, :]
    dPhi_2pt = b2 * (dH2 - dom1 * h20[None, :] - dom2 * h2f[None, :])

    # 1-point CEs for m(0)=m0 and lm(tf)=0
    H1, dH1 = H_all[:, 1:1 + L], dH_all[:, 1:1 + L]
    Phi_m = H1 - h_ends[0, 1:1 + L][None, :]
    dPhi_m = b2 * dH1
    Phi_lm = H1 - h_ends[1, 1:1 + L][None, :]
    dPhi_lm = b2 * dH1

    # Unconstrained for lr, lv
    H0, dH0 = H_all[:, :L], dH_all[:, :L]
    Phi_free = H0
    dPhi_free = b2 * dH0

    Phi_val = np.stack(
        [Phi_2pt] * 6 + [Phi_m] + [Phi_free] * 6 + [Phi_lm], axis=0
    )  # (14, n, L)
    Phi_lhs = np.stack(
        [dPhi_2pt] * 6 + [dPhi_m] + [dPhi_free] * 6 + [dPhi_lm], axis=0
    )  # (14, n, L)

    P_val = np.zeros((len(z), 14), dtype=float)
    P_lhs = np.zeros((len(z), 14), dtype=float)
    P_val[:, 0:3] = om1 * prob.r0[None, :] + om2 * prob.rf[None, :]
    P_lhs[:, 0:3] = b2 * (dom1 * prob.r0[None, :] + dom2 * prob.rf[None, :])
    P_val[:, 3:6] = om1 * prob.v0[None, :] + om2 * prob.vf[None, :]
    P_lhs[:, 3:6] = b2 * (dom1 * prob.v0[None, :] + dom2 * prob.vf[None, :])
    P_val[:, 6] = prob.m0

    rng = np.random.default_rng(seed)
    beta = np.zeros((14, L), dtype=float)
    beta[0:6] = 1e-3 * rng.uniform(0.0, 1.0, (6, L))
    beta[7:13] = 1e-2 * rng.uniform(0.0, 1.0, (6, L))
    mask = z < 1.0 - 1e-12
    beta[13] = np.linalg.lstsq(Phi_lm[mask], np.ones(int(np.sum(mask))), rcond=1e-4)[0]
    xi = beta.reshape(-1)

    def res_only(x, rho):
        b = x.reshape(14, L)
        y = np.einsum("inl,il->ni", Phi_val, b) + P_val
        lhs = np.einsum("inl,il->ni", Phi_lhs, b) + P_lhs
        F = np.asarray(_RHS14_BATCH(jnp.asarray(y), prob.mu, prob.t_max, prob.c, float(rho)))
        r = (lhs - F).T.reshape(-1)
        return r if np.all(np.isfinite(r)) else np.full_like(r, 1e6)

    def res_jac(x, rho):
        b = x.reshape(14, L)
        y = np.einsum("inl,il->ni", Phi_val, b) + P_val
        lhs = np.einsum("inl,il->ni", Phi_lhs, b) + P_lhs
        F = np.asarray(_RHS14_BATCH(jnp.asarray(y), prob.mu, prob.t_max, prob.c, float(rho)))
        Jf = np.asarray(_JRHS14_BATCH(jnp.asarray(y), prob.mu, prob.t_max, prob.c, float(rho)))
        r = (lhs - F).T.reshape(-1)
        J = -np.einsum("nij,jnl->injl", Jf, Phi_val)
        for i in range(14):
            J[i, :, i, :] += Phi_lhs[i]
        J2 = J.reshape(14 * len(z), 14 * L)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(J2))):
            return np.full_like(r, 1e6), np.zeros_like(J2)
        return r, J2

    tot_f = tot_j = 0
    rho_reached = float(rho_schedule[0])
    for k, rho in enumerate(rho_schedule):
        xi_n, r_v, nf, nj, _ = _lm_qr_svd(
            lambda x, r=rho: res_jac(x, r),
            lambda x, r=rho: res_only(x, r),
            xi, scaled=(k > 0), max_iter=80 if k == 0 else 20, lam0=1e-2 if k == 0 else 1e-4,
        )
        tot_f += nf
        tot_j += nj
        y_chk = np.einsum("inl,il->ni", Phi_val, xi_n.reshape(14, L)) + P_val
        if np.all(np.isfinite(xi_n)) and np.all(y_chk[:, 6] > 0.02):
            xi = xi_n
            rho_reached = float(rho)

    dt = time.perf_counter() - t0
    r_fin = res_only(xi, rho_reached)
    rms = float(np.sqrt(np.mean(r_fin ** 2)))

    # Evaluate on dense grid
    z_d = np.linspace(-1.0, 1.0, 2001)
    H_d, _, _ = chebyshev_basis(z_d, L + 2)
    sd = 0.5 * (z_d + 1.0)
    om1d, om2d = (1.0 - sd)[:, None], sd[:, None]
    Phi2d = H_d[:, 2:2 + L] - om1d * h20[None, :] - om2d * h2f[None, :]
    Phimd = H_d[:, 1:1 + L] - h_ends[0, 1:1 + L][None, :]
    Philmd = H_d[:, 1:1 + L] - h_ends[1, 1:1 + L][None, :]
    Phifd = H_d[:, :L]
    b = xi.reshape(14, L)
    r_d = Phi2d @ b[0:3].T + om1d * prob.r0[None, :] + om2d * prob.rf[None, :]
    v_d = Phi2d @ b[3:6].T + om1d * prob.v0[None, :] + om2d * prob.vf[None, :]
    m_d = Phimd @ b[6] + prob.m0
    lv_d = Phifd @ b[10:13].T
    lm_d = Philmd @ b[13]
    lvn_d = np.linalg.norm(lv_d, axis=1)
    alpha_d = -lv_d / np.maximum(lvn_d, 1e-12)[:, None]
    S_d = prob.c * lvn_d / m_d + lm_d - 1.0
    delta_d = np.array([throttle_paper(float(s), rho_reached) for s in S_d])
    t_tu = 0.5 * (z_d + 1.0) * prob.tf
    t_days = t_tu * prob.t_ref_days
    switches = detect_switches(t_days, S_d)
    seq = thrust_sequence_label(S_d, switches)
    sol_rp = propagate_control_scenario_a(prob, t_tu, delta_d, alpha_d)
    if sol_rp.status == 0 and np.all(np.isfinite(sol_rp.y[:, -1])):
        xf = sol_rp.y[:, -1]
        reprop = float(np.linalg.norm(np.concatenate([xf[0:3] - prob.rf, xf[3:6] - prob.vf])))
    else:
        reprop = None
    e_err, _ = energy_balance_error_scenario_a(prob, t_tu, r_d, v_d, delta_d, alpha_d, m_d)
    fuel_kg = float((prob.m0 - m_d[-1]) * prob.m_ref_kg)
    ok = bool(np.all(m_d > 0.05) and rms < 1.5e-2 and reprop is not None and reprop < 5e-2)
    return {
        "seed": seed,
        "success": ok,
        "fuel_kg": fuel_kg,
        "residual_rms": rms,
        "reprop_arrival_residual": reprop,
        "energy_error": float(e_err) if np.isfinite(e_err) else None,
        "n_switches": len(switches),
        "switch_times_days": [float(s["t_days"]) for s in switches],
        "thrust_sequence": seq,
        "total_nfev": tot_f,
        "total_njev": tot_j,
        "runtime_s": dt,
        "n_unknowns": 14 * L,
        "n_residuals": 14 * n_points,
    }


def _pack(res, seed: int) -> dict:
    return {
        "seed": seed,
        "success": bool(res.success),
        "success_reprop_1e1": bool(
            res.residual_rms < 1.5e-2 and np.isfinite(res.reprop_arrival_residual) and res.reprop_arrival_residual < 1e-1
        ),
        "fuel_kg": float(res.fuel_kg),
        "residual_rms": float(res.residual_rms),
        "reprop_arrival_residual": float(res.reprop_arrival_residual) if np.isfinite(res.reprop_arrival_residual) else None,
        "energy_error": float(res.energy_error) if np.isfinite(res.energy_error) else None,
        "n_switches": len(res.switches),
        "switch_times_days": [float(s["t_days"]) for s in res.switches],
        "thrust_sequence": res.thrust_sequence,
        "total_nfev": int(res.total_nfev),
        "total_njev": int(res.total_njev),
        "runtime_s": float(res.runtime_s),
        "n_unknowns": 8 * int(res.n_neurons),
        "n_residuals": 8 * int(res.n_points),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--max-wall-s", type=float, default=500.0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    def _over_budget() -> bool:
        return (time.perf_counter() - t_start) > args.max_wall_s

    prob = ScenarioAProblem()
    data = _load()
    seeds = [1000 + i for i in range(args.n_seeds)]

    # (a) 14-residual vs 8-residual CE
    print("=== Phase A4(a): 14-residual vs 8-residual CE ===")
    ab_a = data["ablation_a_14res_vs_8res"]
    for label in ["8res_coarse_L70", "8res_refined_L80", "14res_L50"]:
        ab_a.setdefault(label, {})
    solver8 = PaperPoNNSolver(n_neurons=70, n_points=140, solver_algo="lm_svd", basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01")
    for s in seeds:
        if _over_budget():
            print("[checkpoint] wall limit in A4(a)"); _save(data); return
        sk = str(s)
        if sk not in ab_a["8res_coarse_L70"] or sk not in ab_a["8res_refined_L80"]:
            rc, rr = solver8.solve(prob, seed=s, refine=True, refine_neurons=80, points_per_switch=100, refine_grid_mode="uniform_dense")
            ab_a["8res_coarse_L70"][sk] = _pack(rc, s)
            ab_a["8res_refined_L80"][sk] = _pack(rr, s)
            print(f"  [8res s={s}] coarse={rc.fuel_kg:.2f} (rp={rc.reprop_arrival_residual:.2e}, {rc.runtime_s:.1f}s) | ref={rr.fuel_kg:.2f} (rp={rr.reprop_arrival_residual:.2e}, {rr.runtime_s:.1f}s)")
            _save(data)
        if _over_budget():
            print("[checkpoint] wall limit in A4(a)"); _save(data); return
        if sk not in ab_a["14res_L50"]:
            r14 = solve_14res_scenario_a(prob, seed=s, n_neurons=50, n_points=120)
            ab_a["14res_L50"][sk] = r14
            print(f"  [14res s={s}] ok={r14['success']} fuel={r14['fuel_kg']:.2f} rms={r14['residual_rms']:.2e} rp={r14['reprop_arrival_residual']} {r14['runtime_s']:.1f}s")
            _save(data)

    # (b) Basis ablation: chebyshev_shifted vs literal chebyshev vs random-weight tanh ELM
    print("=== Phase A4(b): Chebyshev NN vs Random-Weight tanh ELM ===")
    ab_b = data["ablation_b_basis"]
    basis_configs = [
        ("chebyshev_shifted_cgl", "chebyshev_shifted", "cgl"),
        ("chebyshev_literal_cgl", "chebyshev", "cgl"),
        ("tanh_elm_cgl", "tanh_elm", "cgl"),
        ("tanh_elm_uniform", "tanh_elm", "uniform"),
    ]
    for label, btype, gtype in basis_configs:
        ab_b.setdefault(label, {})
        sol_b = PaperPoNNSolver(n_neurons=70, n_points=140, solver_algo="lm_svd", basis_type=btype, grid_type=gtype, init_mode="scaled_u01")
        for s in seeds:
            if _over_budget():
                print(f"[checkpoint] wall limit in A4(b) {label}"); _save(data); return
            sk = str(s)
            if sk in ab_b[label]:
                continue
            rc, rr = sol_b.solve(prob, seed=s, refine=True, refine_neurons=80, points_per_switch=100, refine_grid_mode="uniform_dense", skip_refine_if_coarse_diverged=True)
            ab_b[label][sk] = {"coarse": _pack(rc, s), "refined": _pack(rr if rr is not None else rc, s)}
            rr_eff = rr if rr is not None else rc
            print(f"  [{label:22s} s={s}] ref_ok={rr_eff.success} fuel={rr_eff.fuel_kg:.2f} sw={len(rr_eff.switches)} rp={rr_eff.reprop_arrival_residual:.2e} {rr_eff.runtime_s:.1f}s")
            _save(data)

    # (c) Continuation schedule & step count ablation
    print("=== Phase A4(c): Continuation rho schedule & step count ===")
    ab_c = data["ablation_c_rho_schedule"]
    sched_configs = [
        ("n5_to_1e-10", tuple(float(x) for x in np.logspace(0.0, -10.0, 5))),
        ("n10_to_1e-10", tuple(float(x) for x in np.logspace(0.0, -10.0, 10))),
        ("n20_to_1e-10", DEFAULT_RHO_SCHEDULE),
        ("n40_to_1e-10", tuple(float(x) for x in np.logspace(0.0, -10.0, 40))),
        ("n20_to_1e-3", tuple(float(x) for x in np.logspace(0.0, -3.0, 20))),
        ("n20_to_1e-6", tuple(float(x) for x in np.logspace(0.0, -6.0, 20))),
    ]
    for label, sched in sched_configs:
        ab_c.setdefault(label, {})
        sol_c = PaperPoNNSolver(
            n_neurons=70, n_points=140, rho_schedule=sched,
            solver_algo="lm_svd", basis_type="chebyshev_shifted", grid_type="cgl", init_mode="scaled_u01",
        )
        for s in seeds[:5]:
            if _over_budget():
                print(f"[checkpoint] wall limit in A4(c) {label}"); _save(data); return
            sk = str(s)
            if sk in ab_c[label]:
                continue
            rc, rr = sol_c.solve(prob, seed=s, refine=True, refine_neurons=80, points_per_switch=100, refine_grid_mode="uniform_dense", skip_refine_if_coarse_diverged=True)
            rr_eff = rr if rr is not None else rc
            ab_c[label][sk] = {"coarse": _pack(rc, s), "refined": _pack(rr_eff, s)}
            print(f"  [{label:14s} s={s}] ref_ok={rr_eff.success} fuel={rr_eff.fuel_kg:.2f} sw={len(rr_eff.switches)} rms={rr_eff.residual_rms:.2e} rp={rr_eff.reprop_arrival_residual:.2e} {rr_eff.runtime_s:.1f}s")
            _save(data)

    # (d) Grid & Optimizer ablation (uniform_140 vs cgl_140; lm_svd vs scipy lm vs scipy trf)
    print("=== Phase A4(d): Grid & Optimizer Ablation ===")
    ab_d = data["ablation_d_grid_and_solver"]
    grid_opt_configs = [
        ("uniform_140_lm_svd", "uniform", "lm_svd", "chebyshev_shifted", "scaled_u01", False),
        ("cgl_140_lm_svd", "cgl", "lm_svd", "chebyshev_shifted", "scaled_u01", False),
        ("cgl_140_scipy_lm", "cgl", "lm", "chebyshev_shifted", "scaled_u01", False),
        ("cgl_140_scipy_trf", "cgl", "trf", "chebyshev_shifted", "scaled_u01", False),
        ("uniform_540_lm_svd", "uniform", "lm_svd", "chebyshev_shifted", "scaled_u01", False),
    ]
    for label, gtype, algo, btype, imode, do_ref in grid_opt_configs:
        ab_d.setdefault(label, {})
        n_pts = 540 if "540" in label else 140
        n_neu = 80 if "540" in label else 70
        sol_d = PaperPoNNSolver(
            n_neurons=n_neu, n_points=n_pts, solver_algo=algo,
            basis_type=btype, grid_type=gtype, init_mode=imode,
            max_iter_first=80 if algo in ("lm", "trf") else 100,
            max_iter_cont=25,
        )
        for s in seeds[:5]:
            if _over_budget():
                print(f"[checkpoint] wall limit in A4(d) {label}"); _save(data); return
            sk = str(s)
            if sk in ab_d[label]:
                continue
            rc, _ = sol_d.solve(prob, seed=s, refine=do_ref)
            ab_d[label][sk] = _pack(rc, s)
            print(f"  [{label:20s} s={s}] ok={rc.success} fuel={rc.fuel_kg:.2f} sw={len(rc.switches)} rms={rc.residual_rms:.2e} rp={rc.reprop_arrival_residual:.2e} {rc.runtime_s:.1f}s")
            _save(data)

    print("=== PHASE A4 COMPLETE ===")


if __name__ == "__main__":
    main()
