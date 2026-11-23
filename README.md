# X-TFC Pontryagin Neural Networks — Earth–Mars Trajectory Optimization Benchmark

Exact reproduction, ablation, and perturbation benchmark of **D'Ambrosio & Furfaro (2024)**, *"Learning Fuel-Optimal Trajectories for Space Applications via Pontryagin Neural Networks"*, *Aerospace* 11(3), 228 (`doi:10.3390/aerospace11030228`), Sections 3.1, 3.3, and 4.1, alongside a secondary low-thrust Earth–Mars transfer study (Scenario B).

All tables, figures, and metrics in this repository are generated directly from stored experiment JSONs under `results/` with zero hard-coded outputs.

---

## Key Results Summary — Scenario A (D'Ambrosio & Furfaro 2024, Table 1)

Mission: 3-D heliocentric Earth–Mars rendezvous, $m_0 = 1000	ext{ kg}$, $t_f = 348.795	ext{ d}$, $T_{\max} = 0.5	ext{ N}$, $I_{	ext{sp}} = 2000	ext{ s}$, Sun two-body gravity.

### Single-Seed Reproduction vs. Independent Paper Checks

| Configuration | Fuel (kg) | Delta vs Shoot (kg) | Switches (days) | Sequence | ODE RMS | DOP853 Reprop | Energy Err | Runtime (s) |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Paper check (Shooting, Ref. [31]) | 396.0600 | — | [47, 66, 148, 292] (Fig. 5) | ON-OFF-ON-OFF-ON | — | — | — | — |
| Paper check (PoNN Refined, Sec. 4.1) | 396.8500 | +0.7900 | [47, 66, 148, 292] (Fig. 5) | ON-OFF-ON-OFF-ON | — | — | — | — |
| Phase A1 Indirect Shooting (this repo) | 396.0598 | +0.0000 | [46.58, 68.02, 142.72, 290.25] | ON-OFF-ON-OFF-ON | 3.73e-11 | 3.46e-04 | 9.90e-05 | 5.99 |
| 180-Node LGL Collocation (smooth init) | 396.1793 | +0.1195 | [46.79, 67.89, 142.64, 290.24] | ON-OFF-ON-OFF-ON | 0.00e+00 | 5.12e-04 | 8.22e-04 | 30.88 |
| PoNN Coarse (L=70, n=140 CGL) | 400.6966 | +4.6368 | [46.30, 66.54, 144.98, 291.29] | ON-OFF-ON-OFF-ON | 2.06e-03 | 8.64e-02 | 1.61e-02 | 3.58 |
| PoNN Refined (L=80, n=540 uniform) | 396.5215 | +0.4617 | [49.83, 68.59, 141.72, 290.55] | ON-OFF-ON-OFF-ON | 4.80e-03 | 3.63e-02 | 8.24e-03 | 13.87 |
| PoNN Refined (L=80, n=540 CGL) | 397.5304 | +1.4705 | [50.54, 68.87, 141.56, 290.13] | ON-OFF-ON-OFF-ON | 4.04e-03 | 3.87e-02 | 7.53e-03 | 11.35 |
| PoNN Refined (L=80, switch bracket) | 400.6341 | +4.5743 | [51.78, 72.85, 144.98, 291.29] | ON-OFF-ON-OFF-ON | 2.76e-03 | 1.07e-01 | 1.92e-02 | 26.52 |
| PoNN Literal (L=70, n=140 uni, T0..T69, U(0,1)) | 419.0212 | +22.9614 | 7 switches | ON-OFF-ON-OFF-ON-OFF-ON-OFF | 7.42e-05 | 1.36e+00 | 7.71e-01 | 18.04 |

### 100-Seed Three-Method Benchmark (Phase B)

| Method (100 random seeds) | Conv (<5e-2) [95% CI] | Conv (<1e-2) [95% CI] | Fuel Med [Min, Max] (kg) | Reprop Med | Energy Err Med | Runtime Med (s) | nfev / njev Med |
| --- | :---: | :---: | :---: | ---: | ---: | ---: | :---: |
| PoNN (frozen, scaled U(0,1)) | 92/100 (92.0%) [85.0%, 95.9%] | 0/100 (0.0%) [0.0%, 3.7%] | 396.52 [396.52, 396.52] | 3.63e-02 | 8.24e-03 | 11.78 | 243 / 227 |
| PoNN (frozen, exact paper U(0,1)) | 49/100 (49.0%) [39.4%, 58.7%] | 0/100 (0.0%) [0.0%, 3.7%] | 396.52 [396.52, 396.52] | 3.63e-02 | 8.24e-03 | 16.67 | 692 / 365 |
| Indirect Shooting (7-D costates) | 68/100 (68.0%) [58.3%, 76.3%] | 68/100 (68.0%) [58.3%, 76.3%] | 396.06 [396.06, 396.06] | 3.46e-04 | 9.90e-05 | 6.70 | 456 / 10 |
| 180-Node LGL Collocation (IPOPT) | 55/100 (55.0%) [45.2%, 64.4%] | 18/100 (18.0%) [11.7%, 26.7%] | 397.10 [396.09, 410.00] | 2.48e-02 | 9.04e-04 | 32.14 | 262 / 126 |

### SRP Perturbation Sweep Summary (Phase C)

| Case | SRP Mode | $A/m_0$ (m²/kg) | $k_{\text{SRP}}$ (nd) | Shooting Fuel (kg) [conv] | PoNN Ref Fuel (kg) [conv] | LGL-180 Fuel (kg) [s=0 / conv] | Shoot Switches (days) |
| --- | --- | ---: | ---: | :---: | :---: | :---: | --- |
| `none_0.00` | none | 0.00 | 0.00e+00 | 396.060 (+0.000) [4/5] | 396.522 [4/5] | 396.179 (ok, 3/5) | [46.58, 68.02, 142.72, 290.25] |
| `radial_0.02` | radial | 0.02 | 1.85e-05 | 396.089 (+0.029) [4/5] | 396.530 [4/5] | 396.130 (ok, 3/5) | [46.59, 68.03, 142.72, 290.25] |
| `radial_0.10` | radial | 0.10 | 9.23e-05 | 396.205 (+0.145) [4/5] | 396.565 [4/5] | 396.323 (ok, 1/3) | [46.64, 68.04, 142.71, 290.23] |
| `radial_0.25` | radial | 0.25 | 2.31e-04 | 396.422 (+0.362) [3/5] | 396.630 [4/5] | 396.540 (ok, 2/3) | [46.73, 68.06, 142.70, 290.19] |
| `radial_0.50` | radial | 0.50 | 4.61e-04 | 396.785 (+0.725) [4/5] | 396.739 [4/5] | 396.900 (ok, 2/3) | [46.88, 68.10, 142.69, 290.13] |
| `radial_1.00` | radial | 1.00 | 9.23e-04 | 397.511 (+1.451) [3/5] | 398.146 [4/5] | 397.626 (ok, 2/3) | [47.19, 68.17, 142.67, 290.00] |
| `attitude_0.10` | attitude_flat_plate | 0.10 | 9.23e-05 | 396.177 (+0.117) [4/5] | 396.557 [4/5] | 396.200 (max_it, 0/1) | [46.63, 68.03, 142.71, 290.24] |
| `attitude_0.50` | attitude_flat_plate | 0.50 | 4.61e-04 | 396.647 (+0.587) [3/5] | 396.699 [4/5] | 396.662 (max_it, 0/1) | [46.82, 68.05, 142.69, 290.17] |
| `attitude_1.00` | attitude_flat_plate | 1.00 | 9.23e-04 | 397.236 (+1.176) [4/5] | 398.066 [4/5] | 397.077 (max_it, 0/1) | [47.06, 68.07, 142.66, 290.09] |

### Preregistered Benchmark & Hybrid Evaluation — Fresh Seeds 2000..2099

Locked in `PREREG.md` (commit `3c87713`) with dual convergence thresholds (loose $< 5	imes 10^-2$, strict $< 1	imes 10^-2$) and exact two-sided McNemar test ($p < 0.05$, $\Delta \ge +15.0	ext{ pp}$):

#### Scenario A 4-Method Benchmark (Fresh Seeds 2000..2099)
| Method (100 fresh seeds 2000..2099) | Conv (<5e-2, loose) [95% CI] | Conv (<1e-2, strict) [95% CI] | Fuel Med [Min, Max] (kg) | Reprop Med | Energy Err Med | Runtime Med (s) | nfev / njev Med |
| --- | :---: | :---: | :---: | ---: | ---: | ---: | :---: |
| PoNN (frozen, scaled U(0,1)) | 92/100 (92.0%) [85.0%, 95.9%] | 0/100 (0.0%) [0.0%, 3.7%] | 396.52 [396.52, 396.52] | 3.63e-02 | 8.24e-03 | 19.89 | 241 / 227 |
| PoNN (frozen, exact paper U(0,1)) | 39/100 (39.0%) [30.0%, 48.8%] | 0/100 (0.0%) [0.0%, 3.7%] | 396.52 [396.52, 396.52] | 3.63e-02 | 8.24e-03 | 25.84 | 694 / 365 |
| Indirect Shooting (7-D costates) | 69/100 (69.0%) [59.4%, 77.2%] | 69/100 (69.0%) [59.4%, 77.2%] | 396.06 [396.06, 396.06] | 3.46e-04 | — | 22.26 | 636 / 10 |
| 180-Node LGL Collocation (IPOPT) | 49/100 (49.0%) [39.4%, 58.7%] | 19/100 (19.0%) [12.5%, 27.8%] | 396.80 [396.09, 407.69] | 1.92e-02 | 8.46e-04 | 67.31 | 261 / 127 |

#### Hybrid vs Shooting across Solar Radiation Pressure (Fresh Seeds 2000..2099)
| SRP $A/m_0$ (m²/kg) | Mode | Shooting Strict (<1e-2) [95% CI] | Hybrid Strict (<1e-2) [95% CI] | Strict Delta (pp) | Exact McNemar $p$ | $\chi^2$ (cont. corr.) | Discordant (Shoot only / Hyb only) | Shooting Wall-Clock Med (s) | Hybrid Wall-Clock Med (s) [PoNN + Shoot] | Shoot Propagations Med | Hybrid Evaluations Med | Preregistered Verdict |
| ---: | --- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | :---: |
| 0.00 | None | 69/100 (69.0%) [59.4%, 77.2%] | 95/100 (95.0%) [88.8%, 97.8%] | **+26.0 pp** | **8.68e-07** | 20.83 (p=5.01e-06) | b=2 / c=28 | 22.26 | 19.01 (3.31 + 15.40) | 636 | 273 (26 + 236) | **`SUPPORTED`** |
| 0.50 | Radial | 72/100 (72.0%) [62.5%, 79.9%] | 96/100 (96.0%) [90.2%, 98.4%] | **+24.0 pp** | **8.05e-07** | 20.35 (p=6.46e-06) | b=1 / c=25 | 26.50 | 21.31 (2.85 + 18.26) | 626 | 314 (26 + 277) | **`SUPPORTED`** |
| 1.00 | Radial | 76/100 (76.0%) [66.8%, 83.3%] | 95/100 (95.0%) [88.8%, 97.8%] | **+19.0 pp** | **6.60e-05** | 14.09 (p=1.75e-04) | b=2 / c=21 | 20.54 | 19.07 (3.28 + 15.72) | 482 | 284 (26 + 249) | **`SUPPORTED`** |


---

## Secondary Study — Scenario B (`1218 kg`, `0.5 N`, `Isp = 3100 s`, SRP)

Global feasibility scan (preserved in `results/phase1_results.json`) established $t_{\min} = 260.833	ext{ d}$ at $	heta_f = 192.518^\circ$ for the 1218 kg spacecraft at $T_{\max} = 0.5	ext{ N}$. Testing the **frozen Phase A2 PoNN (no retuning)** on Scenario B ($t_f = 300	ext{ d}, 	heta_f = 225^\circ$):

| Method on Scenario B ($t_f=300$ d, $\theta_f=225^\circ$, $1218$ kg, $I_{\text{sp}}=3100$ s) | Conv Rate [95% Wilson CI] | Fuel Mean ± SD (kg) | Median Runtime (s) | Notes |
| --- | :---: | :---: | ---: | --- |
| 180-Node LGL Collocation (smooth init) | 1/1 | 244.080 | 47.75 | reprop=4.11e-04, 3 switches |
| Frozen Phase A2 PoNN (ZERO retuning, 20 seeds) | 0/20 (0.0%) [0.0%, 16.1%] | — | 13.52 | 8-res Chebyshev CE, L=70->80 |
| Indirect Shooting (random 7-D costates, 20 seeds) | 17/20 (85.0%) [64.0%, 94.8%] | 243.826 ± 0.000 | 7.71 | 20-step rho continuation |
| Hybrid: Shooting warm-started from Frozen PoNN | 20/20 (100.0%) [83.9%, 100.0%] | 243.826 ± 0.000 | 4.73 | lam0 extracted from PoNN |

---

## Figures

- [`results/figures/scenario_a_fig2_coarse_vs_shooting.png`](results/figures/scenario_a_fig2_coarse_vs_shooting.png) — Coarse PoNN vs. Indirect Shooting ($S(t)$ and $\delta(t)$)
- [`results/figures/scenario_a_fig3_trajectory_3d.png`](results/figures/scenario_a_fig3_trajectory_3d.png) — 3-D Earth–Mars Transfer and Optimal Thrust Direction
- [`results/figures/scenario_a_fig5_refined_solution.png`](results/figures/scenario_a_fig5_refined_solution.png) — Refined PoNN vs. Shooting & LGL-180 Collocation ($S, \delta, m, \lambda_m$)
- [`results/figures/scenario_a_phase_a4_ablations.png`](results/figures/scenario_a_phase_a4_ablations.png) — Phase A4 Formulation, Basis, $ho$-Schedule, and Grid/Optimizer Ablations
- [`results/figures/scenario_a_phase_b_benchmark.png`](results/figures/scenario_a_phase_b_benchmark.png) — Phase B 100-Seed Three-Method Benchmark
- [`results/figures/scenario_a_phase_c_srp_sweep.png`](results/figures/scenario_a_phase_c_srp_sweep.png) — Phase C Radial & Attitude Flat-Plate SRP Perturbation Sweep
- [`results/figures/scenario_a_prereg_hybrid_comparison.png`](results/figures/scenario_a_prereg_hybrid_comparison.png) — Preregistered Hybrid vs Shooting Comparison Across SRP (Seeds 2000..2099)

---

## Repository Layout & Reproduction Commands

- `docs/PAPER_SPEC.md` — Complete Phase 0 equation specification, FD verification, and ambiguity list
- `trajectory_optimization/scenario_a.py` — Scenario A dynamics, analytical radial/attitude SRP Jacobians, and Phase A1 shooting solver
- `trajectory_optimization/ponn_paper.py` — Exact 8-group Chebyshev PoNN (Table A6/A1 CEs, JAX analytical Jacobian, `lm_svd` / SciPy `'lm'` / `'trf'`, Sec. 4.1 refinement)
- `trajectory_optimization/scenario_a_collocation.py` — 180-node LGL pseudospectral direct collocation (CasADi + IPOPT)
- `REPORT.md`, `ASSUMPTIONS.md`, `DEVIATIONS.md` — Full scientific report (with paper claim audit), ambiguity log, and deviations log

```bash
# Run full unit test suite (29 tests: Scenario A FD/CE/Jacobian/Shooting/Projection + Scenario B)
.venv/bin/pytest tests/ -v

# Re-run individual phases (checkpointed per seed)
PYTHONPATH=. .venv/bin/python scripts/run_phase_a3.py --n-seeds 100
PYTHONPATH=. .venv/bin/python scripts/run_phase_a4.py --n-seeds 10
PYTHONPATH=. .venv/bin/python scripts/run_phase_b.py --n-seeds 100
PYTHONPATH=. .venv/bin/python scripts/run_phase_c.py --n-seeds 5 --colloc-seeds 1
PYTHONPATH=. .venv/bin/python scripts/run_phase_d.py

# Regenerate all figures, markdown tables, REPORT.md, README.md, ASSUMPTIONS.md, DEVIATIONS.md
PYTHONPATH=. .venv/bin/python scripts/generate_paper_deliverables.py
```
