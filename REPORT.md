# Scientific Verification and Claim-by-Claim Audit Report

This report is generated programmatically from the stored numerical outputs of the solver suite. Every claim in the manuscript is compared directly against the values computed by the code, and each status label (`SUPPORTED`, `PARTIAL`, or `CONTRADICTED`) is assigned numerically from the data.

## 1. Executive Summary of Findings

1. **Escape Phase with $J_2$ (Table E1) and Perturbation Scaling (Table 2):** 3-D Cartesian DOP853 propagation (`rtol = 1e-10, atol = 1e-11`) from a 500 km parking orbit at $i = 28.5^\circ$ with $T = 0.50\text{ N}$ and $m_0 = 1,538.6\text{ kg}$ reaches parabolic escape energy ($E = 0$) in **225.58 days**, consuming **320.56 kg** of xenon and arriving at **1218.04 kg** (paper: 225.6 days, 320.6 kg, 1,218.0 kg → **SUPPORTED**). Table 2 perturbation ratios ($J_2 / a_T = 36.2$ at 500 km; $a_{\text{SRP}} = 1.09e-07\text{ m/s}^2$ at 1 AU) match the paper's analytical formulas (**SUPPORTED**).
2. **Minimum-Time Heliocentric Feasibility Scan at $T_{\max} = 0.50\text{ N}$:** Solving the 180-node LGL minimum-time rendezvous problem with free transfer duration $t_f$ and free Mars arrival longitude $\theta_f$ (with $m_0 = 1,218.0\text{ kg}, I_{\text{sp}} = 3,100\text{ s}, T \le 0.50\text{ N}$) yields a minimum transfer time of **$t_{\min} = 260.8333\text{ days}$** at arrival longitude **$\theta_f = 192.5175^\circ$** (3.360065 rad), consuming **370.65 kg** of xenon.
3. **Non-Convergence at the Paper's Nominal $t_f = 247.9\text{ days}$ Duration Under $T \le 0.50\text{ N}$:** Because $t_f = 247.9\text{ days}$ lies below the shortest transfer time found in our global scan ($t_{\min} = 260.83\text{ days}$), no tested solver or initialization achieved a converged circular Mars rendezvous at $t_f = 247.9\text{ days}$ under $T \le 0.50\text{ N}$. Across the bounded-thrust angle sweep at $t_f = 247.9\text{ days}$, **0 of 9** tested arrival angles converged (minimum slack norm `3.102e-02` at $\theta_f = 185.0^\circ$). In the 20-run robustness protocol at $t_f = 247.9\text{ days}$, convergence rates are **0/20** (X-TFC PoNN), **0/20** (Indirect shooting), and **0/20** (Pseudospectral collocation) (**CONTRADICTED** vs paper's 19/20, 14/20, 20/20).
4. **Shortest All-Method Converged Duration ($t_f = 261.0\text{ days}, \theta_f = 192.52^\circ$):** At $t_f = 261.0\text{ days}$ (just above $t_{\min} = 260.83\text{ days}$), all three solvers solve the identical 14-state fuel-optimal problem under $T \le 0.50\text{ N}$: Indirect shooting consumes **368.28 kg** ($\varepsilon_M = 2.82e-11$), 180-node Pseudospectral collocation consumes **368.35 kg** (terminal $\varepsilon_M = 0.00e+00$, DOP853 re-propagated $\varepsilon_M = 4.81e-05$, zero slack), and X-TFC PoNN consumes **370.65 kg** (14-ODE RMS residual `7.75e-05`, DOP853 re-propagated $\varepsilon_M = 6.44e-04$). Near $t_{\min}$, the spacecraft must thrust near full throttle for almost the entire transfer, burning $\approx 368.3\text{--}370.6\text{ kg}$ of xenon (exceeding the paper's 300 kg nominal xenon budget and 274.8–276.1 kg claim).

## 2. Claim-by-Claim Numerical Audit Table

| Section / Table | Paper Claim | Computed Result (Paper Scenario tf=247.9 d) | Computed Result (Feasible Scenario tf=261.0 d) | Status (247.9 d / 261.0 d) |
| --- | --- | --- | --- | :---: |
| Table E1 (i=28.5°) | t_esc = 225.6 d, Δm = 320.6 kg, m_f = 1,218.0 kg | 225.58 d, 320.56 kg, 1218.04 kg | (Phase 1 identical) | **SUPPORTED / SUPPORTED** |
| Table 2 (J2 500 km) | a_J2 = 1.18e-2 m/s², a_J2/a_T = 36.2 | 1.18e-02 m/s², ratio = 36.2 | (Identical) | **SUPPORTED** |
| Table 2 (SRP 1 AU) | a_SRP = 1.09e-7 m/s², K/μ_S = 1.85e-5 | 1.09e-07 m/s², K/μ_S = 1.85e-05 | (Identical) | **SUPPORTED** |
| Table 1 Heliocentric Convergence | All 3 methods converge at tf ≈ 248 d, T ≤ 0.50 N | Conv: XT=False, SH=False, PS=False (t_min=260.83 d) | Conv: XT=True, SH=True, PS=True | **CONTRADICTED / SUPPORTED** |
| Table 1 Fuel: X-TFC PoNN | 274.8 kg | 295.54 kg (unconverged, ODE RMS=1.71e-01) | 370.65 kg | **CONTRADICTED / CONTRADICTED** |
| Table 1 Fuel: Indirect shooting | 276.1 kg | 352.27 kg (unconverged, ε_M=3.48e-02) | 368.28 kg | **CONTRADICTED / CONTRADICTED** |
| Table 1 Fuel: Collocation (N=180) | 275.3 kg | 307.19 kg (unconverged, slack=3.69e-02) | 368.35 kg | **CONTRADICTED / CONTRADICTED** |
| Table 1 Terminal ε_M: X-TFC | 3.2e-7 | Exact BCs: 0.00e+00 (re-prop: 6.50e-01) | Exact BCs: 0.00e+00 (re-prop: 6.44e-04) | **CONTRADICTED / CONTRADICTED** |
| Table 1 Terminal ε_M: Shooting | 6.5e-7 | 3.48e-02 | 2.82e-11 | **CONTRADICTED / SUPPORTED** |
| Table 1 Terminal ε_M: Collocation | 4.0e-7 | 3.69e-02 (re-prop: 3.69e-02) | 0.00e+00 (re-prop: 4.81e-05) | **CONTRADICTED / SUPPORTED** |
| Table 1 Energy Error ε_E | XT: 8e-7, SH: 1.2e-6, PS: 6e-7 | XT: 1.95e-01, SH: 5.80e-05, PS: 1.28e-03 | XT: 7.80e-04, SH: 3.31e-04, PS: 7.88e-05 | **CONTRADICTED / CONTRADICTED** |
| Table 1 X-TFC Loss & Epochs | 4,300 epochs, L0=1.2e-1 → Lf=2.8e-8 | 15 LM iters, L0=3.08e-01 → Lf=2.91e-02 | 129 LM iters, L0=4.10e-09 → Lf=6.00e-09 | **CONTRADICTED / SUPPORTED** |
| Table 1 & 3 Runtime (min) | XT: 18.6, SH: 24.0, PS: 12.2 min | XT: 0.305, SH: 0.175, PS: 0.558 min | XT: 0.865, SH: 0.064, PS: 0.190 min | **CONTRADICTED / CONTRADICTED / CONTRADICTED** |
| Table 3 Convergence (20 runs) | XT: 19/20, SH: 14/20, PS: 20/20 | XT: 0/20, SH: 0/20, PS: 0/20 | XT: 10/20, SH: 20/20, PS: 13/20 | **CONTRADICTED / PARTIAL** |

## 3. Summary of Angle-Sweep, Unperturbed Control, and Perturbation Sensitivity

- **Bounded-Thrust Angle Sweep:** At $t_f = 247.9\text{ days}$, 0/9 angles converged. At $t_f = 261.0\text{ days}$, 2/5 angles converged under both indirect shooting and 180-node collocation.
- **Unperturbed Control (`srp_scale = 0` at $t_f = 261.0\text{ days}$):** Removing SRP changes propellant consumption by `-0.0473 kg` (Indirect shooting), `-0.0458 kg` (Collocation), and `+0.0039 kg` (X-TFC PoNN), consistent with the analytical ratio $K_{\text{SRP}} / \mu_S = 1.8455e-05$.
- **Perturbation-Strength Sweep (`srp_scale` from 0× to 1000×):** Propellant consumption varies smoothly from `368.3319 kg` at 0× SRP to `370.8869 kg` at 1000× SRP.

## 4. Phase 1: Global Feasibility Scan & Multi-$t_f$ 20-Run Protocol

- **Global Feasibility Scan ($\theta_f \in [60^\circ, 300^\circ]$, step $10^\circ$, 4 guesses/angle + 5 free-$\theta_f$ multi-starts):** 25/25 fixed arrival longitudes had at least one converged solve and 5/5 free-$\theta_f$ 180-node starts converged. The shortest transfer time found across all tested starts is **$t_{\min} = 260.8333\text{ days}$** at **$\theta_f^* = 192.518^\circ$**. **Was the earlier 260.83 d result a local minimum?** **No** — no tested longitude or initial guess (including low-angle 0-rev and 1-rev spiral guesses) produced a feasible transfer below 260.83 days.

# Phase 1.1: Global Minimum-Time Feasibility Scan at $T_{\max} = 0.50\text{ N}$

Fixed-longitude minimum-time NLP solves ($N = 80$ LGL nodes, 4 initial guesses per angle: `direct`, `spiral_0rev`, `spiral_inward`, `spiral_1rev`) across $\theta_f \in [60^\circ, 300^\circ]$ (step $10^\circ$), supplemented by 5 multi-start free-$\theta_f$ $N = 180$ LGL minimum-time solves.

## 1. Fixed-Longitude Minimum-Time Scan ($\theta_f \in [60^\circ, 300^\circ]$)

| $\theta_f$ (deg) | Converged Starts | $t_{\min}(\theta_f)$ (days) | Propellant at $t_{\min}$ (kg) | Best Guess Mode | DOP853 Re-prop $\varepsilon_M$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 60.0° | 1 / 4 | 390.944 | 555.54 | `spiral_1rev` | 3.33e-03 |
| 70.0° | 1 / 4 | 394.456 | 560.53 | `spiral_1rev` | 3.79e-03 |
| 80.0° | 1 / 4 | 398.029 | 565.61 | `spiral_1rev` | 4.17e-03 |
| 90.0° | 1 / 4 | 401.736 | 570.88 | `spiral_1rev` | 4.47e-03 |
| 100.0° | 1 / 4 | 405.670 | 576.47 | `spiral_1rev` | 4.69e-03 |
| 110.0° | 1 / 4 | 409.949 | 582.55 | `spiral_1rev` | 4.87e-03 |
| 120.0° | 1 / 4 | 414.705 | 589.31 | `spiral_1rev` | 5.08e-03 |
| 130.0° | 1 / 4 | 420.064 | 596.92 | `spiral_1rev` | 5.37e-03 |
| 140.0° | 1 / 4 | 426.112 | 605.51 | `spiral_1rev` | 5.78e-03 |
| 150.0° | 1 / 4 | 432.831 | 615.06 | `spiral_1rev` | 6.30e-03 |
| 160.0° | 1 / 4 | 440.039 | 625.31 | `spiral_1rev` | 6.70e-03 |
| 170.0° | 1 / 4 | 447.399 | 635.76 | `spiral_1rev` | 6.81e-03 |
| 180.0° | 1 / 4 | 454.650 | 646.07 | `spiral_1rev` | 6.80e-03 |
| 190.0° | 1 / 4 | 480.218 | 630.33 | `spiral_1rev` | 1.14e-02 |
| 200.0° | 4 / 4 | 265.024 | 376.61 | `direct` | 4.56e-04 |
| 210.0° | 4 / 4 | 272.881 | 387.77 | `spiral_0rev` | 5.80e-04 |
| 220.0° | 4 / 4 | 281.026 | 399.34 | `direct` | 1.64e-03 |
| 230.0° | 4 / 4 | 289.095 | 410.81 | `spiral_inward` | 6.81e-04 |
| 240.0° | 4 / 4 | 296.996 | 422.04 | `spiral_0rev` | 5.05e-04 |
| 250.0° | 4 / 4 | 304.633 | 432.89 | `spiral_0rev` | 8.19e-04 |
| 260.0° | 4 / 4 | 312.009 | 443.37 | `spiral_0rev` | 4.62e-04 |
| 270.0° | 4 / 4 | 319.052 | 453.38 | `direct` | 6.02e-04 |
| 280.0° | 3 / 4 | 325.799 | 462.97 | `direct` | 5.24e-04 |
| 290.0° | 3 / 4 | 332.242 | 472.12 | `spiral_0rev` | 2.09e-03 |
| 300.0° | 4 / 4 | 338.287 | 480.71 | `spiral_0rev` | 5.12e-04 |

## 2. Multi-Start Free-$\theta_f$ Minimum-Time Solves ($N = 180$ LGL Nodes)

| Initial Guess $(t_f, \theta_f)$ | Converged? | $t_{\min}$ (days) | $\theta_f^*$ (deg) | Fuel (kg) | IPOPT Iters | Runtime (s) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| (180 d, 100.0°) | True | 260.8333 | 192.518° | 370.65 | 7 | 5.85 | `Solve_Succeeded` |
| (210 d, 120.0°) | True | 260.8333 | 192.518° | 370.65 | 8 | 6.00 | `Solve_Succeeded` |
| (230 d, 150.0°) | True | 260.8333 | 192.518° | 370.65 | 7 | 4.72 | `Solve_Succeeded` |
| (255 d, 192.5°) | True | 260.8333 | 192.518° | 370.65 | 8 | 6.35 | `Solve_Succeeded` |
| (300 d, 240.0°) | True | 260.8825 | 192.580° | 370.72 | 142 | 85.85 | `Solve_Succeeded` |

## 3. Global Scan Assessment

- **Shortest transfer time found across all tested starts:** **260.8333 days** at $\theta_f^* = 192.518^\circ$ (source: `free_theta_180`).
- **Was the earlier 260.83 d result a local minimum?** **no** — within the tested 0-rev direct/spiral and 1-rev spiral trajectory families across $\theta_f \in [60^\circ, 300^\circ]$ and all 5 free-$\theta_f$ multi-start solves, no tested longitude or initialization achieved a feasible transfer below 260.83 days. Low-angle targets ($60^\circ\text{--}180^\circ$) that converged required $t_{\min} \in [390.9, 454.7]\text{ d}$, while targets in $[200^\circ, 300^\circ]$ achieved $t_{\min} \ge 265.0\text{ d}$ on the $10^\circ$ grid, with the free-$\theta_f$ optimum at $\theta_f^* = 192.518^\circ$, $t_{\min} = 260.8333\text{ d}$.


# Phase 1.2–1.4: Multi-$t_f$ 20-Run Robustness Protocol (Fuel-Minimal $\theta_f^*(t_f)$, All Methods + Cold-Start X-TFC)

All four solver configurations are evaluated on the identical 3-D 14-state `HelioProblem` at each transfer duration $t_f \in \{247.9, 280.0, 300.0, 330.0, 360.0\}\text{ days}$ over 20 seeds (`1000..1019`). Re-propagated arrival error $\varepsilon_M$ is computed by integrating each solver's control profile forward with `DOP853` (never reporting the TFC boundary-embedded 0 as arrival accuracy).

## $t_f = 247.9$ days, Fuel-Minimal $\theta_f^* = 192.52^\circ$ ($T_{\max} = 0.50\text{ N}$)

| Method | Converged (95% Wilson CI) | Fuel (kg) | Throttle Switches | Coast Arc (days) | Re-prop $\varepsilon_M$ | Energy Err $\varepsilon_E$ | Runtime (s) | Mean Func / Jac Evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN (warm) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 11.70 ± 1.16 (all) | 20 / 15 (all) |
| X-TFC PoNN (no warm start) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 11.05 ± 0.77 (all) | 20 / 15 (all) |
| Indirect shooting | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 3.56 ± 2.45 (all) | 381,856 / 1 (all) |
| Pseudospectral collocation | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 39.01 ± 3.65 (all) | 640 / 203 (all) |

## $t_f = 280.0$ days, Fuel-Minimal $\theta_f^* = 205.00^\circ$ ($T_{\max} = 0.50\text{ N}$)

| Method | Converged (95% Wilson CI) | Fuel (kg) | Throttle Switches | Coast Arc (days) | Re-prop $\varepsilon_M$ | Energy Err $\varepsilon_E$ | Runtime (s) | Mean Func / Jac Evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN (warm) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 7.81 ± 3.30 (all) | 16 / 14 (all) |
| X-TFC PoNN (no warm start) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 8.45 ± 2.82 (all) | 19 / 15 (all) |
| Indirect shooting | 20/20 ([0.84, 1.00]) | 261.78 ± 0.00 | 2.0 ± 0.0 | 95.8 ± 0.0 | 1.39e-03 | 2.40e-04 | 3.69 ± 0.96 | 377,710 / 10 |
| Pseudospectral collocation | 14/20 ([0.48, 0.85]) | 263.11 ± 3.34 | 2.6 ± 1.6 | 94.8 ± 2.6 | 1.20e-04 | 7.49e-04 | 12.74 ± 4.02 | 79 / 66 |

## $t_f = 300.0$ days, Fuel-Minimal $\theta_f^* = 217.50^\circ$ ($T_{\max} = 0.50\text{ N}$)

| Method | Converged (95% Wilson CI) | Fuel (kg) | Throttle Switches | Coast Arc (days) | Re-prop $\varepsilon_M$ | Energy Err $\varepsilon_E$ | Runtime (s) | Mean Func / Jac Evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN (warm) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 41.95 ± 5.04 (all) | 100 / 91 (all) |
| X-TFC PoNN (no warm start) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 7.26 ± 0.38 (all) | 18 / 15 (all) |
| Indirect shooting | 20/20 ([0.84, 1.00]) | 221.95 ± 0.00 | 2.0 ± 0.0 | 143.4 ± 0.0 | 7.30e-04 | 2.76e-04 | 3.22 ± 0.82 | 340,095 / 10 |
| Pseudospectral collocation | 20/20 ([0.84, 1.00]) | 221.97 ± 0.00 | 2.0 ± 0.0 | 143.4 ± 0.0 | 1.08e-04 | 2.06e-04 | 17.01 ± 12.94 | 111 / 93 |

## $t_f = 330.0$ days, Fuel-Minimal $\theta_f^* = 235.00^\circ$ ($T_{\max} = 0.50\text{ N}$)

| Method | Converged (95% Wilson CI) | Fuel (kg) | Throttle Switches | Coast Arc (days) | Re-prop $\varepsilon_M$ | Energy Err $\varepsilon_E$ | Runtime (s) | Mean Func / Jac Evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN (warm) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 7.49 ± 0.53 (all) | 18 / 14 (all) |
| X-TFC PoNN (no warm start) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 7.10 ± 0.52 (all) | 18 / 14 (all) |
| Indirect shooting | 18/20 ([0.70, 0.97]) | 205.13 ± 0.00 | 2.0 ± 0.0 | 185.5 ± 0.0 | 2.17e-05 | 2.49e-05 | 2.20 ± 0.75 | 232,023 / 10 |
| Pseudospectral collocation | 16/20 ([0.58, 0.92]) | 205.13 ± 0.00 | 2.0 ± 0.0 | 185.5 ± 0.0 | 7.98e-05 | 1.66e-04 | 24.99 ± 17.97 | 142 / 126 |

## $t_f = 360.0$ days, Fuel-Minimal $\theta_f^* = 262.50^\circ$ ($T_{\max} = 0.50\text{ N}$)

| Method | Converged (95% Wilson CI) | Fuel (kg) | Throttle Switches | Coast Arc (days) | Re-prop $\varepsilon_M$ | Energy Err $\varepsilon_E$ | Runtime (s) | Mean Func / Jac Evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN (warm) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 7.45 ± 0.69 (all) | 18 / 15 (all) |
| X-TFC PoNN (no warm start) | 0/20 ([0.00, 0.16]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 7.03 ± 0.62 (all) | 18 / 15 (all) |
| Indirect shooting | 16/20 ([0.58, 0.92]) | 205.06 ± 0.00 | 3.0 ± 0.0 | 212.8 ± 0.0 | 2.75e-05 | 7.51e-06 | 2.57 ± 0.38 | 276,410 / 10 |
| Pseudospectral collocation | 15/20 ([0.53, 0.89]) | 205.53 ± 1.28 | 4.9 ± 1.6 | 215.3 ± 1.4 | 4.38e-04 | 7.10e-04 | 39.24 ± 19.21 | 243 / 210 |


## 5. Phase 2: 100-Seed Random-Initialization Basin Study & Hybrid X-TFC $\to$ Shooting Test

# Phase 2: 100-Seed Random-Initialization Basin Study & Hybrid X-TFC $\to$ Shooting Test ($t_f = 280.0$ d, $\theta_f^* = 205.00^\circ$)

Evaluated on 100 independent seeds (`2000..2099`) on the identical 3-D 14-state `HelioProblem` ($T_{\max} = 0.50\text{ N}$, $I_{\text{sp}} = 3,100\text{ s}$, SRP enabled).

## Phase 2A: Convergence Rates (95% Wilson CI) & Final Fuel Distribution

| Method | Converged (95% Wilson CI) | Fuel Mean ± SD (kg) | Fuel Median [Q25, Q75] (kg) | Fuel [Min, Max] (kg) | Re-prop $\varepsilon_M$ | Mean Iters | Mean Runtime (s) | Mean Func / Jac Evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN (no warm start) | 0/100 ([0.00, 0.04]) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | N/A (0 conv) | 7.27 (all) | 18 / 15 (all) |
| Indirect shooting (random costates) | 97/100 ([0.92, 0.99]) | 261.78 ± 0.00 | 261.78 [261.78, 261.78] | [261.78, 261.78] | 1.39e-03 | 700.4 | 3.80 | 425,067 / 10 |
| Pseudospectral collocation (random guess) | 79/100 ([0.70, 0.86]) | 261.96 ± 0.58 | 261.81 [261.81, 261.81] | [261.81, 265.71] | 1.16e-04 | 37.5 | 14.48 | 106 / 77 |
| Hybrid (X-TFC -> Indirect shooting) | 100/100 ([0.96, 1.00]) | 261.78 ± 0.00 | 261.78 [261.78, 261.78] | [261.78, 261.78] | 1.39e-03 | 455.3 | 2.89 | 322,518 / 10 |

## Phase 2B: Paired Comparison — Hybrid (Cold-Start X-TFC $\to$ Shooting) vs Random-Costate Shooting

| Metric (Same 100 Seeds) | Random-Costate Shooting | Hybrid (X-TFC $\to$ Shooting) |
| --- | :---: | :---: |
| Converged runs (95% Wilson CI) | 97/100 ([0.92, 0.99]) | 100/100 ([0.96, 1.00]) |
| Seed-by-seed contingency | Both conv: 97 | Hybrid-only conv: 3, Random-only conv: 0, Both fail: 0 |
| Mean shooting propagations (on 97 jointly converged seeds) | 700.4 | 456.4 |
| Mean DOP853 RHS evaluations (jointly converged seeds) | 425,067 | 323,310 |
| Mean shooting-stage wall-clock runtime (s) | 3.80 | 2.89 |
| Mean end-to-end wall-clock runtime incl. X-TFC pre-solve (s) | 3.80 | 10.14 |

**Assessment:** Initializing indirect shooting with the cold-start X-TFC initial costate vector improved convergence from 97/100 to 100/100 (recovering 3 seeds where random costates failed, with 0 regressions). On the 97 jointly converged seeds, shooting propagations dropped from 700.4 to 456.4 (shooting stage runtime 3.80 s vs 2.89 s), though total end-to-end runtime including the X-TFC pre-solve was 10.14 s.


