# Earth–Mars Low-Thrust Trajectory Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end implementation and numerical verification of three trajectory optimization methods for the two-phase Earth–Mars low-thrust transfer problem (Phase 1: 500 km parking-orbit escape spiral with Earth's $J_2$; Phase 2: 3-D heliocentric rendezvous with radial Solar Radiation Pressure):

1. **X-TFC Pontryagin Neural Network (`XTFCPoNNSolver`)** — single-hidden-layer $\tanh$ Extreme Learning Machine with exact Theory of Functional Connections (TFC) boundary expressions, JAX automatic differentiation, and Levenberg-Marquardt training on all 14 state/costate ODE residuals.
2. **Indirect Shooting (`IndirectShootingSolver`)** — 14-state Pontryagin Maximum Principle two-point boundary-value problem integrated with `solve_ivp(DOP853)` and solved via `scipy.optimize.fsolve` (with documented `least_squares(method='lm')` fallback) across a 10-step logistic throttle smoothing homotopy.
3. **Direct Pseudospectral Collocation (`CollocationSolver`)** — 180 Legendre-Gauss-Lobatto (LGL) nodes in all fuel-optimal runs, spectral differentiation-matrix defect constraints, explicit mass state, hard terminal constraints (with last-resort slack continuation recorded when infeasible), solved via CasADi + IPOPT.

All tables, figures, and audit reports in this repository are generated programmatically from stored solver outputs.

---

## Summary of Key Numerical Findings

- **Global Feasibility Scan (Phase 1.1, $\theta_f \in [60^\circ, 300^\circ]$):** Across 25 fixed longitudes (4 guesses/angle, including low-angle spirals) and 5 free-$\theta_f$ 180-node starts, the shortest transfer time found is **$t_{\min} = 260.8333\text{ days}$** at **$\theta_f^* = 192.518^\circ$** (the earlier 260.83 d result was not a local minimum within the tested 0-rev and 1-rev trajectory families).
- **Paper Nominal Duration ($t_f = 247.9\text{ days}$):** Because $247.9\text{ days}$ lies below the shortest transfer time found in the global scan ($t_{\min} = 260.83\text{ days}$), none of the tested solvers or initializations converged at $t_f = 247.9\text{ days}$ under $T \le 0.50\text{ N}$.
- **Shortest All-Method Converged Duration ($t_f = 261.0\text{ days}, \theta_f = 192.52^\circ$):** All three solvers converge on the identical 14-state problem under $T \le 0.50\text{ N}$ and cross-validate against a high-precision indirect shooting reference (`rtol = 1e-13, atol = 1e-14`).
- **Phase 2 100-Seed Basin & Hybrid Study ($t_f = 280.0\text{ d}, \theta_f^* = 205.00^\circ$):** Convergence rates (95% Wilson CI): X-TFC cold-start **0/100** ([0.00, 0.04]), Indirect shooting (random) **97/100** ([0.92, 0.99]), 180-node Collocation **79/100** ([0.70, 0.86]), Hybrid X-TFC $\to$ Shooting **100/100** ([0.96, 1.00]).

---

# Table 1: Initial Mission, Spacecraft, Propulsion, Orbital and Numerical Parameters

## Part A: Problem Specification & Physical Constants

| Parameter Category | Parameter | Symbol | Setting / Computed Value | Unit | Notes |
| --- | --- | :---: | :---: | :---: | --- |
| Mission | Departure / Arrival body | – | Earth → Mars | – | Two-phase low-thrust transfer |
| Mission | Paper nominal heliocentric transfer time | t_f,nom | 247.90 | days | Common fixed time for paper scenario (4.2644 TU) |
| Mission | Minimum feasible transfer time at 0.50 N | t_min | 260.8333 | days | Free-tf, free-θ_f 180-node LGL minimum-time solution (4.4869 TU) |
| Mission | Shortest all-method converged duration | t_f,feas | 261.00 | days | Common fixed time for feasible benchmark (4.4897 TU) |
| Mission | Target Mars arrival longitude | θ_f | 192.5175 (3.360065) | deg (rad) | Common target state for all three methods |
| Spacecraft | Parking-orbit initial wet mass | m_park | 1538.6 | kg | 500 km circular parking orbit |
| Spacecraft | Heliocentric initial mass | m_0 | 1218.0 | kg | Mass after escape spiral (1218.04 kg computed) |
| Spacecraft | Nominal dry mass / xenon budget | m_dry / m_Xe,0 | 918.0 / 300.0 | kg | Paper specification (m_0 = m_dry + m_Xe,0) |
| Propulsion | Thrust bounds | [T_min, T_max] | [0.0, 0.50] | N | Strictly enforced in all three solvers |
| Propulsion | Specific impulse & standard gravity | I_sp, g_0 | 3100.0, 9.80665 | s, m/s² | Exhaust velocity c = 30400.61 m/s |
| Propulsion | Maximum mass-flow rate | ṁ_max | 1.6447e-05 (1.4210) | kg/s (kg/d) | T_max / (I_sp g_0) |
| Earth Departure | Parking altitude & radius | h_p, r_p | 500.0, 6878.137 | km | μ_E = 398600.4418 km³/s², J_2 = 1.08262668e-03 |
| Earth Departure | Escape duration & propellant (i=28.5°) | t_esc, Δm_esc | 225.58, 320.56 | days, kg | DOP853 3D Cartesian propagation to E = 0 (Δv = 7.103 km/s) |
| Earth Departure | Equatorial J_2 acceleration at 500 km | a_J2(500 km) | 1.177e-02 | m/s² | 36.2× max thrust acceleration |
| Heliocentric | Sun gravitational parameter & AU | μ_S, AU | 132712440018.00, 149,597,870.7 | km³/s², km | r_E = 1.000 AU, r_M = 1.524 AU |
| Solar Radiation Pressure | P_0, C_R, A/m | P_0, C_R, A/m | 4.56e-06, 1.20, 0.020 | N/m², –, m²/kg | K_SRP = 2.4492e+15 m³/s² (K_SRP/μ_S = 1.8455e-05) |
| Solar Radiation Pressure | SRP acceleration at 1 AU / 1.524 AU | a_SRP | 1.094e-07 / 4.712e-08 | m/s² | Explicit radial perturbation a_SRP = K_SRP r / r³ |

## Part B: Computed Solver Performance (Seed 1000 Representative Run)

### 1. Paper Nominal Scenario (tf = 247.9 days, θ_f = 192.52°, T_max = 0.50 N — Below Minimum Feasible Time t_min = 260.83 days)

| Method | Converged? | Propellant Δm_Xe (kg) | Final Mass m_f (kg) | Flight Time t_f (days) | Solver Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | 180-LGL Defect RMS | Energy Error ε_E | Max Thrust (N) | Iterations | Func / Jac Evals | Runtime (s) | Runtime (min) | Warm Start / Continuation |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| X-TFC PoNN | False | 295.54 | 922.46 | 247.90 | 0.00e+00 | 6.50e-01 | 1.71e-01 | 1.15e+00 | 1.95e-01 | 0.4998 | 15 (L0=3.08e-01→Lf=2.91e-02) | 23 / 15 | 18.32 | 0.305 | deterministic polar/tangential guess (eps=1 shooting did not converge) |
| Indirect shooting | False | 352.27 | 865.73 | 247.90 | 3.48e-02 | 3.48e-02 | 1.85e-02 | 2.00e-03 | 5.80e-05 | 0.5000 | 2118 | 1,426,644 / 1 | 10.49 | 0.175 | none (random costate draw) |
| Pseudospectral collocation | False | 307.19 | 910.81 | 247.90 | 3.69e-02 | 3.69e-02 | 2.15e+05 | 4.09e-04 | 1.28e-03 | 0.5000 | 100 | 402 / 204 | 33.47 | 0.558 | none (seeded polar interpolation) (slack=3.69e-02) |

### 2. Shortest All-Method Converged Scenario (tf = 261.0 days, θ_f = 192.52°, T_max = 0.50 N)

| Method | Converged? | Propellant Δm_Xe (kg) | Final Mass m_f (kg) | Flight Time t_f (days) | Solver Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | 180-LGL Defect RMS | Energy Error ε_E | Max Thrust (N) | Iterations | Func / Jac Evals | Runtime (s) | Runtime (min) | Warm Start / Continuation |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| X-TFC PoNN | True | 370.65 | 847.35 | 261.00 | 0.00e+00 | 6.44e-04 | 7.75e-05 | 7.20e-04 | 7.80e-04 | 0.5000 | 129 (L0=4.10e-09→Lf=6.00e-09) | 151 / 129 | 51.88 | 0.865 | projected from eps=1 smoothed shooting extremal |
| Indirect shooting | True | 368.28 | 849.72 | 261.00 | 2.82e-11 | 5.61e-04 | 1.16e-03 | 3.58e-03 | 3.31e-04 | 0.5000 | 1105 | 516,950 / 10 | 3.86 | 0.064 | none (random costate draw) |
| Pseudospectral collocation | True | 368.35 | 849.65 | 261.00 | 0.00e+00 | 4.81e-05 | 2.76e+01 | 3.31e-11 | 7.88e-05 | 0.5000 | 48 | 160 / 98 | 11.39 | 0.190 | none (seeded polar interpolation) |


---

# Mars Arrival Angle Sweep and Minimum-Time Feasibility Analysis (T_max = 0.50 N)

## 1. Minimum-Time Rendezvous Problem (Free tf, Free Arrival Longitude θ_f, 180 LGL Nodes)

| Metric | Value |
| --- | --- |
| Solver status | `Solve_Succeeded` (`success = True`) |
| Minimum transfer time `t_min` | **260.8333 days** (4.486880 TU) |
| Optimal arrival longitude `θ_f(t_min)` | **192.5175°** (3.360065 rad) |
| Propellant consumed at `t_min` | **370.65 kg** |
| IPOPT iterations (`N = 180`) | 8 |
| Wall-clock runtime | 5.08 s |

## 2. Bounded-Thrust Angle Sweep at Paper Nominal Duration (tf = 247.9 days)

| θ_f (deg) | θ_f (rad) | Collocation Converged? | Collocation Fuel (kg) | Collocation Slack Norm | Collocation Re-prop ε_M | Shooting Converged? | Shooting Fuel (kg) | Shooting ε_M |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 165.0° | 2.8798 | False | 343.36 | 3.726e-01 | 3.726e-01 | False | 352.27 | 4.041e-01 |
| 175.0° | 3.0543 | False | 307.78 | 1.516e-01 | 1.516e-01 | False | 352.25 | 1.364e-01 |
| 180.0° | 3.1416 | False | 341.07 | 5.110e-02 | 5.116e-02 | False | 352.27 | 4.949e-02 |
| 185.0° | 3.2289 | False | 286.92 | 3.102e-02 | 3.102e-02 | False | 352.27 | 1.942e-02 |
| 190.0° | 3.3161 | False | 291.55 | 3.218e-02 | 3.221e-02 | False | 352.27 | 2.378e-02 |
| 192.5° | 3.3601 | False | 274.01 | 4.008e-02 | 4.012e-02 | False | 352.27 | 2.127e-01 |
| 195.0° | 3.4034 | False | 285.38 | 4.489e-02 | 4.497e-02 | False | 352.27 | 7.624e-02 |
| 200.0° | 3.4907 | False | 228.57 | 7.003e-02 | 7.004e-02 | False | 352.27 | 2.089e-01 |
| 210.0° | 3.6652 | False | 249.44 | 1.064e-01 | 1.064e-01 | False | 352.27 | 7.741e-02 |

## 3. Bounded-Thrust Angle Sweep at Feasible Duration (tf = 261.0 days)

| θ_f (deg) | θ_f (rad) | Collocation Converged? | Collocation Fuel (kg) | Collocation Slack Norm | Collocation Re-prop ε_M | Shooting Converged? | Shooting Fuel (kg) | Shooting ε_M |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 191.5° | 3.3423 | False | 283.28 | 1.653e-02 | 1.660e-02 | False | 370.72 | 3.253e-04 |
| 192.0° | 3.3510 | True | 369.96 | 0.000e+00 | 6.490e-05 | True | 369.93 | 1.177e-14 |
| 192.5° | 3.3601 | True | 368.35 | 0.000e+00 | 4.801e-05 | True | 368.28 | 9.364e-13 |
| 193.0° | 3.3685 | False | 312.62 | 8.824e-03 | 8.962e-03 | True | 368.86 | 2.454e-13 |
| 193.5° | 3.3772 | False | 282.94 | 2.375e-02 | 2.380e-02 | False | 370.89 | 3.395e-05 |


---

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


---

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


---

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


---

# Table 2: Size of the J2 and SRP accelerations compared with thrust and central gravity

| Case | Distance | Perturbing acceleration (m/s²) | Maximum thrust acceleration (m/s²) | Perturbation ÷ max thrust | Perturbation ÷ central gravity |
| --- | --- | :---: | :---: | :---: | :---: |
| J2, 500 km orbit, equatorial (m = 1,538.6 kg) | 6,878 km | 1.18e-02 | 3.25e-04 | 36.2 | 1.40e-03 |
| J2, r = 17,000 km, equatorial (m = 1,538.6 kg) | 17,000 km | 3.15e-04 | 3.25e-04 | 0.970 | 2.29e-04 |
| J2, geostationary radius, equatorial (m = 1,538.6 kg) | 42,164 km | 8.33e-06 | 3.25e-04 | 0.026 | 3.72e-05 |
| SRP, Earth distance (m = 1,218 kg) | 1.000 AU | 1.09e-07 | 4.11e-04 | 2.67e-04 | 1.85e-05 |
| SRP, Mars distance (m = 943.0 kg) | 1.524 AU | 4.71e-08 | 5.30e-04 | 8.89e-05 | 1.85e-05 |

---

# Table 3: Robustness and spread over 20 repeated runs (180 LGL nodes in all collocation runs)

## A. Paper Nominal Transfer Duration (tf = 247.9 days, θ_f = 192.52°, T_max = 0.50 N)

| Method | Converged runs | Converged Fuel (kg) | Flight time (days) | Arrival residual (ε_M) | Re-propagated ε_M | Runtime (s) | Runtime (min) | All-runs Fuel (kg) | All-runs ε_M | Mean Func / Jac evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN | 0 / 20 | N/A (0 converged) | N/A (0 converged) | N/A (0 converged) | N/A (0 converged) | 6.72 ± 0.65 | 0.112 ± 0.011 | 343.55 ± 25.21 | 0.00e+00 ± 0.00e+00 | 20 / 15 |
| Indirect shooting | 0 / 20 | N/A (0 converged) | N/A (0 converged) | N/A (0 converged) | N/A (0 converged) | 2.65 ± 1.86 | 0.044 ± 0.031 | 352.26 ± 0.03 | 2.91e-02 ± 1.29e-03 | 361509 / 1 |
| Pseudospectral collocation | 0 / 20 | N/A (0 converged) | N/A (0 converged) | N/A (0 converged) | N/A (0 converged) | 29.95 ± 3.36 | 0.499 ± 0.056 | 305.07 ± 43.60 | 3.80e-02 ± 8.99e-03 | 677 / 203 |

## B. Shortest All-Method Converged Duration (tf = 261.0 days, θ_f = 192.52°, T_max = 0.50 N)

| Method | Converged runs | Converged Fuel (kg) | Flight time (days) | Arrival residual (ε_M) | Re-propagated ε_M | Runtime (s) | Runtime (min) | All-runs Fuel (kg) | All-runs ε_M | Mean Func / Jac evals |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-TFC PoNN | 10 / 20 | 370.64 ± 0.02 | 261.00 ± 0.00 | 0.00e+00 ± 0.00e+00 | 6.81e-04 ± 4.76e-05 | 59.06 ± 4.62 | 0.984 ± 0.077 | 369.73 ± 1.77 | 0.00e+00 ± 0.00e+00 | 118 / 100 |
| Indirect shooting | 20 / 20 | 368.28 ± 0.00 | 261.00 ± 0.00 | 4.63e-12 ± 5.97e-12 | 5.61e-04 ± 5.15e-09 | 3.56 ± 1.74 | 0.059 ± 0.029 | 368.28 ± 0.00 | 4.63e-12 ± 5.97e-12 | 402686 / 10 |
| Pseudospectral collocation | 13 / 20 | 368.35 ± 0.00 | 261.00 ± 0.00 | 0.00e+00 ± 0.00e+00 | 4.81e-05 ± 2.46e-09 | 12.54 ± 3.18 | 0.209 ± 0.053 | 349.58 ± 37.62 | 4.15e-03 ± 8.59e-03 | 160 / 128 |


---

# Table E1: Numerical escape-phase propagation with and without J2 (prescribed tangential thrust, 0.5 N, initial mass 1,538.6 kg)

| Case | Time to escape energy (days) | Xenon used (kg) | Mass at escape energy (kg) | Final radius (km) | Δv (km/s) | End energy residual (J/kg) | DOP853 RHS evals | Runtime (s) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No J2 (control) | 225.51 | 320.46 | 1218.14 | 867,276 | 7.100 | 6.98e-10 | 314,453 | 2.87 |
| J2, i = 0.0° | 225.57 | 320.54 | 1218.06 | 867,146 | 7.102 | 1.16e-10 | 315,113 | 3.65 |
| J2, i = 28.5° (reference) | 225.58 | 320.56 | 1218.04 | 867,162 | 7.103 | 4.07e-10 | 322,901 | 3.74 |
| J2, i = 51.6° | 225.61 | 320.59 | 1218.01 | 867,268 | 7.103 | 5.82e-11 | 322,793 | 3.73 |

---

# Extended Scientific Verification, Cross-Validation & Sensitivity Studies

## 1. Cross-Validation Matrix & High-Precision Reference Comparison (Feasible Scenario: tf = 261.0 d, θ_f = 192.52°)

- **High-Precision Shooting Reference (`rtol=1e-13, atol=1e-14, eps=5e-4`):** converged = `True`, fuel = **368.2844 kg**, `ε_M` = `8.27e-15`, `λ_m(tf)` = `1.07e-14`, propagations = 956, RHS evals = 961,168.

| Method | Converged? | Fuel (kg) | Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | 180-LGL Defect RMS | H_paper Drift | H_smooth Drift | Pos RMS vs Ref (AU) | Fuel Diff vs Ref (kg) | Iters | Func / Jac Evals | Runtime (s) | Warm Start / Slack |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| X-TFC PoNN | True | 370.645 | 0.00e+00 | 6.44e-04 | 7.75e-05 | 7.20e-04 | 1.19e-03 | 1.19e-03 | 2.18e-04 | 2.361 | 129 | 151 / 129 | 51.88 | projected from eps=1 smoothed shooting extremal |
| Indirect shooting | True | 368.285 | 2.82e-11 | 5.61e-04 | 1.16e-03 | 3.58e-03 | 3.55e-05 | 1.25e-09 | 9.38e-08 | 0.000 | 1105 | 516,950 / 10 | 3.86 | none (random costate draw) |
| Pseudospectral collocation | True | 368.351 | 0.00e+00 | 4.81e-05 | 2.76e+01 | 3.31e-11 | 7.56e+00 | 7.56e+00 | 4.63e-05 | 0.067 | 48 | 160 / 98 | 11.39 | none (seeded polar interpolation) |

## 2. Cross-Validation Matrix on Paper Nominal Duration (tf = 247.9 d, θ_f = 192.52° — Physically Infeasible Under 0.50 N)

| Method | Converged? | Fuel (kg) | Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | 180-LGL Defect RMS | Max Thrust (N) | Mass Balance Err (kg) | Iters | Func / Jac Evals | Runtime (s) | Status / Slack |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| X-TFC PoNN | False | 295.54 | 0.000e+00 | 6.504e-01 | 1.706e-01 | 1.154e+00 | 0.4998 | 6.454e-01 | 15 | 23 / 15 | 18.32 | eps=1: RMS 14-residual 1.71e-01 > 0.0001 after 15 LM iterations |
| Indirect shooting | False | 352.27 | 3.478e-02 | 3.477e-02 | 1.850e-02 | 1.998e-03 | 0.5000 | 2.705e-13 | 2118 | 1,426,644 / 1 | 10.49 | homotopy stopped at eps=1 (fsolve (lm fallback did not improve), |res|=3.48e-02) |
| Pseudospectral collocation | False | 307.19 | 3.690e-02 | 3.691e-02 | 2.154e+05 | 4.093e-04 | 0.5000 | 1.629e+00 | 100 | 402 / 204 | 33.47 | Maximum_Iterations_Exceeded (slack continuation residual |s|=3.690e-02; hard problem infeasible) |

## 3. Unperturbed Control Run (SRP Disabled, `srp_scale = 0`, tf = 261.0 d)

| Method | Fuel with SRP (kg) | Fuel without SRP (kg) | ΔFuel (SRP − No SRP) (kg) | No-SRP DOP853 Re-prop ε_M |
| --- | :---: | :---: | :---: | :---: |
| X-TFC PoNN | 370.6452 | 370.6413 | +0.0039 | 5.57e-04 |
| Indirect shooting | 368.2845 | 368.3319 | -0.0473 | 6.59e-04 |
| Pseudospectral collocation | 368.3509 | 368.3968 | -0.0458 | 4.89e-05 |

## 4. Perturbation-Strength Sweep (`srp_scale` × K_SRP, tf = 261.0 d)

| SRP Multiplier | K_SRP / μ_S | Shooting Fuel (kg) | Shooting ε_M | X-TFC Fuel (kg) | X-TFC 14-ODE RMS | Collocation Fuel (kg) | Collocation Re-prop ε_M |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0× | 0.000e+00 | 368.3319 | 1.43e-11 | 370.6413 | 7.65e-05 | 368.3968 | 4.89e-05 |
| 1× | 1.846e-05 | 368.2845 | 2.82e-11 | 370.6452 | 7.75e-05 | 368.3509 | 4.81e-05 |
| 10× | 1.846e-04 | 367.8683 | 1.03e-10 | 370.6903 | 8.59e-05 | 305.9425 | 1.02e-02 |
| 100× | 1.846e-03 | 364.5762 | 1.56e-11 | 370.3909 | 1.05e-04 | 297.5581 | 1.14e-02 |
| 1000× | 1.846e-02 | 370.8869 | 6.63e-04 | 248.5673 | 1.18e-01 | 238.0895 | 2.80e-02 |

## 5. Neurons and Collocation-Nodes Convergence Study (tf = 261.0 d)

### 5A. X-TFC PoNN Basis Size (`L` Neurons, `N` Collocation Points)

| Neurons `L` | Points `N` | Converged? | 14-ODE RMS | DOP853 Re-prop ε_M | Fuel (kg) | Pos RMS vs Ref (AU) | Fuel Diff vs Ref (kg) | LM Iters | Func / Jac Evals | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 40 | 80 | False | 7.00e-04 | 9.22e-03 | 358.784 | 7.19e-04 | 9.500 | 15 | 17 / 15 | 3.16 |
| 80 | 160 | False | 2.90e-04 | 5.85e-03 | 362.244 | 3.14e-04 | 6.040 | 15 | 19 / 15 | 4.17 |
| 120 | 240 | False | 1.22e-04 | 1.90e-03 | 366.170 | 1.38e-04 | 2.115 | 15 | 17 / 15 | 6.48 |
| 160 | 280 | True | 7.75e-05 | 6.44e-04 | 370.645 | 2.18e-04 | 2.361 | 129 | 151 / 129 | 63.06 |
| 220 | 320 | True | 4.35e-05 | 8.79e-04 | 370.774 | 2.25e-04 | 2.490 | 41 | 60 / 41 | 44.13 |

### 5B. Pseudospectral Collocation LGL Node Count (`N_LGL`)

| LGL Nodes `N` | Converged? | DOP853 Re-prop ε_M | 180-LGL Defect RMS | Fuel (kg) | Pos RMS vs Ref (AU) | Fuel Diff vs Ref (kg) | IPOPT Iters | Func / Jac Evals | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 30 | False | 4.56e-03 | 1.55e-03 | 370.917 | 2.47e-03 | 2.633 | 70 | 470 / 142 | 0.40 |
| 60 | False | 1.27e+00 | 1.17e+01 | 358.907 | 1.30e+00 | 9.377 | 70 | 274 / 142 | 2.66 |
| 90 | True | 1.61e-04 | 2.74e-04 | 368.479 | 1.69e-04 | 0.194 | 51 | 124 / 104 | 3.32 |
| 120 | True | 1.05e-04 | 3.01e-04 | 368.430 | 7.16e-05 | 0.145 | 53 | 128 / 108 | 6.08 |
| 180 | True | 4.81e-05 | 3.31e-11 | 368.351 | 4.63e-05 | 0.067 | 48 | 160 / 98 | 14.58 |

## 6. 3-D Inclined Eccentric Keplerian Mars Ephemeris vs Circular Coplanar Mars (tf = 310.0 d, θ_f = 220.0°)

- **Keplerian target state (`a = 1.52371 AU, e = 0.09339, i = 1.850°`):** `r_M = [-1.2065, -1.0124, 0.0084] AU`, `v_M = [0.5535, -0.5536, -0.0252] V_E`.

| Ephemeris Model | Method | Converged? | Fuel (kg) | Terminal ε_M | DOP853 Re-prop ε_M | 14-ODE RMS | Runtime (s) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Circular Coplanar (1.524 AU) | X-TFC PoNN | False | 214.917 | 0.00e+00 | 1.91e-02 | 1.13e-04 | 41.65 |
| Circular Coplanar (1.524 AU) | Indirect shooting | True | 216.787 | 3.42e-11 | 9.50e-04 | 6.18e-04 | 3.64 |
| Circular Coplanar (1.524 AU) | Pseudospectral collocation | True | 216.805 | 0.00e+00 | 1.52e-04 | 1.82e+01 | 19.86 |
| 3-D Eccentric Keplerian (J2000) | X-TFC PoNN | False | 335.183 | 0.00e+00 | 1.83e-02 | 1.76e-04 | 42.56 |
| 3-D Eccentric Keplerian (J2000) | Indirect shooting | True | 333.370 | 2.46e-11 | 9.33e-04 | 6.49e-04 | 3.04 |
| 3-D Eccentric Keplerian (J2000) | Pseudospectral collocation | True | 333.418 | 0.00e+00 | 1.35e-04 | 3.33e+01 | 19.01 |

---

## Reproducing All Results and Running Tests

```bash
# Run the core verification and benchmark pipeline
.venv/bin/python run_all.py

# Run Phase 1 (Global feasibility scan + multi-tf 20-run protocol)
.venv/bin/python scripts/run_phase1.py

# Run Phase 2 (100-seed random-initialization basin study + hybrid test)
.venv/bin/python scripts/run_phase2.py

# Regenerate all tables, figures, REPORT/README/ASSUMPTIONS/DEVIATIONS from stored data
.venv/bin/python run_all.py

# Run the unit test suite
.venv/bin/pytest tests/ -v
```

See [`REPORT.md`](REPORT.md) for the claim-by-claim audit, [`ASSUMPTIONS.md`](ASSUMPTIONS.md) for all mathematical and modeling assumptions, and [`DEVIATIONS.md`](DEVIATIONS.md) for all software and numerical deviations.

