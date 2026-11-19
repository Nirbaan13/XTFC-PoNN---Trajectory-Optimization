# Scientific Verification and Claim-by-Claim Audit Report

This report is generated programmatically from the stored numerical outputs of the solver suite. Every claim in the manuscript is compared directly against the values computed by the code, and each status label (`SUPPORTED`, `PARTIAL`, or `CONTRADICTED`) is assigned numerically from the data.

## 1. Executive Summary of Findings

1. **Escape Phase with $J_2$ (Table E1) and Perturbation Scaling (Table 2):** 3-D Cartesian DOP853 propagation (`rtol = 1e-10, atol = 1e-11`) from a 500 km parking orbit at $i = 28.5^\circ$ with $T = 0.50\text{ N}$ and $m_0 = 1,538.6\text{ kg}$ reaches parabolic escape energy ($E = 0$) in **225.58 days**, consuming **320.56 kg** of xenon and arriving at **1218.04 kg** (paper: 225.6 days, 320.6 kg, 1,218.0 kg → **SUPPORTED**). Table 2 perturbation ratios ($J_2 / a_T = 36.2$ at 500 km; $a_{\text{SRP}} = 1.09e-07\text{ m/s}^2$ at 1 AU) match the paper's analytical formulas (**SUPPORTED**).
2. **Minimum-Time Heliocentric Feasibility at $T_{\max} = 0.50\text{ N}$:** Solving the 180-node LGL minimum-time rendezvous problem with free transfer duration $t_f$ and free Mars arrival longitude $\theta_f$ (with $m_0 = 1,218.0\text{ kg}, I_{\text{sp}} = 3,100\text{ s}, T \le 0.50\text{ N}$) yields a global minimum transfer time of **$t_{\min} = 260.8333\text{ days}$** at arrival longitude **$\theta_f = 192.5175^\circ$** (3.360065 rad), consuming **370.65 kg** of xenon.
3. **Infeasibility of the Paper's Nominal $t_f = 247.9\text{ days}$ Scenario Under $T \le 0.50\text{ N}$:** Because $t_f = 247.9\text{ days} < t_{\min} = 260.83\text{ days}$, no trajectory with $T \le 0.50\text{ N}$ can satisfy both Earth departure and circular Mars rendezvous in 247.9 days. Across the bounded-thrust angle sweep at $t_f = 247.9\text{ days}$, **0 of 9** tested arrival angles converge (minimum slack norm `3.102e-02` at $\theta_f = 185.0^\circ$). In the 20-run robustness protocol at $t_f = 247.9\text{ days}$, convergence rates are **0/20** (X-TFC PoNN), **0/20** (Indirect shooting), and **0/20** (Pseudospectral collocation) (**CONTRADICTED** vs paper's 19/20, 14/20, 20/20).
4. **Shortest All-Method Converged Duration ($t_f = 261.0\text{ days}, \theta_f = 192.52^\circ$):** At $t_f = 261.0\text{ days}$ (just above $t_{\min} = 260.83\text{ days}$), all three solvers solve the identical 14-state fuel-optimal problem under $T \le 0.50\text{ N}$: Indirect shooting consumes **368.28 kg** ($\varepsilon_M = 2.82e-11$), 180-node Pseudospectral collocation consumes **368.35 kg** (terminal $\varepsilon_M = 0.00e+00$, DOP853 re-propagated $\varepsilon_M = 4.81e-05$, zero slack), and X-TFC PoNN consumes **370.65 kg** (14-ODE RMS residual `7.75e-05`, DOP853 re-propagated $\varepsilon_M = 6.44e-04$). Notice that near $t_{\min}$, the spacecraft must thrust near full throttle for almost the entire transfer, burning $\approx 368.3\text{--}370.6\text{ kg}$ of xenon (exceeding the paper's 300 kg nominal xenon budget and 274.8–276.1 kg claim).

## 2. Claim-by-Claim Numerical Audit Table

| Section / Table | Paper Claim | Computed Result (Paper Scenario tf=247.9 d) | Computed Result (Feasible Scenario tf=261.0 d) | Status (247.9 d / 261.0 d) |
| --- | --- | --- | --- | :---: |
| Table E1 (i=28.5°) | t_esc = 225.6 d, Δm = 320.6 kg, m_f = 1,218.0 kg | 225.58 d, 320.56 kg, 1218.04 kg | (Phase 1 identical) | **SUPPORTED / SUPPORTED** |
| Table 2 (J2 500 km) | a_J2 = 1.18e-2 m/s², a_J2/a_T = 36.2 | 1.18e-02 m/s², ratio = 36.2 | (Identical) | **SUPPORTED** |
| Table 2 (SRP 1 AU) | a_SRP = 1.09e-7 m/s², K/μ_S = 1.85e-5 | 1.09e-07 m/s², K/μ_S = 1.85e-05 | (Identical) | **SUPPORTED** |
| Table 1 Heliocentric Convergence | All 3 methods converge at tf ≈ 248 d, T ≤ 0.50 N | Conv: XT=False, SH=False, PS=False (t_min=260.83 d) | Conv: XT=True, SH=True, PS=True | **CONTRADICTED / SUPPORTED** |
| Table 1 Fuel: X-TFC PoNN | 274.8 kg | 295.54 kg (unconverged, ODE RMS=1.71e-01) | 370.65 kg | **CONTRADICTED / CONTRADICTED** |
| Table 1 Fuel: Indirect shooting | 276.1 kg | 352.27 kg (unconverged, ε_M=3.48e-02) | 368.28 kg | **CONTRADICTED / CONTRADICTED** |
| Table 1 Fuel: Collocation (N=180) | 275.3 kg | 340.84 kg (unconverged, slack=3.17e-02) | 368.35 kg | **CONTRADICTED / CONTRADICTED** |
| Table 1 Terminal ε_M: X-TFC | 3.2e-7 | Exact BCs: 0.00e+00 (re-prop: 6.50e-01) | Exact BCs: 0.00e+00 (re-prop: 6.44e-04) | **CONTRADICTED / CONTRADICTED** |
| Table 1 Terminal ε_M: Shooting | 6.5e-7 | 3.48e-02 | 2.82e-11 | **CONTRADICTED / SUPPORTED** |
| Table 1 Terminal ε_M: Collocation | 4.0e-7 | 3.17e-02 (re-prop: 3.22e-02) | 0.00e+00 (re-prop: 4.81e-05) | **CONTRADICTED / SUPPORTED** |
| Table 1 Energy Error ε_E | XT: 8e-7, SH: 1.2e-6, PS: 6e-7 | XT: 1.95e-01, SH: 5.80e-05, PS: 2.55e-03 | XT: 7.80e-04, SH: 3.31e-04, PS: 7.88e-05 | **CONTRADICTED / CONTRADICTED** |
| Table 1 X-TFC Loss & Epochs | 4,300 epochs, L0=1.2e-1 → Lf=2.8e-8 | 15 LM iters, L0=3.08e-01 → Lf=2.91e-02 | 129 LM iters, L0=4.10e-09 → Lf=6.00e-09 | **CONTRADICTED / SUPPORTED** |
| Table 1 & 3 Runtime (min) | XT: 18.6, SH: 24.0, PS: 12.2 min | XT: 0.380, SH: 0.229, PS: 0.982 min | XT: 1.066, SH: 0.084, PS: 0.202 min | **CONTRADICTED / CONTRADICTED / CONTRADICTED** |
| Table 3 Convergence (20 runs) | XT: 19/20, SH: 14/20, PS: 20/20 | XT: 0/20, SH: 0/20, PS: 0/20 | XT: 10/20, SH: 20/20, PS: 13/20 | **CONTRADICTED / PARTIAL** |

## 3. Summary of Angle-Sweep, Unperturbed Control, and Perturbation Sensitivity

- **Bounded-Thrust Angle Sweep:** At $t_f = 247.9\text{ days}$, 0/9 angles converge. At $t_f = 261.0\text{ days}$, 2/5 angles converge under both indirect shooting and 180-node collocation.
- **Unperturbed Control (`srp_scale = 0` at $t_f = 261.0\text{ days}$):** Removing SRP changes propellant consumption by `-0.0473 kg` (Indirect shooting), `-0.0458 kg` (Collocation), and `+0.0039 kg` (X-TFC PoNN), consistent with the analytical ratio $K_{\text{SRP}} / \mu_S = 1.8455e-05$.
- **Perturbation-Strength Sweep (`srp_scale` from 0× to 1000×):** Propellant consumption varies smoothly from `368.3319 kg` at 0× SRP to `370.8869 kg` at 1000× SRP.

