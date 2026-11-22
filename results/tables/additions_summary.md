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
