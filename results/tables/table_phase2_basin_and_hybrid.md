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

