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

