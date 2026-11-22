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

