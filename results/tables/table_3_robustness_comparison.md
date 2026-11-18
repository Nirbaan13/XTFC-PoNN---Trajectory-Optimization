# Table 3: Success rate and spread over 20 repeated runs (mean ± SD of converged runs)

| Method | Converged runs | Converged Fuel (kg) | Flight time (days) | Arrival residual | Mean Runtime (min) | All-Runs Mean Fuel (kg) | All-Runs Mean Residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| X-TFC PoNN | 0 / 20 | N/A (0 conv) | N/A | N/A | 0.029 ± 0.004 | 331.7 ± 19.33 | 8.3e-10 ± 5.3e-10 |
| Indirect Shooting | 0 / 20 | N/A (0 conv) | N/A | N/A | 0.266 ± 0.093 | 349.3 ± 11.10 | 8.9e-02 ± 1.5e-03 |
| Pseudospectral Collocation | 0 / 20 | N/A (0 conv) | N/A | N/A | 0.048 ± 0.001 | 352.3 ± 0.00 | 8.5e-02 ± 3.4e-15 |

*Note on Convergence:* With nominal thrust strictly capped at $T \le 0.50$ N at $t_f \approx 248$ days, 0 of 20 runs achieve the convergence threshold $\varepsilon_M < 10^{-6}$ because circular Mars rendezvous physically requires $T_{\text{peak}} \ge 0.6745$ N (or $t_f \ge 286$ days). The solvers honestly terminate at the minimum physical distance achievable under the 0.50 N cap (burning ~352 kg of fuel).*