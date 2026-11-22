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

