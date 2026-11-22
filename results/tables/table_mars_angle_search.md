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

