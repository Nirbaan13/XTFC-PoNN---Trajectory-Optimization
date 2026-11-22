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

