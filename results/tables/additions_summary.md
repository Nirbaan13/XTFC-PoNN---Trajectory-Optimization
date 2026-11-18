# Additions: Unperturbed Controls & Cross-Validation Benchmarks

## 1. Unperturbed Control Run Comparison (No SRP vs Perturbed)
| Method | Perturbed Fuel (kg) | Unperturbed Fuel (kg) | Fuel Difference (kg) | Max Position Diff (km) |
| --- | --- | --- | --- | --- |
| Indirect Shooting | 352.98 | 353.06 | 0.08 | 4,208.4 |
| Pseudospectral Collocation | 352.27 | 352.35 | 0.08 | 4,208.4 |
| X-TFC PoNN | 235.47 | 235.55 | 0.08 | 4,208.4 |

## 2. Cross-Validation Across Methods
- **Pseudospectral Control Forward-Propagated via DOP853 (Solve_ivp)**: Terminal Arrival Residual = 9.617e-02
- **Indirect Shooting Trajectory Collocation Defect Norm (LGL grid)**: 1.562e-02
- **Indirect Shooting Hamiltonian Conservation Error**: 2.562e-04
- **X-TFC PoNN Trajectory Collocation Defect Norm (LGL grid)**: 1.742e-02
- **X-TFC PoNN Hamiltonian Conservation Error**: 1.028e-01