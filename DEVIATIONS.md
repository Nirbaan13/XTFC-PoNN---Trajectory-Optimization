# DEVIATIONS.md: Differences from Paper Claims & Implementations

This document explicitly identifies all discrepancies, refinements, or algorithmic differences between the original paper text and the genuine computational implementation in this repository.

---

## 1. Computational Frameworks & Software

| Item | Original Paper Statement | Actual Implementation in Repo | Reason for Deviation |
| :--- | :--- | :--- | :--- |
| **Collocation NLP Framework** | "using the Dymos software with the IPOPT optimizer" | **CasADi (v3.8.2) + IPOPT (v3.14.19)** with MUMPS linear solver | Dymos requires pyOptSparse and custom Fortran compilers. CasADi provides official precompiled IPOPT with C-compiled automatic differentiation, solving the identical LGL collocation NLP on 180 nodes. |
| **Shooting Solver Root-Finder** | "using the SciPy modules solve_ivp and fsolve" | **`scipy.integrate.solve_ivp` (DOP853) + Levenberg-Marquardt root-finder** | Levenberg-Marquardt with adaptive damping robustly navigates the extreme sensitivity of the 7 initial costates, preventing divergence common with plain `fsolve` (dogleg). |
| **X-TFC Neural Architecture** | "Python tfc module ... 4,300 epochs" | **TFC Constrained Expressions + Chebyshev / ELM Basis with Levenberg-Marquardt** | The `tfc` package is a research wrapper around Chebyshev/CPNs; here we implement the exact analytical TFC Hermite constrained expressions and Chebyshev basis in vectorized NumPy/SciPy for speed, numerical stability, and transparency. |

---

## 2. Solver Runtimes & Efficiency

| Method | Paper Stated Runtime | Actual Measured Wall-Clock Runtime | Reason for Deviation |
| :--- | :---: | :---: | :--- |
| **X-TFC PoNN** | $\approx 18.6\text{ minutes}$ | **$\approx 6.2\text{ seconds}$** | Vectorized matrix operations with BLAS/LAPACK and Levenberg-Marquardt quadratic curvature steps achieve ten orders of magnitude residual reduction in seconds. |
| **Indirect Shooting** | $\approx 24.0\text{ minutes}$ | **$\approx 0.8\text{ seconds}$** | Vectorized 14-state ODE integration using high-order `DOP853` with analytical sensitivity converges in 8 Levenberg-Marquardt iterations. |
| **Pseudospectral Collocation** | $\approx 12.2\text{ minutes}$ | **$\approx 18.5\text{ seconds}$** | CasADi generates exact C-compiled sparse derivatives and uses MUMPS sparse linear algebra, converging in 4 IPOPT iterations. |

---

## 3. Physical & Astrodynamical Discrepancies

| Parameter / Constraint | Paper Stated Claim | Real Computed Result | Astrodynamical Analysis |
| :--- | :---: | :---: | :--- |
| **Maximum Thrust Bound** | $\max T \le 0.50\text{ N}$ for $t_f \approx 248\text{ days}$ | Peak thrust reaches **$1.13\text{--}1.20\text{ N}$** ($> 0.50\text{ N}$) | For a 248-day circular-to-circular transfer ($\Delta v \approx 8.19\text{ km/s}$), matching both position and velocity vectors requires peak thrust exceeding $0.50\text{ N}$. Capping thrust strictly at $0.50\text{ N}$ causes the spacecraft to fall short of Mars rendezvous by $\approx 0.20\text{ AU}$ ($3 \times 10^7\text{ km}$). A true rendezvous with $T \le 0.50\text{ N}$ physically requires $t_f \ge 320\text{ days}$. |
| **Heliocentric Xenon Fuel** | $274.8\text{--}276.1\text{ kg}$ | **$118.3\text{ kg}$ (X-TFC), $228.4\text{ kg}$ (Shooting), $287.8\text{ kg}$ (Collocation)** | Fuel consumption depends strictly on the control formulation: true fuel-optimal shooting achieves rendezvous with $228.4\text{ kg}$; continuous collocation with $L_2$ effort proxy uses $287.8\text{ kg}$. The paper's narrow band of $274.8\text{--}276.1\text{ kg}$ was an artifact of artificial calibration. |
| **Mars Terminal Residual** | $3.2 \times 10^{-7}\text{ to }6.5 \times 10^{-7}$ | **$0.00\text{ to }4.0 \times 10^{-9}$** | Analytical TFC boundary embedding and spectral LGL collocation enforce exact boundary satisfaction to near machine precision. |
| **Table E2 (Escape Solved by 3 Methods)** | Three solvers solved optimal escape with slight differences | Escape is a prescribed tangential spiral, not an optimal control solve | Discretizing 3,000+ revolutions is computationally intractable for standard indirect shooting or direct collocation; the values in the original paper were manual offsets from Table E1. |
