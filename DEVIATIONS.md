# DEVIATIONS.md: Differences from Paper Claims & Implementations

This document explicitly identifies all discrepancies, refinements, or algorithmic differences between the original paper text and the genuine computational implementation in this repository.

---

## 1. Computational Frameworks & Software

| Item | Original Paper Statement | Actual Implementation in Repo | Reason for Deviation |
| :--- | :--- | :--- | :--- |
| **Collocation NLP Framework** | "using the Dymos software with the IPOPT optimizer" | **CasADi (v3.8.2) + IPOPT (v3.14.19)** with MUMPS linear solver | Dymos requires pyOptSparse and custom Fortran compilers. CasADi provides official precompiled IPOPT with C-compiled automatic differentiation, solving the identical LGL collocation NLP on 180 nodes. |
| **Collocation Objective & Formulation** | Single-stage optimization minimizing propellant | **Two-stage collocation**: Stage 1 solves energy-optimal problem as initial guess; Stage 2 solves true fuel-optimal NLP minimizing propellant $(m_0 - m_f)$ with explicit mass state $m(t_k)$ and strict $m(t_k)\|\mathbf{u}_k\| \le 0.50\text{ N}$ | Direct fuel-optimal collocation with bang-bang switching exhibits severe local minima and Jacobian singular arcs without warm-starting; Stage 1 energy-optimal profile provides an unbiased initial guess. |
| **Shooting Solver Root-Finder** | "using the SciPy modules solve_ivp and fsolve" | **`scipy.integrate.solve_ivp` (DOP853) + Levenberg-Marquardt root-finder** with $\varepsilon$-homotopy | Plain `fsolve` (Powell's hybrid dogleg) diverges due to extreme sensitivity of the 7 initial costates; Levenberg-Marquardt with continuation stages provides quadratic curvature convergence. |
| **X-TFC Neural Architecture** | "Python tfc module ... 4,300 epochs" | **TFC Constrained Expressions + Chebyshev Basis with Levenberg-Marquardt** | The `tfc` package is a research wrapper around Chebyshev/CPNs; here we implement the exact analytical TFC Hermite constrained expressions and Chebyshev basis in vectorized NumPy for speed, stability, and transparency. Evaluates the full 14-residual norm $\|R_{14}\|$. |

---

## 2. Solver Runtimes & Efficiency

| Method | Paper Stated Runtime | Actual Measured Wall-Clock Runtime | Reason for Deviation |
| :--- | :---: | :---: | :--- |
| **X-TFC PoNN** | $\approx 18.6\text{ minutes}$ | **$\approx 2.8\text{ seconds}$** | Vectorized matrix operations with BLAS/LAPACK and Levenberg-Marquardt quadratic curvature steps achieve loss minimization in seconds. |
| **Indirect Shooting** | $\approx 24.0\text{ minutes}$ | **$\approx 14.3\text{ seconds}$** | Vectorized 14-state ODE integration using high-order `DOP853` with continuation and Levenberg-Marquardt converges in seconds. |
| **Pseudospectral Collocation** | $\approx 12.2\text{ minutes}$ | **$\approx 60.0\text{ seconds}$** | CasADi generates exact C-compiled sparse derivatives and uses MUMPS sparse linear algebra on 180 LGL nodes. |

---

## 3. Physical & Astrodynamical Discrepancies

| Parameter / Constraint | Paper Stated Claim | Real Computed Result | Astrodynamical Analysis |
| :--- | :---: | :---: | :--- |
| **Maximum Thrust Bound** | $\max T \le 0.50\text{ N}$ for $t_f \approx 248\text{ days}$ | Strictly enforced $\max T \le 0.50\text{ N}$; transfer **falls short of Mars by $0.085\text{--}0.089\text{ AU}$** ($1.3 \times 10^7\text{ km}$) | Full parametric sweep over arrival angles $\theta_f \in [2.0, 5.0]\text{ rad}$ proves the required average thrust is $\bar{T} \ge 0.541\text{ N} > 0.50\text{ N}$ across all angles. An engine limited to $0.50\text{ N}$ cannot physically deliver the necessary impulse in $248\text{ days}$. Bounded thrust rendezvous requires $t_f \ge 325\text{ days}$. |
| **Mars Arrival Phase Angle** | Unstated / assumed fixed | Swept across $[2.0, 5.0]\text{ rad}$; global minimum fuel angle is **$\theta_f^* = 3.200\text{ rad} (183.3^\circ)$** | Feasible angle range for $T \le 0.50\text{ N}$ at $t_f \approx 248\text{ days}$ is **Empty set ($\emptyset$)**. |
| **Heliocentric Xenon Fuel** | $274.8\text{--}276.1\text{ kg}$ | **$352.3\text{ kg}$ (Collocation), $353.0\text{ kg}$ (Shooting)** under $0.50\text{ N}$ cap ($228.4\text{ kg}$ if thrust unbounded) | Under strictly bounded $0.50\text{ N}$ thrust, both physical solvers burn full available throttle ($\approx 352\text{--}353\text{ kg}$) attempting to reach Mars. The paper's narrow band of $274.8\text{--}276.1\text{ kg}$ was an artifact of artificial hardcoding. |
| **Mars Terminal Residual** | $3.2 \times 10^{-7}\text{ to }6.5 \times 10^{-7}$ (all $< 10^{-6}$) | **$8.5 \times 10^{-2}$ (Collocation), $8.9 \times 10^{-2}$ (Shooting)** | Because $0.50\text{ N}$ thrust is physically insufficient to achieve circular rendezvous in $248\text{ days}$, both physical solvers terminate at the closest achievable distance, properly reporting non-convergence. |
| **Robustness Success Rate** | 19/20 (X-TFC), 14/20 (Shooting), 20/20 (Collocation) | **0 / 20 converged** across all three methods under $T \le 0.50\text{ N}$ | With $T \le 0.50\text{ N}$ strictly enforced, no run can satisfy $\varepsilon_M < 10^{-6}$ in $248\text{ days}$. The paper's "14 of 20" was produced by a hardcoded `unlucky_seeds` list. |
| **Table E2 (Escape Solved by 3 Methods)** | Three solvers solved optimal escape with slight differences | Escape is a prescribed tangential spiral, not an optimal control solve | Discretizing 3,000+ revolutions is computationally intractable for standard indirect shooting or direct collocation; the values in the original paper were manual offsets from Table E1. |
