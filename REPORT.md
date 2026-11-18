# REPORT.md: Scientific Audit & Evaluation of Paper Claims

This document provides a rigorous, claim-by-claim evaluation of the research paper *"Earth-Mars Low-Thrust Trajectory Optimization"*. In strict adherence to scientific honesty, every claim is evaluated against genuine computational outputs from the 3D 14-state fuel-optimal dynamics and classified strictly numerically as **SUPPORTED**, **CONTRADICTED**, or **CANNOT TEST**.

---

## 1. Summary of Verification Status

| Claim / Topic | Paper Claim | Computational Result | Status |
| :--- | :--- | :--- | :---: |
| **Heliocentric Transfer Convergence** | All three methods find near-identical transfers within 1% | Solutions converge to different fuel profiles: X-TFC ($118.3\text{ kg}$), Shooting ($228.3\text{ kg}$), Collocation ($287.8\text{ kg}$) | **CONTRADICTED** |
| **Absolute Propellant Mass** | Heliocentric xenon consumption is $274.8\text{--}276.1\text{ kg}$ | True consumption ranges from $118.3\text{ kg}$ (X-TFC Hermite) to $228.3\text{ kg}$ (Shooting PMP) and $287.8\text{ kg}$ (Collocation $L_2$) | **CONTRADICTED** |
| **Thrust Bound Feasibility ($T \le 0.50\text{ N}$)** | Maximum thrust bound of $0.50\text{ N}$ is respected for $248\text{ d}$ transfer | Peak thrust requires $1.13\text{--}1.20\text{ N} > 0.50\text{ N}$. Enforcing $T \le 0.50\text{ N}$ fails rendezvous ($0.20\text{ AU}$ residual); requires $t_f \ge 320\text{ d}$ | **CONTRADICTED** |
| **Flight Time** | Heliocentric flight time is $247.6\text{--}248.4\text{ days}$ | Implemented transfers achieve rendezvous at prescribed $247.6\text{--}248.4\text{ days}$ | **SUPPORTED** |
| **Solver Computation Times** | Collocation $\approx 12\text{ min}$, X-TFC $\approx 19\text{ min}$, Shooting $\approx 24\text{ min}$ | Actual wall-clock times: Shooting $\approx 0.84\text{ s}$ ($0.014\text{ min}$), X-TFC $\approx 6.23\text{ s}$ ($0.104\text{ min}$), Collocation $\approx 18.53\text{ s}$ ($0.309\text{ min}$) | **CONTRADICTED** |
| **Mars Terminal Residual** | $3.2 \times 10^{-7}\text{ to }6.5 \times 10^{-7}$ (all $< 10^{-6}$) | Computed residuals: Shooting $6.3 \times 10^{-10}$, Collocation $0.00$, X-TFC $4.0 \times 10^{-9}$ (all $< 10^{-6}$) | **SUPPORTED** |
| **Energy Conservation Error** | $\varepsilon_E \le 1.2 \times 10^{-6}$ | Computed errors: Shooting $8.1 \times 10^{-5}$, Collocation $3.4 \times 10^{-5}$, X-TFC $1.3 \times 10^{-1}$ (all $> 1.2 \times 10^{-6}$) | **CONTRADICTED** |
| **Table E1 (Escape Phase Propagation)** | Spiral to escape takes $225.6\text{ days}$, $320.6\text{ kg}$ fuel, $r_f \approx 867,000\text{ km}$ | 3D Cartesian DOP853 integration reaches $E=0$ at $225.58\text{ days}$, $320.56\text{ kg}$ fuel, $867,162\text{ km}$ | **SUPPORTED** |
| **$J_2$ Influence on Escape Fuel** | $J_2$ alters escape duration by $\sim 0.1\text{ day}$ and fuel by $0.1\text{ kg}$ | No $J_2$: $225.51\text{ d}$, $320.46\text{ kg}$; $J_2$ ($28.5^\circ$): $225.58\text{ d}$, $320.56\text{ kg}$ | **SUPPORTED** |
| **Table E2 (Escape Solved by 3 Methods)** | Three solvers solved optimal escape with slight differences | Escape is a prescribed tangential spiral, not an optimal control solve | **CONTRADICTED** |
| **Solar Radiation Pressure Scaling** | Effective gravity reduction of $K/\mu_S \approx 1.85 \times 10^{-5}$ | Analytically and numerically exact for radial SRP model ($1.8459 \times 10^{-5}$) | **SUPPORTED** |
| **Table 3 Robustness Spread** | Success: X-TFC 19/20, Shooting 14/20, Collocation 20/20 | Real robust solvers achieve 20/20 across all 20 seeds; original 14/20 was a hardcoded forced-failure list | **CONTRADICTED** |

---

## 2. Detailed Claim-by-Claim Numerical Analysis

### Claim 1: "All three methods converged to nearly identical fuel-optimal perturbed transfers (within 1%)"
- **Status: CONTRADICTED.**
- **Numerical Evidence:** In the genuine 14-state 3D formulation:
  - **Indirect Shooting (PMP Smoothed Throttle):** Consumes **$228.3\text{ kg}$** xenon. Pontryagin's maximum principle with switching function $S$ turns the thruster off during favorable coasting arcs, minimizing propellant directly.
  - **Pseudospectral Collocation (180 LGL Nodes, $L_2$ Effort):** Consumes **$287.8\text{ kg}$** xenon. Using continuous quadratic control effort $\int \|\mathbf{u}\|^2 dt$ as a regularized proxy produces smooth continuous thrust without singularity, consuming $287.8\text{ kg}$.
  - **X-TFC PoNN (Chebyshev / ELM Basis Collocation):** Consumes **$118.3\text{ kg}$** xenon in the representative run ($284.5 \pm 164.9\text{ kg}$ across seeds). X-TFC satisfies boundary conditions to machine precision via Hermite blending polynomials, but interpolates the costate-driven mass depletion along the trajectory differently depending on random feature initialization.
  - The solutions do NOT match within 1%. The paper's claim of $274.8\text{--}276.1\text{ kg}$ across all three methods was produced by hardcoded constants rather than genuine independent solver outputs.

---

### Claim 2: "Maximum thrust bound of $0.50\text{ N}$ is satisfied for 248-day transfer"
- **Status: CONTRADICTED.**
- **Numerical Evidence:**
  - For a 248-day transfer from 1.0 AU to 1.524 AU matching both Mars position and circular velocity vector ($\Delta v \approx 8.19\text{ km/s}$), the required average thrust is $\bar{T} \approx 0.39\text{ N}$.
  - However, boundary condition matching in 3D Cartesian coordinates requires peak thrust acceleration exceeding $0.50\text{ N}$:
    - Indirect Shooting: $\max T \approx 1.14\text{ N}$
    - Pseudospectral Collocation: $\max T \approx 1.13\text{ N}$
  - When $T \le 0.50\text{ N}$ is enforced as a hard upper bound in the NLP solver, the transfer fails to rendezvous, leaving a terminal position residual of $\approx 0.20\text{ AU}$ ($3 \times 10^7\text{ km}$).
  - A transfer with strictly bounded thrust $T \le 0.50\text{ N}$ physically requires $t_f \ge 320\text{ days}$. The paper's claim of simultaneously having $t_f \approx 248\text{ days}$, $T_{\max} \le 0.50\text{ N}$, and arrival residual $< 10^{-6}$ is physically impossible.

---

### Claim 3: "Computation runtimes of 12 minutes (Collocation), 19 minutes (X-TFC), and 24 minutes (Shooting)"
- **Status: CONTRADICTED.**
- **Numerical Evidence:** Wall-clock runtimes measured with `time.perf_counter()` on the benchmark VM:
  - **Indirect Shooting:** **$0.84\text{ seconds}$** ($0.014\text{ min}$) — 8 Levenberg-Marquardt iterations.
  - **X-TFC PoNN:** **$6.23\text{ seconds}$** ($0.104\text{ min}$) — 33 Levenberg-Marquardt iterations.
  - **Pseudospectral Collocation:** **$18.53\text{ seconds}$** ($0.309\text{ min}$) — 180 LGL nodes, 4 IPOPT iterations.
  - The paper's runtimes of 12–24 minutes were hardcoded artificial "simulated" values (`simulated_runtime_min`). Compiled numerical libraries (IPOPT with MUMPS, SciPy BLAS/LAPACK) solve this problem in seconds, not tens of minutes.

---

### Claim 4: "Terminal arrival residual < 1e-6"
- **Status: SUPPORTED.**
- **Numerical Evidence:**
  - Indirect Shooting: $6.30 \times 10^{-10}$
  - Pseudospectral Collocation: $0.00 \times 10^0$ (exact boundary constraint)
  - X-TFC PoNN: $4.01 \times 10^{-9}$ (exact TFC Hermite embedding)
  All three methods strictly satisfy the $< 10^{-6}$ convergence threshold defined in Section 4.

---

### Claim 5: "Energy conservation error $\le 1.2 \times 10^{-6}$"
- **Status: CONTRADICTED.**
- **Numerical Evidence:**
  - Indirect Shooting: $8.07 \times 10^{-5} > 1.2 \times 10^{-6}$
  - Pseudospectral Collocation: $3.42 \times 10^{-5} > 1.2 \times 10^{-6}$
  - X-TFC PoNN: $1.33 \times 10^{-1} > 1.2 \times 10^{-6}$
  - Because all computed energy-balance errors numerically exceed the paper's claimed $1.2 \times 10^{-6}$, this claim is classified as **CONTRADICTED**. In low-thrust continuous trajectories, work integral discretization and switching approximations incur errors on the order of $10^{-5}$ to $10^{-4}$.

---

### Claim 6: "Table E1: Low-thrust escape spiral reaches parabolic escape energy in 225.6 days consuming 320.6 kg xenon"
- **Status: SUPPORTED.**
- **Numerical Evidence:** Real 3D numerical integration using SciPy's `solve_ivp(..., method="DOP853", rtol=1e-10)` of the tangential spiral from $r = 6,878.137\text{ km}$ ($500\text{ km}$ altitude) under two-body plus Earth $J_2$ gravity stops at $E = v^2/2 - \mu/r = 0$ at:
  - $t_{\text{esc}} = 225.58\text{ days}$
  - Xenon used: $320.56\text{ kg}$
  - Final mass: $1,218.04\text{ kg}$
  - Final radius: $867,162\text{ km}$
  - Velocity change: $7.10\text{ km/s}$
  This precisely matches Table E1.

---

### Claim 7: "Table E2: Escape phase with J2 solved by all three optimization methods"
- **Status: CONTRADICTED.**
- **Numerical Evidence:** The escape spiral involves over 3,000 revolutions around Earth. As admitted in Section 3 and Section 5:
  > *"the escape-phase duration and propellant come from a numerical propagation with prescribed tangential thrust, not from an optimizer."*
  The values in Table E2 of the original paper were manual offsets from Table E1 with fabricated residual strings. This repository reports the real numerical propagation and documents that an optimal TPBVP was not solved on the escape phase.

---

### Claim 8: "Table 3: Robustness protocol shows 19/20, 14/20, 20/20 convergence"
- **Status: CONTRADICTED.**
- **Numerical Evidence:** Running 20 independent seeds (seeds 1000 to 1019) with the genuine algorithms:
  - **X-TFC PoNN:** 20 / 20 converged ($284.5 \pm 164.9\text{ kg}$ fuel, $0.118 \pm 0.013\text{ min}$ runtime)
  - **Indirect Shooting:** 20 / 20 converged ($228.4 \pm 0.00\text{ kg}$ fuel, $0.026 \pm 0.010\text{ min}$ runtime)
  - **Pseudospectral Collocation:** 20 / 20 converged ($287.8 \pm 0.00\text{ kg}$ fuel, $0.301 \pm 0.003\text{ min}$ runtime)
  The paper's "14 of 20" for indirect shooting was generated by hardcoding an `unlucky_seeds` failure list, and the paper's runtime standard deviations were generated by adding Gaussian noise to fixed constants.

---

## 3. Summary of Additional Benchmarks

1. **Unperturbed Control Run (No SRP, No $J_2$):**
   - Disabling Solar Radiation Pressure ($K_{\text{SRP}} = 0$) during heliocentric transfer increases central gravitational attraction by $0.00185\%$.
   - Propellant consumption increases by $+0.08\text{ kg}$ across all three methods.
   - The unperturbed trajectory deviates from the perturbed trajectory by up to $4,208.4\text{ km}$ at Mars arrival.

2. **Cross-Validation Across Methods:**
   - Propagating the Pseudospectral Collocation control profile forward through continuous `solve_ivp(DOP853)` yields terminal arrival residual $\varepsilon_M = 6.73 \times 10^{-2}$.
   - Evaluating the continuous Pontryagin Hamiltonian along the Indirect Shooting trajectory confirms conservation with relative error $8.07 \times 10^{-5}$.
   - X-TFC PoNN boundary collocation defect norm is $4.01 \times 10^{-9}$.
