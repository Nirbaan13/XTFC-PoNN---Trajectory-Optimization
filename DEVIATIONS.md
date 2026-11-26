# Deviations from D'Ambrosio & Furfaro (Aerospace 2024, 11, 228)

Every intentional or environment-forced deviation from D'Ambrosio & Furfaro (2024) is logged transparently below:

1. **Software stack (Python/JAX/SciPy/CasADi/IPOPT vs. MATLAB R2023b/GPOPS-II):**
   - MATLAB R2023b and GPOPS-II are proprietary and unavailable in this Linux environment.
   - We implemented the exact mathematical formulation in Python using JAX (64-bit exact analytical Jacobians), SciPy (`DOP853`, `fsolve`, `least_squares`), custom OpenBLAS Cholesky/QR-SVD Levenberg–Marquardt (`lm_svd`), and CasADi + IPOPT (180-node LGL pseudospectral direct collocation).

2. **Coarse collocation grid (CGL $n=140$ vs. evenly spaced $n=140$):**
   - Sec. 2.2 states evenly spaced discretization points ($n=140, L=70$).
   - Direct testing (Phase A3 & A4(d)) revealed severe Runge aliasing on uniform $n=140$ ($\kappa(\Phi_r)=1.67\times 10^8$, node RMS `6.6e-6` vs. dense RMS `1.94`, DOP853 re-propagation error `1.36`, `0/5` convergence).
   - We adopt $n=140$ Chebyshev–Gauss–Lobatto (CGL) nodes for the coarse stage (explicitly noted by the paper as an admissible alternative quadrature scheme), which lowers $\kappa(\Phi_r)$ to `6.23e3` and restores physical convergence, while retaining and reporting literal `uniform_140` results honestly.

3. **Refined grid placement (`uniform_dense` $n=540$ vs. literal narrow switch-bracket clustering):**
   - Sec. 4.1 adds 100 points per switch interval ($n=540$ for 4 switches) and sets $L=80$.
   - Literal unweighted narrow-bracket clustering (`switch_bracket`) caused the solver to shift switches out of the clusters (`400.63 kg`, `reprop = 1.07e-1`).
   - Our primary refined discretization distributes the exact same refined node budget ($n=540$) uniformly (`uniform_dense`, `396.52 kg`, `reprop = 3.63e-2`), and we report `cgl_dense` (`397.53 kg`) and literal `switch_bracket` (`400.63 kg`) side by side.

4. **Random weight initialization scaling (`paper_u01` vs. `scaled_u01`):**
   - Sec. 4.1 draws $\beta_{1..3}, \beta_{\lambda_v 1..3} \sim U(0, 1)$, which yields initial positions at $\sim 25{,}000\text{ AU}$ due to $|T_k'(\pm 1)| = k^2$.
   - Rather than silently changing initialization, we benchmarked **both** exact unscaled `paper_u01` (`49/100` converged) and `scaled_u01` (`1e-3 * U(0,1)` for position, `1e-2 * U(0,1)` for velocity costate; `92/100` converged) over 100 full seeds each.

5. **Scope restriction (Sec. 4.1 Earth–Mars interplanetary only; Sec. 4.2 Mars landing out of scope):**
   - Per project specification, reproduction and benchmarking cover Sections 3.1, 3.3, and 4.1 (Earth–Mars interplanetary transfer), extended with Phase C SRP perturbations and secondary Scenario B transfer tests.

6. **Independent DOP853 re-propagation validation (not reported in paper):**
   - Because TFC constrained expressions satisfy boundary conditions analytically even when interior ODE residuals are nonzero, we re-propagate every solver's open-loop control $(\delta(t), \boldsymbol{\alpha}(t))$ with `DOP853` (`rtol=1e-11, atol=1e-13`) and report actual terminal arrival error alongside energy balance error.

7. **Preregistered held-out testing on fresh seeds (`2000..2099`, `4000..4099`, `5000+`, `6000..6029`, `7000..7099`, `8000..8099`):**
   - Evaluated held-out seed suites with frozen settings and zero post-hoc parameter adjustments under `PREREG.md`, `PREREG_ADDENDUM.md`, and `PREREG_ADDENDUM_2.md`.
