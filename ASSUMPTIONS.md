# Assumptions & Paper Ambiguity Log

This document records every modeling assumption made where D'Ambrosio & Furfaro (*Aerospace* **2024**, 11(3), 228) are silent or ambiguous, plus secondary Scenario B modeling assumptions. No assumption was tuned toward paper result numbers.

## Scenario A (D'Ambrosio & Furfaro 2024) — Explicit Resolution of 8 Paper Ambiguities

1. **Chebyshev basis indexing ($\sigma_q = T_{q-1}$ vs. $T_q$) and TFC support-polynomial linear dependence:**
   - Paper Table 2 specifies $\sigma_q(z) = T_{q-1}(w_q z + b_q)$ with $w_q = 1, b_q = 0$ for $q = 1, \dots, L$.
   - Substituting $T_0, T_1, T_2, T_3$ into the Table A6 4-constraint Hermite CE for $r_i(z)$ yields four identically zero columns ($\Phi_r T_k \equiv 0$ for $k=0..3$), and substituting $T_0$ into the Table A1 1-constraint CEs for $m(z)$ and $\lambda_m(z)$ yields identically zero columns ($\Phi_m T_0 = \Phi_{\lambda_m} T_0 \equiv 0$).
   - **Assumption & ablation:** Standard TFC theory requires the free function to be linearly independent of the support polynomials. Our primary validated mode (`chebyshev_shifted`) shifts the free basis to $T_4 \dots T_{L+3}$ for $r_i$, $T_1 \dots T_L$ for $m, \lambda_m$, and $T_0 \dots T_{L-1}$ for $\lambda_v$. We also retain and benchmark the literal $T_0 \dots T_{L-1}$ indexing (`chebyshev`) in Phase A4(b) (`10/10` at `396.52 kg` shifted vs. `9/10` at `399.45 kg` literal).

2. **Solver tolerances:**
   - The original paper specifies Eq. (27) $\|\mathbf{L}\|_2 < \varepsilon$; numerical values for `FunctionTolerance`, `StepTolerance`, and `OptimalityTolerance` are not reported in the original paper.
   - **Assumption:** We set `ftol = 1e-9`, `xtol = 1e-9`, `gtol = 1e-9` uniformly across all PoNN continuation levels and solvers (`lm_svd`, SciPy `'lm'`, SciPy `'trf'`), and `rtol = 1e-10, atol = 1e-12, level_tol = 1e-8` for indirect shooting, `ipopt.tol = 1e-7` for 180-node LGL collocation.

3. **Maximum iterations per $\rho$ continuation level:**
   - Per-level `MaxIterations` and `MaxFunctionEvaluations` for MATLAB `lsqnonlin` are not reported in the original paper.
   - **Assumption:** We allow `max_iter_first = 250` (for unscaled `paper_u01`, matching MATLAB's default iteration order of magnitude) or `100` (for `scaled_u01`) at $\rho = 1.0$, and `max_iter_cont = 25` on subsequent warm-started continuation levels ($k \ge 1$), reducing to `5` iterations if the prior level remained diverged (`rms > 0.03` and $\min m < 0.08$).

4. **Refinement pass ($L = 80$) initialization:**
   - Sec. 4.1 states that after identifying switch intervals, 100 points per switch interval are added, $L$ is increased to $80$, and X-TFC is run again; weight initialization for the $L=80$ refinement pass is not reported in the original paper.
   - **Assumption:** We warm-start the $L=80$ refined basis by zero-padding the smooth coarse $L=70$ Chebyshev coefficients captured at $\rho \approx 0.0886$ (before coarse-grid switch aliasing develops) and running the remaining continuation levels down to $\rho = 10^{-10}$. For `tanh_elm` (where basis functions change with $L$), the smooth coarse trajectory is projected onto the $L=80$ ELM basis via least squares.

5. **Placement of the 100 extra discretization points per switch interval:**
   - The bracket width and distribution of the 100 points per switch interval are not reported in the original paper.
   - **Assumption & ablation:** We tested both literal unweighted bracket clustering (`switch_bracket`: 100 points inside each sign-change bracket $[z_k, z_{k+1}]$) and global redistribution of the same total refined node count $n_{\text{ref}} = n + 100\,n_{\text{sw}} = 540$ uniformly (`uniform_dense`) and on CGL nodes (`cgl_dense`). Because unweighted narrow-bracket clustering causes the optimizer to dodge the dense brackets (`400.63 kg`, `reprop = 1.07e-1`) in our independent implementation, we adopt `uniform_dense` ($n=540$, `396.52 kg`, `reprop = 3.63e-2`) as our primary refined discretization and report all three modes in Phase A3.

6. **Switch detection rule:**
   - An explicit numerical zero-crossing or debouncing algorithm for $S(t)$ is not reported in the original paper.
   - **Assumption:** For indirect methods (Shooting and PoNN), switches are detected by linear interpolation of every sign change of $S(t)$ evaluated on a dense 2001-point uniform grid. For direct LGL-180 collocation (which has no costate $S$), switches are detected from crossings of $\delta(t) = 0.5$, excluding 1-day endpoint quadrature artifacts ($t < 1\text{ d}$ or $t > t_f - 1\text{ d}$).

7. **Nondimensional costate units & Table 1 canonical scaling:**
   - Table 1 gives $R_{\text{ref}} = 149.59787\times 10^6\text{ km}$, $t_{\text{ref}} = 58.132\text{ d}$, $m_{\text{ref}} = 1000\text{ kg}$, and $\mu_\odot = 1.32712440018\times 10^{11}\text{ km}^3/\text{s}^2$, yielding $\mu_{\text{nd}} = 0.9999848462$ and $t_{f,\text{nd}} = 6.0000516067$.
   - **Assumption:** We use the exact dimensional Table 1 constants normalized by $(R_{\text{ref}}, t_{\text{ref}}, m_{\text{ref}})$. With $J = \int (T_{\max}\delta / c)\,dt$ in canonical units, $\lambda_m$ is dimensionless ($\in [0, 1]$), $\boldsymbol{\lambda}_v$ has units $m_{\text{ref}}/V_{\text{ref}}$, and $\boldsymbol{\lambda}_r$ has units $m_{\text{ref}}/R_{\text{ref}}$. Initial $\boldsymbol{\beta}_{\lambda_m}$ is fit to $\lambda_m \approx 1$ on interior nodes ($z < 1 - 10^{-12}$) with truncated SVD (`rcond = 1e-4`) so $\lambda_m(z_f)=0$ remains exact without an initial boundary spike.

8. **SciPy `'lm'`/`'trf'` and QR/Cholesky SVD-LM vs. MATLAB R2023b `lsqnonlin`:**
   - The original paper uses MATLAB R2023b `lsqnonlin` (`'levenberg-marquardt'` in Sec. 4.1, `'trust-region-reflective'` in Sec. 4.2) with analytical/AD Jacobians.
   - **Assumption:** We compute the exact analytical $8n \times 8L$ Jacobian via JAX automatic differentiation and benchmark SciPy MINPACK `'lm'`, SciPy `'trf'` (`x_scale='jac'`), and an OpenBLAS Cholesky-with-QR+SVD-fallback Levenberg–Marquardt solver (`lm_svd`) implementing MATLAB `lsqnonlin`'s default `ScaleProblem='none'` on step 0 and Jacobian column scaling on continuation steps. Phase A4(d) verifies that all three optimizers converge to identical coarse trajectories (`400.70 kg`, `rms = 2.06e-3`), while `lm_svd` is ~7–10x faster.

## Preregistered Benchmark Protocol (Seeds 2000..2099)

- Evaluation protocol, extraction level $k_{\text{ext}} = 1$ ($\rho_{\text{ext}} = 2.976\times 10^{-1}$, selected on dev seeds 1000..1019 only), dual thresholds ($e_{\text{reprop}} < 5\times 10^{-2}$ loose, $< 1\times 10^{-2}$ strict), and exact McNemar decision rules locked in `PREREG.md` at commit `3c87713` before running seeds 2000..2099.
- Costates extracted at $t_0$ via:
  $\boldsymbol{\lambda}_v(t_0) = \boldsymbol{\sigma}(-1)^T \boldsymbol{\beta}_{\lambda_v}$,
  $\boldsymbol{\lambda}_r(t_0) = -b^2 \boldsymbol{\sigma}'(-1)^T \boldsymbol{\beta}_{\lambda_v}$,
  $\lambda_m(t_0) = (\boldsymbol{\sigma}(-1) - \boldsymbol{\sigma}(1))^T \boldsymbol{\beta}_{\lambda_m}$.
- All model parameters, dynamics, bases, and solvers remained strictly frozen with zero algorithmic alterations.

## Additional Modeling Assumptions (Phase C & Scenario B)

- **Prior literature context:** Perturbation-aware PoNNs exist (J2 formation flying, verified abstract; cislunar high-fidelity ephemeris, abstract only; perturbation list unverified).
- **Phase C attitude-dependent SRP (`attitude_flat_plate`):** Modeled as a flat absorbing/reflecting plate whose surface normal aligns with the optimal thrust unit vector $\boldsymbol{\alpha} = -\boldsymbol{\lambda}_v/\|\boldsymbol{\lambda}_v\|$, yielding projected area factor $\phi(\mathbf{r}, \boldsymbol{\alpha}) = 0.5 + 0.5(\hat{\mathbf{r}}\cdot\boldsymbol{\alpha})^2$ and radial SRP acceleration $\mathbf{a}_p = k_{\text{SRP}}\phi\,\mathbf{r}/r^3$. Since $\boldsymbol{\alpha}$ depends only on $\boldsymbol{\lambda}_v$, the position Jacobian $\partial \mathbf{a}_p/\partial \mathbf{r}$ is exact and verified by central FD (`< 1e-9`).
- **Scenario B (secondary):** Retained unchanged from earlier workspace work ($m_0 = 1218\text{ kg}$, $T_{\max} = 0.5\text{ N}$, $I_{\text{sp}} = 3100\text{ s}$, circular coplanar Earth/Mars orbits at $1.0$ and $1.524\text{ AU}$, radial SRP). In Phase D, the frozen Phase A2 PoNN is evaluated on Scenario B with zero retuning.

## Complete Flat-Plate SRP Model Specification (Tier 1.3)

- **Coordinate frame:** Heliocentric J2000 inertial Cartesian frame.
- **Attitude coupling:** **Prescribed** thrust-aligned normal $\hat{\mathbf{n}} = \boldsymbol{\alpha}^* = -\boldsymbol{\lambda}_v / \|\boldsymbol{\lambda}_v\|$ (the solar array / bus normal tracks the optimal thrust direction; attitude is **not** independently optimized as a separate control variable).
- **Eclipse / shadowing:** None (continuous heliocentric illumination; no Earth or Mars umbra/penumbra eclipses).
- **Projected-area law:** $\phi(\mathbf{r}, \boldsymbol{\alpha}) = \tfrac{1}{2} + \tfrac{1}{2}(\hat{\mathbf{r}}\cdot\boldsymbol{\alpha})^2$, with $\mathbf{a}_{\mathrm{SRP}}(\mathbf{r}, \boldsymbol{\alpha}) = k_{\mathrm{SRP}}\,\phi(\mathbf{r}, \boldsymbol{\alpha})\,\mathbf{r}/\|\mathbf{r}\|^3$ and $k_{\mathrm{SRP}} = P_0 C_R (A/m_0)/a_{\mathrm{ref}}$ (constant initial $A/m_0$ cannonball scaling).
- **Escape-phase $J_2$ note:** Earth's oblateness $J_2$ is **NOT** included in any heliocentric trajectory optimization (Scenario A or Scenario B); it appears strictly in the prescribed Phase 1 Earth-departure escape propagation (`escape_phase.py`).
