# Preregistration Addendum: Tiers 1–3 Ablations, Multi-Case Generalization, SRP Audit, Plateau & Mesh Studies, and Split-Domain Gate

**Preregistration status:** Committed **before** executing any Tier 1, Tier 2, or Tier 3 experiment on fresh seeds (`4000+`, `5000+`, `6000+`, `7000+`). All previously locked settings in [PREREG.md](PREREG.md) (commit `3c87713`) remain frozen and unchanged. No solver hyperparameters will be retuned on test seeds.

---

## 1. Common Thresholds, Metrics, & Multiple-Testing Correction

- **Open-loop re-propagation error:** Every solver's open-loop control $(\delta(t), \boldsymbol{\alpha}(t))$ is independently integrated from $(\mathbf{r}_0, \mathbf{v}_0, m_0)$ using `scipy.integrate.solve_ivp(method="DOP853", rtol=1e-11, atol=1e-13)` to obtain:
  $$e_{\text{reprop}} = \left\| \begin{bmatrix} \mathbf{r}_{\text{DOP853}}(t_f) - \mathbf{r}_f \\ \mathbf{v}_{\text{DOP853}}(t_f) - \mathbf{v}_f \end{bmatrix} \right\|_2.$$
- **Dual convergence thresholds (with 95% Wilson score CIs):**
  - **Strict:** $e_{\text{reprop}} < 1\times 10^{-2}$ (plus solver-specific physical validity: finite states, $m_f > 50\text{ kg}$; PoNN `residual_rms < 1.5e-2`; Shooting `arrival_residual < 1e-6` and `rho_reached <= 1e-4`; LGL Collocation IPOPT converged status and `terminal_error < 1e-4`).
  - **Loose:** $e_{\text{reprop}} < 5\times 10^{-2}$ (with the same physical validity conditions).
- **Holm–Bonferroni correction across all Tier 1 hypothesis tests ($M = 13$):**
  - The 13 primary paired tests comprise the 6 extraction-level ablation comparisons (`H1a–H1f`) and the 7 extra-physical-case Hybrid vs. Random-Shooting comparisons (`H2a–H2g`).
  - For each test, we compute the exact two-sided McNemar $p$-value on strict convergence ($e_{\text{reprop}} < 10^{-2}$).
  - Raw $p$-values are sorted $p_{(1)} \le \dots \le p_{(M)}$ and Holm-adjusted as:
    $$\tilde{p}_{(i)} = \min\left(1,\; \max_{j \le i} (M - j + 1)\, p_{(j)}\right).$$
- **Numeric verdict rule (applied strictly without override):**
  - **`SUPPORTED`:** $\Delta_{\text{strict}} = p_{\text{hyb}} - p_{\text{shoot}} \ge +0.15$ ($\ge +15.0\text{ pp}$) **AND** Holm-adjusted $\tilde{p} < 0.05$.
  - **`CONTRADICTED`:** Testable case where $\Delta_{\text{strict}} < +0.15$ **OR** $\tilde{p} \ge 0.05$.
  - **`CANNOT TEST`:** Any case failing the pre-run feasibility criterion $t_{\min} \le 0.95\, t_f$.

---

## 2. Tier 1 Experiments (Must-Do)

### 2.1 Extraction-Level Ablation (Scenario A, Fresh Seeds `4000..4099`, $N = 100$)
- **Problem:** Nominal Scenario A ($A/m_0 = 0.0$, $m_0 = 1000\text{ kg}$, $T_{\max} = 0.5\text{ N}$, $t_f = 348.795\text{ d}$).
- **Extraction formulas at $z_0 = -1$ ($t = t_0$):**
  $$\boldsymbol{\lambda}_v(t_0) = \boldsymbol{\sigma}(-1)^T \boldsymbol{\beta}_{\lambda_v}, \qquad \boldsymbol{\lambda}_r(t_0) = -b^2 \boldsymbol{\sigma}'(-1)^T \boldsymbol{\beta}_{\lambda_v}, \qquad \lambda_m(t_0) = (\boldsymbol{\sigma}(-1) - \boldsymbol{\sigma}(1))^T \boldsymbol{\beta}_{\lambda_m}.$$
- **Paired arms (7 arms on identical seeds `4000..4099`):**
  - Baseline: `random_shooting` (`ScenarioAShootingSolver(stop_at_rho=1e-5)`).
  - `H1a` ($\rho_{\text{ext}} = 1.0$): PoNN coarse ($L=70, n=140$ CGL, `scaled_u01`) through $\rho = 1.0$; shooting continues `DEFAULT_RHO_SCHEDULE` from $1.0$ down to `1.833e-5` (10 levels).
  - `H1b` ($\rho_{\text{ext}} = 2.976\times 10^{-1}$): PoNN through `DEFAULT_RHO_SCHEDULE[1]`; shooting continues from `2.976e-1` down to `1.833e-5` (9 levels).
  - `H1c` ($\rho_{\text{ext}} = 10^{-1}$): PoNN continued one step from `2.976e-1` to exact `1e-1`; shooting continues `(1e-1, 8.859e-2, ..., 1.833e-5)` (9 levels).
  - `H1d` ($\rho_{\text{ext}} = 10^{-2}$): PoNN continued from `2.637e-2` to exact `1e-2`; shooting continues `(1e-2, 7.848e-3, ..., 1.833e-5)` (7 levels).
  - `H1e` ($\rho_{\text{ext}} = 10^{-3}$): PoNN continued from `2.336e-3` to exact `1e-3`; shooting continues `(1e-3, 6.952e-4, ..., 1.833e-5)` (5 levels).
  - `H1f` ($\rho_{\text{ext}} = \text{final}$, $10^{-10}$): full 20-step coarse PoNN; shooting runs directly at terminal level `(1.833e-5,)` (1 level).
- **Reported metrics:** Strict & loose convergence, 95% Wilson CIs, exact paired McNemar $p$ (raw and Holm-adjusted), `nfev`, `njev`, and wall-clock runtime per stage (`t_ponn`, `t_shoot`, `t_total`).

### 2.2 Extra Physical Cases (7 Cases, Fresh Seeds `5000+`, $N = 50$ per Case)
Pre-run deterministic minimum-time feasibility check (80-node LGL collocation, polar spiral initial guess, fixed boundary states, $\delta \equiv 1$):

| Case ID | Scenario & Parameters | $t_f$ (d) | $t_{\min}$ (d) | $t_{\min} / t_f$ | Feasible ($\le 0.95$) | Fresh Seed Block | Hypothesis |
|:---|:---|---:|---:|---:|:---:|:---|:---:|
| `scen_a_m0_800` | Scenario A, $m_0 = 800\text{ kg}$ | 348.795 | 280.884 | 0.8053 | **PASS** | `5000..5049` | `H2a` |
| `scen_a_m0_1200` | Scenario A, $m_0 = 1200\text{ kg}$ | 348.795 | 326.882 | 0.9372 | **PASS** | `5100..5149` | `H2b` |
| `scen_a_tmax_0.4` | Scenario A, $T_{\max} = 0.4\text{ N}$ | 348.795 | 330.919 | 0.9487 | **PASS** | `5200..5249` | `H2c` |
| `scen_a_tmax_0.7` | Scenario A, $T_{\max} = 0.7\text{ N}$ | 348.795 | 265.979 | 0.7626 | **PASS** | `5300..5349` | `H2d` |
| `scen_b_tf_300` | Scenario B (circ.-coplanar, $\theta_f = 210^\circ$), $t_f = 300\text{ d}$ | 300.000 | 272.881 | 0.9096 | **PASS** | `5400..5449` | `H2e` |
| `scen_b_tf_330` | Scenario B (circ.-coplanar, $\theta_f = 235^\circ$*), $t_f = 330\text{ d}$ | 330.000 | 293.082 | 0.8881 | **PASS** | `5700..5749` | `H2f` |
| `scen_b_tf_360` | Scenario B (circ.-coplanar, $\theta_f = 255^\circ$*), $t_f = 360\text{ d}$ | 360.000 | 308.375 | 0.8566 | **PASS** | `5800..5849` | `H2g` |

- Note on Scenario B arrival longitudes: $\theta_f = 210^\circ$ (consistent with `tests/test_phase_studies.py`) is used at $t_f = 300\text{ d}$ because $\theta_f = 225^\circ$ yields $t_{\min} = 285.068\text{ d}$ ($t_{\min}/300 = 0.9502 > 0.95$).
- **Pre-run Amendment (*):** Initial execution of `scen_b_tf_330` at fixed $\theta_f = 210^\circ$ (seeds `5500..5506`) revealed that fixed $\theta_f = 210^\circ$ is **physically infeasible** at $t_f \in \{330, 360\}\text{ d}$ ($t_f > t_{\max}(\theta_f = 210^\circ)$: even 120/180-node LGL IPOPT collocation fails with `Maximum_Iterations_Exceeded` and $e_{\text{reprop}} \approx 0.46\text{--}2.0$). Advancing Mars's arrival longitude along its circular orbit ($\theta_f = 235^\circ$ at $t_f = 330\text{ d}$, $\theta_f = 255^\circ$ at $t_f = 360\text{ d}$) restores full physical feasibility ($t_{\min}/t_f = 0.8881$ and $0.8566$, IPOPT `Solve_Succeeded` with $e_{\text{reprop}} \le 4\times 10^{-4}$). Fresh untouched seed blocks `5700..5749` and `5800..5849` are locked here prior to execution.
- **Methods per case (frozen settings, zero retuning):**
  1. `PoNN (scaled U(0,1))` (`L=70, n=140 CGL -> L=80, n=540 uniform_dense`).
  2. `Indirect Shooting` (random 7-D costates, `stop_at_rho=1e-5`).
  3. `Hybrid (PoNN -> Shooting)` ($k_{\text{ext}} = 1, \rho_{\text{ext}} = 0.2976$, always run with best-effort PoNN costates even if standalone PoNN does not converge).
  4. `180-Node LGL Collocation` (`max_iter=500, tol=1e-7`).
- **Reported metrics:** Strict & loose convergence with 95% Wilson CIs, exact paired McNemar (Hybrid vs. Shooting, Holm-corrected), propellant mass, switch counts and thrust sequence per case.

### 2.3 SRP Documentation, Finite-Difference Check, & Resolution Audit
- **Flat-plate model documentation:** Explicitly document frame (heliocentric J2000 inertial), attitude coupling (prescribed thrust-aligned normal $\hat{\mathbf{n}} = \boldsymbol{\alpha}^* = -\boldsymbol{\lambda}_v/\|\boldsymbol{\lambda}_v\|$, NOT independently optimized as a separate control variable), no planetary eclipse/shadowing, projected-area factor $\phi(\mathbf{r}, \boldsymbol{\alpha}) = \tfrac{1}{2} + \tfrac{1}{2}(\hat{\mathbf{r}}\cdot\boldsymbol{\alpha})^2$, and constant $A/m_0$ cannonball scaling.
- **Finite-difference verification:** Central finite-difference verification of all costate ODE terms ($\dot{\boldsymbol{\lambda}}_r = -\partial H/\partial \mathbf{r}$, $\dot{\boldsymbol{\lambda}}_v = -\partial H/\partial \mathbf{v}$, $\dot{\lambda}_m = -\partial H/\partial m$) at $N_{\text{FD}} = 1000$ random states across $A/m_0 \in \{0.02, 0.10, 0.50, 1.00\}\text{ m}^2/\text{kg}$ for both `radial` and `attitude_flat_plate` modes. Pass threshold: max relative error $< 10^{-7}$.
- **Resolution check:** From stored SRP sweep data, report $|\Delta m_f(A/m_0)| / |m_{\text{method}}(A/m_0) - m_{\text{shoot}}(A/m_0)|$ for each $A/m_0$ and method.

### 2.4 Single-Process Idle-Machine Runtime Audit (Fresh Seeds `4100..4129`, $N = 30$)
- Executed strictly in a **single Python process** (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`) with no concurrent background jobs.
- Records full distributions (mean, std, median, Q25, Q75, Q05, Q95, min, max) for PoNN (`scaled_u01`), Random Shooting, Hybrid (PoNN stage and shooting stage separately), and 180-Node LGL Collocation; separated by strict successes vs. failures; plus `nfev`, `njev`, CPU/RAM/OS, and exact package versions.

---

## 3. Tier 2 Experiments (Strongly Recommended)

### 3.1 PoNN Observed-Plateau Study (Fresh Seeds `6000..6029`, $N = 30$ per Cell)
- **Sweep:** $L \in \{40, 60, 80, 120, 160\} \times n \in \{140, 280, 540, 1080\}$ (noting $(L=160, n=140)$ is underdetermined $8n < 8L$ and reported explicitly).
- **Variants tested on the same seeds:**
  1. Node distribution: `uniform` vs. `cgl`.
  2. Tightened least-squares tolerances: `ftol = xtol = gtol = 1e-12` vs. default `1e-9`.
  3. Re-propagation control law: smoothed $\delta(S, \rho_{\text{final}})$ vs. hard-switched $\delta(S, 0)$.
  4. Error decomposition: control-discretization error (re-propagating shooting's exact switching control interpolated on the PoNN grid) vs. state-approximation error (re-propagating PoNN's own control).
- **Condition number definition:** Report both the column-scaled Jacobian 2-norm condition number $\kappa_2(J_s) = \sigma_{\max}(J_s)/\sigma_{\min}(J_s)$ where $J_s = J\,\operatorname{diag}(\|J_{:,j}\|_2)^{-1}$ is the matrix passed to QR/SVD inside `lm_svd`, and the unscaled $\kappa_2(J)$, evaluated at the terminal $\rho$ solution.
- **Decision rule for wording:** State **"observed plateau"** ONLY if the median open-loop error $e_{\text{reprop}}$ across converged seeds stays within a factor of $2.0$ across the overdetermined $(L, n)$ sweep; otherwise state that it **decreases** and report the exact reduction factor.

### 3.2 Collocation Mesh Study (Fresh Seeds `6000..6029`, $N = 30$ per Mesh)
- **Meshes:** $N \in \{60, 120, 180, 240\}$ plus one two-stage mesh-refinement arm (`N=120 -> N=240` warm-started via LGL barycentric/linear interpolation).
- **Reported metrics:** Strict & loose convergence with 95% Wilson CIs, discretization error ($e_{\text{reprop}}$ and $|m_f - m_{f,\text{shoot}}|$), and wall-clock runtime.

---

## 4. Tier 3 Experiment (Optional, with Mandatory Go/No-Go Gate)

### 4.1 Split-Domain PoNN Refinement (Fresh Seeds `7000..7099`)
- **Only executed after Tiers 1 and 2 are fully complete, committed, and pushed.**
- **Formulation:** Split $[t_0, t_f]$ at switch times $\{t_1, \dots, t_K\}$ into $K+1$ arcs; one Chebyshev CE per arc (Hermite for $\mathbf{r}, \mathbf{v}$); explicit interface continuity of $(\mathbf{r}, \mathbf{v}, m, \boldsymbol{\lambda}_v, \lambda_m, \boldsymbol{\lambda}_r)$; throttle fixed to $\delta \in \{0, 1\}$ per arc (no $\tanh$); switch times $t_k$ as unknowns with $S(t_k) = 0$ conditions; LM with analytic Jacobian (verified against central finite differences $< 10^{-7}$).
- **Mandatory Go/No-Go Gate (5 seeds `7000..7004`):** Initialize with the exact shooting solution projected onto the split-domain basis.
  - **PASS:** Recovers fuel within $0.05\text{ kg}$ of shooting, switch times within $0.5\text{ d}$, and $e_{\text{reprop}} < 10^{-2}$ on all 5/5 gate seeds.
  - **FAIL:** If any gate seed fails recovery, **STOP immediately**, report the failure honestly, mark Tier 3 as `CONTRADICTED` (gate failed), and do **NOT** continue to 100 seeds.
- **Full evaluation (only if Gate PASSES, seeds `7000..7099`, $N = 100$):** Initialized from PoNN coarse switch times.
  - **`SUPPORTED`:** Fuel within $0.05\text{ kg}$ of shooting, switches within $0.5\text{ d}$, and $e_{\text{reprop}} < 10^{-2}$ on $\ge 90/100$ seeds.
  - **`CONTRADICTED`:** Otherwise.
- **Explicit non-scope:** Do NOT implement the solar-electric power model or any other new perturbation.
