# Preregistration: Held-Out Scenario A Benchmark & PoNN–Shooting Hybrid Evaluation (Seeds `2000..2099`)

**Preregistration status:** Committed **before** running any evaluation on fresh seeds `2000..2099`. All design choices and the Hybrid extraction smoothing level $\rho_{\text{ext}}$ were selected strictly using development seeds `1000..1099` (prior commit `ee5b001` and a 20-seed pilot on `1000..1019`). No solver hyperparameters will be modified after this commit.

---

## 1. Held-Out Test Set & Conditions

- **Fresh random seeds:** $N = 100$ independent seeds, `seed = 2000, 2001, ..., 2099` (never previously evaluated on Scenario A).
- **Primary Scenario A benchmark ($A/m_0 = 0.0\text{ m}^2/\text{kg}$, pure two-body Sun gravity, Table 1 of D'Ambrosio & Furfaro 2024):**
  1. `PoNN (scaled U(0,1))` — frozen Phase A2 settings (`L=70, n=140 CGL -> L=80, n=540 uniform_dense`, `chebyshev_shifted`, `lm_svd`, 20-step $\rho \in [1, 10^{-10}]$, $\beta_{1..3} \sim 10^{-3} U(0,1)$, $\beta_{\lambda_v} \sim 10^{-2} U(0,1)$).
  2. `PoNN (exact paper U(0,1))` — identical frozen settings with literal Sec. 4.1 unscaled $\beta_{1..3}, \beta_{\lambda_v} \sim U(0,1)$ (`max_iter_first=250`).
  3. `Indirect Shooting` — frozen Phase A1 settings (`ScenarioAShootingSolver(stop_at_rho=1e-5)`, random 7-D costates $\boldsymbol{\lambda}_r(t_0), \boldsymbol{\lambda}_v(t_0) \sim U(-1,1)^3$, $\lambda_m(t_0) \sim U(0,1)$, 20-step log-spaced $\rho$ continuation stopped at $10^{-5}$).
  4. `180-Node LGL Collocation` — frozen Phase B settings (`ScenarioACollocationSolver(n_nodes=180, max_iter=500, tol=1e-7)`, CasADi + IPOPT `limited-memory`, random 3-D orbital/throttle initial guesses).
  5. `Hybrid (PoNN -> Shooting)` — frozen `PoNN (scaled U(0,1))` run through step `k_ext` (Section 3), extracting $\boldsymbol{\lambda}_v(t_0)$, $\boldsymbol{\lambda}_r(t_0) = -b^2 \boldsymbol{\sigma}'(z_0)^T \boldsymbol{\beta}_{\lambda_v}$, and $\lambda_m(t_0)$ at $z_0 = -1$, then passing $\boldsymbol{\lambda}(t_0)$ into `ScenarioAShootingSolver` continuing the **same** $\rho$ schedule from $\rho_{\text{ext}}$ down to `stop_at_rho = 1e-5`.
- **SRP Perturbation Hybrid vs. Random-Costate Shooting Tests (fresh seeds `2000..2099`, $N = 100$ paired seeds per SRP level):**
  - Radial cannonball SRP ($P_0 = 4.56\times 10^{-6}\text{ N/m}^2$, $C_R = 1.2$) at **$A/m_0 \in \{0.0, 0.5, 1.0\}\text{ m}^2/\text{kg}$**.
  - On each SRP level and each seed $s \in [2000, 2099]$, both `Indirect Shooting` (random 7-D costates) and `Hybrid (PoNN -> Shooting)` are evaluated under identical dynamics (`ScenarioAProblem(srp_mode="radial", srp_area_to_mass=a_over_m)`).

---

## 2. Convergence Definitions & Uncertainty Reporting

Every trajectory's open-loop control $(\delta(t), \boldsymbol{\alpha}(t))$ is independently re-propagated from $(\mathbf{r}_0, \mathbf{v}_0, m_0)$ with `scipy.integrate.solve_ivp(method="DOP853", rtol=1e-11, atol=1e-13)` to compute the physical terminal arrival error:
$$e_{\text{reprop}} = \left\| \begin{bmatrix} \mathbf{r}_{\text{DOP853}}(t_f) - \mathbf{r}_f \\ \mathbf{v}_{\text{DOP853}}(t_f) - \mathbf{v}_f \end{bmatrix} \right\|_2.$$

Two convergence thresholds are preregistered and reported for every method with **95% Wilson score confidence intervals**:
- **Loose threshold:** $e_{\text{reprop}} < 5\times 10^{-2}$ (plus solver-specific physical validity: finite states, $m(t) > 0.05$; PoNN `residual_rms < 1.5e-2`; Shooting `arrival_residual < 1e-6` and `rho_reached <= 1e-4`; LGL-180 IPOPT converged status and `terminal_error < 1e-4`).
- **Strict threshold:** $e_{\text{reprop}} < 1\times 10^{-2}$ (with the same physical validity conditions).

---

## 3. Hybrid Extraction Level $\rho_{\text{ext}}$ (Selected Strictly on Dev Seeds `1000..1019`)

Prior to preregistration, candidate extraction indices $k_{\text{ext}} \in \{0, 1, 2\}$ along `DEFAULT_RHO_SCHEDULE = logspace(0, -10, 20)` were evaluated on 20 development seeds `1000..1019` (Scenario A, $A/m_0 = 0.0$, frozen `PoNN (scaled U(0,1))` coarse $L=70, n=140$ CGL):

| Candidate $k_{\text{ext}}$ | $\rho_{\text{ext}}$ | Dev Strict Conv (`1000..1019`) | Median PoNN Stage (s) | Median Total Runtime (s) | Median Shooting Propagations | PoNN ODE RMS at $\rho_{\text{ext}}$ |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| `0` | `1.000e+00` | `17/20 (85.0%)` | `1.04` | `6.60` | `261` | `2.12e-05` |
| **`1` (SELECTED)** | **`2.976e-01`** | **`17/20 (85.0%)`** | **`0.90`** | **`6.06`** | **`238`** | **`9.54e-10`** |
| `2` | `8.859e-02` | `17/20 (85.0%)` | `1.07` | `6.55` | `199` | `1.52e-05` |

**Frozen Selection:** We lock **$k_{\text{ext}} = 1$ ($\rho_{\text{ext}} = 10^{-10/19} \approx 2.97635144\times 10^{-1}$)**.
- *Rationale:* At $k_{\text{ext}} = 1$, the smooth PoNN coarse pass achieves near-machine-precision ODE residual (`9.54e-10`) in `< 1.0 s` before any sharp-switch Gibbs oscillations arise, minimizes median total wall-clock runtime (`6.06 s`), and hands off to indirect shooting at a smooth homotopy level (`rho = 0.2976`) with a broad continuation basin.
- *Costate extraction formula:* At $z_0 = -1$ ($t = t_0$), $\boldsymbol{\lambda}_v(t_0) = \boldsymbol{\sigma}(-1)^T \boldsymbol{\beta}_{\lambda_v}$, $\boldsymbol{\lambda}_r(t_0) = -b^2 \boldsymbol{\sigma}'(-1)^T \boldsymbol{\beta}_{\lambda_v}$, and $\lambda_m(t_0) = (\boldsymbol{\sigma}(-1) - \boldsymbol{\sigma}(1))^T \boldsymbol{\beta}_{\lambda_m}$.
- *Shooting continuation schedule:* Continues `DEFAULT_RHO_SCHEDULE[1:]` down to `stop_at_rho = 1e-5` (10 levels: `2.976e-1` through `1.833e-5`).
- *Honest runtime accounting:* Reported Hybrid total runtime always equals `t_ponn_stage + t_shooting_stage`, and total evaluations report both stages transparently.

---

## 4. Primary Hypothesis & Numeric Decision Rule (Hybrid vs. Random-Costate Shooting)

For each SRP condition $A/m_0 \in \{0.0, 0.5, 1.0\}\text{ m}^2/\text{kg}$ on paired fresh seeds `2000..2099` ($N = 100$):

- Let $p_{\text{hyb}}$ be the **strict** ($e_{\text{reprop}} < 10^{-2}$) convergence rate of `Hybrid (PoNN -> Shooting)` and $p_{\text{shoot}}$ be the **strict** convergence rate of `Indirect Shooting` (random 7-D costates).
- Let $n_{10}$ be the number of seeds where Hybrid succeeds (strict) and Random Shooting fails, and $n_{01}$ be the number of seeds where Random Shooting succeeds (strict) and Hybrid fails.
- Compute the **exact two-sided McNemar test** $p$-value (exact binomial test on $\min(n_{10}, n_{01})$ out of $n_{10} + n_{01}$ with $H_0: p = 0.5$, plus continuity-corrected $\chi^2$ statistic for completeness).
- **Preregistered verdict rule:**
  - **`SUPPORTED` (Hybrid Success):** $(p_{\text{hyb}} - p_{\text{shoot}}) \ge +0.15$ ($\ge +15.0$ percentage points) **AND** exact two-sided McNemar $p < 0.05$.
  - **`CONTRADICTED` (Hybrid Failure):** Otherwise.

No threshold, seed range, or solver setting will be altered after inspecting any outcome on seeds `2000..2099`.
