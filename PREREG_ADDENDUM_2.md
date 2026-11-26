# Preregistration Addendum 2: Multi-Start & Deterministic Baselines (`seeds 8000..8099`), Exploratory Cell Labeling, and Narrative–Table Consistency Audit

**Status:** LOCKED BEFORE EXECUTION  
**Scope:** Fresh Scenario A baseline comparison on untouched seeds `8000..8099`, exploratory labeling of truncated Tier 2 `L=160` cells, and full narrative-to-table programmatic consistency audit (Items 1–7).  
**Preservation Rule:** All previously implemented solvers, stored JSON datasets, and preregistered results (`PREREG.md`, `PREREG_ADDENDUM.md`, Tiers 1–3) remain strictly intact.

---

## 1. Fresh Baseline Benchmark (`seeds = 8000..8099`, Scenario A, $A/m_0 = 0$)

All arms are evaluated on the identical `ScenarioAProblem` (Table 1, Sun two-body gravity, $m_0 = 1000\text{ kg}$, $t_f = 348.795\text{ d}$, $T_{\max} = 0.5\text{ N}$, $I_{\text{sp}} = 2000\text{ s}$) with dual convergence thresholds unchanged from `PREREG.md`:
- **Strict convergence:** DOP853 open-loop re-propagation arrival residual $e_{\text{reprop}} < 1\times 10^{-2}$ (and solver terminal tolerances met).
- **Loose convergence:** $e_{\text{reprop}} < 5\times 10^{-2}$.

### 1.1 Arms Evaluated on Seeds `8000..8099` ($N = 100$)

1. **Single-Start Random Shooting (`single_shoot`):**
   - Attempt $k=0$ of the multi-start shooting arm (sub-seed $s_0 = 10s + 0$, costate draw $\boldsymbol{\lambda}_r, \boldsymbol{\lambda}_v \sim U(-1,1)^3$, $\lambda_m \sim U(0,1)$), using `ScenarioAShootingSolver(stop_at_rho=1e-5)`.
2. **Multi-Start Random Shooting (`multistart_shoot_5`):**
   - Up to $K = 5$ independent random costate restarts per seed $s \in \{8000, \dots, 8099\}$ using deterministic disjoint sub-seeds $s_k = 10s + k$ for $k \in \{0, 1, 2, 3, 4\}$ (`80000..80994`).
   - Stops immediately at the **first** strict success ($e_{\text{reprop}} < 1\times 10^{-2}$ and `success == True`).
   - Records: strict success rate (with 95% Wilson CI), attempt index $k^* \in \{1..5\}$ of first strict success, total cumulative wall-clock time to first strict success ($\sum_{j=0}^{k^*-1} t_j$ for successes; $\sum_{j=0}^{4} t_j$ for failures), and total propagations.
3. **Deterministic Energy-Optimal Shooting (`det_energy_shoot`):**
   - Solves the $\rho = 1.0$ energy-optimal / maximally smooth BVP from the documented physics-motivated fixed initial costate guess:
     $$\boldsymbol{\lambda}_0^{\text{fix}} = \big[0,\; 0,\; 0,\; -\hat{\mathbf{v}}(t_0)^T,\; 0.5\big]^T, \qquad \hat{\mathbf{v}}(t_0) = \frac{\mathbf{v}_0}{\|\mathbf{v}_0\|}$$
     (initial thrust aligned with Earth's heliocentric velocity vector, zero initial position costate, $\lambda_m(t_0) = 0.5$), then continues the identical 11-step log-spaced $\rho$ schedule from $\rho = 1.0$ down to $\rho = 10^{-5}$.
   - Evaluated deterministically over 5 repeated timing runs on the idle machine to record exact convergence status, fuel, $e_{\text{reprop}}$, propagations, and median wall-clock time (plus IQR/SD).
4. **Hybrid PoNN $\to$ Shooting at $\rho_{\text{ext}} = 1.0$ (`hybrid_rho1`):**
   - Frozen coarse PoNN ($L=70, n=140$ CGL, scaled $U(0,1)$ init with seed $s$) stopped at $k_{\text{ext}} = 0$ ($\rho_{\text{ext}} = 1.0$).
   - Initial costates extracted via the exact constrained-expression formulas:
     $$\boldsymbol{\lambda}_v(t_0) = \boldsymbol{\sigma}(-1)^T \boldsymbol{\beta}_{\lambda_v}, \quad \boldsymbol{\lambda}_r(t_0) = -b^2 \boldsymbol{\sigma}'(-1)^T \boldsymbol{\beta}_{\lambda_v}, \quad \lambda_m(t_0) = (\boldsymbol{\sigma}(-1) - \boldsymbol{\sigma}(1))^T \boldsymbol{\beta}_{\lambda_m}$$
   - Continues `ScenarioAShootingSolver(stop_at_rho=1e-5)` from $\rho = 1.0$ down to $\rho = 10^{-5}$.
   - Records strict success rate, PoNN stage runtime, shooting stage runtime, and total wall-clock runtime.
5. **Split-Domain PoNN Pipeline (`split_domain`):**
   - Frozen coarse PoNN ($L=70, n=140$ CGL, scaled $U(0,1)$ init with seed $s$, full coarse continuation to $\rho = 10^{-10}$) to detect initial switch times, followed by 5-arc Hermite CE `SplitDomainPoNN` refinement (fixed bang-bang throttle $\delta \in \{0, 1\}$, no $\tanh$, unknown switch times with $S(t_k)=0$, analytic JAX Jacobian LM).
   - Records strict success rate ($e_{\text{reprop}} < 1\times 10^{-2}$ and full Tier 3 accuracy criteria $|\Delta m_f| \le 0.05\text{ kg}, |\Delta t_k| \le 0.5\text{ d}$), coarse PoNN stage runtime, split-domain stage runtime, and total wall-clock runtime.

### 1.2 Preregistered Hypothesis Tests & Holm–Bonferroni Correction ($M = 10$ Tests, FWER $\alpha = 0.05$)

All $M = 10$ primary tests below are jointly Holm–Bonferroni corrected at family-wise error rate $\alpha = 0.05$. Verdicts are assigned strictly by numeric rule (`SUPPORTED` or `CONTRADICTED`):

| ID | Comparison | Metric / Test | `SUPPORTED` Rule |
|:---:|:---|:---|:---|
| `M1` | `hybrid_rho1` vs `single_shoot` | Exact two-sided paired McNemar (strict conv.) | $\Delta_{\text{strict}} \ge +15.0\text{ pp}$ and $p_{\text{Holm}} < 0.05$ |
| `M2` | `split_domain` vs `single_shoot` | Exact two-sided paired McNemar (strict conv.) | $\Delta_{\text{strict}} \ge +15.0\text{ pp}$ and $p_{\text{Holm}} < 0.05$ |
| `M3` | `multistart_shoot_5` vs `single_shoot` | Exact two-sided paired McNemar (strict conv.) | $\Delta_{\text{strict}} \ge +15.0\text{ pp}$ and $p_{\text{Holm}} < 0.05$ |
| `M4` | `hybrid_rho1` vs `multistart_shoot_5` | Exact two-sided paired McNemar (strict conv.) | $p_{\text{Holm}} < 0.05$ (significant rate difference) |
| `M5` | `split_domain` vs `multistart_shoot_5` | Exact two-sided paired McNemar (strict conv.) | $p_{\text{Holm}} < 0.05$ (significant rate difference) |
| `M6` | `hybrid_rho1` vs `split_domain` | Exact two-sided paired McNemar (strict conv.) | $p_{\text{Holm}} < 0.05$ (significant rate difference) |
| `T1` | `hybrid_rho1` vs `multistart_shoot_5` | Two-sided paired Wilcoxon signed-rank (time to first strict success, jointly strict seeds) | $\text{med}(t_{\text{hyb}} - t_{\text{ms5}}) < 0$ and $p_{\text{Holm}} < 0.05$ |
| `T2` | `split_domain` vs `multistart_shoot_5` | Two-sided paired Wilcoxon signed-rank (time to first strict success, jointly strict seeds) | $\text{med}(t_{\text{sd}} - t_{\text{ms5}}) < 0$ and $p_{\text{Holm}} < 0.05$ |
| `T3` | `hybrid_rho1` vs `split_domain` | Two-sided paired Wilcoxon signed-rank (total time, jointly strict seeds) | $p_{\text{Holm}} < 0.05$ (significant runtime difference) |
| `T4` | `hybrid_rho1` vs `det_energy_shoot` | Two-sided one-sample Wilcoxon signed-rank (`hybrid_rho1` strict times vs deterministic median) | $\text{med}(t_{\text{hyb}}) < t_{\text{det}}$ and $p_{\text{Holm}} < 0.05$ |

Unconditional wall-clock distributions (all 100 seeds, successes + failures) and conditional distributions (strict successes only) are both reported in the LaTeX and Markdown tables.

---

## 2. Tier 2 Truncated Cells & Rewording

- In `results/tier2_plateau_6000_6029.json`, 24 of 26 cells have complete $N=30$ seeds (`6000..6029`). The two highest-cost cells — `cgl_L160_n540` ($N=6$, ~410 s/seed) and `cgl_L160_n1080` ($N=3$, ~600 s/seed) — are explicitly marked **`[exploratory]`** in `tab6_ponn_plateau.tex`, `fig_tier2_ponn_plateau.pdf`, and `REPORT.md`, and excluded from the primary $N=30$ summary statistic (which already spans `cgl_L80_n280` at `4.83e-2` down to `cgl_L160_n280` at `5.64e-3`, an `8.55x` error decrease).
- All stale references to "plateau", "ceiling", or "Gibbs limit" in `REPORT.md` (§3.2, §9.1) and `README.md` are replaced with accurate data-driven wording reflecting the Tier 2 finding: single-domain PoNN open-loop error is `3.63e-2` at the baseline degree $L=80$, and decreases by `8.55x` to `5.64e-3` (below `1e-2`) at $L=160, n=280$, at the cost of elevated Jacobian condition number ($\kappa_2(J_s) \sim 10^{10}$) and reduced convergence rate.

---

## 3. Audit Fixes (Items 1–7) & Automated Consistency Test

1. Reword `REPORT.md` §3.2 and §9 to remove "plateau", "ceiling", and "Gibbs limit".
2. Generate all narrative numbers in `REPORT.md` (including §9.2 discordant counts `b=2/c=28`, $\chi^2$ $p$-values, and all prose metrics) directly from stored JSONs.
3. Remove stale `results/tables/additions_summary.md` (which contained unflagged non-converged Scenario B SRP rows).
4. Fix `REPORT.md` section ordering (strictly ascending `§1..§13`) and update reproducibility commands to `python3 run_all.py --from-stored` and `pytest tests/ -v`.
5. Mark failed/unphysical A4(d) runs (e.g., `uniform_140_lm_svd` with `0/5` convergence and unphysical `-0.46 kg` fuel) as `NaN` in tables.
6. Soften `REPORT.md` §7 (and `README.md`/`ASSUMPTIONS.md`/`DEVIATIONS.md`) from "established infeasible" at `247.9 d` to "no feasible solution found".
7. Document independent verification of the `22.26 s` random-shooting median runtime across `2000..2099` (`22.2580 s`, $N=100$) and `4100..4129` (`22.2625 s`, $N=30$).
8. Add `tests/test_narrative_table_consistency.py` verifying that key numbers in `REPORT.md`, `README.md`, and `results/tables/*.tex` match stored JSON data and each other, failing on any mismatch.
