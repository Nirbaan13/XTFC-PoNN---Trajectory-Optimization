# Preregistration Addendum 3: Cross-Case Baseline Comparison (`9200..9649`), Fair Equal-Smoothing Accuracy (`9100..9199`), and Final Claims Freeze

**Status:** LOCKED BEFORE EXECUTION  
**Feature Freeze:** No new optimization methods will be introduced after this run. All existing solvers, stored JSON datasets, and prior preregistrations (`PREREG.md`, `PREREG_ADDENDUM.md`, `PREREG_ADDENDUM_2.md`) remain strictly intact.

---

## 1. TASK 1: Does PoNN Help Beyond Simple Baselines Across 9 Cases? (`seeds 9200..9649`)

We evaluate **9 physical cases** (the 7 pre-verified extra physical cases from Tier 1.2 plus Scenario A radial SRP at $A/m_0 \in \{0.5, 1.0\}\text{ m}^2/\text{kg}$), with **50 fresh untouched seeds per case** (450 seeds total):

| # | Case ID | Physical Definition | Fresh Seed Block ($N=50$) | Pre-verified $t_{\min} \le 0.95\,t_f$ |
|---:|:---|:---|:---:|:---:|
| 1 | `scen_a_m0_800` | Scenario A, $m_0 = 800\text{ kg}$ | `9200..9249` | `280.88 d <= 331.36 d` |
| 2 | `scen_a_m0_1200` | Scenario A, $m_0 = 1200\text{ kg}$ | `9250..9299` | `326.88 d <= 331.36 d` |
| 3 | `scen_a_tmax_0.4` | Scenario A, $T_{\max} = 0.4\text{ N}$ | `9300..9349` | `330.92 d <= 331.36 d` |
| 4 | `scen_a_tmax_0.7` | Scenario A, $T_{\max} = 0.7\text{ N}$ | `9350..9399` | `265.98 d <= 331.36 d` |
| 5 | `scen_b_tf_300` | Scenario B ($\theta_f=210^\circ$), $t_f = 300\text{ d}$ | `9400..9449` | `272.88 d <= 285.00 d` |
| 6 | `scen_b_tf_330` | Scenario B ($\theta_f=235^\circ$), $t_f = 330\text{ d}$ | `9450..9499` | `293.08 d <= 313.50 d` |
| 7 | `scen_b_tf_360` | Scenario B ($\theta_f=255^\circ$), $t_f = 360\text{ d}$ | `9500..9549` | `308.38 d <= 342.00 d` |
| 8 | `scen_a_srp_0.5` | Scenario A, radial SRP $A/m_0 = 0.5\text{ m}^2/\text{kg}$ | `9550..9599` | `~316 d <= 331.36 d` |
| 9 | `scen_a_srp_1.0` | Scenario A, radial SRP $A/m_0 = 1.0\text{ m}^2/\text{kg}$ | `9600..9649` | `~317 d <= 331.36 d` |

### 1.1 Five Arms per Case
1. **(a) Single-start random shooting (`single_shoot`):** Attempt $k=0$ of multi-start shooting (sub-seed $10s + 0$, `stop_at_rho=1e-5`).
2. **(b) Multi-start random shooting (`multistart_shoot_5`):** Up to $K=5$ independent random costate restarts per seed $s$ (disjoint sub-seeds $10s + k$, $k \in \{0..4\}$), stopping immediately at the first strict success ($e_{\text{reprop}} < 10^{-2}$). Records cumulative wall-clock time to first strict success.
3. **(c) Deterministic energy-optimal shooting (`det_energy_shoot`):** Fixed prograde initial costate guess from Addendum 2, $\boldsymbol{\lambda}_0^{\text{fix}} = [\mathbf{0}_3, -\hat{\mathbf{v}}_0, 0.5]$, solving $\rho=1.0 \to 10^{-5}$ over 5 idle-machine timing trials per case.
4. **(d) Hybrid PoNN($\rho_{\text{ext}}=1$) $\to$ Shooting (`hybrid_rho1`):** Frozen coarse PoNN ($L=70, n=140$ CGL, scaled $U(0,1)$) stopped at $\rho_{\text{ext}}=1.0$, exact CE costate extraction, followed by shooting continuation ($\rho=1.0 \to 10^{-5}$).
5. **(e) Split-domain PoNN pipeline (`split_domain`):** Frozen coarse PoNN (full coarse continuation to $\rho=10^{-10}$) to detect switch times and arc sequence $\delta \in \{0, 1\}$ per arc, followed by multi-arc Hermite CE `SplitDomainPoNN` refinement (analytic JAX Jacobian LM). If coarse PoNN finds 0 switches or $> 6$ switches or diverges, `split_domain` is recorded cleanly as failed without retuning.

### 1.2 Metrics & Per-Case Numeric Verdicts
- Dual thresholds: strict ($e_{\text{reprop}} < 10^{-2}$) and loose ($e_{\text{reprop}} < 5\times 10^{-2}$), with 95% Wilson CIs.
- For each of the 9 cases, 4 primary tests vs. `multistart_shoot_5` ($9 \times 4 = 36$ tests):
  - `T1_hyb_rate_<case>`: exact two-sided paired McNemar (`hybrid_rho1` vs `multistart_shoot_5`, strict conv.) — `SUPPORTED` iff $\Delta_{\text{strict}} > 0$ and $p_{\text{Holm}} < 0.05$.
  - `T1_sd_rate_<case>`: exact two-sided paired McNemar (`split_domain` vs `multistart_shoot_5`, strict conv.) — `SUPPORTED` iff $\Delta_{\text{strict}} > 0$ and $p_{\text{Holm}} < 0.05$.
  - `T1_hyb_time_<case>`: two-sided paired Wilcoxon signed-rank (`hybrid_rho1` vs `multistart_shoot_5`, time to first strict success on jointly strict seeds) — `SUPPORTED` iff $\text{med}(t_{\text{hyb}} - t_{\text{ms5}}) < 0$ and $p_{\text{Holm}} < 0.05$.
  - `T1_sd_time_<case>`: two-sided paired Wilcoxon signed-rank (`split_domain` vs `multistart_shoot_5`, time to first strict success on jointly strict seeds) — `SUPPORTED` iff $\text{med}(t_{\text{sd}} - t_{\text{ms5}}) < 0$ and $p_{\text{Holm}} < 0.05$.
- Plain-language per-case summary verdict:
  - **`HELPS (RATE + SPEED)`** if rate `SUPPORTED` and time `SUPPORTED`.
  - **`HELPS (SPEED, MATCHED RATE)`** if time `SUPPORTED` and `hybrid`/`split` strict rate $\ge$ `multistart_shoot_5` $- 5\text{ pp}$.
  - **`HELPS (RATE)`** if rate `SUPPORTED`.
  - **`NO ADVANTAGE`** if neither rate nor speed is `SUPPORTED` (or if `det_energy_shoot` already solves the case deterministically in less or equal time and higher/equal reliability — explicitly noted!).

---

## 2. TASK 2: Fair Equal-Smoothing Accuracy Comparison (`seeds 9100..9199`, $N = 100$)

On fresh untouched seeds `9100..9199` (Scenario A, $A/m_0 = 0$):
1. **`shoot_rho_1e8`:** Indirect shooting continued down to $\rho = 1.27\times 10^{-8}$ (`stop_at_rho=1e-8`). Open-loop DOP853 error is evaluated BOTH on the uniform 2001-point grid (`reprop_grid_2001`) AND with exact event-aware piecewise-constant hard-switch integration (`reprop_exact_arcs`).
2. **`shoot_hard_polish`:** Starting from the converged `shoot_rho_1e8` costates $\boldsymbol{\lambda}_0$, solves the exact hard-switched bang-bang BVP ($\delta = \mathbf{1}_{S \ge 0}$, event-triggered `DOP853` at $S(t)=0$, `rtol=1e-11, atol=1e-13`) via `fsolve` (`xtol=1e-12`). Records total runtime (`shoot_rho_1e8` + polish), exact open-loop arrival residual `reprop_exact_arcs`, and propellant consumption.
3. **`split_domain`:** Frozen coarse PoNN $\to$ 5-arc Hermite CE `SplitDomainPoNN` (exact hard bang-bang $\delta \in \{0, 1\}$, no $\tanh$, evaluated by exact arc-by-arc `DOP853` open-loop integration `rtol=1e-11, atol=1e-13`).

### 2.1 Four Primary Accuracy Tests (Jointly Holm–Bonferroni Corrected with Task 1, Total $M = 40$ Tests, FWER $\alpha = 0.05$)
- `A1`: Exact two-sided paired McNemar (`split_domain` vs `shoot_rho_1e8`, strict conv.) — `SUPPORTED` iff $\Delta_{\text{strict}} \ge +15.0\text{ pp}$ and $p_{\text{Holm}} < 0.05$.
- `A2`: Two-sided paired Wilcoxon (`split_domain` vs `shoot_rho_1e8` on `reprop_exact_arcs`, jointly strict seeds) — `SUPPORTED` (split-domain more accurate than $\rho=10^{-8}$ shooting) iff $\text{med}(e_{\text{sd}} - e_{\text{sh8}}) < 0$ and $p_{\text{Holm}} < 0.05$.
- `A3`: Two-sided paired Wilcoxon (`split_domain` vs `shoot_hard_polish` on `reprop_exact_arcs`, jointly strict seeds) — `SUPPORTED` (split-domain more accurate at equal hard-switch smoothing) iff $\text{med}(e_{\text{sd}} - e_{\text{pol}}) < 0$ and $p_{\text{Holm}} < 0.05$.
- `A4`: Two-sided paired Wilcoxon (`split_domain` vs `shoot_hard_polish`, total wall-clock time on jointly strict seeds) — `SUPPORTED` iff $\text{med}(t_{\text{sd}} - t_{\text{pol}}) < 0$ and $p_{\text{Holm}} < 0.05$.

**Reporting Rule:** Split-domain PoNN will **NOT** be described as more accurate than shooting unless `A3` is `SUPPORTED` (i.e., lower open-loop error at equal exact hard-switch smoothing).

---

## 3. TASK 3 & TASK 4: Language Audit, `CLAIMS.md`, and `v1.0` Reproducibility

- Neutral attribution across `README.md`, `REPORT.md`, `ASSUMPTIONS.md`, `DEVIATIONS.md`, and tables ("not reported in the original paper", "in our independent implementation").
- Exact wording for prior perturbation-aware PoNN literature: *"perturbation-aware PoNNs exist (J2 formation flying, verified abstract; cislunar high-fidelity ephemeris, abstract only; perturbation list unverified)"*.
- Create `CLAIMS.md` listing every paper claim (one line each) with evidence file/table, computation method, and exact wording.
- Regenerate all deliverables via `python3 run_all.py --from-stored`, confirm `pytest tests/ -v` passes (including `tests/test_narrative_table_consistency.py`), verify clean-checkout reproducibility, tag `v1.0`, and push.
