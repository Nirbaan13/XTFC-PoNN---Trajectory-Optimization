# Reference Specification: D'Ambrosio & Furfaro (Aerospace 2024, 11, 228) — Scenario A

**Paper:** Andrea D'Ambrosio and Roberto Furfaro, *"Learning Fuel-Optimal Trajectories for Space Applications via Pontryagin Neural Networks"*, *Aerospace* 2024, 11(3), 228, DOI: `10.3390/aerospace11030228` (Sections 2, 3.1, 3.3, 4.1, Tables 1–2, Appendix A Tables A1 & A6).

This specification is written prior to solver implementation (Phase 0) to record all equations, nondimensionalization constants, unknowns, residuals, finite-difference verification against existing repository dynamics, and paper silences/ambiguities. Paper result numbers are recorded strictly as **independent checks**, never as optimization targets, clipping bounds, or loss terms.

---

## 1. Scenario A Mission & Nondimensionalization Data (Table 1)

| Quantity | Symbol | Value (Dimensional) | Nondimensional Value ($R_{\text{ref}}, t_{\text{ref}}, m_{\text{ref}}$) |
| --- | :---: | --- | --- |
| Reference distance | $R_{\text{ref}}$ | $149.59787 \times 10^6\text{ km}$ | $1.0$ |
| Reference time | $t_{\text{ref}}$ | $58.132\text{ days} = 5{,}022{,}604.8\text{ s}$ | $1.0$ |
| Reference velocity | $V_{\text{ref}} = R_{\text{ref}}/t_{\text{ref}}$ | $29.78491758\text{ km/s}$ | $1.0$ |
| Reference mass | $m_{\text{ref}}$ | $1000\text{ kg}$ | $1.0$ |
| Sun gravitational parameter | $\mu_{\odot}$ | $1.32712440018 \times 10^{11}\text{ km}^3/\text{s}^2$ | $\mu = 0.9999848462$ (see Ambiguity #1) |
| Standard gravity | $g_0$ | $9.80665\text{ m/s}^2$ | — |
| Specific impulse | $I_{\text{sp}}$ | $2000\text{ s}$ | $c = I_{\text{sp}} g_0 / V_{\text{ref}} = 0.6584977094$ |
| Maximum thrust | $T_{\max}$ | $0.5\text{ N}$ | $T_{\max,\text{nd}} = 0.0843145660$ |
| Minimum thrust | $T_{\min}$ | $0.0\text{ N}$ | $0.0$ |
| Initial mass | $m_0$ | $1000\text{ kg}$ | $1.0$ |
| Fixed time of flight | $t_f$ | $348.795\text{ days}$ ($t_0 = 0$) | $6.0000516067\text{ TU}$ ($\approx 6.0$) |
| Initial position | $\mathbf{r}_0$ | $[-140{,}699{,}693,\; -51{,}614{,}428,\; 980]\text{ km}$ | $[-0.9405193603,\; -0.3450211423,\; 6.5508954\times 10^{-6}]$ |
| Initial velocity | $\mathbf{v}_0$ | $[9.774596,\; -28.07828,\; 4.337725\times 10^{-4}]\text{ km/s}$ | $[0.3281726728,\; -0.9427012826,\; 1.4563495\times 10^{-5}]$ |
| Final position | $\mathbf{r}_f$ | $[-172{,}682{,}023,\; 176{,}959{,}469,\; 7{,}948{,}912]\text{ km}$ | $[-1.1543080326,\; 1.1829010065,\; 0.0531351951]$ |
| Final velocity | $\mathbf{v}_f$ | $[-16.427384,\; -14.860506,\; 9.21486\times 10^{-2}]\text{ km/s}$ | $[-0.5515336380,\; -0.4989272154,\; 3.0938007\times 10^{-3}]$ |
| Perturbations (Scenario A) | — | None (Sun two-body gravity only) | $a_p = \mathbf{0}$ |

**Independent Paper Check Values (NOT tuning targets):**
- Indirect shooting propellant consumption: **$396.06\text{ kg}$** (from Ref. [31] & paper Sec. 4.1).
- PoNN refined propellant consumption: **$396.85\text{ kg}$** ($+0.79\text{ kg}$ above shooting).
- Control structure (read off Fig. 5): **4 switches**, thrust structure **ON–OFF–ON–OFF–ON** at approximately **47, 66, 148, 292 days** ($\pm$ a few days).

---

## 2. Optimal Control Problem & PMP TPBVP (Eqs. 28–45)

Cost functional (Eqs. 28–29), with thrust vector $\mathbf{T} = T_{\max}\,\delta\,\boldsymbol{\alpha}$, $\delta \in [0, 1]$, $\|\boldsymbol{\alpha}\| = 1$, and $c = I_{\text{sp}} g_0$:
$$J = \frac{T_{\max}}{c}\int_{t_0}^{t_f} \delta(t)\,dt$$

State equations (Eqs. 32–34):
$$\dot{\mathbf{r}} = \mathbf{v}, \qquad \dot{\mathbf{v}} = -\frac{\mu}{r^3}\mathbf{r} + \frac{T_{\max}\,\delta}{m}\boldsymbol{\alpha}, \qquad \dot{m} = -\frac{T_{\max}\,\delta}{c}$$

Hamiltonian (Eq. 35):
$$H = \frac{T_{\max}\delta}{c} + \boldsymbol{\lambda}_r \cdot \mathbf{v} + \boldsymbol{\lambda}_v \cdot \left(-\frac{\mu}{r^3}\mathbf{r} + \frac{T_{\max}\delta}{m}\boldsymbol{\alpha}\right) - \lambda_m\frac{T_{\max}\delta}{c}$$

Optimal control laws from Pontryagin's Minimum Principle (Eqs. 36–38):
$$\boldsymbol{\alpha}^* = -\frac{\boldsymbol{\lambda}_v}{\|\boldsymbol{\lambda}_v\|}, \qquad S = \frac{c\|\boldsymbol{\lambda}_v\|}{m} + \lambda_m - 1$$
Note that substituting $\boldsymbol{\alpha}^*$ into $H$ yields the coefficient of $\frac{T_{\max}\delta}{c}$ as $(1 - \lambda_m - \frac{c\|\boldsymbol{\lambda}_v\|}{m}) = -S$, so minimizing $H$ gives:
$$\delta^* = \begin{cases} 1 & S > 0 \\ 0 & S < 0 \\ \in [0, 1] & S = 0 \end{cases}$$

Hyperbolic tangent throttle smoothing (Eqs. 44–45) for $\delta \in [0, 1]$ with continuation parameter $\rho \to 0^+$:
$$\delta(S, \rho) = \frac{1}{2}\left(1 + \tanh\!\left(\frac{S}{\rho}\right)\right)$$

Costate ODEs and free-final-mass transversality condition (Eqs. 39–42):
$$\dot{\boldsymbol{\lambda}}_r = -\frac{\partial H}{\partial \mathbf{r}} = \frac{\mu}{r^3}\boldsymbol{\lambda}_v - \frac{3\mu(\mathbf{r}\cdot\boldsymbol{\lambda}_v)}{r^5}\mathbf{r}, \qquad \dot{\boldsymbol{\lambda}}_v = -\frac{\partial H}{\partial \mathbf{v}} = -\boldsymbol{\lambda}_r, \qquad \dot{\lambda}_m = -\frac{\partial H}{\partial m} = -\frac{T_{\max}\delta}{m^2}\|\boldsymbol{\lambda}_v\|$$
$$\lambda_m(t_f) = 0$$

Terminal shooting residual (Eq. 43) for unknown $\boldsymbol{\lambda}(t_0) \in \mathbb{R}^7$:
$$\mathbf{\Phi}(\boldsymbol{\lambda}(t_0)) = \big[\mathbf{r}(t_f) - \mathbf{r}_f,\; \mathbf{v}(t_f) - \mathbf{v}_f,\; \lambda_m(t_f)\big]^T = \mathbf{0}$$

### Finite-Difference Cross-Check Against Existing `helio_dynamics.py`
Verified numerically before implementation (Phase 0):
- Central finite differences ($\Delta = 10^{-7}$) on $H$ match analytical costate ODEs to:
  - $\|-\partial H/\partial \mathbf{r} - \dot{\boldsymbol{\lambda}}_r\| = 2.22\times 10^{-10}$
  - $\|-\partial H/\partial \mathbf{v} - \dot{\boldsymbol{\lambda}}_v\| = 1.90\times 10^{-10}$
  - $|-\partial H/\partial m - \dot{\lambda}_m| = 2.05\times 10^{-10}$
- Exact algebraic mapping to existing `trajectory_optimization/helio_dynamics.py`:
  - $S_{\text{repo}} = \frac{1 - \lambda_m}{c} - \frac{\|\boldsymbol{\lambda}_v\|}{m} = -S_{\text{paper}} / c$ (checked difference: `0.0`)
  - $\delta_{\text{paper}}(S_{\text{paper}}, \rho) = \frac{1}{2}(1 + \tanh(S_{\text{paper}}/\rho))$ equals $T_{\text{repo}}(S_{\text{repo}}, \varepsilon) / T_{\max}$ when $\rho = 2 c \varepsilon$ (checked difference: `0.0`).

---

## 3. PoNN / X-TFC Formulation (Section 2.2, Section 3.3, Table 2, Appendix A)

### Domain Mapping & Chebyshev Neural Network Basis (Table 2)
- Computational domain: $z \in [z_0, z_f] = [-1, 1]$, $\Delta z = z_f - z_0 = 2$, $z^* = z - z_0 = z + 1$.
- Linear mapping: $z = z_0 + c_{\text{map}}(t - t_0)$ with $c_{\text{map}} = b^2 = \frac{z_f - z_0}{t_f - t_0} = \frac{2}{t_f}$.
- Time derivatives: $\frac{d g}{d t} = b^2 \frac{d g}{d z}$, $\frac{d^2 g}{d t^2} = b^4 \frac{d^2 g}{d z^2}$.
- Free function: single-layer Chebyshev Neural Network $g(z) = \boldsymbol{\sigma}(z)^T \boldsymbol{\beta}$ with $\sigma_q(z) = T_{q-1}(w_q z + b_q)$, fixed $w_q = 1$, $b_q = 0$, $q = 1, \dots, L$.
- Baseline discretization (Table 2): $L = 70$ neurons, $n = 140$ evenly spaced points in $[-1, 1]$.
- Continuation schedule (Sec. 4.1): $\rho \in \text{logspace}(0, -10, 20)$ (20 log-spaced steps from $1$ to $10^{-10}$), each step warm-started from the previous converged/final weights.
- Refinement pass (Sec. 4.1): detect switch intervals from sign changes of $S(t)$, add $100$ points per switch interval to $n$, set $L = 80$, and rerun.

### Appendix A Switching Functions $\Omega$
**Table A1** (single-point constraint $f(z_0)$ or $f(z_f)$):
$$\Omega_1(z) = 1, \qquad \Omega_1'(z) = 0$$

**Table A6** (two-point value + derivative constraints $f(z_0), f(z_f), f'(z_0), f'(z_f)$):
$$\Omega_1(z) = 1 + \frac{2(z^*)^3}{\Delta z^3} - \frac{3(z^*)^2}{\Delta z^2}, \qquad \Omega_1'(z) = \frac{6(z^*)^2}{\Delta z^3} - \frac{6 z^*}{\Delta z^2}, \qquad \Omega_1''(z) = \frac{12 z^*}{\Delta z^3} - \frac{6}{\Delta z^2}$$
$$\Omega_2(z) = -\frac{2(z^*)^3}{\Delta z^3} + \frac{3(z^*)^2}{\Delta z^2}, \qquad \Omega_2'(z) = -\frac{6(z^*)^2}{\Delta z^3} + \frac{6 z^*}{\Delta z^2}, \qquad \Omega_2''(z) = -\frac{12 z^*}{\Delta z^3} + \frac{6}{\Delta z^2}$$
$$\Omega_3(z) = z^* + \frac{(z^*)^3}{\Delta z^2} - \frac{2(z^*)^2}{\Delta z}, \qquad \Omega_3'(z) = 1 + \frac{3(z^*)^2}{\Delta z^2} - \frac{4 z^*}{\Delta z}, \qquad \Omega_3''(z) = \frac{6 z^*}{\Delta z^2} - \frac{4}{\Delta z}$$
$$\Omega_4(z) = \frac{(z^*)^3}{\Delta z^2} - \frac{(z^*)^2}{\Delta z}, \qquad \Omega_4'(z) = \frac{3(z^*)^2}{\Delta z^2} - \frac{2 z^*}{\Delta z}, \qquad \Omega_4''(z) = \frac{6 z^*}{\Delta z^2} - \frac{2}{\Delta z}$$

### Constrained Expressions (Eqs. 54–65)
For $i \in \{1, 2, 3\}$ (Cartesian coordinates):
$$r_i(z) = \big[\boldsymbol{\sigma}(z) - \Omega_1\boldsymbol{\sigma}_0 - \Omega_2\boldsymbol{\sigma}_f - \Omega_3\boldsymbol{\sigma}_0' - \Omega_4\boldsymbol{\sigma}_f'\big]^T \boldsymbol{\beta}_i + \Omega_1 r_{0,i} + \Omega_2 r_{f,i} + \frac{\Omega_3 v_{0,i} + \Omega_4 v_{f,i}}{b^2}$$
$$v_i(z) = b^2\Big(\big[\boldsymbol{\sigma}' - \Omega_1'\boldsymbol{\sigma}_0 - \Omega_2'\boldsymbol{\sigma}_f - \Omega_3'\boldsymbol{\sigma}_0' - \Omega_4'\boldsymbol{\sigma}_f'\big]^T \boldsymbol{\beta}_i + \Omega_1' r_{0,i} + \Omega_2' r_{f,i}\Big) + \Omega_3' v_{0,i} + \Omega_4' v_{f,i}$$
$$a_i(z) = b^4\Big(\big[\boldsymbol{\sigma}'' - \Omega_1''\boldsymbol{\sigma}_0 - \Omega_2''\boldsymbol{\sigma}_f - \Omega_3''\boldsymbol{\sigma}_0' - \Omega_4''\boldsymbol{\sigma}_f'\big]^T \boldsymbol{\beta}_i + \Omega_1'' r_{0,i} + \Omega_2'' r_{f,i}\Big) + b^2(\Omega_3'' v_{0,i} + \Omega_4'' v_{f,i})$$
Mass (Table A1, $\Omega_1 = 1$, $m(t_0) = m_0$ exact):
$$m(z) = (\boldsymbol{\sigma}(z) - \boldsymbol{\sigma}_0)^T \boldsymbol{\beta}_m + m_0, \qquad \dot{m}(z) = b^2 \boldsymbol{\sigma}'(z)^T \boldsymbol{\beta}_m$$
Velocity costate (unconstrained; eliminates $\boldsymbol{\lambda}_r = -\dot{\boldsymbol{\lambda}}_v$ and $\dot{\boldsymbol{\lambda}}_r = -\ddot{\boldsymbol{\lambda}}_v$ exactly):
$$\lambda_{v,i}(z) = \boldsymbol{\sigma}(z)^T \boldsymbol{\beta}_{\lambda_v,i}, \qquad \dot{\lambda}_{v,i}(z) = b^2 \boldsymbol{\sigma}'(z)^T \boldsymbol{\beta}_{\lambda_v,i}, \qquad \ddot{\lambda}_{v,i}(z) = b^4 \boldsymbol{\sigma}''(z)^T \boldsymbol{\beta}_{\lambda_v,i}$$
Mass costate (Table A1, $\Omega_1 = 1$, $\lambda_m(t_f) = 0$ exact):
$$\lambda_m(z) = (\boldsymbol{\sigma}(z) - \boldsymbol{\sigma}_f)^T \boldsymbol{\beta}_{\lambda_m}, \qquad \dot{\lambda}_m(z) = b^2 \boldsymbol{\sigma}'(z)^T \boldsymbol{\beta}_{\lambda_m}$$

### Unknowns & 8 Residual Components per Point
- Unknown vector: $\boldsymbol{\Xi} = [\boldsymbol{\beta}_1, \boldsymbol{\beta}_2, \boldsymbol{\beta}_3, \boldsymbol{\beta}_m, \boldsymbol{\beta}_{\lambda_{v1}}, \boldsymbol{\beta}_{\lambda_{v2}}, \boldsymbol{\beta}_{\lambda_{v3}}, \boldsymbol{\beta}_{\lambda_m}]^T \in \mathbb{R}^{8L}$ ($560$ unknowns for $L=70$; $640$ for $L=80$).
- Residual vector $\mathbf{L}(\boldsymbol{\Xi}) \in \mathbb{R}^{8n}$ evaluated at the $n$ points ($8$ equations per point; Eqs. 32, 40, 42 hold identically by construction):
  1. $\mathbf{L}_a = \mathbf{a} - \left(-\frac{\mu}{r^3}\mathbf{r} + \frac{T_{\max}\delta}{m}\boldsymbol{\alpha}\right)$ (3 components)
  2. $L_m = \dot{m} + \frac{T_{\max}\delta}{c}$ (1 component)
  3. $\mathbf{L}_{\lambda_v} = -\ddot{\boldsymbol{\lambda}}_v - \left(\frac{\mu}{r^3}\boldsymbol{\lambda}_v - \frac{3\mu(\mathbf{r}\cdot\boldsymbol{\lambda}_v)}{r^5}\mathbf{r}\right)$ (3 components)
  4. $L_{\lambda_m} = \dot{\lambda}_m + \frac{T_{\max}\delta}{m^2}\|\boldsymbol{\lambda}_v\|$ (1 component)

### Initialization (Sec. 4.1)
- $\boldsymbol{\beta}_1, \boldsymbol{\beta}_2, \boldsymbol{\beta}_3, \boldsymbol{\beta}_{\lambda_{v1}}, \boldsymbol{\beta}_{\lambda_{v2}}, \boldsymbol{\beta}_{\lambda_{v3}} \sim U(0, 1)$ (seeded).
- $\boldsymbol{\beta}_m$: least-squares fit so $m(z) = m_0$ (which gives $\boldsymbol{\beta}_m = \mathbf{0}$ identically since $m(z) = (\boldsymbol{\sigma} - \boldsymbol{\sigma}_0)^T \boldsymbol{\beta}_m + m_0$).
- $\boldsymbol{\beta}_{\lambda_m}$: least-squares fit of $(\boldsymbol{\sigma}(z) - \boldsymbol{\sigma}_f)^T \boldsymbol{\beta}_{\lambda_m} \approx 1$ across the collocation grid (retaining exact $\lambda_m(z_f) = 0$).

---

## 4. Paper Silences & Ambiguities (Logged Here & in `ASSUMPTIONS.md`)

1. **Nondimensional $\mu$ vs rounded $t_{\text{ref}}$:** Table 1 lists $R_{\text{ref}} = 149.59787\times 10^6\text{ km}$, $t_{\text{ref}} = 58.132\text{ d}$, and $\mu_\odot = 1.32712440018\times 10^{11}\text{ km}^3/\text{s}^2$, which gives $\mu_{\text{nd}} = 0.9999848462$ and $t_{f,\text{nd}} = 6.0000516067$ (rather than exactly $1.0$ and $6.0$, which come from $t_{\text{ref}} = \sqrt{R_{\text{ref}}^3/\mu_\odot} = 58.13244087\text{ d}$). We use exact Table 1 dimensional constants with $R_{\text{ref}}, t_{\text{ref}}, m_{\text{ref}}$ so $\mu_{\text{nd}} = 0.9999848462$ and verify sensitivity against $\mu_{\text{nd}} = 1.0$.
2. **Chebyshev basis indexing ($\sigma_q = T_{q-1}$ vs $T_q$) & linear dependence with Hermite CEs:** Using $\sigma_q(z) = T_{q-1}(z)$ for $q=1\dots L$ makes the first 4 constrained basis columns of $r_i$ ($T_0..T_3$) and the first constrained column of $m, \lambda_m$ ($T_0$) identically zero because the Hermite/point projectors reproduce those polynomials exactly. The paper does not state whether those redundant columns are kept (relying on LM damping) or excluded (shifting the free basis to $T_4..T_{L+3}$ / $T_1..T_L$). We test and document the exact behavior in Phase A2 and `ASSUMPTIONS.md`.
3. **Solver tolerances & iteration caps:** The paper states MATLAB R2023b `lsqnonlin` with `'levenberg-marquardt'` (Sec. 4.1) and `'trust-region-reflective'` (Sec. 4.2), and Eq. (27) mentions $\|\mathbf{L}\|_2 < \varepsilon$, without printing exact `FunctionTolerance`, `StepTolerance`, `OptimalityTolerance`, or `MaxIterations` per $\rho$-step.
4. **SciPy `'lm'` / `'trf'` vs MATLAB `lsqnonlin`:** MATLAB `lsqnonlin('levenberg-marquardt')` uses a modified Moré/Marquardt damping with optional `ScaleProblem='jacobian'` and supports rank-deficient Jacobians cleanly via damped QR/normal equations, whereas SciPy `least_squares(method='lm')` wraps MINPACK `lmdif`/`lmder` (which expects $m \ge n$ and handles zero Jacobian columns via column pivoting) and `method='trf'` uses trust-region SVD/QR steps. We test both `'lm'` and `'trf'` (plus exact JAX analytical Jacobian).
5. **Refinement pass initialization:** Sec. 4.1 states *"once the switch intervals are identified, 100 discretization points per switch interval are added to $n$, and the number of neurons $L$ has been increased to 80. Hence, the X-TFC framework has been run again"*, without specifying whether the $L=80$ run is (a) initialized by padding the converged $L=70$ weights with zeros (at the final $\rho$ or re-running the $\rho$-continuation), (b) refit by least squares to the $L=70$ trajectory, or (c) restarted from random weights.
6. **Switch detection & placement of the 100 extra points per switch interval:** The paper does not specify the exact half-width $\Delta z_{\text{sw}}$ around each zero-crossing of $S(z)$ or whether the 100 points per interval are evenly spaced inside the bracket $[z_k, z_{k+1}]$ where $S$ changes sign (or within a padded window $[z_{\text{sw}} - w, z_{\text{sw}} + w]$).
7. **Nondimensional costate scaling:** The cost functional $J = \int (T_{\max}\delta/c)\,dt$ uses nondimensional $T_{\max,\text{nd}}, c_{\text{nd}}, t_{\text{nd}}$; thus $\lambda_m$ is dimensionless ($\lambda_m \in [0, 1]$), $\boldsymbol{\lambda}_v$ has units of $m_{\text{ref}}/V_{\text{ref}}$, and $\boldsymbol{\lambda}_r$ has units of $m_{\text{ref}}/R_{\text{ref}}$ (all dimensionless $O(1)$ in canonical units).
8. **Initialization of $\boldsymbol{\beta}_{\lambda_m}$:** Fitting $\lambda_m(z) = (\boldsymbol{\sigma}(z) - \boldsymbol{\sigma}_f)^T \boldsymbol{\beta}_{\lambda_m} \approx 1$ on all $n$ points including $z_f = 1$ (where $\lambda_m(1) = 0$ identically) produces a sharp boundary layer at $z = 1$ if unregularized; fitting with truncated SVD (`rcond`) or on interior points yields a smooth initial $\lambda_m \approx 1$ decaying to $0$ at $z_f$.
