# ASSUMPTIONS.md: Model & Methodological Assumptions

This document lists all explicit modeling, numerical, and astrodynamical assumptions used in this repository, in strict adherence to scientific honesty.

---

## 1. Heliocentric Problem Formulation & Coordinate System

1. **Planetary Bodies & Ephemerides:**
   - **Earth Departure ($t = 0$):** Assumed circular coplanar orbit at $r_E = 1.0\text{ AU} = 149,597,870.7\text{ km}$, orbital velocity $v_E = \sqrt{\mu_S / r_E} \approx 29.7847\text{ km/s}$. Departure occurs with zero hyperbolic excess velocity ($v_\infty = 0$) at position $\mathbf{r}_0 = [1.0, 0.0, 0.0]^T\text{ AU}$ and velocity $\mathbf{v}_0 = [0.0, 1.0, 0.0]^T\text{ VU}$.
   - **Mars Arrival ($t = t_f$):** Assumed circular coplanar orbit at $r_M = 1.524\text{ AU} = 227,987,154.9\text{ km}$, circular orbital velocity $v_M = \sqrt{\mu_{\text{eff}} / r_M} \approx 24.126\text{ km/s}$.
   - **Mars Arrival Phase Angle ($\theta_f$):** 
     - *Parametric Angle Search:* Rather than fixing an arbitrary arrival phase angle, the Mars arrival angle was treated as a free optimization parameter and swept across the orbital domain $\theta_f \in [2.0, 5.0]\text{ rad}$ ($115^\circ$ to $286^\circ$).
     - *Global Minimum Fuel Geometry:* The numerical sweep proves that the global minimum-fuel and minimum-peak-thrust configuration occurs at $\theta_f^* = 3.200\text{ rad} (183.3^\circ)$ (near $\pi\text{ rad}$, corresponding to a near-half revolution low-thrust transfer arc), requiring $293.5\text{ kg}$ xenon fuel.
     - *Thrust Requirements Across All Angles:* Even at the optimal angle $\theta_f^* = 3.200\text{ rad}$, circular-to-circular rendezvous requires a minimum peak thrust of $1.016\text{ N}$ and a trajectory-averaged thrust of $\bar{T} = 0.542\text{ N}$. Across all other angles, required peak thrust increases up to $4.70\text{ N}$.
     - *Feasible Range for $T \le 0.50\text{ N}$ at $t_f \approx 248\text{ days}$:* **Empty set ($\emptyset$)**. Because the minimum average thrust across the transfer physically exceeds the engine's maximum capability ($\bar{T} = 0.542\text{ N} > 0.50\text{ N}$), **no arrival angle anywhere on Mars's orbit allows rendezvous within 248 days under $T \le 0.50\text{ N}$**.

2. **Flight Time Feasibility Boundary ($t_f \ge 325\text{ days}$ for $T_{\max} \le 0.50\text{ N}$):**
   - When the hardware ceiling $T \le 0.50\text{ N}$ is strictly enforced at $t_f = 247.9\text{ days}$, all three solvers honestly terminate with terminal position shortfalls of $\approx 0.085\text{--}0.089\text{ AU}$ ($1.27\text{--}1.32 \times 10^7\text{ km}$) and maximum fuel burn ($\approx 352\text{--}354\text{ kg}$).
   - Parameter sweeps over flight time reveal that circular Earth-Mars transfer under strictly bounded thrust $T \le 0.50\text{ N}$ becomes physically feasible only when transfer duration is extended to $t_f \ge 325\text{ days}$ (e.g. at $t_f = 330\text{ days}$, required peak thrust drops to $0.476\text{ N} \le 0.50\text{ N}$).
   - Consequently, the paper's simultaneous claim of achieving circular rendezvous in $248\text{ days}$ while strictly observing $T_{\max} \le 0.50\text{ N}$ is **physically and mathematically contradicted**.

3. **Canonical Astrodynamic Units:**
   To ensure numerical conditioning and avoid machine precision loss from large distances ($\sim 10^{11}\text{ m}$) and small mass flow rates ($\sim 10^{-5}\text{ kg/s}$):
   - Length unit: $1\text{ AU} = 1.495978707 \times 10^{11}\text{ m}$
   - Velocity unit: $V_U = \sqrt{\mu_S / 1\text{ AU}} \approx 29.78469\text{ km/s}$
   - Time unit: $T_U = 1\text{ AU} / V_U \approx 5.02264 \times 10^6\text{ s} \approx 58.1324\text{ days}$
   - Mass unit: $M_U = m_0 = 1,218.0\text{ kg}$
   - Force unit: $F_U = M_U V_U^2 / 1\text{ AU} \approx 7.2228\text{ N}$

4. **Solar Radiation Pressure (SRP) Model:**
   - SRP acceleration is purely radial: $\mathbf{a}_{\text{SRP}} = K \frac{\mathbf{r}}{r^3}$, where $K = P_0 C_R (A/m) (1\text{ AU})^2 \approx 2.45 \times 10^{15}\text{ m}^3/\text{s}^2$.
   - Effective solar gravity: $\mu_{\text{eff}} = \mu_S - K$, reducing central gravity by the dimensionless fraction $K / \mu_S \approx 1.8459 \times 10^{-5}$.
   - No eclipse modeling during heliocentric transfer.

---

## 2. Pontryagin Maximum Principle & Fuel-Optimal Control (14-State System)

1. **Hamiltonian & 14-State System:**
   - State: $\mathbf{x} = [\mathbf{r}, \mathbf{v}, m]^T \in \mathbb{R}^7$.
   - Costate: $\boldsymbol{\lambda} = [\boldsymbol{\lambda}_r, \boldsymbol{\lambda}_v, \lambda_m]^T \in \mathbb{R}^7$.
   - Hamiltonian:
     $$H = \frac{T}{c} + \boldsymbol{\lambda}_r \cdot \mathbf{v} + \boldsymbol{\lambda}_v \cdot \left(\mathbf{g}(\mathbf{r}) + \mathbf{a}_{\text{SRP}}(\mathbf{r}) + \frac{T}{m} \mathbf{u}\right) - \lambda_m \frac{T}{c}$$
   - Optimal thrust direction: $\mathbf{u}^* = -\frac{\boldsymbol{\lambda}_v}{\|\boldsymbol{\lambda}_v\|}$.
   - Switching function:
     $$S = \frac{1 - \lambda_m}{c} - \frac{\|\boldsymbol{\lambda}_v\|}{m}$$
   - State ODEs:
     $$\dot{\mathbf{r}} = \mathbf{v}, \quad \dot{\mathbf{v}} = \mathbf{g}(\mathbf{r}) + \mathbf{a}_{\text{SRP}}(\mathbf{r}) + \frac{T}{m} \mathbf{u}^*, \quad \dot{m} = -\frac{T}{c}$$
   - Costate ODEs:
     $$\dot{\boldsymbol{\lambda}}_r = -\left[\frac{\partial \mathbf{g}}{\partial \mathbf{r}} + \frac{\partial \mathbf{a}_{\text{SRP}}}{\partial \mathbf{r}}\right]^T \boldsymbol{\lambda}_v, \quad \dot{\boldsymbol{\lambda}}_v = -\boldsymbol{\lambda}_r, \quad \dot{\lambda}_m = -\frac{T}{m^2} \|\boldsymbol{\lambda}_v\|$$
   - Free final mass transversality condition: $\lambda_m(t_f) = 0$.

2. **Throttle Law Smoothing:**
   - Bang-bang control: $T = T_{\max}$ when $S < 0$, $T = 0$ when $S > 0$.
   - Smoothed via homotopy parameter $\varepsilon > 0$:
     $$T(S; \varepsilon) = \frac{T_{\max}}{1 + e^{S / \varepsilon}}$$

3. **Collocation Formulation:**
   - Solved in two stages: Stage 1 minimizes quadratic control effort $\sum w_k \|\mathbf{u}_k\|^2$ as a smooth initial guess; Stage 2 solves the true fuel-optimal NLP minimizing propellant $(m_0 - m_f)$ subject to explicit mass depletion $m_k - m_{k-1} = -\Delta t_k \bar{T}_k / c$ and strict thrust limits $m(t_k) \|\mathbf{u}(t_k)\| \le 0.50\text{ N}$.

4. **X-TFC Neural Architecture:**
   - Single hidden layer network with Chebyshev orthogonal polynomial basis.
   - Exact analytical Hermite boundary embedding guarantees $r(0) = r_0$, $r(t_f) = r_f$, $v(0) = v_0$, $v(t_f) = v_f$, $m(0) = m_0$, and $\lambda_m(t_f) = 0$ to machine precision.
   - Evaluates the complete 14-residual norm $\|R_{14}\|$ across all collocation nodes; converges only if $\|R_{14}\| < 10^{-4}$ and arrival residual $< 10^{-6}$.

---

## 3. Earth Parking Orbit Escape Phase

1. **Dynamics:**
   - Initial 500 km altitude circular orbit: $r_p = R_E + 500\text{ km} = 6,878.137\text{ km}$.
   - Thrust: tangential along velocity vector $\mathbf{u} = \mathbf{v} / \|\mathbf{v}\|$ at maximum magnitude $T = 0.50\text{ N}$.
   - Earth $J_2 = 1.08262668 \times 10^{-3}$.
   - Termination condition: two-body specific energy reaches parabolic escape $E = \frac{v^2}{2} - \frac{\mu_E}{r} = 0$.
