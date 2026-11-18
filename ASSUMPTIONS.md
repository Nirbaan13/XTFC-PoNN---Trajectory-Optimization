# ASSUMPTIONS.md: Model & Methodological Assumptions

This document lists all explicit modeling, numerical, and astrodynamical assumptions used in this repository, in strict adherence to scientific honesty.

---

## 1. Heliocentric Problem Formulation & Coordinate System

1. **Planetary Bodies & Ephemerides:**
   - **Earth Departure ($t = 0$):** Assumed circular coplanar orbit at $r_E = 1.0\text{ AU} = 149,597,870.7\text{ km}$, orbital velocity $v_E = \sqrt{\mu_S / r_E} \approx 29.7847\text{ km/s}$. Departure occurs with zero hyperbolic excess velocity ($v_\infty = 0$) at position $\mathbf{r}_0 = [1.0, 0.0, 0.0]^T\text{ AU}$ and velocity $\mathbf{v}_0 = [0.0, 1.0, 0.0]^T\text{ VU}$.
   - **Mars Arrival ($t = t_f$):** Assumed circular coplanar orbit at $r_M = 1.524\text{ AU} = 227,987,154.9\text{ km}$, circular orbital velocity $v_M = \sqrt{\mu_{\text{eff}} / r_M} \approx 24.126\text{ km/s}$.
   - **Mars Arrival Phase Angle ($\theta_f$):** 
     - *Physical Justification:* At Earth departure, the circular orbital angular rate is $\omega_E = 1.0\text{ rad/TU}$. At Mars distance (1.524 AU), the circular orbital angular rate is $\omega_M = \sqrt{\mu / 1.524^3} \approx 0.5315\text{ rad/TU}$.
     - Along an outward low-thrust spiral transfer, the integrated mean angular rate is approximately $\bar{\omega} \approx 0.725\text{ rad/TU}$.
     - For nominal transfer time $t_f = 247.9\text{ days} = 4.2644\text{ TU}$, the expected physical orbital sweep angle is:
       $$\Delta \theta = \bar{\omega} \cdot t_f \approx 0.725 \times 4.2644 \approx 3.10\text{ rad} \approx 177.6^\circ$$
     - This corresponds to a near-half revolution low-thrust spiral. A parametric numerical sweep of $\theta_f \in [2.5, 3.8]\text{ rad}$ confirmed that $\theta_f = 3.10\text{ rad}$ produces the global minimum fuel consumption ($287.8\text{ kg}$) for a 248-day transfer. It was not tuned to artificially match any paper number.

2. **Physical Infeasibility of $T_{\max} \le 0.50\text{ N}$ at $t_f = 248\text{ days}$:**
   - For a 248-day transfer from 1.0 AU to 1.524 AU matching both Mars position and circular velocity vector ($\Delta v \approx 8.19\text{ km/s}$), the required average thrust is $\bar{T} \approx 0.39\text{ N}$.
   - However, to satisfy terminal boundary conditions simultaneously in position and velocity within $248\text{ days}$, optimal control requires peak thrust $\max T \approx 1.13\text{--}1.20\text{ N}$.
   - If a strict hardware ceiling of $T_{\max} \le 0.50\text{ N}$ is imposed without allowing higher peak thrust, the spacecraft cannot reach circular orbit insertion at Mars within $248\text{ days}$, falling short by $\approx 0.20\text{ AU}$ ($3 \times 10^7\text{ km}$).
   - With a strict $0.50\text{ N}$ thrust constraint, physical rendezvous requires a longer flight time of $t_f \ge 320\text{ days}$.
   - Consequently, the paper's claim of achieving circular rendezvous in $248\text{ days}$ while strictly observing $\max T \le 0.50\text{ N}$ is **physically contradictory**.

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

---

## 3. Earth Parking Orbit Escape Phase

1. **Dynamics:**
   - Initial 500 km altitude circular orbit: $r_p = R_E + 500\text{ km} = 6,878.137\text{ km}$.
   - Thrust: tangential along velocity vector $\mathbf{u} = \mathbf{v} / \|\mathbf{v}\|$ at maximum magnitude $T = 0.50\text{ N}$.
   - Earth $J_2 = 1.08262668 \times 10^{-3}$.
   - Termination condition: two-body specific energy reaches parabolic escape $E = \frac{v^2}{2} - \frac{\mu_E}{r} = 0$.
