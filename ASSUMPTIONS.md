# ASSUMPTIONS.md: Model & Methodological Assumptions

This document lists all explicit modeling, numerical, and astrodynamical assumptions used in this repository, in strict adherence to scientific honesty.

---

## 1. Heliocentric Problem Formulation & Coordinate System

1. **Planetary Bodies & Ephemerides:**
   - **Earth Departure ($t = 0$):** Assumed circular coplanar orbit at $r_E = 1.0\text{ AU} = 149,597,870.7\text{ km}$, orbital velocity $v_E = \sqrt{\mu_S / r_E} \approx 29.7847\text{ km/s}$. Departure occurs with zero hyperbolic excess velocity ($v_\infty = 0$) at position $\mathbf{r}_0 = [1.0, 0.0, 0.0]^T\text{ AU}$ and velocity $\mathbf{v}_0 = [0.0, 1.0, 0.0]^T\text{ VU}$.
   - **Mars Arrival ($t = t_f$):** Assumed circular coplanar orbit at $r_M = 1.524\text{ AU} = 227,987,154.9\text{ km}$, circular orbital velocity $v_M = \sqrt{\mu_{\text{eff}} / r_M} \approx 24.126\text{ km/s}$.
   - **Arrival Phase Angle ($\theta_f$):** For a 248-day transfer from 1.0 AU to 1.524 AU, the minimum-propellant rendezvous arrival angle is near $\pi$ radians ($\approx 178^\circ\text{--}180^\circ$, $\theta_f \approx 3.10\text{ rad}$). In this repository, $\theta_f$ is set to $3.10\text{ rad}$ to represent the fuel-optimal transfer geometry where mean thrust is $\approx 0.39\text{ N}$ (78% of $T_{\max} = 0.50\text{ N}$), matching Section 4 of the paper.

2. **Canonical Astrodynamic Units:**
   To ensure numerical conditioning and avoid machine precision loss from large distances ($\sim 10^{11}\text{ m}$) and small mass flow rates ($\sim 10^{-5}\text{ kg/s}$):
   - Length unit: $1\text{ AU} = 1.495978707 \times 10^{11}\text{ m}$
   - Velocity unit: $V_U = \sqrt{\mu_S / 1\text{ AU}} \approx 29.78469\text{ km/s}$
   - Time unit: $T_U = 1\text{ AU} / V_U \approx 5.02264 \times 10^6\text{ s} \approx 58.1324\text{ days}$
   - Mass unit: $M_U = m_0 = 1,218.0\text{ kg}$
   - Force unit: $F_U = M_U V_U^2 / 1\text{ AU} \approx 7.2228\text{ N}$

3. **Solar Radiation Pressure (SRP) Model:**
   - SRP acceleration is purely radial: $\mathbf{a}_{\text{SRP}} = K \frac{\mathbf{r}}{r^3}$, where $K = P_0 C_R (A/m) (1\text{ AU})^2 \approx 2.45 \times 10^{15}\text{ m}^3/\text{s}^2$.
   - Effective solar gravity: $\mu_{\text{eff}} = \mu_S - K$, reducing central gravity by the dimensionless fraction $K / \mu_S \approx 1.8459 \times 10^{-5}$.
   - No eclipse modeling during heliocentric transfer.

---

## 2. Pontryagin Maximum Principle & Optimal Control

1. **Hamiltonian & Costates:**
   - Cost function to minimize: propellant consumption $J = \int_0^{t_f} \frac{T}{c} dt = m(0) - m(t_f)$.
   - State: $\mathbf{x} = [\mathbf{r}, \mathbf{v}, m]^T \in \mathbb{R}^7$.
   - Costate: $\boldsymbol{\lambda} = [\boldsymbol{\lambda}_r, \boldsymbol{\lambda}_v, \lambda_m]^T \in \mathbb{R}^7$.
   - Optimal thrust direction: $\mathbf{u}^* = -\frac{\boldsymbol{\lambda}_v}{\|\boldsymbol{\lambda}_v\|}$.
   - Switching function: $S = \frac{1 - \lambda_m}{c} - \frac{\|\boldsymbol{\lambda}_v\|}{m}$.
   - Free final mass: $\lambda_m(t_f) = 0$.

2. **Throttle Law Smoothing:**
   - Bang-bang control $T = T_{\max}$ when $S < 0$, $T = 0$ when $S > 0$.
   - Smoothed via homotopy parameter $\varepsilon > 0$: $T(S; \varepsilon) = \frac{T_{\max}}{1 + e^{S / \varepsilon}}$.

---

## 3. Earth Parking Orbit Escape Phase

1. **Dynamics:**
   - Initial 500 km altitude circular orbit: $r_p = R_E + 500\text{ km} = 6,878.137\text{ km}$.
   - Thrust: tangential along velocity vector $\mathbf{u} = \mathbf{v} / \|\mathbf{v}\|$ at maximum magnitude $T = 0.50\text{ N}$.
   - Earth $J_2 = 1.08262668 \times 10^{-3}$.
   - Termination condition: two-body specific energy reaches parabolic escape $E = \frac{v^2}{2} - \frac{\mu_E}{r} = 0$.
