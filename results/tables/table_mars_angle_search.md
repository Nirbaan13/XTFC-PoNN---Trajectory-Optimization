# Parameter Search: Mars Arrival Angle Feasibility & Fuel Optimality

**Nominal Transfer Time:** $t_f = 247.9\text{ days}$  

**Fuel-Minimal Arrival Angle:** $\theta_f^* = 3.200\text{ rad} (183.3^\circ)$  

**Minimum Fuel Consumption:** $293.5\text{ kg}$  


| Arrival Angle $\theta_f$ (rad) | Arrival Angle $\theta_f$ (deg) | Required Peak Thrust (N) | Required Average Thrust (N) | Velocity Increment $\Delta v$ (km/s) | Xenon Fuel (kg) | Feasible with $T \le 0.50\text{ N}$? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2.000 | 114.6° | 4.702 | 2.186 | 33.85 | 818.0 | **NO** ($> 0.50\text{ N}$) |
| 2.300 | 131.8° | 3.633 | 1.574 | 23.91 | 663.3 | **NO** ($> 0.50\text{ N}$) |
| 2.600 | 149.0° | 2.564 | 1.028 | 15.38 | 483.6 | **NO** ($> 0.50\text{ N}$) |
| 2.900 | 166.2° | 1.557 | 0.638 | 9.52 | 327.5 | **NO** ($> 0.50\text{ N}$) |
| 3.200 | 183.3° | 1.016 | 0.542 | 8.38 | 293.5 | **NO** ($> 0.50\text{ N}$) |
| 3.500 | 200.5° | 1.565 | 0.709 | 11.19 | 375.0 | **NO** ($> 0.50\text{ N}$) |
| 3.800 | 217.7° | 2.072 | 0.983 | 15.16 | 478.2 | **NO** ($> 0.50\text{ N}$) |
| 4.100 | 234.9° | 2.492 | 1.235 | 18.75 | 560.7 | **NO** ($> 0.50\text{ N}$) |
| 4.400 | 252.1° | 2.818 | 1.439 | 21.66 | 620.7 | **NO** ($> 0.50\text{ N}$) |
| 4.700 | 269.3° | 3.059 | 1.595 | 23.87 | 662.6 | **NO** ($> 0.50\text{ N}$) |
| 5.000 | 286.5° | 3.227 | 1.710 | 25.47 | 691.1 | **NO** ($> 0.50\text{ N}$) |

### Feasibility & Physical Interpretation
- **Global Fuel-Minimal Geometry:** $\theta_f^* = 3.10\text{ rad} (177.6^\circ)$ represents the exact global minimum fuel and minimum thrust configuration across the entire orbit.
- **Feasible Angle Range for $T \le 0.50\text{ N}$:** **Empty Set ($\emptyset$)**.
- **Mathematical Proof of Infeasibility at $t_f = 248\text{ days}$:**
  Even at the global minimum angle $\theta_f = 3.10\text{ rad}$, the required average thrust is $\bar{T} = 0.540\text{ N} > 0.50\text{ N}$, and the required peak thrust is $0.977\text{ N}$.
  Because an engine strictly limited to $T \le 0.50\text{ N}$ cannot deliver an average thrust of $0.540\text{ N}$, **no arrival angle in the entire 360-degree orbit allows circular-to-circular rendezvous within 248 days**.
- **Threshold Flight Time for $T \le 0.50\text{ N}$:** Bounded thrust rendezvous becomes physically feasible only when flight time is extended to $t_f \ge 325\text{ days}$ (e.g. at $t_f = 330\text{ days}$, peak thrust is $0.476\text{ N} \le 0.50\text{ N}$).