"""Earth Parking Orbit Escape Phase Numerical Propagation (Table E1).

Simulates the low-thrust tangential escape spiral from a 500 km circular
parking orbit to parabolic escape energy (E = v^2/2 - mu/r = 0), evaluating Earth
J2 gravitational perturbations across multiple orbital inclinations (0 deg, 28.5 deg, 51.6 deg)
and a no-J2 control case, using DOP853 numerical integration in full 3D Cartesian coordinates.

Note on Table E2:
Section 3 and Section 5 of the paper state:
"the escape-phase duration and propellant come from a numerical propagation
with prescribed tangential thrust, not from an optimizer."
Accordingly, only Table E1 (numerical propagation) is produced; the redundant
Table E2 has been removed in accordance with requirement F.
"""

from dataclasses import dataclass
import time
import numpy as np
from scipy.integrate import solve_ivp

from .constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    M_PARKING, T_MAX, C_EXHAUST,
    R_PARK, INCLINATION_REF_DEG
)
from .dynamics import j2_acceleration


@dataclass
class EscapeResult:
    case_name: str
    escape_time_days: float
    xenon_used_kg: float
    final_mass_kg: float
    final_radius_km: float
    delta_v_km_s: float
    runtime_seconds: float
    end_state_residual: float = 0.0
    n_rhs_evals: int = 0


def propagate_escape_phase(
    use_j2: bool = True,
    inclination_deg: float = INCLINATION_REF_DEG,
    m0: float = M_PARKING,
    t_max: float = T_MAX,
    rtol: float = 1e-10,
    atol: float = 1e-11
) -> EscapeResult:
    """Propagates the low-thrust tangential escape spiral in 3D Cartesian coordinates.
    
    Stopped when two-body specific mechanical energy reaches parabolic escape energy:
    E = 0.5 * ||v||^2 - mu_E / ||r|| = 0.
    """
    start_time = time.perf_counter()
    inc_rad = np.radians(inclination_deg)
    mdot = t_max / C_EXHAUST
    
    r_init = R_PARK
    v_circ = np.sqrt(MU_EARTH / r_init)
    
    y0 = np.array([
        r_init, 0.0, 0.0,
        0.0, v_circ * np.cos(inc_rad), v_circ * np.sin(inc_rad),
        m0
    ])

    def ode(t: float, y: np.ndarray) -> np.ndarray:
        r = y[0:3]
        v = y[3:6]
        m = y[6]
        
        rn = np.linalg.norm(r)
        vn = np.linalg.norm(v)
        
        g = -MU_EARTH * r / (rn ** 3)
        a_j2 = j2_acceleration(r, mu=MU_EARTH, r_e=R_EARTH, j2=J2_EARTH) if use_j2 else np.zeros(3)
        a_thrust = (t_max / m) * (v / vn)
        
        vdot = g + a_j2 + a_thrust
        rdot = v
        return np.concatenate([rdot, vdot, [-mdot]])

    def escape_energy_event(t: float, y: np.ndarray) -> float:
        rn = np.linalg.norm(y[0:3])
        vn = np.linalg.norm(y[3:6])
        return 0.5 * (vn ** 2) - (MU_EARTH / rn)

    escape_energy_event.terminal = True
    escape_energy_event.direction = 1

    t_span = (0.0, 260.0 * 86400.0)
    sol = solve_ivp(
        ode,
        t_span,
        y0,
        events=escape_energy_event,
        method="DOP853",
        rtol=rtol,
        atol=atol
    )

    runtime = time.perf_counter() - start_time

    if len(sol.t_events[0]) > 0:
        t_esc_sec = sol.t_events[0][0]
        y_esc = sol.y_events[0][0]
    else:
        t_esc_sec = sol.t[-1]
        y_esc = sol.y[:, -1]

    t_escape_days = float(t_esc_sec / 86400.0)
    final_m = float(y_esc[6])
    final_r_km = float(np.linalg.norm(y_esc[0:3]) / 1e3)
    xenon_used = float(m0 - final_m)
    final_dv = float(C_EXHAUST * np.log(m0 / final_m) / 1e3)
    end_energy = 0.5 * (np.linalg.norm(y_esc[3:6]) ** 2) - (MU_EARTH / np.linalg.norm(y_esc[0:3]))

    case_label = "No J2 (control)" if not use_j2 else f"J2, i = {inclination_deg}°"
    if use_j2 and abs(inclination_deg - INCLINATION_REF_DEG) < 1e-2:
        case_label += " (reference)"

    return EscapeResult(
        case_name=case_label,
        escape_time_days=float(t_escape_days),
        xenon_used_kg=float(xenon_used),
        final_mass_kg=float(final_m),
        final_radius_km=float(final_r_km),
        delta_v_km_s=float(final_dv),
        runtime_seconds=float(runtime),
        end_state_residual=abs(float(end_energy)),
        n_rhs_evals=int(sol.nfev),
    )


def generate_table_e1() -> list[EscapeResult]:
    """Generates the four cases for Table E1 via real 3D Cartesian DOP853 integration."""
    return [
        propagate_escape_phase(use_j2=False, inclination_deg=0.0),
        propagate_escape_phase(use_j2=True, inclination_deg=0.0),
        propagate_escape_phase(use_j2=True, inclination_deg=28.5),
        propagate_escape_phase(use_j2=True, inclination_deg=51.6),
    ]
