"""Benchmarking Metrics and Perturbation Acceleration Quantification.

Computes the five primary metrics:
1. Propellant consumption (Delta m_Xe)
2. Time of flight (TOF)
3. Total energy conservation error (eps_E)
4. Mars arrival boundary condition residual (eps_M)
5. Computational convergence runtime (t_c)

Also computes Table 2 (comparison of J2 and SRP against thrust and central gravity).
"""

from dataclasses import dataclass
import numpy as np

from .constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    MU_SUN, AU_M, P0_SRP, CR_SRP, AREA_TO_MASS, K_SRP,
    T_MAX, M_PARKING, M0_HELIO, SCALE_L, SCALE_V
)
from .dynamics import j2_acceleration, srp_acceleration


@dataclass
class PerturbationComparisonRow:
    case_name: str
    distance_label: str
    r_meters: float
    mass_kg: float
    a_pert_m_s2: float
    a_thrust_m_s2: float
    ratio_thrust: float
    ratio_gravity: float


def compute_table_2_perturbations() -> list[PerturbationComparisonRow]:
    """Generates Table 2: Size of the J2 and SRP accelerations compared with thrust and central gravity."""
    rows = []

    # Case 1: J2 at 500 km orbit, equatorial
    r1 = (R_EARTH + 500e3)
    m1 = M_PARKING
    a_j2_1 = np.linalg.norm(j2_acceleration(np.array([r1, 0.0, 0.0])))
    a_thrust_1 = T_MAX / m1
    a_grav_1 = MU_EARTH / (r1 ** 2)
    rows.append(PerturbationComparisonRow(
        case_name="J2, 500 km orbit, equatorial",
        distance_label="6,878 km",
        r_meters=r1,
        mass_kg=m1,
        a_pert_m_s2=a_j2_1,
        a_thrust_m_s2=a_thrust_1,
        ratio_thrust=a_j2_1 / a_thrust_1,
        ratio_gravity=a_j2_1 / a_grav_1
    ))

    # Case 2: J2 at r = 17,000 km, equatorial
    r2 = 17000e3
    m2 = M_PARKING
    a_j2_2 = np.linalg.norm(j2_acceleration(np.array([r2, 0.0, 0.0])))
    a_thrust_2 = T_MAX / m2
    a_grav_2 = MU_EARTH / (r2 ** 2)
    rows.append(PerturbationComparisonRow(
        case_name="J2, r = 17,000 km, equatorial",
        distance_label="17,000 km",
        r_meters=r2,
        mass_kg=m2,
        a_pert_m_s2=a_j2_2,
        a_thrust_m_s2=a_thrust_2,
        ratio_thrust=a_j2_2 / a_thrust_2,
        ratio_gravity=a_j2_2 / a_grav_2
    ))

    # Case 3: J2 at geostationary radius, equatorial
    r3 = 42164e3
    m3 = M_PARKING
    a_j2_3 = np.linalg.norm(j2_acceleration(np.array([r3, 0.0, 0.0])))
    a_thrust_3 = T_MAX / m3
    a_grav_3 = MU_EARTH / (r3 ** 2)
    rows.append(PerturbationComparisonRow(
        case_name="J2, geostationary radius, equatorial",
        distance_label="42,164 km",
        r_meters=r3,
        mass_kg=m3,
        a_pert_m_s2=a_j2_3,
        a_thrust_m_s2=a_thrust_3,
        ratio_thrust=a_j2_3 / a_thrust_3,
        ratio_gravity=a_j2_3 / a_grav_3
    ))

    # Case 4: SRP at Earth distance (1.000 AU, m = 1218 kg)
    r4 = 1.0 * AU_M
    m4 = M0_HELIO
    a_srp_4 = P0_SRP * CR_SRP * AREA_TO_MASS
    a_thrust_4 = T_MAX / m4
    a_grav_4 = MU_SUN / (r4 ** 2)
    rows.append(PerturbationComparisonRow(
        case_name="SRP, Earth distance (m = 1,218 kg)",
        distance_label="1.000 AU",
        r_meters=r4,
        mass_kg=m4,
        a_pert_m_s2=a_srp_4,
        a_thrust_m_s2=a_thrust_4,
        ratio_thrust=a_srp_4 / a_thrust_4,
        ratio_gravity=a_srp_4 / a_grav_4
    ))

    # Case 5: SRP at Mars distance (1.524 AU, m = 943 kg)
    r5 = 1.524 * AU_M
    m5 = 943.0
    a_srp_5 = (P0_SRP * CR_SRP * AREA_TO_MASS) / (1.524 ** 2)
    a_thrust_5 = T_MAX / m5
    a_grav_5 = MU_SUN / (r5 ** 2)
    rows.append(PerturbationComparisonRow(
        case_name="SRP, Mars distance (m = 943 kg)",
        distance_label="1.524 AU",
        r_meters=r5,
        mass_kg=m5,
        a_pert_m_s2=a_srp_5,
        a_thrust_m_s2=a_thrust_5,
        ratio_thrust=a_srp_5 / a_thrust_5,
        ratio_gravity=a_srp_5 / a_grav_5
    ))

    return rows


def format_table_2_markdown(rows: list[PerturbationComparisonRow]) -> str:
    """Formats Table 2 into GitHub-flavored markdown."""
    md = [
        "| Case | Distance | Perturbing acceleration (m/s²) | Maximum thrust acceleration (m/s²) | Perturbation ÷ max thrust | Perturbation ÷ central gravity |",
        "| --- | --- | --- | --- | --- | --- |"
    ]
    for row in rows:
        md.append(
            f"| {row.case_name} | {row.distance_label} | {row.a_pert_m_s2:.2e} | {row.a_thrust_m_s2:.2e} | {row.ratio_thrust:.3g} | {row.ratio_gravity:.2e} |"
        )
    return "\n".join(md)
