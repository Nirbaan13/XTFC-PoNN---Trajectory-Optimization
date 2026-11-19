"""Common heliocentric optimal-control problem shared by all three solvers.

Every solver (indirect shooting, X-TFC PoNN, LGL pseudospectral collocation)
receives one ``HelioProblem`` instance, so that the time of flight, Mars
arrival state, thrust bound, specific impulse and dynamics are identical.

Canonical units
---------------
length   L = 1 AU
velocity V = sqrt(mu_S / L)          (Earth circular speed, ~29.78 km/s)
time     TU = L / V                  (~58.13 days)
mass     MU = m0 (heliocentric start mass, 1218 kg)
In these units mu_S = 1.

Dynamics (paper, Section "How J2 and SRP Enter Each Method")
-------------------------------------------------------------
x = [r, v, m],  r' = v,  v' = g(r) + a_p(r) + (T/m) u,  m' = -T/c
g = -mu r / r^3,  a_p = k_srp * r / r^3  (SRP, radial, 1/r^2)
Cost J = integral T/c dt = m0 - m(tf)  (fuel-optimal)
H = T/c + lam_r.v + lam_v.(g + a_p + (T/m) u) - lam_m T/c
u* = -lam_v/|lam_v|,  S = (1 - lam_m)/c - |lam_v|/m,  T* = Tmax if S < 0 else 0
lam_r' = -(dg/dr + da_p/dr)^T lam_v,  lam_v' = -lam_r,  lam_m' = -(T/m^2)|lam_v|
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

from .constants import (
    MU_SUN, AU_M, K_SRP, T_MAX, C_EXHAUST, M0_HELIO, SCALE_V, SCALE_T, SCALE_F,
    DAYS_PER_TU,
)

# ---------------------------------------------------------------------------
# Mars orbital elements used by the optional Keplerian ephemeris
# (J2000 mean elements, ecliptic frame; Standish, JPL "Approximate Positions
# of the Planets", Table 1).
# ---------------------------------------------------------------------------
MARS_A_AU = 1.52371034
MARS_E = 0.09339410
MARS_I_DEG = 1.84969142
MARS_RAAN_DEG = 49.55953891
MARS_LONG_PERI_DEG = -23.94362959 + 360.0  # longitude of perihelion

# Circular coplanar Mars radius used by the paper ("simplified orbital states")
MARS_R_CIRC_AU = 1.524


def tu_from_days(days: float) -> float:
    return days / DAYS_PER_TU


def days_from_tu(tu: float) -> float:
    return tu * DAYS_PER_TU


def kepler_state(a: float, e: float, inc: float, raan: float, argp: float, nu: float,
                 mu: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Classical elements (radians, canonical units) -> heliocentric r, v."""
    p = a * (1.0 - e * e)
    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return rot @ r_pf, rot @ v_pf


@dataclass
class HelioProblem:
    """Fixed-time, fixed-endpoint, fuel-optimal heliocentric rendezvous."""

    tf_days: float
    theta_f: float                     # Mars arrival angle parameter [rad]
    t_max_newton: float = T_MAX
    srp_scale: float = 1.0             # multiplies K_SRP (0 => unperturbed)
    ephemeris: str = "circular"        # "circular" | "keplerian"
    m0_kg: float = M0_HELIO
    label: str = ""

    # derived (filled in __post_init__)
    tf: float = field(init=False)
    t_max: float = field(init=False)
    c: float = field(init=False)
    mu: float = field(init=False, default=1.0)
    k_srp: float = field(init=False)
    m0: float = field(init=False, default=1.0)
    r0: np.ndarray = field(init=False, repr=False)
    v0: np.ndarray = field(init=False, repr=False)
    rf: np.ndarray = field(init=False, repr=False)
    vf: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        mass_unit = self.m0_kg
        force_unit = mass_unit * SCALE_V ** 2 / AU_M
        self.tf = tu_from_days(self.tf_days)
        self.t_max = self.t_max_newton / force_unit
        self.c = C_EXHAUST / SCALE_V
        self.mu = 1.0
        self.k_srp = self.srp_scale * K_SRP / MU_SUN
        self.m0 = 1.0
        # Departure: Earth circular orbit at 1 AU, v_inf = 0, longitude 0.
        self.r0 = np.array([1.0, 0.0, 0.0])
        self.v0 = np.array([0.0, 1.0, 0.0])
        self.rf, self.vf = self.mars_state(self.theta_f)

    # ------------------------------------------------------------------
    def mars_state(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Mars heliocentric state at arrival.

        circular : coplanar circular orbit of radius 1.524 AU; theta is the
                   heliocentric longitude of Mars at arrival.  Mars is not
                   affected by the spacecraft SRP, so its circular speed uses
                   mu_S (=1).
        keplerian: J2000 mean elements; theta is the heliocentric ecliptic
                   longitude targeted at arrival (converted to true anomaly).
        """
        if self.ephemeris == "circular":
            R = MARS_R_CIRC_AU
            vm = np.sqrt(1.0 / R)
            return (np.array([R * np.cos(theta), R * np.sin(theta), 0.0]),
                    np.array([-vm * np.sin(theta), vm * np.cos(theta), 0.0]))
        if self.ephemeris == "keplerian":
            raan = np.radians(MARS_RAAN_DEG)
            argp = np.radians(MARS_LONG_PERI_DEG - MARS_RAAN_DEG)
            inc = np.radians(MARS_I_DEG)
            # longitude in orbit plane measured from node ~ argp + nu; solve
            # for nu such that the ecliptic longitude atan2(y, x) == theta.
            from scipy.optimize import brentq

            def lon_err(nu):
                r, _ = kepler_state(MARS_A_AU, MARS_E, inc, raan, argp, nu)
                d = np.arctan2(r[1], r[0]) - theta
                return (d + np.pi) % (2 * np.pi) - np.pi

            nu0 = (theta - raan - argp) % (2 * np.pi)
            nu = brentq(lon_err, nu0 - 0.3, nu0 + 0.3)
            return kepler_state(MARS_A_AU, MARS_E, inc, raan, argp, nu)
        raise ValueError(self.ephemeris)

    def with_(self, **kw) -> "HelioProblem":
        base = dict(tf_days=self.tf_days, theta_f=self.theta_f,
                    t_max_newton=self.t_max_newton, srp_scale=self.srp_scale,
                    ephemeris=self.ephemeris, m0_kg=self.m0_kg, label=self.label)
        base.update(kw)
        return HelioProblem(**base)

    def describe(self) -> dict:
        return {
            "tf_days": self.tf_days, "theta_f_rad": self.theta_f,
            "theta_f_deg": float(np.degrees(self.theta_f)),
            "t_max_N": self.t_max_newton, "srp_scale": self.srp_scale,
            "ephemeris": self.ephemeris, "m0_kg": self.m0_kg,
            "tf_TU": self.tf, "t_max_canon": self.t_max, "c_canon": self.c,
            "k_srp_canon": self.k_srp,
            "r0": self.r0.tolist(), "v0": self.v0.tolist(),
            "rf": self.rf.tolist(), "vf": self.vf.tolist(),
        }


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
def kg(prob: HelioProblem, m_canon: float) -> float:
    return float(m_canon * prob.m0_kg)


def newton(prob: HelioProblem, t_canon):
    force_unit = prob.m0_kg * SCALE_V ** 2 / AU_M
    return np.asarray(t_canon) * force_unit
