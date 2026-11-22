"""Scenario A (D'Ambrosio & Furfaro, Aerospace 2024, 11, 228):
Exact Table 1 Earth-Mars interplanetary fuel-optimal problem, PMP dynamics
(Eqs. 28-45), optional radial/attitude-dependent SRP perturbations (Phase C),
and Phase A1 reference indirect shooting solver.

All paper numbers are used strictly as independent verification checks,
never as optimization targets, clipping bounds, or loss terms.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, least_squares

# ---------------------------------------------------------------------------
# Table 1 exact constants (D'Ambrosio & Furfaro 2024, Aerospace 11, 228)
# ---------------------------------------------------------------------------
R_REF_KM = 149.59787e6                 # km
T_REF_DAYS = 58.132                    # days
T_REF_S = T_REF_DAYS * 86400.0         # s
M_REF_KG = 1000.0                      # kg
V_REF_KM_S = R_REF_KM / T_REF_S        # km/s
A_REF_KM_S2 = R_REF_KM / (T_REF_S ** 2)

MU_SUN_KM3_S2 = 1.32712440018e11       # km^3/s^2
G0_M_S2 = 9.80665                      # m/s^2
ISP_A_S = 2000.0                       # s
T_MAX_A_N = 0.5                        # N
M0_A_KG = 1000.0                       # kg
TF_A_DAYS = 348.795                    # days

R0_A_KM = np.array([-140699693.0, -51614428.0, 980.0], dtype=float)
V0_A_KM_S = np.array([9.774596, -28.07828, 4.337725e-4], dtype=float)
RF_A_KM = np.array([-172682023.0, 176959469.0, 7948912.0], dtype=float)
VF_A_KM_S = np.array([-16.427384, -14.860506, 9.21486e-2], dtype=float)

# SRP physical constants (Phase C)
P0_SRP_N_M2 = 4.56e-6                  # N/m^2 at 1 AU
CR_SRP_DEFAULT = 1.2                   # reflectivity coefficient

DEFAULT_RHO_SCHEDULE = tuple(float(x) for x in np.logspace(0.0, -10.0, 20))


@dataclass
class ScenarioAProblem:
    """Nondimensional 3-D fuel-optimal Earth-Mars rendezvous (Table 1 of D'Ambrosio & Furfaro 2024)."""

    r0_km: np.ndarray = field(default_factory=lambda: R0_A_KM.copy())
    v0_km_s: np.ndarray = field(default_factory=lambda: V0_A_KM_S.copy())
    rf_km: np.ndarray = field(default_factory=lambda: RF_A_KM.copy())
    vf_km_s: np.ndarray = field(default_factory=lambda: VF_A_KM_S.copy())
    m0_kg: float = M0_A_KG
    tf_days: float = TF_A_DAYS
    t_max_N: float = T_MAX_A_N
    isp_s: float = ISP_A_S
    g0: float = G0_M_S2
    mu_km3_s2: float = MU_SUN_KM3_S2
    r_ref_km: float = R_REF_KM
    t_ref_days: float = T_REF_DAYS
    m_ref_kg: float = M_REF_KG
    # Optional Phase C SRP perturbation
    srp_area_to_mass: float = 0.0        # A/m0 in m^2/kg (0 => pure two-body Scenario A)
    srp_cr: float = CR_SRP_DEFAULT
    srp_p0: float = P0_SRP_N_M2
    srp_mode: str = "none"               # "none" | "radial" | "attitude_flat_plate"
    label: str = "scenario_a"

    # Derived nondimensional quantities
    t_ref_s: float = field(init=False)
    v_ref_km_s: float = field(init=False)
    a_ref_m_s2: float = field(init=False)
    mu: float = field(init=False)
    c: float = field(init=False)          # dimensionless Isp * g0
    t_max: float = field(init=False)
    m0: float = field(init=False)
    tf: float = field(init=False)
    r0: np.ndarray = field(init=False, repr=False)
    v0: np.ndarray = field(init=False, repr=False)
    rf: np.ndarray = field(init=False, repr=False)
    vf: np.ndarray = field(init=False, repr=False)
    k_srp: float = field(init=False)      # dimensionless SRP coefficient: a_srp_rad = k_srp * r / r^3

    def __post_init__(self):
        self.t_ref_s = self.t_ref_days * 86400.0
        self.v_ref_km_s = self.r_ref_km / self.t_ref_s
        self.a_ref_m_s2 = (self.r_ref_km * 1e3) / (self.t_ref_s ** 2)
        self.mu = self.mu_km3_s2 / (self.r_ref_km ** 3 / self.t_ref_s ** 2)
        self.c = (self.isp_s * self.g0 / 1000.0) / self.v_ref_km_s
        self.t_max = self.t_max_N / (self.m_ref_kg * self.a_ref_m_s2)
        self.m0 = self.m0_kg / self.m_ref_kg
        self.tf = self.tf_days / self.t_ref_days
        self.r0 = np.asarray(self.r0_km, dtype=float) / self.r_ref_km
        self.v0 = np.asarray(self.v0_km_s, dtype=float) / self.v_ref_km_s
        self.rf = np.asarray(self.rf_km, dtype=float) / self.r_ref_km
        self.vf = np.asarray(self.vf_km_s, dtype=float) / self.v_ref_km_s
        if self.srp_mode == "none" or self.srp_area_to_mass <= 0.0:
            self.k_srp = 0.0
        else:
            # At 1 AU (r_nd = 1), a_srp_dim = P0 * CR * (A/m0) [m/s^2]
            # Note: fixed A/m0 cannonball scaling in radial mode:
            #   a_srp(r) = (P0 * CR * (A/m0) / a_ref) * (1 / r^2) * r_hat
            #            = k_srp * r / r^3
            a_srp_1au = self.srp_p0 * self.srp_cr * self.srp_area_to_mass
            self.k_srp = a_srp_1au / self.a_ref_m_s2

    def with_(self, **kw) -> "ScenarioAProblem":
        base = dict(
            r0_km=self.r0_km.copy(),
            v0_km_s=self.v0_km_s.copy(),
            rf_km=self.rf_km.copy(),
            vf_km_s=self.vf_km_s.copy(),
            m0_kg=self.m0_kg,
            tf_days=self.tf_days,
            t_max_N=self.t_max_N,
            isp_s=self.isp_s,
            g0=self.g0,
            mu_km3_s2=self.mu_km3_s2,
            r_ref_km=self.r_ref_km,
            t_ref_days=self.t_ref_days,
            m_ref_kg=self.m_ref_kg,
            srp_area_to_mass=self.srp_area_to_mass,
            srp_cr=self.srp_cr,
            srp_p0=self.srp_p0,
            srp_mode=self.srp_mode,
            label=self.label,
        )
        base.update(kw)
        return ScenarioAProblem(**base)


# ---------------------------------------------------------------------------
# Perturbation acceleration & analytical Jacobian (Phase C)
# ---------------------------------------------------------------------------
def srp_accel_and_jacobian(
    prob: ScenarioAProblem,
    r: np.ndarray,
    alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (a_srp, da_srp/dr) in nondimensional units.

    Modes:
      - 'none'  : zero perturbation
      - 'radial': a_p(r) = k_srp * r / r^3 (constant A/m cannonball SRP)
                  da_p/dr = (k_srp / r^3) * (I - 3 r_hat r_hat^T)
      - 'attitude_flat_plate':
                  Spacecraft bus / solar array normal n_hat tracks thrust
                  direction alpha (or fixed Sun-cone when coasting). For a
                  flat absorbing/specular plate whose normal follows the
                  Sun-relative thrust cone angle cos(gamma) = |r_hat . alpha|,
                  the effective cross-section scales with projected area
                  (0.5 + 0.5 * (r_hat . alpha)^2) along r_hat, where
                  alpha = -lam_v / ||lam_v|| depends only on lam_v (not r),
                  yielding exact analytical Jacobian da_p/dr!
    """
    k = prob.k_srp
    if k == 0.0 or prob.srp_mode == "none":
        return np.zeros(3), np.zeros((3, 3))

    r2 = float(np.dot(r, r))
    rn = math.sqrt(r2)
    ir3 = 1.0 / (r2 * rn)
    rhat = r / rn

    if prob.srp_mode == "radial":
        a_p = k * ir3 * r
        J_p = k * ir3 * (np.eye(3) - 3.0 * np.outer(rhat, rhat))
        return a_p, J_p

    if prob.srp_mode == "attitude_flat_plate":
        # Projected-area flat-plate radial SRP with attitude normal aligned to alpha:
        #   phi(r, alpha) = 0.5 + 0.5 * (r_hat . alpha)^2
        #   a_p(r, alpha) = k * phi(r, alpha) * r / r^3
        # Because alpha = -lam_v / ||lam_v|| is independent of r:
        #   d(r_hat . alpha)/dr = (alpha - (r_hat . alpha) r_hat) / r
        #   dphi/dr = (r_hat . alpha) * (alpha - (r_hat . alpha) r_hat) / r
        #   da_p/dr = k * phi * (1/r^3)(I - 3 r_hat r_hat^T) + k * (r / r^3) (dphi/dr)^T
        if alpha is None:
            alpha = rhat
        u_dot = float(np.dot(rhat, alpha))
        phi = 0.5 + 0.5 * (u_dot ** 2)
        a_p = k * phi * ir3 * r
        dphi_dr = (u_dot / rn) * (alpha - u_dot * rhat)
        J_p = k * phi * ir3 * (np.eye(3) - 3.0 * np.outer(rhat, rhat)) + k * ir3 * np.outer(r, dphi_dr)
        return a_p, J_p

    raise ValueError(f"Unknown srp_mode: {prob.srp_mode}")


# ---------------------------------------------------------------------------
# Paper Eqs. 28-45 dynamics
# ---------------------------------------------------------------------------
def switching_function_paper(c: float, lam_v_norm: float, m: float, lam_m: float) -> float:
    """Eq. (38): S = (Isp * g0 * ||lam_v||) / m + lam_m - 1."""
    return c * lam_v_norm / m + lam_m - 1.0


def throttle_paper(S: float, rho: float) -> float:
    """Eq. (44): delta(S, rho) = 0.5 * (1 + tanh(S / rho))."""
    if rho <= 0.0:
        return 1.0 if S > 0.0 else 0.0
    x = S / rho
    if x > 35.0:
        return 1.0
    if x < -35.0:
        return 0.0
    return 0.5 * (1.0 + math.tanh(x))


def hamiltonian_paper(prob: ScenarioAProblem, rho: float, y: np.ndarray) -> float:
    """Eq. (35) with optimal alpha* = -lam_v / ||lam_v|| and optional SRP."""
    r, v, m = y[0:3], y[3:6], y[6]
    lr, lv, lm = y[7:10], y[10:13], y[13]
    rn = float(np.linalg.norm(r))
    lvn = float(np.linalg.norm(lv))
    alpha = -lv / lvn if lvn > 0.0 else np.zeros(3)
    S = switching_function_paper(prob.c, lvn, m, lm)
    delta = throttle_paper(S, rho)
    a_p, _ = srp_accel_and_jacobian(prob, r, alpha)
    acc = -prob.mu * r / (rn ** 3) + a_p + (prob.t_max * delta / m) * alpha
    return float(
        prob.t_max * delta / prob.c
        + np.dot(lr, v)
        + np.dot(lv, acc)
        - lm * prob.t_max * delta / prob.c
    )


def make_scenario_a_rhs14(prob: ScenarioAProblem, rho: float, counter: list | None = None):
    """14-state PMP ODE y = [r(3), v(3), m(1), lam_r(3), lam_v(3), lam_m(1)]."""
    mu = prob.mu
    tmax = prob.t_max
    c = prob.c
    k_srp = prob.k_srp
    srp_mode = prob.srp_mode

    def rhs(t, y):
        if counter is not None:
            counter[0] += 1
        x, yy, z, vx, vy, vz, m, lx, ly, lz, ax, ay, az, lm = y
        r2 = x * x + yy * yy + z * z
        rn = math.sqrt(r2)
        ir3 = 1.0 / (r2 * rn)
        lvn = math.sqrt(ax * ax + ay * ay + az * az)
        if lvn > 0.0:
            ux, uy, uz = -ax / lvn, -ay / lvn, -az / lvn
        else:
            ux = uy = uz = 0.0

        S = c * lvn / m + lm - 1.0
        delta = throttle_paper(S, rho)
        ta = tmax * delta / m

        if k_srp == 0.0 or srp_mode == "none":
            gx, gy, gz = -mu * x * ir3, -mu * yy * ir3, -mu * z * ir3
            rdl = (x * ax + yy * ay + z * az) / r2
            lr_dot_x = mu * ir3 * (ax - 3.0 * x * rdl)
            lr_dot_y = mu * ir3 * (ay - 3.0 * yy * rdl)
            lr_dot_z = mu * ir3 * (az - 3.0 * z * rdl)
        elif srp_mode == "radial":
            mk = mu - k_srp
            gx, gy, gz = -mk * x * ir3, -mk * yy * ir3, -mk * z * ir3
            rdl = (x * ax + yy * ay + z * az) / r2
            lr_dot_x = mk * ir3 * (ax - 3.0 * x * rdl)
            lr_dot_y = mk * ir3 * (ay - 3.0 * yy * rdl)
            lr_dot_z = mk * ir3 * (az - 3.0 * z * rdl)
        else:
            r_vec = np.array([x, yy, z])
            lv_vec = np.array([ax, ay, az])
            alpha_vec = np.array([ux, uy, uz])
            a_p, J_p = srp_accel_and_jacobian(prob, r_vec, alpha_vec)
            gx, gy, gz = -mu * x * ir3 + a_p[0], -mu * yy * ir3 + a_p[1], -mu * z * ir3 + a_p[2]
            rdl = (x * ax + yy * ay + z * az) / r2
            j_srp_t_lv = J_p.T @ lv_vec
            lr_dot_x = mu * ir3 * (ax - 3.0 * x * rdl) - j_srp_t_lv[0]
            lr_dot_y = mu * ir3 * (ay - 3.0 * yy * rdl) - j_srp_t_lv[1]
            lr_dot_z = mu * ir3 * (az - 3.0 * z * rdl) - j_srp_t_lv[2]

        return np.array([
            vx, vy, vz,
            gx + ta * ux, gy + ta * uy, gz + ta * uz,
            -tmax * delta / c,
            lr_dot_x, lr_dot_y, lr_dot_z,
            -lx, -ly, -lz,
            -(tmax * delta / (m * m)) * lvn,
        ])

    return rhs


def propagate_control_scenario_a(
    prob: ScenarioAProblem,
    t_nodes: np.ndarray,
    delta_nodes: np.ndarray,
    alpha_nodes: np.ndarray,
    rtol: float = 1e-11,
    atol: float = 1e-12,
):
    """DOP853 forward re-propagation of a control history (delta(t), alpha(t)) on Scenario A."""
    d_clip = np.clip(delta_nodes, 0.0, 1.0)
    u_norm = np.maximum(np.linalg.norm(alpha_nodes, axis=1), 1e-12)
    u_unit = alpha_nodes / u_norm[:, None]

    def rhs(t, x):
        r = x[0:3]
        v = x[3:6]
        m = x[6]
        d = float(np.interp(t, t_nodes, d_clip))
        u = np.array([np.interp(t, t_nodes, u_unit[:, i]) for i in range(3)])
        un = float(np.linalg.norm(u))
        if un > 0.0:
            u = u / un
        rn = float(np.linalg.norm(r))
        a_p, _ = srp_accel_and_jacobian(prob, r, u)
        acc = -prob.mu * r / (rn ** 3) + a_p + (prob.t_max * d / m) * u
        return np.array([v[0], v[1], v[2], acc[0], acc[1], acc[2], -prob.t_max * d / prob.c])

    x0 = np.concatenate([prob.r0, prob.v0, [prob.m0]])
    return solve_ivp(
        rhs, (0.0, prob.tf), x0, method="DOP853",
        rtol=rtol, atol=atol, dense_output=True, max_step=prob.tf / 400.0,
    )


def energy_balance_error_scenario_a(
    prob: ScenarioAProblem,
    t_tu: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    delta: np.ndarray,
    alpha: np.ndarray,
    m: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Relative mechanical energy balance error max_t |E(t) - E(t0) - W(t)| / |E(t0)|."""
    rn = np.linalg.norm(r, axis=1)
    E = 0.5 * np.sum(v * v, axis=1) - prob.mu / rn
    a_nc = np.zeros_like(r)
    for k in range(len(t_tu)):
        a_p, _ = srp_accel_and_jacobian(prob, r[k], alpha[k])
        a_nc[k] = a_p + (prob.t_max * delta[k] / m[k]) * alpha[k]
    integrand = np.sum(a_nc * v, axis=1)
    W = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t_tu))])
    prof = np.abs(E - E[0] - W) / abs(E[0])
    return float(np.max(prof)), prof


def detect_switches(t_days: np.ndarray, S_vals: np.ndarray) -> list[dict]:
    """Detect zero-crossings of switching function S(t) via linear interpolation."""
    switches = []
    for k in range(len(S_vals) - 1):
        s0, s1 = float(S_vals[k]), float(S_vals[k + 1])
        if s0 == 0.0:
            continue
        if (s0 > 0.0 and s1 < 0.0) or (s0 < 0.0 and s1 > 0.0):
            frac = abs(s0) / (abs(s0) + abs(s1))
            t_sw = float(t_days[k] + frac * (t_days[k + 1] - t_days[k]))
            direction = "ON->OFF" if s0 > 0.0 else "OFF->ON"
            switches.append({
                "index": int(k),
                "t_days": t_sw,
                "t_bracket_days": [float(t_days[k]), float(t_days[k + 1])],
                "direction": direction,
            })
    return switches


def thrust_sequence_label(S_vals: np.ndarray, switches: list[dict]) -> str:
    """Human-readable bang-off-bang thrust sequence, e.g. ON-OFF-ON-OFF-ON."""
    cur = "ON" if float(S_vals[0]) >= 0.0 else "OFF"
    seq = [cur]
    for sw in switches:
        cur = "OFF" if sw["direction"] == "ON->OFF" else "ON"
        seq.append(cur)
    return "-".join(seq)


# ---------------------------------------------------------------------------
# Phase A1: Reference Indirect Shooting Solver
# ---------------------------------------------------------------------------
@dataclass
class ScenarioAShootingResult:
    success: bool
    status: str
    lam0: np.ndarray
    rho_reached: float | None
    t_tu: np.ndarray
    t_days: np.ndarray
    Y: np.ndarray                    # M x 14
    S: np.ndarray                    # M
    delta: np.ndarray                # M
    alpha: np.ndarray                # M x 3
    fuel_kg: float
    final_mass_kg: float
    arrival_residual: float
    reprop_arrival_residual: float
    lam_m_tf: float
    energy_error: float
    switches: list[dict]
    thrust_sequence: str
    n_propagations: int
    n_rhs: int
    n_jacobian_builds: int
    runtime_s: float
    level_log: list[dict]


def random_costates_scenario_a(seed: int) -> np.ndarray:
    """Random initial costates: lam_r0, lam_v0 ~ U(-1, 1)^3, lam_m0 ~ U(0, 1)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3), rng.uniform(0.0, 1.0, 1)])


class ScenarioAShootingSolver:
    """Indirect shooting solver for Scenario A using tanh smoothing and rho continuation."""

    def __init__(
        self,
        rho_schedule: tuple[float, ...] = DEFAULT_RHO_SCHEDULE,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        level_tol: float = 1e-8,
        maxfev_per_level: int = 200,
        n_dense: int = 2001,
        stop_at_rho: float | None = 1e-5,
    ):
        # Note: DOP853 with tanh(S/rho) becomes stiff below rho ~ 1e-5 unless events are used;
        # we follow the full schedule down to stop_at_rho (or full 1e-10 when stop_at_rho is None).
        if stop_at_rho is not None:
            self.rho_schedule = tuple(r for r in rho_schedule if r >= stop_at_rho - 1e-15)
            if not self.rho_schedule:
                self.rho_schedule = (float(stop_at_rho),)
        else:
            self.rho_schedule = tuple(rho_schedule)
        self.rtol = rtol
        self.atol = atol
        self.level_tol = level_tol
        self.maxfev = maxfev_per_level
        self.n_dense = n_dense

    def propagate(self, prob: ScenarioAProblem, lam0: np.ndarray, rho: float, dense: bool = False, counter=None):
        y0 = np.concatenate([prob.r0, prob.v0, [prob.m0], lam0])
        rhs = make_scenario_a_rhs14(prob, rho, counter)
        max_step = prob.tf / 200.0 if rho <= 1e-3 else np.inf
        return solve_ivp(
            rhs, (0.0, prob.tf), y0, method="DOP853",
            rtol=self.rtol, atol=self.atol, dense_output=dense, max_step=max_step,
        )

    def residual(self, prob: ScenarioAProblem, lam0: np.ndarray, rho: float, counter=None) -> np.ndarray:
        try:
            sol = self.propagate(prob, lam0, rho, dense=False, counter=counter)
        except Exception:
            return np.full(7, 1e3)
        if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
            return np.full(7, 1e3)
        yf = sol.y[:, -1]
        if yf[6] <= 0.02 or np.linalg.norm(yf[0:3]) < 0.1:
            return np.full(7, 1e3)
        return np.concatenate([yf[0:3] - prob.rf, yf[3:6] - prob.vf, [yf[13]]])

    def solve(
        self,
        prob: ScenarioAProblem,
        seed: int | None = 1000,
        lam0_guess: np.ndarray | None = None,
        rho_schedule: tuple[float, ...] | None = None,
    ) -> ScenarioAShootingResult:
        t0 = time.perf_counter()
        sched = tuple(rho_schedule) if rho_schedule is not None else self.rho_schedule
        z = random_costates_scenario_a(seed) if lam0_guess is None else np.array(lam0_guess, dtype=float)
        rhs_counter = [0]
        n_prop = [0]
        jac_builds = 0
        level_log = []
        rho_reached = None
        ok_all = True
        status = ""
        zf = z.copy()

        for rho in sched:
            def F(zz, rho=rho):
                n_prop[0] += 1
                return self.residual(prob, zz, rho, rhs_counter)

            z_start = z.copy()
            p0 = n_prop[0]
            zf, info, ier, msg = fsolve(F, z_start, full_output=True, xtol=1e-12, maxfev=self.maxfev)
            jac_builds += 1
            res_norm = float(np.linalg.norm(info["fvec"]))
            used = "fsolve"
            if not (np.all(np.isfinite(zf)) and res_norm < self.level_tol):
                lm = least_squares(
                    F, z_start, method="lm",
                    xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=self.maxfev * 2,
                )
                res_lm = float(np.linalg.norm(lm.fun))
                jac_builds += int(lm.njev) if lm.njev is not None else 0
                if res_lm < res_norm or not np.isfinite(res_norm):
                    zf, res_norm, used = lm.x, res_lm, "lm"
                else:
                    used = "fsolve (lm fallback did not improve)"
            lvl_ok = bool(np.all(np.isfinite(zf)) and res_norm < self.level_tol)
            level_log.append({
                "rho": float(rho),
                "solver": used,
                "residual": float(res_norm),
                "ok": lvl_ok,
                "propagations": int(n_prop[0] - p0),
            })
            if not lvl_ok:
                ok_all = False
                status = f"homotopy stopped at rho={rho:.2e} ({used}, |res|={res_norm:.2e})"
                break
            z = zf
            rho_reached = float(rho)

        rho_eval = rho_reached if rho_reached is not None else float(sched[0])
        z_eval = z if rho_reached is not None else zf
        sol = self.propagate(prob, z_eval, rho_eval, dense=True)
        t_tu = np.linspace(0.0, prob.tf, self.n_dense)
        t_days = t_tu * prob.t_ref_days
        if sol.status == 0:
            Y = sol.sol(t_tu).T
            yf = Y[-1]
            arr_res = float(np.linalg.norm(np.concatenate([yf[0:3] - prob.rf, yf[3:6] - prob.vf])))
            lam_m_tf = float(yf[13])
            lv = Y[:, 10:13]
            lvn = np.linalg.norm(lv, axis=1)
            S = prob.c * lvn / Y[:, 6] + Y[:, 13] - 1.0
            delta = np.array([throttle_paper(s, rho_eval) for s in S])
            alpha = -lv / np.maximum(lvn, 1e-12)[:, None]
            sol_rp = propagate_control_scenario_a(prob, t_tu, delta, alpha)
            if sol_rp.status == 0:
                xf_rp = sol_rp.y[:, -1]
                reprop_res = float(np.linalg.norm(np.concatenate([xf_rp[0:3] - prob.rf, xf_rp[3:6] - prob.vf])))
            else:
                reprop_res = float("nan")
            e_err, _ = energy_balance_error_scenario_a(prob, t_tu, Y[:, 0:3], Y[:, 3:6], delta, alpha, Y[:, 6])
            switches = detect_switches(t_days, S)
            seq = thrust_sequence_label(S, switches)
            final_mass_kg = float(yf[6] * prob.m_ref_kg)
            fuel_kg = float(prob.m0_kg - final_mass_kg)
        else:
            Y = np.full((self.n_dense, 14), np.nan)
            S = np.full(self.n_dense, np.nan)
            delta = np.full(self.n_dense, np.nan)
            alpha = np.full((self.n_dense, 3), np.nan)
            arr_res = reprop_res = lam_m_tf = e_err = final_mass_kg = fuel_kg = float("nan")
            switches = []
            seq = "FAILED"

        success = bool(ok_all and np.isfinite(arr_res) and arr_res < 1e-6 and abs(lam_m_tf) < 1e-6)
        if ok_all:
            status = "converged through full rho schedule" if success else "schedule finished but tolerance not met"

        return ScenarioAShootingResult(
            success=success,
            status=status,
            lam0=z_eval,
            rho_reached=rho_reached,
            t_tu=t_tu,
            t_days=t_days,
            Y=Y,
            S=S,
            delta=delta,
            alpha=alpha,
            fuel_kg=fuel_kg,
            final_mass_kg=final_mass_kg,
            arrival_residual=arr_res,
            reprop_arrival_residual=reprop_res,
            lam_m_tf=lam_m_tf,
            energy_error=e_err,
            switches=switches,
            thrust_sequence=seq,
            n_propagations=int(n_prop[0]),
            n_rhs=int(rhs_counter[0]),
            n_jacobian_builds=int(jac_builds),
            runtime_s=time.perf_counter() - t0,
            level_log=level_log,
        )
