"""Step 1 Verification Script: Dynamics, Jacobians, Hamiltonian & Costate Equations.

Executes rigorous checks against numerical finite differences:
1. Central gravity Jacobian vs. finite differences
2. SRP acceleration & Jacobian vs. finite differences
3. Earth J2 acceleration & Jacobian vs. finite differences (checking symmetry)
4. Hamiltonian derivative vs. Costate ODEs:
   -dH/dr == dlambda_r/dt
   -dH/dv == dlambda_v/dt
   -dH/dm == dlambda_m/dt
5. Conservation of Hamiltonian dH/dt == 0 along autonomous optimal trajectory
"""

import numpy as np
from scipy.integrate import solve_ivp
from trajectory_optimization.constants import (
    MU_SUN, AU_M, K_SRP, MU_EFF_CANON,
    MU_EARTH, R_EARTH, J2_EARTH,
    T_MAX_CANON, C_CANON
)
from trajectory_optimization.dynamics import (
    central_gravity, central_gravity_jacobian,
    srp_acceleration, srp_jacobian,
    j2_acceleration, j2_jacobian,
    optimal_thrust_direction, optimal_throttle, switching_function,
    heliocentric_state_costate_ode
)


def verify_gravity_jacobians():
    print("=" * 80)
    print("1. CENTRAL GRAVITY & SRP JACOBIAN VERIFICATION")
    print("=" * 80)
    
    r_test = np.array([1.12, -0.45, 0.23])
    h = 1e-6
    
    # Central Gravity
    J_grav_ana = central_gravity_jacobian(r_test, mu=MU_EFF_CANON)
    J_grav_num = np.zeros((3, 3))
    for j in range(3):
        rp = r_test.copy(); rp[j] += h
        rm = r_test.copy(); rm[j] -= h
        J_grav_num[:, j] = (central_gravity(rp, mu=MU_EFF_CANON) - central_gravity(rm, mu=MU_EFF_CANON)) / (2 * h)
    
    diff_grav = np.max(np.abs(J_grav_ana - J_grav_num))
    print(f"Central Gravity Jacobian Max Absolute Difference: {diff_grav:.3e}")
    assert diff_grav < 1e-8, "Central gravity Jacobian failed verification!"
    
    # SRP
    J_srp_ana = srp_jacobian(r_test, k_srp=1.84589e-5)
    J_srp_num = np.zeros((3, 3))
    for j in range(3):
        rp = r_test.copy(); rp[j] += h
        rm = r_test.copy(); rm[j] -= h
        J_srp_num[:, j] = (srp_acceleration(rp, k_srp=1.84589e-5) - srp_acceleration(rm, k_srp=1.84589e-5)) / (2 * h)
        
    diff_srp = np.max(np.abs(J_srp_ana - J_srp_num))
    print(f"SRP Acceleration Jacobian Max Absolute Difference:  {diff_srp:.3e}")
    assert diff_srp < 1e-8, "SRP Jacobian failed verification!"
    print("  -> Central Gravity & SRP Jacobians: PASSED (Machine Precision)")


def verify_j2_jacobian():
    print("\n" + "=" * 80)
    print("2. EARTH J2 ACCELERATION & JACOBIAN VERIFICATION")
    print("=" * 80)
    
    test_positions = [
        np.array([6878.137e3, 0.0, 0.0]),          # 500 km equatorial
        np.array([4500.0e3, 3200.0e3, 2800.0e3]),  # Inclined orbit point
        np.array([1200.0e3, -5000.0e3, 4000.0e3])  # General 3D point
    ]
    
    h = 1e-3
    for idx, r_pos in enumerate(test_positions, 1):
        J_j2_ana = j2_jacobian(r_pos)
        J_j2_num = np.zeros((3, 3))
        for j in range(3):
            rp = r_pos.copy(); rp[j] += h
            rm = r_pos.copy(); rm[j] -= h
            J_j2_num[:, j] = (j2_acceleration(rp) - j2_acceleration(rm)) / (2 * h)
            
        diff_j2 = np.max(np.abs(J_j2_ana - J_j2_num))
        is_sym = np.allclose(J_j2_ana, J_j2_ana.T, atol=1e-15)
        print(f"Position {idx} ({np.linalg.norm(r_pos)/1e3:.1f} km): Max diff = {diff_j2:.3e} | Symmetric: {is_sym}")
        assert diff_j2 < 1e-12, f"J2 Jacobian failed at position {idx}!"
        assert is_sym, f"J2 Jacobian is not symmetric at position {idx}!"
    print("  -> Earth J2 Analytical Jacobian: PASSED (Exact & Symmetric)")


def compute_hamiltonian(y, eps_smooth=0.005):
    r = y[0:3]
    v = y[3:6]
    m = y[6]
    lam_r = y[7:10]
    lam_v = y[10:13]
    lam_m = y[13]
    
    g_eff = central_gravity(r, mu=MU_EFF_CANON)
    u = optimal_thrust_direction(lam_v)
    s = switching_function(lam_v, lam_m, m, c=C_CANON)
    T = optimal_throttle(s, t_max=T_MAX_CANON, smoothing_eps=eps_smooth)
    
    # H = (T/c)*(1 - lam_m) + lam_r . v + lam_v . (g_eff + (T/m)*u)
    return float((T / C_CANON) * (1.0 - lam_m) + np.dot(lam_r, v) + np.dot(lam_v, g_eff + (T / m) * u))


def verify_costate_equations():
    print("\n" + "=" * 80)
    print("3. PONTRYAGIN COSTATE EQUATIONS VS. NUMERICAL GRADIENT OF HAMILTONIAN")
    print("=" * 80)
    
    y0 = np.array([
        1.1, -0.3, 0.05,       # r
        0.1, 0.95, -0.02,      # v
        0.92,                  # m
        -0.15, -0.25, 0.04,    # lambda_r
        -0.35, -0.45, 0.02,    # lambda_v
        0.02                   # lambda_m
    ])
    
    eps_smooth = 0.005
    ydot = heliocentric_state_costate_ode(0.0, y0, smoothing_eps=eps_smooth)
    dlam_r_dt = ydot[7:10]
    dlam_v_dt = ydot[10:13]
    dlam_m_dt = ydot[13]
    
    h = 1e-6
    # -dH/dr
    minus_dH_dr = np.zeros(3)
    for j in range(3):
        yp = y0.copy(); yp[j] += h
        ym = y0.copy(); ym[j] -= h
        minus_dH_dr[j] = -(compute_hamiltonian(yp, eps_smooth) - compute_hamiltonian(ym, eps_smooth)) / (2 * h)
        
    diff_lam_r = np.max(np.abs(dlam_r_dt - minus_dH_dr))
    print(f"d(lambda_r)/dt vs -dH/dr Max Difference: {diff_lam_r:.3e}")
    assert diff_lam_r < 1e-6, "Costate lambda_r equation inconsistent with -dH/dr!"
    
    # -dH/dv
    minus_dH_dv = np.zeros(3)
    for j in range(3):
        yp = y0.copy(); yp[3 + j] += h
        ym = y0.copy(); ym[3 + j] -= h
        minus_dH_dv[j] = -(compute_hamiltonian(yp, eps_smooth) - compute_hamiltonian(ym, eps_smooth)) / (2 * h)
        
    diff_lam_v = np.max(np.abs(dlam_v_dt - minus_dH_dv))
    print(f"d(lambda_v)/dt vs -dH/dv Max Difference: {diff_lam_v:.3e}")
    assert diff_lam_v < 1e-6, "Costate lambda_v equation inconsistent with -dH/dv!"
    
    # -dH/dm
    yp = y0.copy(); yp[6] += h
    ym = y0.copy(); ym[6] -= h
    minus_dH_dm = -(compute_hamiltonian(yp, eps_smooth) - compute_hamiltonian(ym, eps_smooth)) / (2 * h)
    
    diff_lam_m = abs(dlam_m_dt - minus_dH_dm)
    print(f"d(lambda_m)/dt vs -dH/dm Max Difference: {diff_lam_m:.3e}")
    assert diff_lam_m < 1e-6, "Costate lambda_m equation inconsistent with -dH/dm!"
    print("  -> Pontryagin Costate ODEs: PASSED (100% Consistent with Canonical Hamiltonian)")


def verify_hamiltonian_conservation():
    print("\n" + "=" * 80)
    print("4. HAMILTONIAN CONSERVATION ALONG OPTIMAL TRAJECTORY (dH/dt == 0)")
    print("=" * 80)
    
    y0 = np.array([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0,
        -0.13, -0.05, 0.0,
        -0.06, -0.15, 0.0,
        0.01
    ])
    
    eps_smooth = 0.002
    def ode(t, y):
        return heliocentric_state_costate_ode(t, y, smoothing_eps=eps_smooth)
        
    sol = solve_ivp(ode, (0.0, 2.0), y0, method="DOP853", rtol=1e-11, atol=1e-12)
    H_profile = np.array([compute_hamiltonian(sol.y[:, i], eps_smooth) for i in range(sol.y.shape[1])])
    
    dH_max = np.max(np.abs(H_profile - H_profile[0]))
    dH_rel = dH_max / abs(H_profile[0])
    
    print(f"Initial Hamiltonian H(0):       {H_profile[0]:.8f}")
    print(f"Final Hamiltonian H(tf):        {H_profile[-1]:.8f}")
    print(f"Max Absolute Variation |dH|:    {dH_max:.3e}")
    print(f"Max Relative Variation |dH/H0|: {dH_rel:.3e}")
    assert dH_rel < 1e-6, "Hamiltonian is not conserved along the autonomous trajectory!"
    print("  -> Hamiltonian Conservation dH/dt == 0: PASSED")


if __name__ == "__main__":
    verify_gravity_jacobians()
    verify_j2_jacobian()
    verify_costate_equations()
    verify_hamiltonian_conservation()
    print("\n" + "=" * 80)
    print("ALL STEP 1 DYNAMICS & PONTRYAGIN TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
