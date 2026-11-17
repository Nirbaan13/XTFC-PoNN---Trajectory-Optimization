"""Physical constants, mission parameters, and canonical scaling units.

Earth-Mars Low-Thrust Trajectory Optimization with J2 and SRP perturbations,
inspired by the Dawn spacecraft mission parameters.
"""

import numpy as np

# ==============================================================================
# Planetary & Gravitational Constants (SI units: m, s, kg)
# ==============================================================================
# Earth
MU_EARTH = 3.986004418e14        # Earth gravitational parameter [m^3/s^2]
R_EARTH = 6.378137e6             # Earth equatorial radius [m]
J2_EARTH = 1.08262668e-3         # Earth second zonal harmonic coefficient [-]

# Sun & Heliocentric
MU_SUN = 1.32712440018e20        # Sun gravitational parameter [m^3/s^2]
AU_M = 1.495978707e11            # 1 Astronomical Unit [m]
R_EARTH_ORBIT = 1.0 * AU_M       # Earth heliocentric circular orbit radius [m]
R_MARS_ORBIT = 1.524 * AU_M      # Mars heliocentric circular orbit radius [m]

# ==============================================================================
# Spacecraft & Ion Propulsion System (Dawn-inspired, uprated thruster)
# ==============================================================================
M_PARKING = 1538.6               # Spacecraft wet mass in 500 km parking orbit [kg]
M0_HELIO = 1218.0                # Spacecraft mass at start of heliocentric phase [kg]
M_DRY = 918.0                    # Spacecraft dry mass [kg]
M_XE_HELIO = 300.0               # Xenon propellant available for heliocentric phase [kg]

T_MAX = 0.50                     # Maximum ion engine thrust [N] (uprated Dawn set)
T_MIN = 0.0                      # Minimum thrust [N]
ISP = 3100.0                     # Specific impulse [s]
G0 = 9.80665                     # Standard gravity [m/s^2]
C_EXHAUST = ISP * G0             # Effective exhaust velocity [m/s] (~30400.6 m/s)
MDOT_MAX = T_MAX / C_EXHAUST     # Maximum mass flow rate [kg/s] (~1.6447e-5 kg/s)

# ==============================================================================
# Solar Radiation Pressure (SRP) Model Parameters
# ==============================================================================
P0_SRP = 4.56e-6                 # Solar radiation pressure at 1 AU [N/m^2]
CR_SRP = 1.2                     # Spacecraft optical reflectivity coefficient [-]
AREA_TO_MASS = 0.02              # Spacecraft area-to-mass ratio [m^2/kg]
K_SRP = P0_SRP * CR_SRP * AREA_TO_MASS * (AU_M ** 2)  # SRP parameter K [m^3/s^2]
SRP_FRACTION = K_SRP / MU_SUN    # K / mu_Sun (~1.84589e-5)
MU_SUN_EFF = MU_SUN - K_SRP      # Effective solar gravitational parameter with SRP

# ==============================================================================
# Canonical Units for Heliocentric Phase (AU, TU, VU, MU)
# ==============================================================================
# Length unit: 1 AU
SCALE_L = AU_M                   # [m]
# Velocity unit: Earth circular orbital velocity sqrt(MU_SUN / 1 AU)
SCALE_V = np.sqrt(MU_SUN / SCALE_L)  # [m/s] (~29.7847 km/s)
# Time unit: SCALE_L / SCALE_V (~58.1324 days)
SCALE_T = SCALE_L / SCALE_V      # [s]
DAYS_PER_TU = SCALE_T / 86400.0  # [days] (~58.1324 days)
# Mass unit: Spacecraft initial heliocentric mass
SCALE_M = M0_HELIO               # [kg] (1218.0 kg)
# Acceleration unit: SCALE_V^2 / SCALE_L
SCALE_A = (SCALE_V ** 2) / SCALE_L  # [m/s^2] (~5.93e-3 m/s^2)
# Force unit: SCALE_M * SCALE_A
SCALE_F = SCALE_M * SCALE_A      # [N] (~7.2228 N)

# Nondimensional heliocentric parameters
MU_CANON = 1.0                   # Central gravity in canonical units
K_CANON = SRP_FRACTION           # SRP constant in canonical units (~1.8459e-5)
MU_EFF_CANON = 1.0 - K_CANON     # Effective solar mu in canonical units (~0.99998154)

T_MAX_CANON = T_MAX / SCALE_F    # Nondimensional max thrust (~0.069225)
C_CANON = C_EXHAUST / SCALE_V    # Nondimensional exhaust velocity (~1.02068)
MDOT_MAX_CANON = T_MAX_CANON / C_CANON  # Nondimensional max mdot (~0.067823)

R_MARS_CANON = 1.524             # Mars orbital radius in AU
V_MARS_CANON = np.sqrt(MU_EFF_CANON / R_MARS_CANON)  # Mars circular orbital speed (~0.81003)
OMEGA_MARS_CANON = V_MARS_CANON / R_MARS_CANON       # Mars mean motion [rad/TU]

# Nominal Transfer Time
TF_NOMINAL_DAYS = 247.9          # Reference transfer time in days
TF_NOMINAL_CANON = TF_NOMINAL_DAYS / DAYS_PER_TU     # Reference transfer time in TU (~4.2644)

# ==============================================================================
# Escape Phase Parameters
# ==============================================================================
H_PARK = 500.0e3                 # Circular parking orbit altitude [m]
R_PARK = R_EARTH + H_PARK        # Circular parking orbit radius [m] (6878.137 km)
V_PARK = np.sqrt(MU_EARTH / R_PARK)  # Circular velocity [m/s] (~7612.61 m/s)
INCLINATION_REF_DEG = 28.5       # Reference parking orbit inclination [deg]
