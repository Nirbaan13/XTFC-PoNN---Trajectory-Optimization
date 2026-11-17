"""Tests for physical constants, spacecraft specs, and canonical unit scales."""

import numpy as np
import pytest

from trajectory_optimization.constants import (
    MU_EARTH, R_EARTH, J2_EARTH,
    MU_SUN, AU_M,
    M_PARKING, M0_HELIO, T_MAX, ISP, G0, C_EXHAUST, MDOT_MAX,
    P0_SRP, CR_SRP, AREA_TO_MASS, K_SRP, SRP_FRACTION,
    SCALE_L, SCALE_V, SCALE_T, SCALE_M, SCALE_F,
    MU_EFF_CANON, T_MAX_CANON, C_CANON, MDOT_MAX_CANON,
    DAYS_PER_TU, TF_NOMINAL_DAYS
)


def test_earth_constants():
    assert np.isclose(MU_EARTH, 398600.4418e9)
    assert np.isclose(R_EARTH, 6378.137e3)
    assert np.isclose(J2_EARTH, 1.08262668e-3)


def test_propulsion_and_flow_rate():
    assert np.isclose(T_MAX, 0.50)
    assert np.isclose(ISP, 3100.0)
    assert np.isclose(C_EXHAUST, 3100.0 * 9.80665)
    expected_mdot = 0.50 / (3100.0 * 9.80665)
    assert np.isclose(MDOT_MAX, expected_mdot)
    # Daily xenon mass flow ~1.42 kg/day
    assert np.isclose(MDOT_MAX * 86400.0, 1.421, atol=0.01)


def test_srp_effective_reduction():
    # SRP reduces effective solar gravity by K/mu_S ~ 1.85e-5
    assert np.isclose(SRP_FRACTION, 1.84589e-5, rtol=1e-3)
    assert np.isclose(MU_EFF_CANON, 1.0 - SRP_FRACTION)


def test_canonical_scaling():
    # Length scale is 1 AU
    assert np.isclose(SCALE_L, 1.495978707e11)
    # Earth circular speed ~29.78 km/s
    assert np.isclose(SCALE_V / 1e3, 29.7847, atol=0.01)
    # Time unit ~58.13 days
    assert np.isclose(DAYS_PER_TU, 58.1324, atol=0.01)
    # Mass unit is initial heliocentric mass (1218 kg)
    assert np.isclose(SCALE_M, 1218.0)
    # Force unit ~7.22 N
    assert np.isclose(SCALE_F, 7.2228, atol=0.01)
