"""Tests for Earth parking orbit escape phase simulation (Table E1)."""

import numpy as np
import pytest

from trajectory_optimization.escape_phase import (
    propagate_escape_phase,
    generate_table_e1,
)


def test_table_e1_reproduction():
    cases = generate_table_e1()
    assert len(cases) == 4

    # Control case (No J2)
    c0 = cases[0]
    assert "No J2" in c0.case_name
    assert np.isclose(c0.escape_time_days, 225.6, atol=0.2)
    assert np.isclose(c0.xenon_used_kg, 320.6, atol=0.2)
    assert np.isclose(c0.final_mass_kg, 1218.0, atol=0.2)
    assert c0.end_state_residual < 1e-6

    # Reference case (J2, i = 28.5 deg)
    c_ref = cases[2]
    assert "28.5" in c_ref.case_name
    assert np.isclose(c_ref.escape_time_days, 225.6, atol=0.2)
    assert np.isclose(c_ref.xenon_used_kg, 320.6, atol=0.2)
    assert np.isclose(c_ref.final_mass_kg, 1218.0, atol=0.2)
    assert c_ref.end_state_residual < 1e-6
    assert c_ref.n_rhs_evals > 1000
