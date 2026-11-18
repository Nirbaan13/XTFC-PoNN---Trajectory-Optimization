"""Tests for Earth parking orbit escape phase simulation (Tables E1 and E2)."""

import numpy as np
import pytest

from trajectory_optimization.escape_phase import (
    propagate_escape_phase,
    generate_table_e1,
    generate_table_e2
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

    # Reference case (J2, i = 28.5 deg)
    c_ref = cases[2]
    assert "28.5" in c_ref.case_name
    assert np.isclose(c_ref.escape_time_days, 225.6, atol=0.2)
    assert np.isclose(c_ref.xenon_used_kg, 320.6, atol=0.2)
    assert np.isclose(c_ref.final_mass_kg, 1218.0, atol=0.2)


def test_table_e2_reproduction():
    methods = generate_table_e2()
    assert len(methods) == 3

    m_names = [m["method"] for m in methods]
    assert "X-TFC PoNN" in m_names
    assert "Indirect shooting" in m_names
    assert "Pseudospectral collocation" in m_names

    for m in methods:
        assert 225.0 <= m["escape_time_days"] <= 226.0
        assert 320.0 <= m["xenon_used_kg"] <= 321.0
        assert 1218.0 <= m["final_mass_kg"] <= 1218.5
