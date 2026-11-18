"""Tests for Indirect Shooting method."""

import numpy as np
import pytest

from trajectory_optimization.indirect_shooting import IndirectShootingSolver
from trajectory_optimization.constants import SCALE_M


def test_indirect_shooting_solve():
    solver = IndirectShootingSolver(flight_time_days=248.4)
    sol = solver.solve(seed=1000, perturb_guess=False)

    assert sol.converged
    assert np.isclose(sol.flight_time_days, 248.4, atol=0.1)
    # Check physical fuel bounds
    assert 200.0 <= sol.xenon_used_kg <= 320.0
    assert np.isclose(sol.final_mass_kg, SCALE_M - sol.xenon_used_kg, atol=0.1)
    assert sol.arrival_residual <= 1e-5
    assert sol.energy_error <= 1e-4
    assert len(sol.t_days) == 250
    assert len(sol.r_au) == 250
    assert sol.runtime_seconds > 0.0
    assert sol.iterations > 0
