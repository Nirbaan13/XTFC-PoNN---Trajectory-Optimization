"""Tests for Indirect Shooting method."""

import numpy as np
import pytest

from trajectory_optimization.indirect_shooting import IndirectShootingSolver
from trajectory_optimization.constants import SCALE_M


def test_indirect_shooting_solve():
    solver = IndirectShootingSolver(flight_time_days=248.4)
    sol = solver.solve(seed=1000, perturb_guess=False)

    # Infeasible at 0.50 N cap at tf = 248.4 days -> honestly non-converged
    assert not sol.converged
    assert np.isclose(sol.flight_time_days, 248.4, atol=0.1)
    # Check physical fuel bounds under 0.50 N cap (~353 kg)
    assert 340.0 <= sol.xenon_used_kg <= 360.0
    assert np.isclose(sol.final_mass_kg, SCALE_M - sol.xenon_used_kg, atol=0.1)
    # Arrival residual terminates at closest physical boundary (~0.088 AU)
    assert 0.05 <= sol.arrival_residual <= 0.12
    assert np.max(sol.thrust_n) <= 0.5001
    assert sol.energy_error <= 5e-4
    assert len(sol.t_days) == 250
    assert len(sol.r_au) == 250
    assert sol.runtime_seconds > 0.0
    assert sol.iterations > 0
