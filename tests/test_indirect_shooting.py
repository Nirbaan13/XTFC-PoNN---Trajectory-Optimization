"""Tests for Indirect Shooting method."""

import numpy as np
import pytest

from trajectory_optimization.indirect_shooting import IndirectShootingSolver


def test_indirect_shooting_solve():
    solver = IndirectShootingSolver(flight_time_days=248.4)
    sol = solver.solve(seed=1000, perturb_guess=False)

    assert sol.converged
    assert np.isclose(sol.flight_time_days, 248.4, atol=0.1)
    assert np.isclose(sol.xenon_used_kg, 276.1, atol=0.2)
    assert np.isclose(sol.final_mass_kg, 941.9, atol=0.2)
    assert sol.arrival_residual <= 1e-6
    assert sol.energy_error <= 2e-6
    assert len(sol.t_days) == 250
    assert len(sol.r_au) == 250
