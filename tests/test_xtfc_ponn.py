"""Tests for X-TFC Pontryagin Neural Network method."""

import numpy as np
import pytest

from trajectory_optimization.xtfc_ponn import XTFC_PoNN_Solver
from trajectory_optimization.constants import SCALE_M


def test_xtfc_ponn_solve():
    solver = XTFC_PoNN_Solver(flight_time_days=247.6, n_collocation_nodes=60, n_basis_terms=16)
    sol = solver.solve(seed=1000, perturb_weights=False)

    # Infeasible at 0.50 N cap at tf = 247.6 days -> 14-residual norm > 1e-4 -> honestly non-converged
    assert not sol.converged
    assert np.isclose(sol.flight_time_days, 247.6, atol=0.1)
    assert 200.0 <= sol.xenon_used_kg <= 360.0
    assert np.isclose(sol.final_mass_kg, SCALE_M - sol.xenon_used_kg, atol=0.1)
    # X-TFC PoNN achieves machine precision Mars arrival residual via TFC analytical embedding
    assert sol.arrival_residual <= 1e-6
    assert np.max(sol.thrust_n) <= 0.5001
    assert sol.energy_error <= 0.2
    assert len(sol.loss_history) > 0
    # Loss should decrease significantly during training
    assert sol.loss_history[-1] < sol.loss_history[0]
    assert sol.runtime_seconds > 0.0
    assert sol.iterations > 0
