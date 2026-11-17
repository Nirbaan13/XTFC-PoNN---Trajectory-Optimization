"""Tests for X-TFC Pontryagin Neural Network method."""

import numpy as np
import pytest

from trajectory_optimization.xtfc_ponn import XTFC_PoNN_Solver


def test_xtfc_ponn_solve():
    solver = XTFC_PoNN_Solver(flight_time_days=247.6, n_epochs=500)
    sol = solver.solve(seed=1000, perturb_weights=False)

    assert sol.converged
    assert np.isclose(sol.flight_time_days, 247.6, atol=0.1)
    assert np.isclose(sol.xenon_used_kg, 274.8, atol=0.2)
    assert np.isclose(sol.final_mass_kg, 943.2, atol=0.2)
    # X-TFC PoNN achieves smallest Mars arrival residual
    assert sol.arrival_residual <= 5e-7
    assert sol.energy_error <= 1.5e-6
    assert len(sol.loss_history) > 0
    # Loss should decrease exponentially
    assert sol.loss_history[-1] < sol.loss_history[0]
