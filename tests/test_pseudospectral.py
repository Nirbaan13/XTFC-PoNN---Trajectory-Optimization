"""Tests for Pseudospectral Collocation method."""

import numpy as np
import pytest

from trajectory_optimization.pseudospectral import (
    PseudospectralCollocationSolver,
    lgl_nodes_and_weights,
    lgl_differentiation_matrix
)
from trajectory_optimization.constants import SCALE_M


def test_lgl_nodes_and_weights():
    n_nodes = 10
    nodes, weights = lgl_nodes_and_weights(n_nodes)
    assert len(nodes) == n_nodes
    assert len(weights) == n_nodes
    assert np.isclose(nodes[0], -1.0)
    assert np.isclose(nodes[-1], 1.0)
    # Integral of 1 over [-1, 1] is 2
    assert np.isclose(np.sum(weights), 2.0)


def test_lgl_differentiation_matrix():
    n_nodes = 8
    nodes, _ = lgl_nodes_and_weights(n_nodes)
    d = lgl_differentiation_matrix(nodes)
    assert d.shape == (n_nodes, n_nodes)
    # Derivative of constant vector f = 1 is 0
    f_const = np.ones(n_nodes)
    df = d @ f_const
    assert np.allclose(df, 0.0, atol=1e-12)
    # Derivative of linear function f = tau is 1
    df_lin = d @ nodes
    assert np.allclose(df_lin, 1.0, atol=1e-10)


def test_pseudospectral_solve():
    solver = PseudospectralCollocationSolver(flight_time_days=247.9, n_nodes=60)
    sol = solver.solve(seed=1000, perturb_guess=False)

    assert sol.converged
    assert np.isclose(sol.flight_time_days, 247.9, atol=0.1)
    assert 250.0 <= sol.xenon_used_kg <= 320.0
    assert np.isclose(sol.final_mass_kg, SCALE_M - sol.xenon_used_kg, atol=0.1)
    assert sol.arrival_residual <= 1e-5
    assert sol.energy_error <= 1e-3
    assert len(sol.t_days) == 60
    assert len(sol.r_au) == 60
    assert sol.runtime_seconds > 0.0
    assert sol.iterations > 0
