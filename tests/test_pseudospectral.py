"""Tests for Pseudospectral Collocation method (180 LGL nodes)."""

import numpy as np

from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.pseudospectral import (
    CollocationSolver,
    lgl_nodes_and_weights,
    lgl_differentiation_matrix,
)
from trajectory_optimization.metrics import evaluate_solution


def test_lgl_nodes_and_weights():
    n_nodes = 180
    nodes, weights = lgl_nodes_and_weights(n_nodes)
    assert len(nodes) == n_nodes
    assert len(weights) == n_nodes
    assert np.isclose(nodes[0], -1.0)
    assert np.isclose(nodes[-1], 1.0)
    assert np.isclose(np.sum(weights), 2.0)


def test_lgl_differentiation_matrix():
    n_nodes = 180
    nodes, _ = lgl_nodes_and_weights(n_nodes)
    d = lgl_differentiation_matrix(nodes)
    assert d.shape == (n_nodes, n_nodes)
    assert np.allclose(d @ np.ones(n_nodes), 0.0, atol=1e-11)
    assert np.allclose(d @ nodes, 1.0, atol=1e-10)
    assert np.allclose(d @ np.sin(3.0 * nodes), 3.0 * np.cos(3.0 * nodes), atol=1e-10)


def test_pseudospectral_solve():
    solver = CollocationSolver(n_nodes=180, allow_slack=True)
    p261 = HelioProblem(tf_days=261.0, theta_f=3.360065)
    res = solver.solve(p261, seed=1000)
    ev = evaluate_solution(p261, res)
    assert res.success
    assert not res.slack_used
    assert res.terminal_violation < 1e-8
    assert ev["max_thrust_N"] <= 0.500001
    assert ev["thrust_bound_ok"]
    assert res.ipopt_iterations > 0
    assert res.runtime_s > 0.0

