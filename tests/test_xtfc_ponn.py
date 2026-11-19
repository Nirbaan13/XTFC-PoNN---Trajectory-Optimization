"""Tests for X-TFC Pontryagin Neural Network method."""

import numpy as np

from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.xtfc_ponn import XTFCPoNNSolver
from trajectory_optimization.metrics import evaluate_solution


def test_xtfc_ponn_solve():
    solver = XTFCPoNNSolver(n_neurons=160, n_points=280)

    # 1. Paper nominal duration (247.9 d) -> infeasible under 0.50 N -> reports non-convergence
    p248 = HelioProblem(tf_days=247.9, theta_f=3.360065)
    r248 = solver.solve(p248, seed=1000)
    assert not r248.success
    assert r248.arrival_residual < 1e-12  # TFC analytical boundary embedding is exact

    # 2. Feasible duration (261.0 d) -> converges through full homotopy
    p261 = HelioProblem(tf_days=261.0, theta_f=3.360065)
    r261 = solver.solve(p261, seed=1000)
    ev = evaluate_solution(p261, r261)
    assert r261.success
    assert r261.arrival_residual < 1e-12
    assert r261.residual_rms < 1e-4
    assert ev["max_thrust_N"] <= 0.500001
    assert len(r261.loss_history) == r261.iterations
    assert r261.runtime_s > 0.0
