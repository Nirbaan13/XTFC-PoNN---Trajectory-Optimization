"""Tests for Indirect Shooting method on both infeasible (247.9 d) and feasible (261.0 d) problems."""

import numpy as np

from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.indirect_shooting import IndirectShootingSolver
from trajectory_optimization.metrics import evaluate_solution


def test_indirect_shooting_solve():
    solver = IndirectShootingSolver()

    # 1. Paper nominal duration (247.9 d) -> physically infeasible under 0.50 N
    p248 = HelioProblem(tf_days=247.9, theta_f=3.360065)
    r248 = solver.solve(p248, seed=1000)
    assert not r248.success

    # 2. Feasible duration (261.0 d) -> converges through full homotopy
    p261 = HelioProblem(tf_days=261.0, theta_f=3.360065)
    r261 = solver.solve(p261, seed=1000)
    ev = evaluate_solution(p261, r261)
    assert r261.success
    assert r261.arrival_residual < 1e-6
    assert abs(r261.lam_m_tf) < 1e-6
    assert ev["max_thrust_N"] <= 0.500001
    assert ev["thrust_bound_ok"]
    assert ev["runtime_s"] > 0.0
    assert r261.n_propagations > 0
    assert r261.n_rhs > 0
