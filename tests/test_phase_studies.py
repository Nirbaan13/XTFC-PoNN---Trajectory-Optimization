"""Unit tests for Phase 1 / Phase 2 utilities (throttle switches, Wilson CI, fixed-theta min-time)."""

import numpy as np

from trajectory_optimization.metrics import analyze_throttle_profile, wilson_ci
from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.pseudospectral import solve_min_time_fixed_theta


def test_analyze_throttle_profile_bang_off_bang():
    t_days = np.linspace(0.0, 300.0, 301)
    T = np.full_like(t_days, 0.50)
    T[(t_days >= 100.0) & (t_days <= 200.0)] = 0.0
    switches, coast_days = analyze_throttle_profile(t_days, T, t_max_newton=0.50)
    assert switches == 2
    assert 99.0 <= coast_days <= 102.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0
    assert 0.14 < hi < 0.18
    lo2, hi2 = wilson_ci(20, 20)
    assert 0.82 < lo2 < 0.86
    assert hi2 == 1.0
    lo3, hi3 = wilson_ci(80, 100)
    assert 0.70 < lo3 < 0.72
    assert 0.86 < hi3 < 0.88


def test_solve_min_time_fixed_theta_210deg():
    prob = HelioProblem(tf_days=270.0, theta_f=float(np.radians(210.0)))
    res = solve_min_time_fixed_theta(prob, theta_f_rad=float(np.radians(210.0)), n_nodes=60, guess_mode="direct", max_iter=150)
    assert res["success"]
    assert 260.0 < res["tf_days"] < 290.0
    assert res["reprop_arrival_residual"] < 5e-2
