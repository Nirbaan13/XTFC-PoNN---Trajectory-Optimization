"""Tests for Tier 3.9 Split-Domain PoNN analytic Jacobian verification."""

from __future__ import annotations

import numpy as np

from trajectory_optimization.scenario_a import ScenarioAProblem
from trajectory_optimization.split_domain_ponn import SplitDomainPoNN


def test_split_domain_analytic_jacobian_matches_fd():
    """Verify JAX analytic Jacobian of SplitDomainPoNN against central finite differences."""
    prob = ScenarioAProblem()
    sd = SplitDomainPoNN(prob, thrust_arcs=[1.0, 0.0, 1.0, 0.0, 1.0], n_neurons_arc=6, n_points_arc=10)
    rng = np.random.default_rng(7000)

    # Build a physically reasonable test vector theta (non-degenerate r, m > 0, ordered switches)
    theta = np.zeros(sd.n_theta, dtype=float)
    betas = np.zeros((sd.M, 8, sd.L))
    betas[:, 0:3, :] = rng.uniform(-1e-3, 1e-3, size=(sd.M, 3, sd.L))
    betas[0, 3, :] = rng.uniform(-1e-2, 1e-2, size=sd.L)
    betas[1:, 3, 0] = 0.8  # positive mass ~0.8
    betas[1:, 3, 1:] = rng.uniform(-1e-2, 1e-2, size=(sd.M - 1, sd.L - 1))
    betas[:, 4:7, 0] = rng.uniform(0.2, 0.8, size=(sd.M, 3))
    betas[:, 4:7, 1:] = rng.uniform(-1e-2, 1e-2, size=(sd.M, 3, sd.L - 1))
    betas[:, 7, :] = rng.uniform(-0.1, 0.1, size=(sd.M, sd.L))

    t_sw = np.linspace(0.2 * prob.tf, 0.8 * prob.tf, sd.K)
    r_if = np.array([prob.r0 + (k + 1) / (sd.K + 1) * (prob.rf - prob.r0) for k in range(sd.K)])
    v_if = np.array([prob.v0 + (k + 1) / (sd.K + 1) * (prob.vf - prob.v0) for k in range(sd.K)])
    theta = np.concatenate([betas.reshape(-1), r_if.reshape(-1), v_if.reshape(-1), t_sw])

    r0, J_an = sd.residual_and_jacobian(theta)
    assert np.all(np.isfinite(r0))
    assert np.all(np.isfinite(J_an))

    # Spot-check 40 columns including ALL switch-time columns and interface columns
    cols = list(range(sd.n_beta, sd.n_theta)) + list(rng.choice(sd.n_beta, size=15, replace=False))
    eps = 1e-7
    max_rel = 0.0
    for c_idx in cols:
        tp = theta.copy(); tp[c_idx] += eps
        tm = theta.copy(); tm[c_idx] -= eps
        j_fd = (sd.residual(tp) - sd.residual(tm)) / (2.0 * eps)
        err = float(np.linalg.norm(J_an[:, c_idx] - j_fd))
        rel = err / max(float(np.linalg.norm(J_an[:, c_idx])), 1.0)
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-7
