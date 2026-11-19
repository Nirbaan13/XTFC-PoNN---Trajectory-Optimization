"""Script to run Pseudospectral Collocation optimization on 180 LGL nodes."""

import argparse
import numpy as np
from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.pseudospectral import CollocationSolver
from trajectory_optimization.metrics import evaluate_solution


def main():
    parser = argparse.ArgumentParser(description="Run Pseudospectral Collocation Method")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--tof", type=float, default=261.0, help="Flight time in days")
    parser.add_argument("--theta-deg", type=float, default=192.5175, help="Mars arrival angle in degrees")
    parser.add_argument("--nodes", type=int, default=180, help="Number of LGL nodes")
    args = parser.parse_args()

    prob = HelioProblem(tf_days=args.tof, theta_f=float(np.deg2rad(args.theta_deg)))
    print(
        f"Solving Earth-Mars transfer with Pseudospectral Collocation "
        f"({args.nodes} nodes, seed={args.seed}, TOF={args.tof} d)..."
    )
    solver = CollocationSolver(n_nodes=args.nodes, allow_slack=True)
    sol = solver.solve(prob, seed=args.seed)
    ev = evaluate_solution(prob, sol)

    print("\n--- Pseudospectral Collocation Results ---")
    print(f"Status:                 {'CONVERGED' if ev['converged'] else 'NON-CONVERGED'} ({ev['status']})")
    print(f"Discretization Nodes:   {args.nodes} Legendre-Gauss-Lobatto")
    print(f"IPOPT Iterations:       {ev['iterations']}")
    print(f"Function / Jac Evals:   {ev['func_evals']} / {ev['jac_evals']}")
    print(f"Slack Fallback Used:    {sol.slack_used} (norm={sol.slack_norm:.2e})")
    print(f"Flight Time:            {ev['tf_days']:.2f} days")
    print(f"Propellant Used:        {ev['fuel_kg']:.2f} kg")
    print(f"Final Spacecraft Mass:  {ev['final_mass_kg']:.2f} kg")
    print(f"Nodal Arrival Residual: {ev['arrival_residual']:.2e}")
    print(f"Propagated Residual:    {ev['reprop_arrival_residual']:.2e}")
    print(f"Energy Balance Error:   {ev['energy_error']:.2e}")
    print(f"Wall-Clock Time:        {ev['runtime_s']:.2f} s")


if __name__ == "__main__":
    main()
