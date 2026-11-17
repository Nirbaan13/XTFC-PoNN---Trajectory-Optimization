"""Script to run Pseudospectral Collocation optimization on 180 LGL nodes."""

import argparse
from trajectory_optimization.pseudospectral import PseudospectralCollocationSolver


def main():
    parser = argparse.ArgumentParser(description="Run Pseudospectral Collocation Method")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--tof", type=float, default=247.9, help="Flight time in days")
    parser.add_argument("--nodes", type=int, default=180, help="Number of LGL nodes")
    args = parser.parse_args()

    print(f"Solving Earth-Mars transfer with Pseudospectral Collocation ({args.nodes} nodes, seed={args.seed}, TOF={args.tof} d)...")
    solver = PseudospectralCollocationSolver(flight_time_days=args.tof, n_nodes=args.nodes)
    sol = solver.solve(seed=args.seed)

    print("\n--- Pseudospectral Collocation Results ---")
    print(f"Status:                 {'CONVERGED' if sol.converged else 'FAILED'}")
    print(f"Discretization Nodes:   {args.nodes} Legendre-Gauss-Lobatto")
    print(f"NLP Iterations:         {sol.iterations}")
    print(f"Flight Time:            {sol.flight_time_days:.1f} days")
    print(f"Propellant Used:        {sol.xenon_used_kg:.1f} kg")
    print(f"Final Spacecraft Mass:  {sol.final_mass_kg:.1f} kg")
    print(f"Mars Arrival Residual:  {sol.arrival_residual:.2e}")
    print(f"Energy Loss Error:      {sol.energy_error:.2e}")
    print(f"Wall-Clock Time:        {sol.runtime_seconds/60.0:.1f} min")


if __name__ == "__main__":
    main()
