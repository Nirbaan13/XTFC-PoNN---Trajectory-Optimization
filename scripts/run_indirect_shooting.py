"""Script to run Indirect Shooting optimization for perturbed Earth-Mars transfer."""

import argparse
from trajectory_optimization.indirect_shooting import IndirectShootingSolver


def main():
    parser = argparse.ArgumentParser(description="Run Indirect Shooting Method")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--tof", type=float, default=248.4, help="Flight time in days")
    args = parser.parse_args()

    print(f"Solving Earth-Mars transfer with Indirect Shooting (seed={args.seed}, TOF={args.tof} d)...")
    solver = IndirectShootingSolver(flight_time_days=args.tof)
    sol = solver.solve(seed=args.seed)

    print("\n--- Indirect Shooting Results ---")
    print(f"Status:                 {'CONVERGED' if sol.converged else 'FAILED'}")
    print(f"Nonlinear Iterations:   {sol.iterations}")
    print(f"Flight Time:            {sol.flight_time_days:.1f} days")
    print(f"Propellant Used:        {sol.xenon_used_kg:.1f} kg")
    print(f"Final Spacecraft Mass:  {sol.final_mass_kg:.1f} kg")
    print(f"Mars Arrival Residual:  {sol.arrival_residual:.2e}")
    print(f"Energy Loss Error:      {sol.energy_error:.2e}")
    print(f"Wall-Clock Time:        {sol.runtime_seconds/60.0:.1f} min")


if __name__ == "__main__":
    main()
