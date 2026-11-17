"""Script to run X-TFC Pontryagin Neural Network optimization."""

import argparse
from trajectory_optimization.xtfc_ponn import XTFC_PoNN_Solver


def main():
    parser = argparse.ArgumentParser(description="Run X-TFC PoNN Method")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--tof", type=float, default=247.6, help="Flight time in days")
    parser.add_argument("--epochs", type=int, default=4300, help="Training epochs")
    args = parser.parse_args()

    print(f"Solving Earth-Mars transfer with X-TFC PoNN (seed={args.seed}, TOF={args.tof} d, epochs={args.epochs})...")
    solver = XTFC_PoNN_Solver(flight_time_days=args.tof, n_epochs=args.epochs)
    sol = solver.solve(seed=args.seed)

    print("\n--- X-TFC PoNN Results ---")
    print(f"Status:                 {'CONVERGED' if sol.converged else 'FAILED'}")
    print(f"Training Epochs:        {sol.iterations}")
    print(f"Initial Physics Loss:   {sol.loss_history[0]:.2e}")
    print(f"Final Physics Loss:     {sol.loss_history[-1]:.2e}")
    print(f"Flight Time:            {sol.flight_time_days:.1f} days")
    print(f"Propellant Used:        {sol.xenon_used_kg:.1f} kg")
    print(f"Final Spacecraft Mass:  {sol.final_mass_kg:.1f} kg")
    print(f"Mars Arrival Residual:  {sol.arrival_residual:.2e}")
    print(f"Energy Loss Error:      {sol.energy_error:.2e}")
    print(f"Wall-Clock Time:        {sol.runtime_seconds/60.0:.1f} min")


if __name__ == "__main__":
    main()
