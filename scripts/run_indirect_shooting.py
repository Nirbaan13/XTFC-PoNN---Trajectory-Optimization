"""Script to run Indirect Shooting optimization for perturbed Earth-Mars transfer."""

import argparse
import numpy as np
from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.indirect_shooting import IndirectShootingSolver
from trajectory_optimization.metrics import evaluate_solution


def main():
    parser = argparse.ArgumentParser(description="Run Indirect Shooting Method")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--tof", type=float, default=261.0, help="Flight time in days")
    parser.add_argument("--theta-deg", type=float, default=192.5175, help="Mars arrival angle in degrees")
    args = parser.parse_args()

    prob = HelioProblem(tf_days=args.tof, theta_f=float(np.deg2rad(args.theta_deg)))
    print(
        f"Solving Earth-Mars transfer with Indirect Shooting "
        f"(seed={args.seed}, TOF={args.tof} d, theta_f={args.theta_deg} deg)..."
    )
    solver = IndirectShootingSolver()
    sol = solver.solve(prob, seed=args.seed)
    ev = evaluate_solution(prob, sol)

    print("\n--- Indirect Shooting Results ---")
    print(f"Status:                 {'CONVERGED' if ev['converged'] else 'NON-CONVERGED'} ({ev['status']})")
    print(f"Nonlinear Iterations:   {ev['iterations']}")
    print(f"Function / Jac Evals:   {ev['func_evals']} / {ev['jac_evals']}")
    print(f"Flight Time:            {ev['tf_days']:.2f} days")
    print(f"Propellant Used:        {ev['fuel_kg']:.2f} kg")
    print(f"Final Spacecraft Mass:  {ev['final_mass_kg']:.2f} kg")
    print(f"Mars Arrival Residual:  {ev['arrival_residual']:.2e}")
    print(f"Energy Balance Error:   {ev['energy_error']:.2e}")
    print(f"14-ODE RMS Residual:    {ev['ode14_rms']:.2e}")
    print(f"Max Thrust:             {ev['max_thrust_N']:.4f} N")
    print(f"Wall-Clock Time:        {ev['runtime_s']:.2f} s")


if __name__ == "__main__":
    main()
