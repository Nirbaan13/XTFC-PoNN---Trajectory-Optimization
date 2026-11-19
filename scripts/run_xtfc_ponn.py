"""Script to run X-TFC Pontryagin Neural Network optimization."""

import argparse
import numpy as np
from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.xtfc_ponn import XTFCPoNNSolver
from trajectory_optimization.metrics import evaluate_solution


def main():
    parser = argparse.ArgumentParser(description="Run X-TFC PoNN Method")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--tof", type=float, default=261.0, help="Flight time in days")
    parser.add_argument("--theta-deg", type=float, default=192.5175, help="Mars arrival angle in degrees")
    parser.add_argument("--neurons", type=int, default=160, help="Hidden neurons L")
    parser.add_argument("--nodes", type=int, default=280, help="Collocation points N")
    args = parser.parse_args()

    prob = HelioProblem(tf_days=args.tof, theta_f=float(np.deg2rad(args.theta_deg)))
    print(
        f"Solving Earth-Mars transfer with X-TFC PoNN "
        f"(seed={args.seed}, TOF={args.tof} d, L={args.neurons}, N={args.nodes})..."
    )
    solver = XTFCPoNNSolver(n_neurons=args.neurons, n_points=args.nodes)
    sol = solver.solve(prob, seed=args.seed)
    ev = evaluate_solution(prob, sol)

    print("\n--- X-TFC PoNN Results ---")
    print(f"Status:                 {'CONVERGED' if ev['converged'] else 'NON-CONVERGED'} ({ev['status']})")
    print(f"LM Iterations:          {ev['iterations']}")
    print(f"Function / Jac Evals:   {ev['func_evals']} / {ev['jac_evals']}")
    print(f"Warm Start Disclosed:   {sol.warm_start}")
    print(f"Initial Physics Loss:   {sol.loss_history[0]:.2e}")
    print(f"Final Physics Loss:     {sol.loss_history[-1]:.2e}")
    print(f"Flight Time:            {ev['tf_days']:.2f} days")
    print(f"Propellant Used:        {ev['fuel_kg']:.2f} kg")
    print(f"Final Spacecraft Mass:  {ev['final_mass_kg']:.2f} kg")
    print(f"TFC Boundary Residual:  {ev['arrival_residual']:.2e}")
    print(f"Propagated Residual:    {ev['reprop_arrival_residual']:.2e}")
    print(f"Energy Balance Error:   {ev['energy_error']:.2e}")
    print(f"14-ODE RMS Residual:    {ev['ode14_rms']:.2e}")
    print(f"Wall-Clock Time:        {ev['runtime_s']:.2f} s")


if __name__ == "__main__":
    main()
