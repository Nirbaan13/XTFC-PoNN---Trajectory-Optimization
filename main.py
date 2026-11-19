"""Unified Command-Line Interface (CLI) for Earth-Mars Low-Thrust Trajectory Optimization.

Usage examples:
    python main.py run-all
    python main.py escape
    python main.py method xtfc --tf 261.0
    python main.py method shooting --tf 261.0
    python main.py method pseudospectral --tf 261.0
    python main.py benchmark --runs 20
    python main.py plot
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.indirect_shooting import IndirectShootingSolver
from trajectory_optimization.xtfc_ponn import XTFCPoNNSolver
from trajectory_optimization.pseudospectral import CollocationSolver
from trajectory_optimization.escape_phase import generate_table_e1
from trajectory_optimization.metrics import evaluate_solution
from trajectory_optimization.benchmarks import run_three_methods, run_robustness_protocol, format_table_3_markdown
from trajectory_optimization.plotting import (
    plot_figure_4_1,
    plot_figure_4_2_schematic_3d,
    plot_figure_4_3_top_view,
    plot_perturbation_magnitudes,
)


def cmd_run_all(args):
    import run_all
    run_all.main()


def cmd_escape(args):
    print("--- Table E1: Numerical Escape Phase Propagation ---")
    e1_cases = generate_table_e1()
    for c in e1_cases:
        print(
            f"{c.case_name:30s} | Time: {c.escape_time_days:6.2f} d | "
            f"Fuel: {c.xenon_used_kg:6.2f} kg | Final Mass: {c.final_mass_kg:7.2f} kg | "
            f"End E res: {c.end_state_residual:.2e}"
        )


def cmd_method(args):
    m = args.name.lower()
    seed = args.seed
    prob = HelioProblem(tf_days=args.tf, theta_f=float(np.radians(args.theta_deg)))
    print(f"Executing method: {m} (seed={seed}, tf={args.tf} d, theta_f={args.theta_deg} deg)")

    if m in ["xtfc", "xtfc_ponn", "ponn"]:
        solver = XTFCPoNNSolver()
        raw = solver.solve(prob, seed=seed)
    elif m in ["shooting", "indirect", "indirect_shooting"]:
        solver = IndirectShootingSolver()
        raw = solver.solve(prob, seed=seed)
    elif m in ["pseudospectral", "collocation", "ps"]:
        solver = CollocationSolver(n_nodes=180, allow_slack=True)
        raw = solver.solve(prob, seed=seed)
    else:
        print(f"Unknown method '{args.name}'. Choose from: xtfc, shooting, pseudospectral")
        sys.exit(1)

    ev = evaluate_solution(prob, raw)
    print("\n" + "=" * 60)
    print(f"Results for {ev['method']}")
    print("=" * 60)
    print(f"Converged:              {ev['converged']} ({ev['status']})")
    print(f"Iterations:             {ev['iterations']}")
    print(f"Flight Time:            {ev['tf_days']:.2f} days")
    print(f"Xenon Fuel Used:        {ev['fuel_kg']:.2f} kg")
    print(f"Final Spacecraft Mass:  {ev['final_mass_kg']:.2f} kg")
    print(f"Mars Arrival Residual:  {ev['arrival_residual']:.2e} (re-prop: {ev['reprop_arrival_residual']:.2e})")
    print(f"14-ODE RMS Residual:    {ev['ode14_rms']:.2e}")
    print(f"180-LGL Defect RMS:     {ev['lgl180_defect_rms']:.2e}")
    print(f"Energy Balance Error:   {ev['energy_error']:.2e}")
    print(f"Max Thrust:             {ev['max_thrust_N']:.4f} N")
    print(f"Convergence Runtime:    {ev['runtime_s']:.2f} s ({ev['runtime_min']:.3f} min)")


def cmd_benchmark(args):
    prob248 = HelioProblem(tf_days=247.9, theta_f=float(np.radians(192.52)))
    prob261 = HelioProblem(tf_days=261.0, theta_f=float(np.radians(192.52)))
    s248 = run_robustness_protocol(prob248, n_runs=args.runs)
    s261 = run_robustness_protocol(prob261, n_runs=args.runs)
    print("\n" + format_table_3_markdown(s248, s261, 247.9, 261.0, 192.52))


def cmd_plot(args):
    prob = HelioProblem(tf_days=261.0, theta_f=float(np.radians(192.52)))
    bundle = run_three_methods(prob, seed=1000, compute_ref=False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_figure_4_1(bundle, out_dir / "figure_4_1_heliocentric_results.png")
    plot_figure_4_2_schematic_3d(bundle, out_dir / "figure_4_2_schematic_3d.png")
    plot_figure_4_3_top_view(bundle, out_dir / "figure_4_3_top_view_2d.png")
    plot_perturbation_magnitudes(out_dir / "perturbation_accelerations.png")
    print(f"All plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Earth-Mars Low-Thrust Trajectory Optimization Suite (X-TFC PoNN, Indirect Shooting, Pseudospectral Collocation)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    sub_runall = subparsers.add_parser("run-all", help="Run full pipeline: all methods, tables, and figures")
    sub_runall.set_defaults(func=cmd_run_all)

    sub_escape = subparsers.add_parser("escape", help="Run parking orbit escape phase simulation")
    sub_escape.set_defaults(func=cmd_escape)

    sub_method = subparsers.add_parser("method", help="Run a single optimization method")
    sub_method.add_argument("name", choices=["xtfc", "shooting", "pseudospectral"], help="Method name")
    sub_method.add_argument("--seed", type=int, default=1000, help="Random seed")
    sub_method.add_argument("--tf", type=float, default=261.0, help="Transfer time in days")
    sub_method.add_argument("--theta-deg", type=float, default=192.52, help="Mars arrival longitude in deg")
    sub_method.set_defaults(func=cmd_method)

    sub_bench = subparsers.add_parser("benchmark", help="Run 20-run robustness Monte Carlo benchmark")
    sub_bench.add_argument("--runs", type=int, default=20, help="Number of runs per method")
    sub_bench.set_defaults(func=cmd_benchmark)

    sub_plot = subparsers.add_parser("plot", help="Generate all figures")
    sub_plot.add_argument("--output-dir", default="results/figures", help="Output directory for plots")
    sub_plot.set_defaults(func=cmd_plot)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
