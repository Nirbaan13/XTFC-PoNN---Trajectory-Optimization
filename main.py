"""Unified Command-Line Interface (CLI) for Earth-Mars Low-Thrust Trajectory Optimization.

Usage examples:
    python main.py run-all
    python main.py escape
    python main.py method xtfc
    python main.py method shooting
    python main.py method pseudospectral
    python main.py benchmark --runs 20
    python main.py plot
"""

import argparse
import sys
from pathlib import Path

from trajectory_optimization.indirect_shooting import IndirectShootingSolver
from trajectory_optimization.xtfc_ponn import XTFC_PoNN_Solver
from trajectory_optimization.pseudospectral import PseudospectralCollocationSolver
from trajectory_optimization.escape_phase import generate_table_e1, generate_table_e2
from trajectory_optimization.benchmarks import run_robustness_protocol, format_table_3_markdown
from trajectory_optimization.plotting import (
    plot_figure_4_1,
    plot_figure_4_2_schematic_3d,
    plot_figure_4_3_top_view,
    plot_perturbation_magnitudes
)


def cmd_run_all(args):
    import run_all
    run_all.main()


def cmd_escape(args):
    print("--- Table E1: Numerical Escape Phase Propagation ---")
    e1_cases = generate_table_e1()
    for c in e1_cases:
        print(f"{c.case_name:30s} | Time: {c.escape_time_days:5.1f} d | Fuel: {c.xenon_used_kg:5.1f} kg | Final Mass: {c.final_mass_kg:6.1f} kg")

    print("\n--- Table E2: Escape Phase Solved by the Three Methods ---")
    e2_cases = generate_table_e2()
    for c in e2_cases:
        print(f"{c['method']:30s} | Time: {c['escape_time_days']:6.2f} d | Fuel: {c['xenon_used_kg']:5.1f} kg | Residual: {c['end_state_residual']}")


def cmd_method(args):
    m = args.name.lower()
    seed = args.seed
    print(f"Executing method: {m} (seed={seed})")
    
    if m in ["xtfc", "xtfc_ponn", "ponn"]:
        solver = XTFC_PoNN_Solver()
        sol = solver.solve(seed=seed)
    elif m in ["shooting", "indirect", "indirect_shooting"]:
        solver = IndirectShootingSolver()
        sol = solver.solve(seed=seed)
    elif m in ["pseudospectral", "collocation", "ps"]:
        solver = PseudospectralCollocationSolver()
        sol = solver.solve(seed=seed)
    else:
        print(f"Unknown method '{args.name}'. Choose from: xtfc, shooting, pseudospectral")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Results for {sol.method_name}")
    print("=" * 60)
    print(f"Converged:              {sol.converged}")
    print(f"Iterations:             {sol.iterations}")
    print(f"Flight Time:            {sol.flight_time_days:.1f} days")
    print(f"Xenon Fuel Used:        {sol.xenon_used_kg:.1f} kg")
    print(f"Final Spacecraft Mass:  {sol.final_mass_kg:.1f} kg")
    print(f"Mars Arrival Residual:  {sol.arrival_residual:.2e}")
    print(f"Energy Conservation Err:{sol.energy_error:.2e}")
    print(f"Convergence Runtime:    {sol.runtime_seconds/60.0:.1f} min")


def cmd_benchmark(args):
    print(f"Executing robustness protocol with {args.runs} runs per method...")
    stats = run_robustness_protocol(n_runs=args.runs)
    print("\n" + format_table_3_markdown(stats))


def cmd_plot(args):
    from trajectory_optimization.benchmarks import run_representative_benchmarks
    solutions = run_representative_benchmarks(seed=1000)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_figure_4_1(solutions, out_dir / "figure_4_1_heliocentric_results.png")
    plot_figure_4_2_schematic_3d(solutions, out_dir / "figure_4_2_schematic_3d.png")
    plot_figure_4_3_top_view(solutions, out_dir / "figure_4_3_top_view_2d.png")
    plot_perturbation_magnitudes(out_dir / "perturbation_accelerations.png")
    print(f"All plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Earth-Mars Low-Thrust Trajectory Optimization Suite (X-TFC PoNN, Indirect Shooting, Pseudospectral Collocation)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # run-all
    sub_runall = subparsers.add_parser("run-all", help="Run full pipeline: all methods, tables, and figures")
    sub_runall.set_defaults(func=cmd_run_all)

    # escape
    sub_escape = subparsers.add_parser("escape", help="Run parking orbit escape phase simulation")
    sub_escape.set_defaults(func=cmd_escape)

    # method
    sub_method = subparsers.add_parser("method", help="Run a single optimization method")
    sub_method.add_argument("name", choices=["xtfc", "shooting", "pseudospectral"], help="Method name")
    sub_method.add_argument("--seed", type=int, default=1000, help="Random seed")
    sub_method.set_defaults(func=cmd_method)

    # benchmark
    sub_bench = subparsers.add_parser("benchmark", help="Run 20-run robustness Monte Carlo benchmark")
    sub_bench.add_argument("--runs", type=int, default=20, help="Number of runs per method")
    sub_bench.set_defaults(func=cmd_benchmark)

    # plot
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
