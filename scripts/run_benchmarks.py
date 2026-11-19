"""Script to run 20-run robustness Monte Carlo benchmark and print Table 3."""

import argparse
import numpy as np
from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.benchmarks import run_robustness_protocol, format_table_3_markdown


def main():
    parser = argparse.ArgumentParser(description="Run 20-run Monte Carlo Robustness Benchmarks")
    parser.add_argument("--runs", type=int, default=20, help="Runs per method")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed")
    parser.add_argument("--theta-deg", type=float, default=192.5175, help="Mars arrival angle in degrees")
    args = parser.parse_args()

    theta_rad = float(np.deg2rad(args.theta_deg))
    prob_248 = HelioProblem(tf_days=247.9, theta_f=theta_rad, label="paper_247.9d")
    prob_feas = HelioProblem(tf_days=261.0, theta_f=theta_rad, label="feasible_261.0d")
    s_248 = run_robustness_protocol(prob_248, n_runs=args.runs, base_seed=args.seed)
    s_feas = run_robustness_protocol(prob_feas, n_runs=args.runs, base_seed=args.seed)
    print("\n" + "=" * 80)
    print(f"TABLE 3: SUCCESS RATE AND SPREAD OVER {args.runs} REPEATED RUNS")
    print("=" * 80)
    print(format_table_3_markdown(s_248, s_feas, 247.9, 261.0, args.theta_deg))


if __name__ == "__main__":
    main()
