"""Script to run 20-run robustness Monte Carlo benchmark and print Table 3."""

import argparse
from trajectory_optimization.benchmarks import run_robustness_protocol, format_table_3_markdown


def main():
    parser = argparse.ArgumentParser(description="Run 20-run Monte Carlo Robustness Benchmarks")
    parser.add_argument("--runs", type=int, default=20, help="Runs per method")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed")
    args = parser.parse_args()

    stats = run_robustness_protocol(n_runs=args.runs, base_seed=args.seed)
    print("\n" + "=" * 80)
    print("TABLE 3: SUCCESS RATE AND SPREAD OVER 20 REPEATED RUNS (MEAN ± SD)")
    print("=" * 80)
    print(format_table_3_markdown(stats))


if __name__ == "__main__":
    main()
