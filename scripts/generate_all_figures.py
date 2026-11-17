"""Script to generate all publication-grade figures: Figure 4.1, 4.2, 4.3 and Table 2."""

from pathlib import Path
from trajectory_optimization.benchmarks import run_representative_benchmarks
from trajectory_optimization.plotting import (
    plot_figure_4_1,
    plot_figure_4_2_schematic_3d,
    plot_figure_4_3_top_view,
    plot_perturbation_magnitudes
)


def main():
    print("Generating all figures...")
    solutions = run_representative_benchmarks(seed=1000)

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_figure_4_1(solutions, fig_dir / "figure_4_1_heliocentric_results.png")
    plot_figure_4_2_schematic_3d(solutions, fig_dir / "figure_4_2_schematic_3d.png")
    plot_figure_4_3_top_view(solutions, fig_dir / "figure_4_3_top_view_2d.png")
    plot_perturbation_magnitudes(fig_dir / "perturbation_accelerations.png")

    print(f"All figures generated successfully in {fig_dir}/")


if __name__ == "__main__":
    main()
