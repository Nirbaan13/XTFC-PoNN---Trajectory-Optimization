"""Script to generate all publication-grade figures: Figure 4.1, 4.2, 4.3 and perturbation plots."""

from pathlib import Path
import numpy as np
from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.benchmarks import run_three_methods
from trajectory_optimization.plotting import (
    plot_figure_4_1,
    plot_figure_4_2_schematic_3d,
    plot_figure_4_3_top_view,
    plot_perturbation_magnitudes,
)


def main():
    print("Generating figures from feasible common scenario (tf=261.0 d, theta_f=192.5175 deg)...")
    prob = HelioProblem(tf_days=261.0, theta_f=float(np.deg2rad(192.5175)))
    bundle = run_three_methods(prob, seed=1000, compute_ref=False)

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_figure_4_1(bundle, fig_dir / "figure_4_1_heliocentric_results.png")
    plot_figure_4_2_schematic_3d(bundle, fig_dir / "figure_4_2_schematic_3d.png")
    plot_figure_4_3_top_view(bundle, fig_dir / "figure_4_3_top_view_2d.png")
    plot_perturbation_magnitudes(fig_dir / "perturbation_accelerations.png")

    print(f"All figures generated in {fig_dir}/")


if __name__ == "__main__":
    main()
