"""Master execution script for Earth-Mars Low-Thrust Trajectory Optimization.

Executes all methods, computes all five performance metrics, generates all
tables (Table 1, Table 2, Table 3, Table E1, Table E2), renders all publication
figures (Figure 4.1, Figure 4.2, Figure 4.3, Perturbation accelerations), and
saves all results to the results/ directory.
"""

from pathlib import Path
import time
import numpy as np

from trajectory_optimization.benchmarks import (
    run_representative_benchmarks,
    run_robustness_protocol,
    format_table_3_markdown,
    format_table_e1_markdown,
    format_table_e2_markdown
)
from trajectory_optimization.metrics import (
    compute_table_2_perturbations,
    format_table_2_markdown
)
from trajectory_optimization.plotting import (
    plot_figure_4_1,
    plot_figure_4_2_schematic_3d,
    plot_figure_4_3_top_view,
    plot_perturbation_magnitudes
)
from trajectory_optimization.additions import (
    run_unperturbed_control_benchmarks,
    run_cross_validation
)


def generate_table_1_markdown(solutions: dict) -> str:
    """Generates Table 1 comparing initial parameters and converged results."""
    xtfc = solutions["X-TFC PoNN"]
    shooting = solutions["Indirect Shooting"]
    ps = solutions["Pseudospectral Collocation"]

    md = [
        "# Table 1: Initial Mission, Spacecraft, Propulsion, Orbital and Numerical Parameters\n",
        "| Parameter Category | Parameter | Symbol | Assumed Initial Value / Setting | Unit | Application / Justification |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Mission | Departure body | – | Earth | – | Initial planetary body |",
        "| Mission | Arrival body | – | Mars | – | Terminal planetary body |",
        "| Mission | Mission type | – | Low-thrust interplanetary transfer | – | Common mission definition for all algorithms |",
        "| Mission | Transfer type | – | Earth–Mars heliocentric transfer | – | Same for X-TFC, indirect shooting and collocation |",
        "| Mission | Nominal transfer time | tf | 247.6–248.4 | days | Heliocentric transfer time; nominal target ≈248 days |",
        "| Mission | Reference transfer time | tf,ref | 247.9 | days | Representative common value for numerical initialization |",
        "| Mission | Total mission time | ttot | ≈474 | days | Escape (≈226 days) plus heliocentric (≈248 days) |",
        "| Mission | Propulsion mode | – | Solar-electric ion propulsion | – | Consistent with Dawn-type spacecraft configuration |",
        "| Mission | Eclipse modeling | – | Not included | – | Explicit methodological assumption |",
        "| Spacecraft | Mass at start of heliocentric phase | m0 | 1,218 | kg | Dawn-class mass used at start of heliocentric phase |",
        "| Spacecraft | Xenon available for heliocentric | mXe,0 | ≈300 | kg | Must exceed reported consumption of 274.8–276.1 kg |",
        "| Spacecraft | Final spacecraft mass | mf | ≈941.9–943.2 | kg | Initial mass minus xenon consumption |",
        "| Spacecraft | Propellant | – | Xenon | – | Ion propulsion propellant |",
        "| Spacecraft | Propellant consumption | Δmprop | 274.8–276.1 | kg | Heliocentric phase |",
        "| Spacecraft | Spacecraft dry mass | mdry | ≈918 | kg | m0 − mXe,0 = 1,218 − 300 kg |",
        "| Spacecraft | Solar-array configuration | – | Large deployable solar arrays | – | Consistent with Dawn-type configuration |",
        "| Propulsion | Maximum thrust | Tmax | ≈0.50 | N | Uprated Dawn-derived thruster set |",
        "| Propulsion | Minimum thrust | Tmin | 0 | N | Allows coast/low-thrust control |",
        "| Propulsion | Specific impulse | Isp | ≈3,100 | s | Representative ion-propulsion assumption |",
        "| Propulsion | Standard gravity | g0 | 9.80665 | m/s² | Standard gravitational acceleration |",
        "| Propulsion | Maximum mass-flow rate | ṁmax | ≈1.64 × 10⁻⁵ | kg/s | Tmax / (Isp * g0) |",
        "| Earth Departure | Parking-orbit altitude | hp | ≈500 | km | Initial circular parking orbit |",
        "| Earth Departure | Earth equatorial radius | RE | 6,378.137 | km | Earth gravitational model |",
        "| Earth Departure | Earth gravitational parameter | μE | 398,600.4418 | km³/s² | Earth departure dynamics |",
        "| Earth Departure | Earth J2 coefficient | J2 | 1.08262668 × 10⁻³ | – | Used in J2 acceleration model |",
        "| Earth Departure | Departure perturbation | – | Earth J2 | – | Applied during parking-orbit escape |",
        "| Heliocentric Dynamics | Sun gravitational parameter | μS | 1.32712440018 × 10¹¹ | km³/s² | Main heliocentric gravitational parameter |",
        "| Heliocentric Dynamics | Earth heliocentric orbit radius | rE | ≈1.0 | AU | Initial heliocentric condition |",
        "| Heliocentric Dynamics | Mars heliocentric orbit radius | rM | ≈1.524 | AU | Terminal heliocentric condition |",
        "| Heliocentric Dynamics | Astronomical unit | AU | 149,597,870.7 | km | Distance normalization |",
        "| Solar Radiation Pressure | SRP included | – | Yes | – | Applied during heliocentric transfer |",
        "| Solar Radiation Pressure | Solar pressure at 1 AU | P0 | ≈4.56 × 10⁻⁶ | N/m² | Reference solar radiation pressure |",
        "| Solar Radiation Pressure | Reflectivity coefficient | CR | ≈1.2 | – | Spacecraft optical coefficient |",
        "| Solar Radiation Pressure | Area-to-mass ratio | A/m | ≈0.02 | m²/kg | Large-solar-array configuration |",
        "| Solar Radiation Pressure | SRP acceleration at 1 AU | aSRP,1AU | ≈1.09 × 10⁻⁷ | m/s² | P0 * CR * A/m |",
        "| Solar Radiation Pressure | SRP acceleration at Mars | aSRP,Mars | ≈4.7 × 10⁻⁸ | m/s² | Scaled by 1/r² to 1.524 AU |",
        "\n### Representative Method Performance (Seed 1000)\n",
        "| Method | Propellant Used (kg) | Flight Time (days) | Final Mass (kg) | Mars Arrival Residual | Energy Error | Runtime (s) | Runtime (min) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| X-TFC PoNN | {xtfc.xenon_used_kg:.1f} | {xtfc.flight_time_days:.1f} | {xtfc.final_mass_kg:.1f} | {xtfc.arrival_residual:.1e} | {xtfc.energy_error:.1e} | {xtfc.runtime_seconds:.2f} s | {xtfc.runtime_seconds/60.0:.3f} |",
        f"| Indirect Shooting | {shooting.xenon_used_kg:.1f} | {shooting.flight_time_days:.1f} | {shooting.final_mass_kg:.1f} | {shooting.arrival_residual:.1e} | {shooting.energy_error:.1e} | {shooting.runtime_seconds:.2f} s | {shooting.runtime_seconds/60.0:.3f} |",
        f"| Pseudospectral Collocation | {ps.xenon_used_kg:.1f} | {ps.flight_time_days:.1f} | {ps.final_mass_kg:.1f} | {ps.arrival_residual:.1e} | {ps.energy_error:.1e} | {ps.runtime_seconds:.2f} s | {ps.runtime_seconds/60.0:.3f} |"
    ]
    return "\n".join(md)


def main():
    print("=" * 80)
    print("EARTH-MARS LOW-THRUST TRAJECTORY OPTIMIZATION")
    print("Benchmarking X-TFC PoNN, Indirect Shooting, and Pseudospectral Collocation")
    print("=" * 80)

    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Representative Runs (Seed 1000)
    print("\n[Step 1/5] Solving Representative Heliocentric Transfers (Seed 1000)...")
    solutions = run_representative_benchmarks(seed=1000)

    # 2. Generate Table 1 & Save
    table_1_md = generate_table_1_markdown(solutions)
    (tables_dir / "table_1_mission_parameters.md").write_text(table_1_md)
    print("  -> Table 1 saved to results/tables/table_1_mission_parameters.md")

    # 3. Generate Table 2 (Perturbation Magnitudes)
    print("\n[Step 2/5] Quantifying J2 and SRP Perturbation Accelerations (Table 2)...")
    t2_rows = compute_table_2_perturbations()
    table_2_md = "# Table 2: Size of the J2 and SRP accelerations compared with thrust and central gravity\n\n" + format_table_2_markdown(t2_rows)
    (tables_dir / "table_2_perturbation_magnitudes.md").write_text(table_2_md)
    print("  -> Table 2 saved to results/tables/table_2_perturbation_magnitudes.md")

    # 4. Generate Escape Phase Tables (Table E1 & Table E2)
    print("\n[Step 3/5] Propagating 500 km Parking Orbit Escape Phase (Tables E1 & E2)...")
    table_e1_md = "# Table E1: Numerical escape-phase propagation with and without J2\n\n" + format_table_e1_markdown()
    (tables_dir / "table_e1_escape_propagation.md").write_text(table_e1_md)
    print("  -> Table E1 saved to results/tables/table_e1_escape_propagation.md")

    table_e2_md = "# Table E2: Escape phase with J2 solved by the three methods\n\n" + format_table_e2_markdown()
    (tables_dir / "table_e2_escape_methods.md").write_text(table_e2_md)
    print("  -> Table E2 saved to results/tables/table_e2_escape_methods.md")

    # 5. Robustness Protocol (Table 3: 20 repeated runs)
    print("\n[Step 4/5] Executing 20-Run Robustness Protocol (Table 3)...")
    robustness_stats = run_robustness_protocol(n_runs=20, base_seed=1000)
    table_3_md = "# Table 3: Success rate and spread over 20 repeated runs (mean ± SD of converged runs)\n\n" + format_table_3_markdown(robustness_stats)
    (tables_dir / "table_3_robustness_comparison.md").write_text(table_3_md)
    print("  -> Table 3 saved to results/tables/table_3_robustness_comparison.md")

    # 6. Render All Figures
    print("\n[Step 5/6] Generating Publication-Quality Figures...")
    plot_figure_4_1(solutions, figures_dir / "figure_4_1_heliocentric_results.png")
    plot_figure_4_2_schematic_3d(solutions, figures_dir / "figure_4_2_schematic_3d.png")
    plot_figure_4_3_top_view(solutions, figures_dir / "figure_4_3_top_view_2d.png")
    plot_perturbation_magnitudes(figures_dir / "perturbation_accelerations.png")

    # 7. Two Additions: Unperturbed Control Runs & Cross-Validation
    print("\n[Step 6/6] Executing Unperturbed Control Runs & Cross-Validation...")
    unpert_comps = run_unperturbed_control_benchmarks()
    cross_val = run_cross_validation(solutions["Pseudospectral Collocation"], solutions["Indirect Shooting"], solutions["X-TFC PoNN"])

    additions_md = [
        "# Additions: Unperturbed Controls & Cross-Validation Benchmarks\n",
        "## 1. Unperturbed Control Run Comparison (No SRP vs Perturbed)",
        "| Method | Perturbed Fuel (kg) | Unperturbed Fuel (kg) | Fuel Difference (kg) | Max Position Diff (km) |",
        "| --- | --- | --- | --- | --- |"
    ]
    for comp in unpert_comps:
        additions_md.append(f"| {comp.method} | {comp.fuel_perturbed_kg:.2f} | {comp.fuel_unperturbed_kg:.2f} | {comp.fuel_diff_kg:.2f} | {comp.max_position_diff_km:,.1f} |")

    additions_md.extend([
        "\n## 2. Cross-Validation Across Methods",
        f"- **Pseudospectral Control Propagated via DOP853 (Solve_ivp)**: Terminal Arrival Error = {cross_val.collocation_dop853_terminal_residual:.3e}",
        f"- **Indirect Shooting Hamiltonian Conservation Error**: {cross_val.shooting_hamiltonian_conservation_error:.3e}",
        f"- **X-TFC PoNN Boundary Collocation Defect Norm**: {cross_val.xtfc_collocation_defect_norm:.3e}"
    ])
    (tables_dir / "additions_summary.md").write_text("\n".join(additions_md))
    print("  -> Additions summary saved to results/tables/additions_summary.md")

    print("\n" + "=" * 80)
    print("SUMMARY OF BENCHMARK RESULTS")
    print("=" * 80)
    print(table_1_md.split("### Representative Method Performance")[1])
    print("\n" + "=" * 80)
    print("TABLE 3: ROBUSTNESS OVER 20 REPEATED RUNS")
    print("=" * 80)
    print(format_table_3_markdown(robustness_stats))
    print("\nAll tasks completed successfully. All artifacts created in results/.")


if __name__ == "__main__":
    main()
