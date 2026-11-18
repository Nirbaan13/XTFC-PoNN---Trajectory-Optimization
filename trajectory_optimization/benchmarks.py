"""Statistical Benchmarking and Robustness Protocol Runner.

Executes:
1. Representative runs (seed 1000) for Table 1 and Figure 4.1.
2. 20-run Monte Carlo robustness evaluation for Table 3.
3. Escape phase evaluations for Table E1 and Table E2.
4. Generates formatted markdown tables from actual computed solver outputs.
"""

from dataclasses import dataclass
import numpy as np

from .indirect_shooting import IndirectShootingSolver, TrajectorySolution
from .xtfc_ponn import XTFC_PoNN_Solver
from .pseudospectral import PseudospectralCollocationSolver
from .escape_phase import generate_table_e1, generate_table_e2
from .metrics import compute_table_2_perturbations, format_table_2_markdown


@dataclass
class RobustnessStats:
    method_name: str
    total_runs: int
    converged_runs: int
    fuel_mean: float
    fuel_std: float
    tof_mean: float
    tof_std: float
    residual_mean: float
    residual_std: float
    runtime_mean_min: float
    runtime_std_min: float


def run_representative_benchmarks(seed: int = 1000) -> dict[str, TrajectorySolution]:
    """Runs all three optimization methods for the reference run (seed 1000)."""
    print(f"Running representative benchmarks with seed {seed}...")
    
    # Method 1: X-TFC PoNN
    print("  -> Executing X-TFC PoNN...")
    xtfc_solver = XTFC_PoNN_Solver(flight_time_days=247.6)
    xtfc_sol = xtfc_solver.solve(seed=seed, perturb_weights=False)

    # Method 2: Indirect Shooting
    print("  -> Executing Indirect Shooting...")
    shooting_solver = IndirectShootingSolver(flight_time_days=248.4)
    shooting_sol = shooting_solver.solve(seed=seed, perturb_guess=False)

    # Method 3: Pseudospectral Collocation
    print("  -> Executing Pseudospectral Collocation...")
    pseudospectral_solver = PseudospectralCollocationSolver(flight_time_days=247.9)
    ps_sol = pseudospectral_solver.solve(seed=seed, perturb_guess=False)

    return {
        "X-TFC PoNN": xtfc_sol,
        "Indirect Shooting": shooting_sol,
        "Pseudospectral Collocation": ps_sol
    }


def run_robustness_protocol(n_runs: int = 20, base_seed: int = 1000) -> dict[str, RobustnessStats]:
    """Runs 20 repeated runs per method with distinct random seeds (Table 3 protocol)."""
    print(f"\nRunning 20-run robustness protocol across {n_runs} seeds...")
    
    seeds = [base_seed + i for i in range(n_runs)]
    methods = ["X-TFC PoNN", "Indirect Shooting", "Pseudospectral Collocation"]
    stats_dict = {}

    for method_name in methods:
        print(f"  Evaluating {method_name} over {n_runs} seeds...")
        converged_list = []
        fuel_list = []
        tof_list = []
        res_list = []
        runtime_list = []

        for s in seeds:
            if method_name == "X-TFC PoNN":
                solver = XTFC_PoNN_Solver(flight_time_days=247.6)
                sol = solver.solve(seed=s, perturb_weights=True)
            elif method_name == "Indirect Shooting":
                solver = IndirectShootingSolver(flight_time_days=248.4)
                sol = solver.solve(seed=s, perturb_guess=True)
            else:
                solver = PseudospectralCollocationSolver(flight_time_days=247.9)
                sol = solver.solve(seed=s, perturb_guess=True)

            runtime_min = sol.runtime_seconds / 60.0
            runtime_list.append(runtime_min)

            if sol.converged:
                converged_list.append(sol)
                fuel_list.append(sol.xenon_used_kg)
                tof_list.append(sol.flight_time_days)
                res_list.append(sol.arrival_residual)

        # Statistical calculations from actual converged runs
        n_conv = len(converged_list)
        stat = RobustnessStats(
            method_name=method_name,
            total_runs=n_runs,
            converged_runs=n_conv,
            fuel_mean=round(float(np.mean(fuel_list)), 1) if n_conv > 0 else 0.0,
            fuel_std=round(float(np.std(fuel_list)), 2) if n_conv > 1 else 0.0,
            tof_mean=round(float(np.mean(tof_list)), 1) if n_conv > 0 else 0.0,
            tof_std=round(float(np.std(tof_list)), 2) if n_conv > 1 else 0.0,
            residual_mean=float(np.mean(res_list)) if n_conv > 0 else 0.0,
            residual_std=float(np.std(res_list)) if n_conv > 1 else 0.0,
            runtime_mean_min=round(float(np.mean(runtime_list)), 3) if runtime_list else 0.0,
            runtime_std_min=round(float(np.std(runtime_list)), 3) if len(runtime_list) > 1 else 0.0
        )
        stats_dict[method_name] = stat

    return stats_dict


def format_table_3_markdown(stats_dict: dict[str, RobustnessStats]) -> str:
    """Formats Table 3 into markdown."""
    md = [
        "| Method | Converged runs | Fuel (kg) | Flight time (days) | Arrival residual | Runtime (min) |",
        "| --- | --- | --- | --- | --- | --- |"
    ]
    for m in ["X-TFC PoNN", "Indirect Shooting", "Pseudospectral Collocation"]:
        st = stats_dict[m]
        md.append(
            f"| {st.method_name} | {st.converged_runs} / {st.total_runs} | "
            f"{st.fuel_mean:.1f} ± {st.fuel_std:.2f} | "
            f"{st.tof_mean:.1f} ± {st.tof_std:.2f} | "
            f"{st.residual_mean:.1e} ± {st.residual_std:.1e} | "
            f"{st.runtime_mean_min:.3f} ± {st.runtime_std_min:.3f} |"
        )
    return "\n".join(md)


def format_table_e1_markdown() -> str:
    """Formats Table E1 into markdown."""
    e1_rows = generate_table_e1()
    md = [
        "| Case | Time to escape energy (days) | Xenon used (kg) | Mass at escape energy (kg) |",
        "| --- | --- | --- | --- |"
    ]
    for row in e1_rows:
        md.append(f"| {row.case_name} | {row.escape_time_days:.1f} | {row.xenon_used_kg:.1f} | {row.final_mass_kg:,.1f} |")
    return "\n".join(md)


def format_table_e2_markdown() -> str:
    """Formats Table E2 into markdown."""
    e2_rows = generate_table_e2()
    md = [
        "| Method | Escape time (days) | Xenon used (kg) | Mass at escape energy (kg) | End-state residual |",
        "| --- | --- | --- | --- | --- |"
    ]
    for row in e2_rows:
        md.append(
            f"| {row['method']} | {row['escape_time_days']:.2f} | {row['xenon_used_kg']:.1f} | {row['final_mass_kg']:.1f} | {row['end_state_residual']} |"
        )
    return "\n".join(md)
