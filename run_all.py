"""Master execution script for Earth-Mars Low-Thrust Trajectory Optimization.

Executes:
1. Phase 1 escape-spiral 3-D DOP853 propagation (Table E1).
2. Table 2 perturbation magnitude comparison.
3. 180-node LGL minimum-time problem (free tf, free Mars arrival longitude theta_f)
   at T_max = 0.50 N to determine t_min and optimal arrival longitude theta_f*.
4. Bounded-thrust Mars arrival angle sweep (180 LGL nodes + indirect shooting)
   at both the paper's nominal duration (tf = 247.9 days) and the shortest
   all-method converged duration (tf = 261.0 days).
5. Representative seed=1000 solve of all three methods + high-precision
   indirect shooting reference at both tf = 247.9 days and tf = 261.0 days.
6. 20-run robustness protocol (180 LGL nodes in all collocation runs) at both
   tf = 247.9 days and tf = 261.0 days (Table 3).
7. Extended additions: unperturbed control (SRP = 0), cross-validation matrix,
   perturbation-strength sweep, neurons/nodes convergence study, evaluation
   counts, and 3-D inclined eccentric Keplerian Mars ephemeris comparison.
8. Programmatic regeneration of all tables, figures, REPORT.md, ASSUMPTIONS.md,
   DEVIATIONS.md, and README.md directly from the computed data dictionary.
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from trajectory_optimization.problem import HelioProblem
from trajectory_optimization.escape_phase import generate_table_e1
from trajectory_optimization.metrics import compute_table_2_perturbations, format_table_2_markdown
from trajectory_optimization.pseudospectral import solve_min_time
from trajectory_optimization.benchmarks import (
    run_three_methods,
    run_robustness_protocol,
    run_mars_angle_sweep,
    format_table_e1_markdown,
    format_table_3_markdown,
    format_angle_sweep_markdown,
)
from trajectory_optimization.additions import (
    run_unperturbed_control,
    run_perturbation_strength_sweep,
    run_discretization_convergence_study,
    run_keplerian_ephemeris_comparison,
    format_additions_markdown,
)
from trajectory_optimization.plotting import (
    plot_figure_4_1,
    plot_figure_4_2_schematic_3d,
    plot_figure_4_3_top_view,
    plot_perturbation_magnitudes,
)
from trajectory_optimization.reporting import (
    generate_table_1_markdown,
    generate_report_markdown,
    generate_assumptions_markdown,
    generate_deviations_markdown,
    generate_readme_markdown,
)


def main():
    import pickle
    import subprocess
    import sys

    if "--from-stored" in sys.argv or (
        "--recompute-legacy" not in sys.argv
        and Path("results/prereg_fresh_2000_2099.json").exists()
    ):
        print("==================================================================", flush=True)
        print("Reproducing all tables, figures, and reports from stored JSON data", flush=True)
        print("==================================================================", flush=True)
        env = dict(**__import__("os").environ)
        env["PYTHONPATH"] = "."
        subprocess.run([sys.executable, "scripts/generate_paper_deliverables.py"], check=True, env=env)
        if Path("scripts/generate_tier1_deliverables.py").exists() and Path("results/tier1_extraction_ablation_4000_4099.json").exists():
            subprocess.run([sys.executable, "scripts/generate_tier1_deliverables.py"], check=True, env=env)
        if Path("scripts/generate_tier2_deliverables.py").exists() and Path("results/tier2_plateau_6000_6029.json").exists():
            subprocess.run([sys.executable, "scripts/generate_tier2_deliverables.py"], check=True, env=env)
        if Path("scripts/generate_tier3_deliverables.py").exists() and Path("results/tier3_split_domain_7000.json").exists():
            subprocess.run([sys.executable, "scripts/generate_tier3_deliverables.py"], check=True, env=env)
        if Path("scripts/generate_addendum2_deliverables.py").exists() and Path("results/addendum2_baselines_8000_8099.json").exists():
            subprocess.run([sys.executable, "scripts/generate_addendum2_deliverables.py"], check=True, env=env)
        print("ALL STORED DELIVERABLES REPRODUCED SUCCESSFULLY.", flush=True)
        return

    t0_all = time.perf_counter()
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_file = Path("results/.pipeline_cache.pkl")
    cache = {}
    if "--fresh" in sys.argv and cache_file.exists():
        cache_file.unlink()
    if cache_file.exists():
        try:
            cache = pickle.loads(cache_file.read_bytes())
        except Exception:
            cache = {}

    def save_cache():
        cache_file.write_bytes(pickle.dumps(cache))

    # Remove deleted Table E2 if present
    e2_file = tables_dir / "table_e2_escape_methods.md"
    if e2_file.exists():
        e2_file.unlink()

    print("==================================================================", flush=True)
    print("STEP 1: Phase 1 Escape Spiral Propagation (Table E1) & Table 2", flush=True)
    print("==================================================================", flush=True)
    if "step1" not in cache:
        e1_cases = generate_table_e1()
        table_e1_md = format_table_e1_markdown(e1_cases)
        table_2 = compute_table_2_perturbations(final_helio_mass_kg=943.0)
        table_2_md = format_table_2_markdown(table_2)
        cache["step1"] = (e1_cases, table_e1_md, table_2, table_2_md)
        save_cache()
    else:
        e1_cases, table_e1_md, table_2, table_2_md = cache["step1"]
        print("  [loaded from cache]", flush=True)
    (tables_dir / "table_e1_escape_propagation.md").write_text(table_e1_md, encoding="utf-8")
    (tables_dir / "table_2_perturbation_magnitudes.md").write_text(table_2_md, encoding="utf-8")

    print("==================================================================", flush=True)
    print("STEP 2: 180-Node Minimum-Time Rendezvous Problem (T_max = 0.50 N)", flush=True)
    print("==================================================================", flush=True)
    if "step2" not in cache:
        base_prob = HelioProblem(tf_days=247.9, theta_f=np.radians(192.5))
        min_time_out = solve_min_time(base_prob, n_nodes=180, seed=1000, tf_guess_days=255.0, theta_guess=np.radians(192.5))
        cache["step2"] = min_time_out
        save_cache()
    else:
        min_time_out = cache["step2"]
        print("  [loaded from cache]", flush=True)
    theta_opt_rad = float(min_time_out["theta_f_rad"])
    theta_opt_deg = float(np.degrees(theta_opt_rad))
    tf_feas_days = float(np.ceil(min_time_out["tf_days"] * 10.0) / 10.0)
    if tf_feas_days - min_time_out["tf_days"] < 0.15:
        tf_feas_days = float(np.ceil(min_time_out["tf_days"]))  # 261.0 days
    print(f"  -> t_min = {min_time_out['tf_days']:.4f} d, theta_f* = {theta_opt_deg:.4f} deg, tf_feas = {tf_feas_days:.1f} d", flush=True)

    prob_248 = HelioProblem(tf_days=247.9, theta_f=theta_opt_rad, label="paper_247.9d")
    prob_feas = HelioProblem(tf_days=tf_feas_days, theta_f=theta_opt_rad, label=f"feasible_{tf_feas_days:.1f}d")

    print("==================================================================", flush=True)
    print("STEP 3: Representative Seed=1000 Runs (247.9 d and Feasible tf)", flush=True)
    print("==================================================================", flush=True)
    if "step3" not in cache:
        rep_248 = run_three_methods(prob_248, seed=1000, compute_ref=False)
        rep_feas = run_three_methods(prob_feas, seed=1000, compute_ref=True)
        cache["step3"] = (rep_248, rep_feas)
        save_cache()
    else:
        rep_248, rep_feas = cache["step3"]
        print("  [loaded from cache]", flush=True)

    print("==================================================================", flush=True)
    print("STEP 4: Bounded-Thrust Mars Arrival Angle Sweeps (180 LGL Nodes)", flush=True)
    print("==================================================================", flush=True)
    if "step4a" not in cache:
        sweep_angles_248 = [165.0, 175.0, 180.0, 185.0, 190.0, round(theta_opt_deg, 2), 195.0, 200.0, 210.0]
        def _prog_4a(rows):
            cache["step4a_partial"] = rows
            save_cache()
        sweep_248 = run_mars_angle_sweep(
            prob_248, sweep_angles_248, n_nodes=180,
            partial_rows=cache.get("step4a_partial"), on_progress=_prog_4a
        )
        cache["step4a"] = sweep_248
        save_cache()
    else:
        sweep_248 = cache["step4a"]
        print("  [4a loaded from cache]", flush=True)
    if "step4b" not in cache:
        sweep_angles_feas = [191.5, 192.0, round(theta_opt_deg, 2), 193.0, 193.5]
        def _prog_4b(rows):
            cache["step4b_partial"] = rows
            save_cache()
        sweep_feas = run_mars_angle_sweep(
            prob_feas, sweep_angles_feas, n_nodes=180,
            partial_rows=cache.get("step4b_partial"), on_progress=_prog_4b
        )
        cache["step4b"] = sweep_feas
        save_cache()
    else:
        sweep_feas = cache["step4b"]
        print("  [4b loaded from cache]", flush=True)
    angle_sweep_md = format_angle_sweep_markdown(min_time_out, sweep_248, sweep_feas)
    (tables_dir / "table_mars_angle_search.md").write_text(angle_sweep_md, encoding="utf-8")

    print("==================================================================", flush=True)
    print("STEP 5A: 20-Run Robustness Protocol on Paper Duration (247.9 d, 180 LGL Nodes)", flush=True)
    print("==================================================================", flush=True)
    if "step5a" not in cache:
        def _prog_5a(r):
            cache["step5a_partial"] = r
            save_cache()
        rob_248 = run_robustness_protocol(
            prob_248, n_runs=20, base_seed=1000,
            partial_runs=cache.get("step5a_partial"), on_progress=_prog_5a
        )
        cache["step5a"] = rob_248
        save_cache()
    else:
        rob_248 = cache["step5a"]
        print("  [loaded from cache]", flush=True)

    print("==================================================================", flush=True)
    print("STEP 5B: 20-Run Robustness Protocol on Feasible Duration (261.0 d, 180 LGL Nodes)", flush=True)
    print("==================================================================", flush=True)
    if "step5b" not in cache:
        def _prog_5b(r):
            cache["step5b_partial"] = r
            save_cache()
        rob_feas = run_robustness_protocol(
            prob_feas, n_runs=20, base_seed=1000,
            partial_runs=cache.get("step5b_partial"), on_progress=_prog_5b
        )
        cache["step5b"] = rob_feas
        save_cache()
    else:
        rob_feas = cache["step5b"]
        print("  [loaded from cache]", flush=True)
    table_3_md = format_table_3_markdown(rob_248, rob_feas, prob_248.tf_days, prob_feas.tf_days, theta_opt_deg)
    (tables_dir / "table_3_robustness_comparison.md").write_text(table_3_md, encoding="utf-8")

    print("==================================================================", flush=True)
    print("STEP 6: Extended Additions (Unperturbed, Sweep, Convergence, Keplerian)", flush=True)
    print("==================================================================", flush=True)
    if "step6_unpert" not in cache:
        unpert_feas = run_unperturbed_control(prob_feas, seed=1000)
        cache["step6_unpert"] = unpert_feas
        save_cache()
    else:
        unpert_feas = cache["step6_unpert"]
    if "step6_pert" not in cache:
        pert_sweep = run_perturbation_strength_sweep(prob_feas, srp_scales=(0.0, 1.0, 10.0, 100.0, 1000.0), seed=1000)
        cache["step6_pert"] = pert_sweep
        save_cache()
    else:
        pert_sweep = cache["step6_pert"]
    if "step6_disc" not in cache:
        disc_conv = run_discretization_convergence_study(prob_feas, rep_feas["reference"], seed=1000)
        cache["step6_disc"] = disc_conv
        save_cache()
    else:
        disc_conv = cache["step6_disc"]
    if "step6_kepl" not in cache:
        kepl_comp = run_keplerian_ephemeris_comparison(tf_days=310.0, theta_deg=220.0, seed=1000)
        cache["step6_kepl"] = kepl_comp
        save_cache()
    else:
        kepl_comp = cache["step6_kepl"]
    additions_md = format_additions_markdown(rep_248, rep_feas, unpert_feas, pert_sweep, disc_conv, kepl_comp)
    (tables_dir / "additions_summary.md").write_text(additions_md, encoding="utf-8")

    print("==================================================================", flush=True)
    print("STEP 7: Generating Publication Figures (4.1, 4.2, 4.3, Perturbations)", flush=True)
    print("==================================================================", flush=True)
    plot_figure_4_1(
        rep_feas,
        output_path=figures_dir / "figure_4_1_heliocentric_results.png",
        scenario_title=f"Figure 4.1: Heliocentric Results at Shortest All-Method Converged Duration (tf = {prob_feas.tf_days:.1f} d, θ_f = {theta_opt_deg:.2f}°, T_max = 0.50 N)",
    )
    plot_figure_4_1(
        rep_248,
        output_path=figures_dir / "figure_4_1_paper_248d_unconverged.png",
        scenario_title=f"Figure 4.1 (Paper Nominal Duration tf = {prob_248.tf_days:.1f} d < t_min = {min_time_out['tf_days']:.2f} d — Non-Converged Under T_max = 0.50 N)",
    )
    plot_figure_4_2_schematic_3d(rep_feas, output_path=figures_dir / "figure_4_2_schematic_3d.png")
    plot_figure_4_3_top_view(rep_feas, output_path=figures_dir / "figure_4_3_top_view_2d.png")
    plot_perturbation_magnitudes(output_path=figures_dir / "perturbation_accelerations.png")

    # Load stored Phase 1 and Phase 2 JSON data if available and regenerate their tables & figures
    import json
    from trajectory_optimization.phase_studies import (
        format_phase1_scan_markdown,
        format_phase1_multi_tf_markdown,
        format_phase2_markdown,
    )
    from trajectory_optimization.plotting import (
        plot_phase1_feasibility_scan,
        plot_phase1_multi_tf_robustness,
        plot_phase2_basin_and_hybrid,
    )

    p1_data = None
    p1_scan_md = None
    p1_multi_tf_md = None
    p1_json = Path("results/phase1_results.json")
    if p1_json.exists():
        p1_data = json.loads(p1_json.read_text(encoding="utf-8"))
        if "scan" in p1_data:
            p1_scan_md = format_phase1_scan_markdown(p1_data["scan"])
            (tables_dir / "table_phase1_feasibility_scan.md").write_text(p1_scan_md, encoding="utf-8")
            plot_phase1_feasibility_scan(
                p1_data["scan"],
                output_path=figures_dir / "figure_phase1_feasibility_scan.png",
            )
        if "multi_tf_robustness" in p1_data:
            rob_map = {float(k): v for k, v in p1_data["multi_tf_robustness"].items()}
            ang_map = {float(k): v for k, v in p1_data["selected_angles"].items()}
            p1_multi_tf_md = format_phase1_multi_tf_markdown(ang_map, rob_map)
            (tables_dir / "table_phase1_multi_tf_robustness.md").write_text(p1_multi_tf_md, encoding="utf-8")
            plot_phase1_multi_tf_robustness(
                rob_map,
                output_path=figures_dir / "figure_phase1_multi_tf_robustness.png",
            )

    p2_data = None
    p2_md = None
    p2_json = Path("results/phase2_results.json")
    if p2_json.exists():
        p2_data = json.loads(p2_json.read_text(encoding="utf-8"))
        p2_md = format_phase2_markdown(p2_data)
        (tables_dir / "table_phase2_basin_and_hybrid.md").write_text(p2_md, encoding="utf-8")
        plot_phase2_basin_and_hybrid(
            p2_data,
            output_path=figures_dir / "figure_phase2_basin_and_hybrid.png",
        )

    print("==================================================================", flush=True)
    print("STEP 8: Generating Table 1, REPORT.md, ASSUMPTIONS.md, DEVIATIONS.md, README.md", flush=True)
    print("==================================================================", flush=True)
    data = {
        "e1_cases": e1_cases,
        "table_e1_md": table_e1_md,
        "table_2": table_2,
        "table_2_md": table_2_md,
        "min_time": min_time_out,
        "rep_248": rep_248,
        "rep_feas": rep_feas,
        "sweep_248": sweep_248,
        "sweep_feas": sweep_feas,
        "angle_sweep_md": angle_sweep_md,
        "rob_248": rob_248,
        "rob_feas": rob_feas,
        "table_3_md": table_3_md,
        "unpert_feas": unpert_feas,
        "pert_sweep": pert_sweep,
        "disc_conv": disc_conv,
        "kepl_comp": kepl_comp,
        "additions_md": additions_md,
        "phase1": p1_data,
        "phase1_scan_md": p1_scan_md,
        "phase1_multi_tf_md": p1_multi_tf_md,
        "phase2": p2_data,
        "phase2_md": p2_md,
    }
    table_1_md = generate_table_1_markdown(data)
    (tables_dir / "table_1_mission_parameters.md").write_text(table_1_md, encoding="utf-8")

    Path("REPORT.md").write_text(generate_report_markdown(data), encoding="utf-8")
    Path("ASSUMPTIONS.md").write_text(generate_assumptions_markdown(data), encoding="utf-8")
    Path("DEVIATIONS.md").write_text(generate_deviations_markdown(data), encoding="utf-8")
    Path("README.md").write_text(generate_readme_markdown(data), encoding="utf-8")

    total_s = time.perf_counter() - t0_all
    print(f"ALL STEPS COMPLETED in {total_s:.2f} s ({total_s/60.0:.2f} min).", flush=True)


if __name__ == "__main__":
    main()

