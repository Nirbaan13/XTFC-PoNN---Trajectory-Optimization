"""Run Phase 1: Global Feasibility Scan (1.1), Fuel-Minimal Angle Selection (1.2),
and Multi-tf 20-Run Robustness Protocol including cold-start X-TFC (1.3–1.4).

Saves resumable checkpoint cache to results/.phase1_cache.pkl, compact JSON data
to results/phase1_results.json, and markdown tables to results/tables/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_optimization.phase_studies import (
    run_global_feasibility_scan,
    select_fuel_minimal_angles,
    run_multi_tf_robustness,
    print_phase1_1_summary,
    print_phase1_robustness_summary,
    format_phase1_scan_markdown,
    format_phase1_multi_tf_markdown,
    save_json_clean,
)


def main():
    t0 = time.perf_counter()
    cache_path = ROOT / "results" / ".phase1_cache.pkl"
    tables_dir = ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if "--fresh" in sys.argv and cache_path.exists():
        cache_path.unlink()

    only_scan = "--scan-only" in sys.argv
    n_workers = 1

    print("==================================================================", flush=True)
    print("PHASE 1.1: Global Feasibility Scan at T_max = 0.50 N (60..300 deg)", flush=True)
    print("==================================================================", flush=True)
    scan_data = run_global_feasibility_scan(
        cache_path=cache_path,
        n_workers=n_workers,
    )
    print_phase1_1_summary(scan_data)
    scan_md = format_phase1_scan_markdown(scan_data)
    (tables_dir / "table_phase1_feasibility_scan.md").write_text(scan_md, encoding="utf-8")

    print("==================================================================", flush=True)
    print("PHASE 1.2: Fuel-Minimal Mars Arrival Angle Selection per tf", flush=True)
    print("==================================================================", flush=True)
    tf_list = (247.9, 280.0, 300.0, 330.0, 360.0)
    selected_angles = select_fuel_minimal_angles(
        scan_data,
        tf_list=tf_list,
        cache_path=cache_path,
        n_workers=n_workers,
    )

    if only_scan:
        save_json_clean(
            ROOT / "results" / "phase1_results.json",
            {
                "scan": scan_data,
                "selected_angles": {str(k): v for k, v in selected_angles.items()},
            },
        )
        print(f"Phase 1.1–1.2 completed in {time.perf_counter() - t0:.2f} s.", flush=True)
        return

    print("==================================================================", flush=True)
    print("PHASE 1.3–1.4: Multi-tf 20-Run Protocol (All Methods + Cold X-TFC)", flush=True)
    print("==================================================================", flush=True)
    rob_by_tf = run_multi_tf_robustness(
        selected_angles,
        tf_list=tf_list,
        n_runs=20,
        base_seed=1000,
        cache_path=cache_path,
        n_workers=n_workers,
    )
    print_phase1_robustness_summary(rob_by_tf)

    multi_tf_md = format_phase1_multi_tf_markdown(selected_angles, rob_by_tf)
    (tables_dir / "table_phase1_multi_tf_robustness.md").write_text(multi_tf_md, encoding="utf-8")

    payload = {
        "scan": scan_data,
        "selected_angles": {str(k): v for k, v in selected_angles.items()},
        "multi_tf_robustness": {str(k): v for k, v in rob_by_tf.items()},
    }
    save_json_clean(ROOT / "results" / "phase1_results.json", payload)
    print(f"PHASE 1 COMPLETED in {time.perf_counter() - t0:.2f} s.", flush=True)


if __name__ == "__main__":
    main()
