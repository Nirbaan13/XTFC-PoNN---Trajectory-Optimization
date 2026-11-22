"""Run Phase 2: 100-Seed Random-Initialization Basin Study (2A) and
Hybrid X-TFC -> Indirect Shooting Test (2B) on the same 100 seeds.

Reads the fuel-minimal theta_f*(280.0 d) from results/phase1_results.json,
saves resumable checkpoints to results/.phase2_cache.pkl, compact JSON to
results/phase2_results.json, and markdown table to results/tables/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_optimization.phase_studies import (
    run_phase2_basin_and_hybrid,
    print_phase2_summary,
    format_phase2_markdown,
    save_json_clean,
)


def main():
    t0 = time.perf_counter()
    cache_path = ROOT / "results" / ".phase2_cache.pkl"
    p1_json = ROOT / "results" / "phase1_results.json"
    tables_dir = ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if "--fresh" in sys.argv and cache_path.exists():
        cache_path.unlink()

    tf_days = 280.0
    theta_f_deg = 205.0
    if p1_json.exists():
        p1 = json.loads(p1_json.read_text(encoding="utf-8"))
        ang_map = p1.get("selected_angles", {})
        if str(tf_days) in ang_map:
            theta_f_deg = float(ang_map[str(tf_days)]["theta_f_deg"])

    print("==================================================================", flush=True)
    print(f"PHASE 2A & 2B: 100-Seed Basin & Hybrid Study (tf={tf_days:.1f} d, theta_f*={theta_f_deg:.2f} deg)", flush=True)
    print("==================================================================", flush=True)
    p2 = run_phase2_basin_and_hybrid(
        tf_days=tf_days,
        theta_f_deg=theta_f_deg,
        n_seeds=100,
        base_seed=2000,
        cache_path=cache_path,
        n_workers=1,
    )
    print_phase2_summary(p2)

    p2_md = format_phase2_markdown(p2)
    (tables_dir / "table_phase2_basin_and_hybrid.md").write_text(p2_md, encoding="utf-8")
    save_json_clean(ROOT / "results" / "phase2_results.json", p2)
    print(f"PHASE 2 COMPLETED in {time.perf_counter() - t0:.2f} s.", flush=True)


if __name__ == "__main__":
    main()
