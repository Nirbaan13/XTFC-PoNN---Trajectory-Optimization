"""Automated audit test: fails if narrative text in REPORT.md / README.md differs from tables or stored JSONs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"


def test_no_stale_additions_summary():
    """Item 3: stale Scenario B additions_summary.md must be removed."""
    assert not (TABLES / "additions_summary.md").exists()


def test_report_section_ordering_and_repro_command():
    """Item 4: REPORT.md sections must be strictly ascending and use run_all.py."""
    txt = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    nums = [int(m.group(1)) for m in re.finditer(r"^## (\d+)\.", txt, flags=re.MULTILINE)]
    assert nums == list(range(1, len(nums) + 1)), f"Non-ascending REPORT.md sections: {nums}"
    assert "python3 run_all.py --from-stored" in txt
    assert ".venv/bin/" not in txt
    assert "scripts/generate_paper_deliverables.py" not in txt


def test_no_plateau_ceiling_gibbs_in_sec3_and_sec8():
    """Item 1: REPORT.md §3.2 and §8 must not claim an irreducible plateau/ceiling/Gibbs limit."""
    txt = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    sec3 = txt.split("\n## 3.")[1].split("\n## 4.")[0]
    sec8 = txt.split("\n## 8.")[1].split("\n## 9.")[0]
    for sec_name, body in [("Sec 3", sec3), ("Sec 8", sec8)]:
        for forbidden in ("plateau", "ceiling", "gibbs"):
            assert forbidden not in body.lower(), f"Forbidden stale term '{forbidden}' found in {sec_name}"


def test_a4d_failed_fuel_marked_nan():
    """Item 5: A4(d) uniform_140_lm_svd failed row must show NaN, not -0.46 kg."""
    txt = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    assert "-0.46" not in txt
    m = re.search(r"\|\s*A4\(d\) Grid/Opt\s*\|\s*uniform_140_lm_svd\s*\|([^|\n]*\|){2}\s*([^|\n]+)\|", txt)
    assert m is not None, "Could not find A4(d) uniform_140_lm_svd row in REPORT.md"
    assert m.group(2).strip() == "NaN", f"Expected NaN fuel for uniform_140_lm_svd, got {m.group(2)!r}"


def test_scenario_b_247_9_softened_wording():
    """Item 6: §7 and README must soften 247.9 d from 'established infeasible' to 'no feasible solution found'."""
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sec7 = report.split("\n## 7.")[1].split("\n## 8.")[0]
    assert "is infeasible" not in sec7.lower()
    assert "established infeasible" not in sec7.lower()
    assert "no feasible solution found" in sec7.lower()
    assert "no feasible solution found" in readme.lower()


def test_prereg_2000_narrative_matches_table_and_json():
    """Item 2: Every narrative number in REPORT.md §8.2 must match stored JSON and table."""
    prereg = json.loads((RESULTS / "prereg_fresh_2000_2099.json").read_text(encoding="utf-8"))
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    sec8 = report.split("\n## 8.")[1].split("\n## 9.")[0]

    # Split narrative after the table
    assert "**Statistical Hypothesis Testing & Verdict" in sec8
    narr = sec8.split("**Statistical Hypothesis Testing & Verdict")[1]

    for arm_key in ("paired_srp_0.0", "paired_srp_0.5", "paired_srp_1.0"):
        s = prereg[arm_key]["summary"]
        sh, hy, cp = s["shooting"], s["hybrid"], s["comparison"]
        n_tot = s["n_total"]
        assert f"{hy['n_strict']}/{n_tot} ({100*hy['rate_strict']:.1f}%)" in narr
        assert f"{sh['n_strict']}/{n_tot} ({100*sh['rate_strict']:.1f}%)" in narr
        assert f"`{cp['delta_strict_pp']:+.1f}` percentage points" in narr
        assert f"`b={cp['discordant_shoot_only_n01']}`" in narr
        assert f"`c={cp['discordant_hyb_only_n10']}`" in narr
        assert f"`p = {cp['mcnemar_exact_p']:.2e}`" in narr
        assert f"\\chi^2 = {cp['mcnemar_chi2']:.2f}, p = {cp['mcnemar_chi2_p']:.2e}" in narr
        assert f"{hy['runtime_total_s']['median']:.2f} s" in narr
        assert f"{sh['runtime_s']['median']:.2f} s" in narr


def test_shooting_runtime_22_26_independently_computed():
    """Item 7: Confirm 2000..2099 and 4100..4129 shooting medians are disjoint and independently computed."""
    p = json.loads((RESULTS / "prereg_fresh_2000_2099.json").read_text(encoding="utf-8"))
    rt = json.loads((RESULTS / "tier1_runtime_single_process.json").read_text(encoding="utf-8"))
    seeds_a = set(int(k) for k in p["paired_srp_0.0"]["seeds"].keys())
    seeds_b = set(int(k) for k in rt["seeds"].keys())
    assert seeds_a.isdisjoint(seeds_b)
    med_a = float(np.median([v["shooting"]["runtime_s"] for v in p["paired_srp_0.0"]["seeds"].values()]))
    med_b = float(np.median([v["shooting"]["runtime_s"] for v in rt["seeds"].values()]))
    assert med_a != med_b
    assert f"{med_a:.2f}" == f"{med_b:.2f}" == "22.26"
    tab5 = (TABLES / "tab5_runtime_distributions.tex").read_text(encoding="utf-8")
    assert f"{med_b:.2f}" in tab5


def test_tier2_exploratory_cells_marked():
    """Tier 2 truncated L=160 (n=540, 1080) cells must be explicitly marked exploratory."""
    tab6 = (TABLES / "tab6_ponn_plateau.tex").read_text(encoding="utf-8")
    assert "$L=160$, $n=540$, 1e-09$^\\dagger$ [exploratory]" in tab6
    assert "$L=160$, $n=1080$, 1e-09$^\\dagger$ [exploratory]" in tab6


def test_tier3_and_addendum2_match_tables_and_json():
    """Tier 3 and Addendum 2 narrative numbers in REPORT.md must match JSON and LaTeX tables."""
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    t3 = json.loads((RESULTS / "tier3_split_domain_7000.json").read_text(encoding="utf-8"))
    f3 = t3["full_100"]
    n_p = sum(1 for v in f3.values() if v.get("passed_prereg"))
    n_t = len(f3)
    assert f"{n_p}/{n_t}" in report
    tab8 = (TABLES / "tab8_split_domain_tier3.tex").read_text(encoding="utf-8")
    assert f"{n_p}/{n_t}" in tab8

    add2_path = RESULTS / "addendum2_baselines_8000_8099.json"
    if add2_path.exists():
        add2 = json.loads(add2_path.read_text(encoding="utf-8"))
        n = add2["summary"]["n_seeds"]
        if n > 0:
            tab9 = (TABLES / "tab9_addendum2_baselines.tex").read_text(encoding="utf-8")
            for arm_key in ("single_shoot", "multistart_shoot_5", "hybrid_rho1", "split_domain"):
                ks = add2["summary"]["arms"][arm_key]["n_strict"]
                assert f"{ks}/{n}" in report
                assert f"{ks}/{n}" in tab9
                assert f"{ks}/{n}" in readme


def test_no_broken_latex_escaping_in_docs():
    """README.md must have no literal tab/CR from unescaped LaTeX; REPORT.md no double-escaped \\\\text."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    assert "\t" not in readme, "Literal tab found in README.md (unescaped \\text or \\theta)"
    assert "\r" not in readme, "Literal CR found in README.md (unescaped \\rho)"
    assert "\\\\text" not in report, "Double-escaped \\\\text found in REPORT.md"
    assert "\\\\rho" not in report, "Double-escaped \\\\rho found in REPORT.md"


def test_task3_language_audit_and_claims_md():
    """Task 3: neutral wording across docs, exact perturbation-aware PoNN sentence, and CLAIMS.md present."""
    for fname in ("README.md", "REPORT.md", "ASSUMPTIONS.md", "DEVIATIONS.md"):
        txt = (ROOT / fname).read_text(encoding="utf-8").lower()
        assert "they did not" not in txt, f"'they did not' found in {fname}"
        assert "published method fails" not in txt, f"'published method fails' found in {fname}"
    expected_lit = "perturbation-aware ponns exist (j2 formation flying, verified abstract; cislunar high-fidelity ephemeris, abstract only; perturbation list unverified)"
    assert expected_lit in (ROOT / "REPORT.md").read_text(encoding="utf-8").lower()
    assert expected_lit in (ROOT / "ASSUMPTIONS.md").read_text(encoding="utf-8").lower()
    claims_path = ROOT / "CLAIMS.md"
    if claims_path.exists():
        ctxt = claims_path.read_text(encoding="utf-8")
        assert "| `C1` |" in ctxt and "| `C11` |" in ctxt
        assert expected_lit in ctxt.lower()

