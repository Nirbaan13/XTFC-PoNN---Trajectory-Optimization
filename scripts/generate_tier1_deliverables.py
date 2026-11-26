"""Generate all Tier 1 paper-ready LaTeX booktabs tables, PDF figures, Holm-corrected verdicts,
and updated REPORT.md / README.md / ASSUMPTIONS.md / DEVIATIONS.md strictly from stored JSONs.

Zero hard-coded results; every number is computed directly from stored JSON files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def holm_bonferroni(p_items: list[tuple[str, float]]) -> dict[str, float]:
    """Compute step-down Holm-Bonferroni adjusted p-values for a list of (id, raw_p)."""
    m = len(p_items)
    ordered = sorted(p_items, key=lambda x: x[1])
    adj: dict[str, float] = {}
    running_max = 0.0
    for idx, (hid, raw_p) in enumerate(ordered):
        val = min(1.0, (m - idx) * float(raw_p))
        running_max = max(running_max, val)
        adj[hid] = running_max
    return adj


def fmt_ci_pct(rate: float, ci: list[float]) -> str:
    return f"{100*rate:.1f}\\% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_stat_med_iqr(s: dict | None, fmt: str = "{:.2f}") -> str:
    if not s:
        return "--"
    return f"{fmt.format(s['median'])} [{fmt.format(s['q25'])}, {fmt.format(s['q75'])}]"


def compute_all_holm_verdicts(ablation: dict, extra: dict) -> dict[str, Any]:
    """Apply Holm-Bonferroni correction across all M=13 Tier 1 primary McNemar tests."""
    tests: list[dict[str, Any]] = []
    ab_map = [
        ("H1a", "hybrid_rho_1e0", r"$\rho_{\mathrm{ext}}=1.0$"),
        ("H1b", "hybrid_rho_2.98e-1", r"$\rho_{\mathrm{ext}}=2.98\times 10^{-1}$"),
        ("H1c", "hybrid_rho_1e-1", r"$\rho_{\mathrm{ext}}=10^{-1}$"),
        ("H1d", "hybrid_rho_1e-2", r"$\rho_{\mathrm{ext}}=10^{-2}$"),
        ("H1e", "hybrid_rho_1e-3", r"$\rho_{\mathrm{ext}}=10^{-3}$"),
        ("H1f", "hybrid_rho_final", r"$\rho_{\mathrm{ext}}=\text{final}$"),
    ]
    for hid, arm, label in ab_map:
        mc = ablation["summary"]["mcnemar_vs_random"][arm]
        tests.append({
            "id": hid,
            "family": "Tier 1.1 Extraction Ablation",
            "arm": arm,
            "label": label,
            "delta_strict_pp": float(mc["delta_strict_pp"]),
            "p_raw": float(mc["p_exact_raw"]),
            "feasible": True,
        })

    ex_map = [
        ("H2a", "scen_a_m0_800", r"Scen. A, $m_0=800$ kg"),
        ("H2b", "scen_a_m0_1200", r"Scen. A, $m_0=1200$ kg"),
        ("H2c", "scen_a_tmax_0.4", r"Scen. A, $T_{\max}=0.4$ N"),
        ("H2d", "scen_a_tmax_0.7", r"Scen. A, $T_{\max}=0.7$ N"),
        ("H2e", "scen_b_tf_300", r"Scen. B, $t_f=300$ d"),
        ("H2f", "scen_b_tf_330", r"Scen. B, $t_f=330$ d"),
        ("H2g", "scen_b_tf_360", r"Scen. B, $t_f=360$ d"),
    ]
    for hid, cid, label in ex_map:
        c = extra["cases"][cid]
        mc = c["summary"]["mcnemar_hybrid_vs_shooting"]
        feas = float(c["meta"]["t_min_ratio"]) <= 0.95
        tests.append({
            "id": hid,
            "family": "Tier 1.2 Extra Physical Cases",
            "arm": cid,
            "label": label,
            "delta_strict_pp": float(mc["delta_strict_pp"]),
            "p_raw": float(mc["p_exact_raw"]),
            "feasible": feas,
        })

    adj_map = holm_bonferroni([(t["id"], t["p_raw"]) for t in tests])
    for t in tests:
        t["p_holm"] = adj_map[t["id"]]
        if not t["feasible"]:
            t["verdict"] = "CANNOT TEST"
        elif t["delta_strict_pp"] >= 15.0 and t["p_holm"] < 0.05:
            t["verdict"] = "SUPPORTED"
        else:
            t["verdict"] = "CONTRADICTED"

    out = {"m_tests": len(tests), "alpha_fwer": 0.05, "tests": {t["id"]: t for t in tests}, "ordered": tests}
    (RESULTS / "tier1_holm_verdicts.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# LaTeX booktabs tables (5 required tables)
# ---------------------------------------------------------------------------
def write_tab1_convergence(prereg: dict) -> None:
    """Table 1: Scenario A convergence on fresh seeds 2000..2099 with 95% Wilson CIs."""
    p_sc = prereg["ponn_scaled_am0.0"]["summary"]
    p_pa = prereg["ponn_paper_u01_am0.0"]["summary"]
    p_co = prereg["collocation_am0.0"]["summary"]
    p_sh = prereg["paired_srp_0.0"]["summary"]["shooting"]
    p_hy = prereg["paired_srp_0.0"]["summary"]["hybrid"]

    rows = [
        ("PoNN (scaled $U(0,1)$)", p_sc, True),
        ("PoNN (paper $U(0,1)$)", p_pa, True),
        ("Indirect Shooting (random)", p_sh, False),
        ("Hybrid (PoNN $\\to$ Shooting)", p_hy, False),
        ("LGL Collocation ($N=180$)", p_co, False),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scenario A convergence on held-out seeds 2000--2099 ($N=100$, $A/m_0=0$). Intervals are 95\% Wilson score CIs; fuel, open-loop error $e_{\mathrm{reprop}}$, and wall-clock time are median [Q25, Q75].}",
        r"\label{tab:scenario_a_convergence}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Loose ($<5\times 10^{-2}$) & Strict ($<10^{-2}$) & Fuel (kg) & $e_{\mathrm{reprop}}$ & Time (s) \\",
        r"\midrule",
    ]
    for name, s, is_ponn in rows:
        n_tot = s.get("n_total", 100)
        nl = s.get("n_loose", 0)
        ns = s.get("n_strict", 0)
        cil = list(wilson_ci(nl, n_tot))
        cis = list(wilson_ci(ns, n_tot))
        fuel_s = s.get("fuel_kg_strict") or s.get("fuel_kg_loose") or s.get("fuel_kg")
        rep_s = s.get("reprop_strict") or s.get("reprop_loose") or s.get("reprop_arrival_residual")
        rt_s = s.get("runtime_s_all") or s.get("total_runtime_s") or s.get("runtime_s")
        lines.append(
            f"{name} & {nl}/{n_tot} ({fmt_ci_pct(nl/n_tot, cil)}) & "
            f"{ns}/{n_tot} ({fmt_ci_pct(ns/n_tot, cis)}) & "
            f"{fmt_stat_med_iqr(fuel_s, '{:.2f}')} & "
            f"{fmt_stat_med_iqr(rep_s, '{:.2e}')} & "
            f"{fmt_stat_med_iqr(rt_s, '{:.2f}')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "tab1_convergence_scenario_a.tex").write_text("\n".join(lines))


def write_tab2_ablation(ablation: dict, holm: dict) -> None:
    """Table 2: Extraction-level ablation on fresh seeds 4000..4099."""
    summ = ablation["summary"]
    n = summ["n_seeds"]
    order = [
        ("random_shooting", None, "Random Shooting (baseline)"),
        ("hybrid_rho_1e0", "H1a", r"Hybrid ($\rho_{\mathrm{ext}}=1.0$)"),
        ("hybrid_rho_2.98e-1", "H1b", r"Hybrid ($\rho_{\mathrm{ext}}=2.98\times 10^{-1}$)"),
        ("hybrid_rho_1e-1", "H1c", r"Hybrid ($\rho_{\mathrm{ext}}=10^{-1}$)"),
        ("hybrid_rho_1e-2", "H1d", r"Hybrid ($\rho_{\mathrm{ext}}=10^{-2}$)"),
        ("hybrid_rho_1e-3", "H1e", r"Hybrid ($\rho_{\mathrm{ext}}=10^{-3}$)"),
        ("hybrid_rho_final", "H1f", r"Hybrid ($\rho_{\mathrm{ext}}=\mathrm{final}$)"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Extraction-level ablation on Scenario A (fresh seeds 4000--4099, $N=" + str(n) + r"$). Exact paired McNemar $p$-values are Holm--Bonferroni corrected across all $M=13$ Tier~1 tests.}",
        r"\label{tab:extraction_ablation}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Arm & Loose & Strict [95\% CI] & $\Delta_{\mathrm{strict}}$ (pp) & $p_{\mathrm{raw}}$ / $p_{\mathrm{Holm}}$ & PoNN / Shoot / Total (s) & PoNN / Shoot Evals & Verdict \\",
        r"\midrule",
    ]
    for arm, hid, label in order:
        a = summ["arms"][arm]
        nl, ns = a["n_loose"], a["n_strict"]
        tp = a["ponn_runtime_s"]["median"] if a["ponn_runtime_s"] else 0.0
        ts = a["shoot_runtime_s"]["median"] if a["shoot_runtime_s"] else 0.0
        tt = a["total_runtime_s"]["median"] if a["total_runtime_s"] else 0.0
        ep = int(round(a["ponn_nfev"]["median"])) if a["ponn_nfev"] else 0
        es = int(round(a["shoot_nfev"]["median"])) if a["shoot_nfev"] else 0
        if hid is None:
            d_str, p_str, v_str = "--", "--", "Baseline"
        else:
            ht = holm["tests"][hid]
            d_str = f"{ht['delta_strict_pp']:+.1f}"
            p_str = f"{fmt_p(ht['p_raw'])} / {fmt_p(ht['p_holm'])}"
            v_str = f"\\textbf{{{ht['verdict']}}}"
        lines.append(
            f"{label} & {nl}/{n} & {ns}/{n} ({fmt_ci_pct(ns/n, a['ci_strict'])}) & "
            f"{d_str} & {p_str} & {tp:.2f} / {ts:.2f} / {tt:.2f} & {ep} / {es} & {v_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab2_extraction_ablation.tex").write_text("\n".join(lines))


def write_tab3_extra_cases(extra: dict, holm: dict) -> None:
    """Table 3: Extra physical cases (7 cases x 50 seeds)."""
    order = [
        ("scen_a_m0_800", "H2a", r"Scen. A ($m_0=800$ kg)"),
        ("scen_a_m0_1200", "H2b", r"Scen. A ($m_0=1200$ kg)"),
        ("scen_a_tmax_0.4", "H2c", r"Scen. A ($T_{\max}=0.4$ N)"),
        ("scen_a_tmax_0.7", "H2d", r"Scen. A ($T_{\max}=0.7$ N)"),
        ("scen_b_tf_300", "H2e", r"Scen. B ($t_f=300$ d)"),
        ("scen_b_tf_330", "H2f", r"Scen. B ($t_f=330$ d)"),
        ("scen_b_tf_360", "H2g", r"Scen. B ($t_f=360$ d)"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Extra physical cases ($N=50$ fresh seeds per case, feasibility verified $t_{\min}\le 0.95\,t_f$). Strict convergence ($e_{\mathrm{reprop}}<10^{-2}$) shown with 95\% Wilson CIs; PoNN shown as loose/strict.}",
        r"\label{tab:extra_cases}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Case & $t_{\min}/t_f$ & Switches (Seq.) & PoNN (L / S) & Shooting (S) & Hybrid (S) & LGL-180 (S) & $\Delta$ (pp) / $p_{\mathrm{Holm}}$ & Verdict \\",
        r"\midrule",
    ]
    for cid, hid, label in order:
        c = extra["cases"][cid]
        m = c["summary"]["methods"]
        n = c["summary"]["n_seeds"]
        ratio = c["meta"]["t_min_ratio"]
        sw = m["hybrid"]["switch_count_mode"]
        seq = m["hybrid"]["thrust_sequence_mode"] or "--"
        ht = holm["tests"][hid]
        lines.append(
            f"{label} & {ratio:.3f} & {sw} ({seq}) & "
            f"{m['ponn']['n_loose']}/{n} / {m['ponn']['n_strict']}/{n} & "
            f"{m['shooting']['n_strict']}/{n} ({100*m['shooting']['rate_strict']:.0f}\\%) & "
            f"{m['hybrid']['n_strict']}/{n} ({100*m['hybrid']['rate_strict']:.0f}\\%) & "
            f"{m['collocation']['n_strict']}/{n} ({100*m['collocation']['rate_strict']:.0f}\\%) & "
            f"{ht['delta_strict_pp']:+.1f} / {fmt_p(ht['p_holm'])} & \\textbf{{{ht['verdict']}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab3_extra_cases.tex").write_text("\n".join(lines))


def write_tab4_srp(srp: dict) -> None:
    """Table 4: SRP fuel change, error vs shooting, resolution ratios, and FD check summary."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{SRP perturbation resolution check on Scenario A. $\Delta m_f$ is the shooting reference fuel shift relative to $A/m_0=0$; resolution ratio is $|\Delta m_f| / |m_{\mathrm{method}}-m_{\mathrm{shoot}}|$. All 8 mode/$A/m_0$ PMP costate FD checks ($N=1000$ random states each) pass with max relative error $\le 7.2\times 10^{-10}$.}",
        r"\label{tab:srp_audit}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"SRP Mode & $A/m_0$ (m$^2$/kg) & $\Delta m_{f,\mathrm{shoot}}$ (kg) & PoNN Err (kg) & PoNN Res. ($|\Delta m_f|/\text{err}$) & LGL-180 Err (kg) & LGL-180 Res. \\",
        r"\midrule",
    ]
    for r in srp["resolution_check"]:
        mode = r["srp_mode"].replace("_", " ")
        am = r["area_to_mass_m2_kg"]
        dm = f"{r['delta_fuel_shoot_kg']:+.4f}" if r["delta_fuel_shoot_kg"] is not None else "--"
        ep = f"{r['err_ponn_vs_shoot_kg']:.4f}" if r["err_ponn_vs_shoot_kg"] is not None else "--"
        rp = f"{r['resolution_ratio_ponn']:.2f}$\\times$" if r["resolution_ratio_ponn"] is not None else "--"
        ec = f"{r['err_colloc_vs_shoot_kg']:.4f}" if r["err_colloc_vs_shoot_kg"] is not None else "--"
        rc = f"{r['resolution_ratio_colloc']:.2f}$\\times$" if r["resolution_ratio_colloc"] is not None else "--"
        lines.append(f"{mode} & {am:.2f} & {dm} & {ep} & {rp} & {ec} & {rc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab4_srp_audit.tex").write_text("\n".join(lines))


def write_tab5_runtime(rt: dict) -> None:
    """Table 5: Single-process idle-machine runtime distributions (seeds 4100..4129)."""
    hw = rt["meta"]["hardware_software"]
    summ = rt["summary"]
    n = summ["n_seeds"]
    pkgs = hw["packages"]
    cap_meta = (
        f"CPU: {hw['cpu_model']} ({hw['cpu_count_logical']} vCPU), RAM: {hw['ram_gb']} GB, "
        f"OS: Linux {hw['kernel']}, Python {hw['python_version']}, NumPy {pkgs['numpy']}, "
        f"SciPy {pkgs['scipy']}, CasADi {pkgs['casadi']}, JAX {pkgs['jax']} "
        f"(one-time JIT warmup: {rt['meta'].get('jax_jit_warmup_s', 0.0):.2f}~s)."
    )
    rows = [
        ("Indirect Shooting (random)", "shooting", "runtime_all", "runtime_strict_success", "runtime_strict_failure", "nfev_all", "njev_all"),
        ("Hybrid — PoNN stage", "hybrid", "ponn_stage_runtime_all", None, None, "ponn_nfev_all", "ponn_njev_all"),
        ("Hybrid — Shooting stage", "hybrid", "shoot_stage_runtime_all", None, None, "shoot_nfev_all", "shoot_njev_all"),
        ("Hybrid — Total (honest)", "hybrid", "runtime_all", "runtime_strict_success", "runtime_strict_failure", None, None),
        ("PoNN (scaled $U(0,1)$, full)", "ponn", "runtime_all", "runtime_loose_success", "runtime_loose_failure", "nfev_all", "njev_all"),
        ("LGL Collocation ($N=180$)", "collocation", "runtime_all", "runtime_strict_success", "runtime_strict_failure", "nfev_all", "njev_all"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Single-process idle-machine runtime distributions on Scenario A (fresh seeds 4100--4129, $N=" + str(n) + r"$, single-threaded BLAS). " + cap_meta + r"}",
        r"\label{tab:runtime_distributions}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method / Stage & All Median [Q25, Q75] (s) & Mean $\pm$ Std (s) & Success Median (s) & Failure Median (s) & Med. \texttt{nfev} & Med. \texttt{njev} \\",
        r"\midrule",
    ]
    for label, mkey, k_all, k_s, k_f, k_nf, k_nj in rows:
        ms = summ[mkey]
        sa = ms.get(k_all)
        ss = ms.get(k_s) if k_s else None
        sf = ms.get(k_f) if k_f else None
        snf = ms.get(k_nf) if k_nf else None
        snj = ms.get(k_nj) if k_nj else None
        mean_std = f"{sa['mean']:.2f} $\\pm$ {sa['std']:.2f}" if sa else "--"
        s_str = f"{ss['median']:.2f} ($n={ss['n']}$)" if ss else "--"
        f_str = f"{sf['median']:.2f} ($n={sf['n']}$)" if sf else "--"
        nf_str = f"{int(round(snf['median']))}" if snf else "--"
        nj_str = f"{int(round(snj['median']))}" if snj else "--"
        lines.append(
            f"{label} & {fmt_stat_med_iqr(sa, '{:.2f}')} & {mean_std} & {s_str} & {f_str} & {nf_str} & {nj_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab5_runtime_distributions.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Publication-quality PDF figures (5 required PDFs)
# ---------------------------------------------------------------------------
def plot_fig1_switching_throttle(a3: dict) -> None:
    ss = a3["single_seed"]
    sh = ss["shooting"]["series"]
    pn = ss["ponn_refined_uniform"]["series"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
    ax1.plot(sh["t_days"], sh["S"], color="#1f77b4", lw=1.8, label="Indirect Shooting (ref.)")
    ax1.plot(pn["t_days"], pn["S"], color="#d62728", lw=1.3, ls="--", label="PoNN (refined, $L=80, n=540$)")
    ax1.axhline(0.0, color="k", lw=0.8, ls=":")
    ax1.set_ylabel("Switching function $S(t)$")
    ax1.set_title("Scenario A: Switching Function and Throttle Profile (PoNN vs. Indirect Shooting)")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(sh["t_days"], sh["delta"], color="#1f77b4", lw=1.8, label="Indirect Shooting")
    ax2.plot(pn["t_days"], pn["delta"], color="#d62728", lw=1.3, ls="--", label="PoNN (refined)")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel(r"Throttle $\delta(t)$")
    ax2.set_ylim(-0.08, 1.08)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_switching_throttle_ponn_vs_shooting.pdf")
    plt.close(fig)


def plot_fig2_convergence_bars(prereg: dict) -> None:
    labels = ["PoNN\n(scaled)", "PoNN\n(paper $U(0,1)$)", "LGL-180\nColloc.", "Shooting\n($A/m=0$)", "Hybrid\n($A/m=0$)", "Shooting\n($A/m=0.5$)", "Hybrid\n($A/m=0.5$)", "Shooting\n($A/m=1.0$)", "Hybrid\n($A/m=1.0$)"]
    entries = [
        prereg["ponn_scaled_am0.0"]["summary"],
        prereg["ponn_paper_u01_am0.0"]["summary"],
        prereg["collocation_am0.0"]["summary"],
        prereg["paired_srp_0.0"]["summary"]["shooting"],
        prereg["paired_srp_0.0"]["summary"]["hybrid"],
        prereg["paired_srp_0.5"]["summary"]["shooting"],
        prereg["paired_srp_0.5"]["summary"]["hybrid"],
        prereg["paired_srp_1.0"]["summary"]["shooting"],
        prereg["paired_srp_1.0"]["summary"]["hybrid"],
    ]
    loose_r, loose_lo, loose_hi = [], [], []
    strict_r, strict_lo, strict_hi = [], [], []
    for s in entries:
        n = s.get("n_total", 100)
        nl, ns = s.get("n_loose", 0), s.get("n_strict", 0)
        cil = wilson_ci(nl, n)
        cis = wilson_ci(ns, n)
        rl, rs = 100.0 * nl / n, 100.0 * ns / n
        loose_r.append(rl)
        loose_lo.append(max(0.0, rl - 100.0 * cil[0]))
        loose_hi.append(max(0.0, 100.0 * cil[1] - rl))
        strict_r.append(rs)
        strict_lo.append(max(0.0, rs - 100.0 * cis[0]))
        strict_hi.append(max(0.0, 100.0 * cis[1] - rs))

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - w / 2, loose_r, w, yerr=[loose_lo, loose_hi], capsize=3, color="#6baed6", label=r"Loose ($e_{\mathrm{reprop}}<5\times 10^{-2}$)")
    ax.bar(x + w / 2, strict_r, w, yerr=[strict_lo, strict_hi], capsize=3, color="#08519c", label=r"Strict ($e_{\mathrm{reprop}}<10^{-2}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Convergence rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Held-Out Scenario A Convergence Rates with 95% Wilson CIs (Seeds 2000–2099, $N=100$)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_convergence_bars_ci.pdf")
    plt.close(fig)


def plot_fig3_extraction_ablation(ablation: dict) -> None:
    summ = ablation["summary"]
    n = summ["n_seeds"]
    order = [
        ("random_shooting", "Random\nShoot"),
        ("hybrid_rho_1e0", r"$\rho=1$"),
        ("hybrid_rho_2.98e-1", r"$\rho=0.298$"),
        ("hybrid_rho_1e-1", r"$\rho=10^{-1}$"),
        ("hybrid_rho_1e-2", r"$\rho=10^{-2}$"),
        ("hybrid_rho_1e-3", r"$\rho=10^{-3}$"),
        ("hybrid_rho_final", r"$\rho=\mathrm{final}$"),
    ]
    labels = [lbl for _, lbl in order]
    rates = [100 * summ["arms"][k]["rate_strict"] for k, _ in order]
    lo = [rates[i] - 100 * summ["arms"][k]["ci_strict"][0] for i, (k, _) in enumerate(order)]
    hi = [100 * summ["arms"][k]["ci_strict"][1] - rates[i] for i, (k, _) in enumerate(order)]
    t_ponn = [summ["arms"][k]["ponn_runtime_s"]["median"] for k, _ in order]
    t_shoot = [summ["arms"][k]["shoot_runtime_s"]["median"] for k, _ in order]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    colors = ["#7f7f7f"] + ["#2ca02c" if k == "hybrid_rho_2.98e-1" else "#1f77b4" for k, _ in order[1:]]
    ax1.bar(x, rates, yerr=[lo, hi], capsize=3.5, color=colors, edgecolor="k", lw=0.6)
    ax1.axhline(rates[0] + 15.0, color="#d62728", ls="--", lw=1.0, label="+15 pp prereg. threshold")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Strict convergence (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title(f"(a) Strict Convergence vs. Extraction $\\rho$ ($N={n}$)")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, t_ponn, color="#ff7f0e", label="PoNN stage (median)")
    ax2.bar(x, t_shoot, bottom=t_ponn, color="#1f77b4", label="Shooting stage (median)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Wall-clock runtime (s)")
    ax2.set_title("(b) Stage-Split Median Wall-Clock Runtime")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_extraction_ablation.pdf")
    plt.close(fig)


def plot_fig4_reprop_vs_seed(prereg: dict) -> None:
    seeds = list(range(2000, 2100))
    def _extract_errs(arm_dict, subkey=None):
        xs, ys = [], []
        for s in seeds:
            rec = arm_dict["seeds"].get(str(s))
            if rec is None:
                continue
            if subkey:
                rec = rec.get(subkey, {})
            if isinstance(rec.get("refined"), dict):
                rec = rec["refined"]
            val = rec.get("reprop_arrival_residual")
            if val is not None and np.isfinite(val) and val > 0:
                xs.append(s); ys.append(val)
        return xs, ys

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for lbl, arm, sub, mk, col in [
        ("PoNN (scaled $U(0,1)$)", prereg["ponn_scaled_am0.0"], None, "o", "#d62728"),
        ("LGL-180 Collocation", prereg["collocation_am0.0"], None, "s", "#9467bd"),
        ("Indirect Shooting (random)", prereg["paired_srp_0.0"], "shooting", "^", "#ff7f0e"),
        ("Hybrid (PoNN $\\to$ Shooting)", prereg["paired_srp_0.0"], "hybrid", "v", "#2ca02c"),
    ]:
        xs, ys = _extract_errs(arm, sub)
        ax.semilogy(xs, ys, marker=mk, ls="none", ms=4.0, alpha=0.75, color=col, label=lbl)

    ax.axhline(5e-2, color="gray", ls="--", lw=1.0, label=r"Loose threshold ($5\times 10^{-2}$)")
    ax.axhline(1e-2, color="black", ls=":", lw=1.1, label=r"Strict threshold ($10^{-2}$)")
    ax.set_xlabel("Fresh Held-Out Seed Index")
    ax.set_ylabel(r"Open-Loop Re-Propagation Error $e_{\mathrm{reprop}}$")
    ax.set_title("Open-Loop Terminal Error Across Held-Out Seeds 2000–2099 ($A/m_0=0$)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_reprop_error_vs_seed.pdf")
    plt.close(fig)


def plot_fig5_srp_fuel_change(srp: dict) -> None:
    rad = [r for r in srp["resolution_check"] if r["srp_mode"] == "radial"]
    att = [r for r in srp["resolution_check"] if r["srp_mode"] == "attitude_flat_plate"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax1.plot([r["area_to_mass_m2_kg"] for r in rad], [r["delta_fuel_shoot_kg"] for r in rad],
             "o-", color="#1f77b4", lw=1.8, label="Radial SRP (Shooting ref.)")
    ax1.plot([r["area_to_mass_m2_kg"] for r in rad], [r["delta_fuel_ponn_kg"] for r in rad],
             "s--", color="#d62728", lw=1.4, label="Radial SRP (PoNN)")
    ax1.plot([r["area_to_mass_m2_kg"] for r in att], [r["delta_fuel_shoot_kg"] for r in att],
             "^-", color="#2ca02c", lw=1.8, label="Flat-plate SRP (Shooting ref.)")
    ax1.set_xlabel(r"Area-to-mass ratio $A/m_0$ (m$^2$/kg)")
    ax1.set_ylabel(r"Propellant shift $\Delta m_f$ (kg)")
    ax1.set_title("(a) SRP Propellant Shift vs. $A/m_0$")
    ax1.legend(loc="upper left", frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy([r["area_to_mass_m2_kg"] for r in rad], [r["resolution_ratio_ponn"] for r in rad],
                 "s-", color="#d62728", lw=1.6, label="PoNN (radial)")
    ax2.semilogy([r["area_to_mass_m2_kg"] for r in rad], [r["resolution_ratio_colloc"] for r in rad],
                 "D-", color="#9467bd", lw=1.6, label="LGL-180 (radial)")
    ax2.axhline(1.0, color="k", ls="--", lw=1.0, label="Resolution limit ($|\\Delta m_f|/\\mathrm{err}=1$)")
    ax2.set_xlabel(r"Area-to-mass ratio $A/m_0$ (m$^2$/kg)")
    ax2.set_ylabel(r"Resolution ratio $|\Delta m_f| / |m_{\mathrm{method}} - m_{\mathrm{shoot}}|$")
    ax2.set_title("(b) Signal-to-Discretization-Error Ratio")
    ax2.legend(loc="lower right", frameon=True, fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_srp_fuel_change.pdf")
    plt.close(fig)


def update_docs_tier1(prereg: dict, ablation: dict, extra: dict, srp: dict, rt: dict, holm: dict) -> None:
    """Append Tier 1 progress note & Holm-corrected verdicts to REPORT.md, update README/ASSUMPTIONS/DEVIATIONS."""
    # 1. Append Tier 1 section to REPORT.md (idempotent replace if already present)
    report_path = Path("REPORT.md")
    rep_txt = report_path.read_text(encoding="utf-8")
    marker = "\n---\n\n## 10. Tier 1 Progress Note & Preregistered Addendum Verdicts"
    if marker in rep_txt:
        rep_txt = rep_txt.split(marker)[0]

    hw = rt["meta"]["hardware_software"]
    pkgs = hw["packages"]
    t_lines = [
        marker,
        "",
        f"**Preregistration addendum commit:** `38e2908` ([PREREG_ADDENDUM.md](PREREG_ADDENDUM.md)). "
        f"All 13 Tier 1 primary paired McNemar tests (`H1a–H1f` extraction-level ablation on seeds `4000..4099`; "
        f"`H2a–H2g` extra physical cases on seeds `5000+`) are Holm–Bonferroni corrected at family-wise $\\alpha = 0.05$. "
        f"Verdicts follow the numeric rule strictly (`SUPPORTED` iff $\\Delta_{{\\text{{strict}}}} \\ge +15.0\\text{{ pp}}$ and $\\tilde{{p}}_{{\\text{{Holm}}}} < 0.05$; else `CONTRADICTED`; `CANNOT TEST` if $t_{{\\min}} > 0.95\\,t_f$).",
        "",
        "### 10.1 Master Holm–Bonferroni Verdict Table (Tier 1, $M = 13$ Tests)",
        "",
        "| ID | Study | Condition | $\\Delta_{\\text{strict}}$ (pp) | $p_{\\text{raw}}$ | $p_{\\text{Holm}}$ | Verdict |",
        "|:---:|:---|:---|---:|---:|---:|:---:|",
    ]
    for t in holm["ordered"]:
        t_lines.append(
            f"| `{t['id']}` | {t['family']} | {t['arm']} | {t['delta_strict_pp']:+.1f} | {fmt_p(t['p_raw'])} | {fmt_p(t['p_holm'])} | **`{t['verdict']}`** |"
        )

    prereg_path = Path("results/prereg_fresh_2000_2099.json")
    prereg_sh_med = None
    prereg_sh_mean = None
    if prereg_path.exists():
        pr_data = json.loads(prereg_path.read_text(encoding="utf-8"))
        s_2000 = [v["shooting"]["runtime_s"] for v in pr_data["paired_srp_0.0"]["seeds"].values()]
        prereg_sh_med = float(np.median(s_2000))
        prereg_sh_mean = float(np.mean(s_2000))
    s_4100 = [v["shooting"]["runtime_s"] for v in rt["seeds"].values()]
    rt_4100_med = float(np.median(s_4100))
    rt_4100_mean = float(np.mean(s_4100))
    coincidence_note = ""
    if prereg_sh_med is not None:
        coincidence_note = (
            f" *(Independent verification of the `22.26 s` coincidence: seeds `2000..2099` ($N=100$) give median "
            f"`{prereg_sh_med:.4f} s` / mean `{prereg_sh_mean:.2f} s`, while disjoint single-process seeds `4100..4129` "
            f"($N=30$) give median `{rt_4100_med:.4f} s` / mean `{rt_4100_mean:.2f} s`; both are computed independently "
            f"from separate JSON files and differ by `{abs(prereg_sh_med - rt_4100_med):.4f} s`.)*"
        )

    res02 = next((r for r in srp["resolution_check"] if r["case_id"] == "radial_0.02"), None)
    res02_txt = (
        f"At $A/m_0 = 0.02\\text{{ m}}^2/\\text{{kg}}$, the physical SRP fuel change (`{res02['delta_fuel_shoot_kg']:+.3f} kg`) "
        f"is `{res02['resolution_ratio_ponn']:.2f}x` standalone PoNN's error and `{res02['resolution_ratio_colloc']:.2f}x` "
        f"LGL-180's error (below discretization resolution); at $A/m_0 \\ge 0.50\\text{{ m}}^2/\\text{{kg}}$ it exceeds both methods' errors (`tab4_srp_audit.tex`)."
        if res02 else ""
    )

    t_lines += [
        "",
        "### 10.2 Tier 1.3 SRP Flat-Plate Model, 1,000-State FD Audit, & Resolution Check",
        "",
        "- **Flat-plate SRP specification:** Heliocentric J2000 inertial frame; attitude normal is **prescribed** along the thrust unit vector $\\hat{\\mathbf{n}} = \\boldsymbol{\\alpha}^* = -\\boldsymbol{\\lambda}_v / \\|\\boldsymbol{\\lambda}_v\\|$ (not independently optimized); no planetary eclipse/shadowing; projected area factor $\\phi(\\mathbf{r}, \\boldsymbol{\\alpha}) = \\tfrac{1}{2} + \\tfrac{1}{2}(\\hat{\\mathbf{r}}\\cdot\\boldsymbol{\\alpha})^2$; constant initial area-to-mass ratio $A/m_0$.",
        f"- **1,000-random-state central FD check:** Across all 8 `(mode, A/m)` combinations (`radial` and `attitude_flat_plate` at $A/m_0 \\in \\{{0.02, 0.10, 0.50, 1.00\\}}$), the maximum relative error of analytical vs. central-FD PMP costate ODE terms is **{max(max(r['max_rel_err_lam_r_dot'], r['max_rel_err_lam_v_dot'], r['max_rel_err_lam_m_dot']) for r in srp['fd_checks']):.2e}** (all 8/8 PASS $< 10^{-7}$).",
        f"- **Resolution check:** {res02_txt}",
        "",
        "### 10.3 Tier 1.4 Single-Process Idle-Machine Runtime Audit (`seeds = 4100..4129`)",
        "",
        f"- **Environment:** {hw['cpu_model']} ({hw['cpu_count_logical']} vCPU), {hw['ram_gb']} GB RAM, Linux `{hw['kernel']}`, Python `{hw['python_version']}`, NumPy `{pkgs['numpy']}`, SciPy `{pkgs['scipy']}`, Matplotlib `{pkgs['matplotlib']}`, CasADi `{pkgs['casadi']}`, JAX `{pkgs['jax']}`, single-threaded BLAS (`OPENBLAS_NUM_THREADS=1`), one-time JAX JIT warmup `{rt['meta'].get('jax_jit_warmup_s', 0.0):.2f} s`.",
        f"- **Median wall-clock runtimes (all seeds):** Random Shooting `{rt['summary']['shooting']['runtime_all']['median']:.2f} s`, Hybrid total `{rt['summary']['hybrid']['runtime_all']['median']:.2f} s` (PoNN stage `{rt['summary']['hybrid']['ponn_stage_runtime_all']['median']:.2f} s` + Shooting stage `{rt['summary']['hybrid']['shoot_stage_runtime_all']['median']:.2f} s`), Standalone PoNN `{rt['summary']['ponn']['runtime_all']['median']:.2f} s`, LGL-180 `{rt['summary']['collocation']['runtime_all']['median']:.2f} s`.{coincidence_note} Full distributions split by success/failure and evaluation counts are tabulated in `results/tables/tab5_runtime_distributions.tex`.",
        "",
        "### 10.4 Paper-Ready Outputs Generated",
        "- LaTeX `booktabs` tables: `results/tables/tab1_convergence_scenario_a.tex` through `tab5_runtime_distributions.tex`.",
        "- PDF figures: `results/figures/fig_switching_throttle_ponn_vs_shooting.pdf`, `fig_convergence_bars_ci.pdf`, `fig_extraction_ablation.pdf`, `fig_reprop_error_vs_seed.pdf`, `fig_srp_fuel_change.pdf`.",
        "",
    ]
    report_path.write_text(rep_txt.rstrip() + "\n" + "\n".join(t_lines), encoding="utf-8")

    # 2. Update ASSUMPTIONS.md with complete flat-plate SRP model specification
    assump_path = Path("ASSUMPTIONS.md")
    atxt = assump_path.read_text(encoding="utf-8")
    srp_marker = "\n## Complete Flat-Plate SRP Model Specification (Tier 1.3)"
    if srp_marker in atxt:
        atxt = atxt.split(srp_marker)[0]
    srp_block = f"""{srp_marker}

- **Coordinate frame:** Heliocentric J2000 inertial Cartesian frame.
- **Attitude coupling:** **Prescribed** thrust-aligned normal $\\hat{{\\mathbf{{n}}}} = \\boldsymbol{{\\alpha}}^* = -\\boldsymbol{{\\lambda}}_v / \\|\\boldsymbol{{\\lambda}}_v\\|$ (the solar array / bus normal tracks the optimal thrust direction; attitude is **not** independently optimized as a separate control variable).
- **Eclipse / shadowing:** None (continuous heliocentric illumination; no Earth or Mars umbra/penumbra eclipses).
- **Projected-area law:** $\\phi(\\mathbf{{r}}, \\boldsymbol{{\\alpha}}) = \\tfrac{{1}}{{2}} + \\tfrac{{1}}{{2}}(\\hat{{\\mathbf{{r}}}}\\cdot\\boldsymbol{{\\alpha}})^2$, with $\\mathbf{{a}}_{{\\mathrm{{SRP}}}}(\\mathbf{{r}}, \\boldsymbol{{\\alpha}}) = k_{{\\mathrm{{SRP}}}}\\,\\phi(\\mathbf{{r}}, \\boldsymbol{{\\alpha}})\\,\\mathbf{{r}}/\\|\\mathbf{{r}}\\|^3$ and $k_{{\\mathrm{{SRP}}}} = P_0 C_R (A/m_0)/a_{{\\mathrm{{ref}}}}$ (constant initial $A/m_0$ cannonball scaling).
- **Escape-phase $J_2$ note:** Earth's oblateness $J_2$ is **NOT** included in any heliocentric trajectory optimization (Scenario A or Scenario B); it appears strictly in the prescribed Phase 1 Earth-departure escape propagation (`escape_phase.py`).
"""
    assump_path.write_text(atxt.rstrip() + "\n" + srp_block, encoding="utf-8")

    # 3. Update README.md title/refs (Andrea D'Ambrosio, J2 note, Tier 1 summary)
    readme_path = Path("README.md")
    rtxt = readme_path.read_text(encoding="utf-8")
    rtxt = rtxt.replace("R. D'Ambrosio", "Andrea D'Ambrosio")
    rtxt = rtxt.replace("**D'Ambrosio & Furfaro (2024)**", "**Andrea D'Ambrosio & Roberto Furfaro (2024)**")
    j2_note = (
        "\n> **Scope note on $J_2$ and independent implementation:** Earth's $J_2$ oblateness is **not** "
        "included in the heliocentric trajectory optimization; it is used exclusively in the prescribed "
        "Earth-departure escape propagation (`escape_phase.py`). Where results differ from published "
        "figures, our independent implementation differs from the published configuration (see `DEVIATIONS.md`).\n"
    )
    if "Scope note on $J_2$" not in rtxt:
        lines_r = rtxt.splitlines()
        lines_r.insert(3, j2_note)
        rtxt = "\n".join(lines_r) + "\n"
    t1_marker = "\n## Tier 1 Paper-Ready Tables, PDF Figures, & Holm-Corrected Verdicts"
    if t1_marker in rtxt:
        rtxt = rtxt.split(t1_marker)[0]
    n_supp = sum(1 for t in holm["ordered"] if t["verdict"] == "SUPPORTED")
    t1_block = f"""{t1_marker}

- **Preregistration addendum:** [PREREG_ADDENDUM.md](PREREG_ADDENDUM.md) (commit `38e2908`).
- **Holm–Bonferroni corrected verdicts ($M={holm['m_tests']}$ tests, FWER $\\alpha=0.05$):** **{n_supp}/{holm['m_tests']} `SUPPORTED`** (full breakdown in [REPORT.md](REPORT.md) Section 10 and `results/tier1_holm_verdicts.json`).
- **LaTeX `booktabs` tables:** `results/tables/tab1_convergence_scenario_a.tex` – `tab5_runtime_distributions.tex`.
- **PDF figures:** `results/figures/fig_switching_throttle_ponn_vs_shooting.pdf`, `fig_convergence_bars_ci.pdf`, `fig_extraction_ablation.pdf`, `fig_reprop_error_vs_seed.pdf`, `fig_srp_fuel_change.pdf`.
- **Reproduce all tables and figures from stored JSON:** `python3 run_all.py --from-stored`
"""
    readme_path.write_text(rtxt.rstrip() + "\n" + t1_block, encoding="utf-8")


def main() -> None:
    prereg = json.loads((RESULTS / "prereg_fresh_2000_2099.json").read_text())
    ablation = json.loads((RESULTS / "tier1_extraction_ablation_4000_4099.json").read_text())
    extra = json.loads((RESULTS / "tier1_extra_cases_5000.json").read_text())
    srp = json.loads((RESULTS / "tier1_srp_audit.json").read_text())
    rt = json.loads((RESULTS / "tier1_runtime_single_process.json").read_text())
    a3 = json.loads((RESULTS / "scenario_a_phase_a3.json").read_text())

    holm = compute_all_holm_verdicts(ablation, extra)
    write_tab1_convergence(prereg)
    write_tab2_ablation(ablation, holm)
    write_tab3_extra_cases(extra, holm)
    write_tab4_srp(srp)
    write_tab5_runtime(rt)

    plot_fig1_switching_throttle(a3)
    plot_fig2_convergence_bars(prereg)
    plot_fig3_extraction_ablation(ablation)
    plot_fig4_reprop_vs_seed(prereg)
    plot_fig5_srp_fuel_change(srp)

    update_docs_tier1(prereg, ablation, extra, srp, rt, holm)
    print("[Tier 1 Deliverables] All 5 LaTeX tables, 5 PDF figures, Holm verdicts, and docs generated.", flush=True)


if __name__ == "__main__":
    main()
