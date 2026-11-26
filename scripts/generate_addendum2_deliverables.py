"""Generate Addendum 2 LaTeX table, PDF figure, and REPORT.md / README.md sections from stored JSON.

Reads:
  - results/addendum2_baselines_8000_8099.json
Writes:
  - results/tables/tab9_addendum2_baselines.tex
  - results/figures/fig_addendum2_baselines.pdf
Updates:
  - REPORT.md (Section 13)
  - README.md (Addendum 2 summary)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def write_tab9_baselines(data: dict) -> None:
    summ = data["summary"]
    arms = summ["arms"]
    det = data["det_energy_shoot"]
    n = summ["n_seeds"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Preregistration Addendum~2 baseline comparison on Scenario~A (fresh seeds 8000--8099, $N="
        + str(n)
        + r"$, $A/m_0=0$). Multi-start shooting runs up to 5 random restarts and stops at the first strict success ($e_{\mathrm{reprop}} < 10^{-2}$). Deterministic energy-optimal shooting solves $\rho=1 \to 10^{-5}$ from the fixed prograde guess $\boldsymbol{\lambda}_0^{\mathrm{fix}} = [\mathbf{0}_3, -\hat{\mathbf{v}}_0, 0.5]$ (5 idle-machine timing trials).}",
        r"\label{tab:addendum2_baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Strict ($<10^{-2}$) [95\% CI] & Med. Time All (s) [IQR] & Med. Time Strict (s) & Med. $e_{\mathrm{reprop}}$ & Notes \\",
        r"\midrule",
    ]

    # Deterministic baseline row
    dt_med = det["runtime_stats"]["median"]
    dt_q25 = det["runtime_stats"]["q25"]
    dt_q75 = det["runtime_stats"]["q75"]
    dt_rp = det["reprop_arrival_residual"]
    dt_ok = "1/1 (det.)" if det["success_strict"] else "0/1 (det.)"
    lines.append(
        f"Deterministic Energy-Opt. Shoot & {dt_ok} & {dt_med:.2f} [{dt_q25:.2f}, {dt_q75:.2f}] & {dt_med:.2f} & {dt_rp:.2e} & fixed prograde $\\boldsymbol{{\\lambda}}_0^{{\\mathrm{{fix}}}}$ \\\\"
    )

    arm_specs = [
        ("single_shoot", "Single-Start Random Shooting ($k=0$)", lambda a: "1 random costate"),
        (
            "multistart_shoot_5",
            "Multi-Start Random Shooting ($K \\le 5$)",
            lambda a: f"med. {a['attempts_strict']['median']:.0f} att. (mean {a['attempts_strict']['mean']:.2f})" if a.get("attempts_strict") else "—",
        ),
        (
            "hybrid_rho1",
            "Hybrid (PoNN $\\rho_{\\mathrm{ext}}=1 \\to$ Shoot)",
            lambda a: f"PoNN {a['ponn_time_s']['median']:.2f}s + Shoot {a['shoot_time_s']['median']:.2f}s",
        ),
        (
            "split_domain",
            "Split-Domain PoNN Pipeline (5-arc)",
            lambda a: f"Coarse {a['ponn_time_s']['median']:.2f}s + Split {a['split_time_s']['median']:.2f}s" if a.get("split_time_s") else "—",
        ),
    ]
    for key, label, note_fn in arm_specs:
        a = arms[key]
        ks = a["n_strict"]
        ci = a["ci_strict"]
        ta = a["time_all_s"]
        ts = a["time_strict_s"]
        rp = a["reprop_strict"]
        ts_str = f"{ts['median']:.2f}" if ts else "---"
        rp_str = f"{rp['median']:.2e}" if rp else "---"
        lines.append(
            f"{label} & {ks}/{n} ({100*ks/n:.1f}\\% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]) & "
            f"{ta['median']:.2f} [{ta['q25']:.2f}, {ta['q75']:.2f}] & "
            f"{ts_str} & {rp_str} & {note_fn(a)} \\\\"
        )

    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Holm--Bonferroni corrected pairwise tests ($M=" + str(len(summ["holm_tests"])) + r"$, FWER $\alpha=0.05$):}} \\",
    ]
    for t in summ["holm_tests"]:
        if t["type"] == "mcnemar":
            eff = f"$\\Delta={t['delta_pp']:+.1f}$ pp (A-only={t['a_only']}, B-only={t['b_only']})"
        else:
            eff = f"$\\Delta t_{{\\mathrm{{med}}}}={t['median_diff_s']:+.2f}$ s ($n={t['n_paired']}$)"
        lines.append(
            f"\\texttt{{{t['id']}}}: {t['comparison'].replace('_', r'\_')} & {eff} & $p_{{\\mathrm{{raw}}}}={_fmt_p(t['p_raw'])}$ & $p_{{\\mathrm{{Holm}}}}={_fmt_p(t['p_holm'])}$ & \\multicolumn{{2}}{{c}}{{\\textbf{{{t['verdict']}}}}} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab9_addendum2_baselines.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_fig_addendum2(data: dict) -> None:
    summ = data["summary"]
    arms = summ["arms"]
    det = data["det_energy_shoot"]
    n = summ["n_seeds"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["Single\nShoot", "Multi-Start\nShoot (K≤5)", "Hybrid\n(ρ_ext=1)", "Split-Domain\nPoNN"]
    keys = ["single_shoot", "multistart_shoot_5", "hybrid_rho1", "split_domain"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]

    rates = [100.0 * arms[k]["rate_strict"] for k in keys]
    yerr_lo = [max(0.0, rates[i] - 100.0 * arms[k]["ci_strict"][0]) for i, k in enumerate(keys)]
    yerr_hi = [max(0.0, 100.0 * arms[k]["ci_strict"][1] - rates[i]) for i, k in enumerate(keys)]

    x = np.arange(len(keys))
    bars = ax1.bar(x, rates, color=colors, alpha=0.88, yerr=[yerr_lo, yerr_hi], capsize=5, edgecolor="black", lw=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Strict Convergence Rate (%) [e_reprop < 1e-2]")
    ax1.set_ylim(0, 112)
    ax1.set_title(f"Strict Convergence (Fresh Seeds 8000–8099, N={n})")
    ax1.grid(axis="y", alpha=0.3)
    for b, r in zip(bars, rates):
        ax1.text(b.get_x() + b.get_width() / 2, r + 3.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right panel: wall-clock time to first strict success (boxplots on strict successes + det line)
    rows = [data["seeds"][k] for k in sorted(data["seeds"].keys(), key=int)]
    time_lists = []
    for k, tk in [
        ("single_shoot", "runtime_s"),
        ("multistart_shoot_5", "total_runtime_s"),
        ("hybrid_rho1", "total_runtime_s"),
        ("split_domain", "total_runtime_s"),
    ]:
        time_lists.append([r[k][tk] for r in rows if r[k]["success_strict"]])

    bp = ax2.boxplot(time_lists, positions=x, widths=0.55, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.6)
    dt_med = det["runtime_stats"]["median"]
    ax2.axhline(dt_med, color="#c44e52", ls="--", lw=1.6, label=f"Det. Energy-Opt. Shoot ({dt_med:.1f} s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Total Wall-Clock Time to First Strict Success (s)")
    ax2.set_title("Time-to-First-Strict-Success Distribution")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_addendum2_baselines.pdf")
    plt.close(fig)


def update_report_and_readme(data: dict) -> None:
    summ = data["summary"]
    arms = summ["arms"]
    det = data["det_energy_shoot"]
    n = summ["n_seeds"]

    report_path = Path("REPORT.md")
    txt = report_path.read_text(encoding="utf-8")
    marker = "\n---\n\n## 13. Preregistration Addendum 2: Multi-Start & Deterministic Baselines (`seeds = 8000..8099`)"
    if marker in txt:
        txt = txt.split(marker)[0]

    dt_med = det["runtime_stats"]["median"]
    dt_std = det["runtime_stats"]["std"]
    lines = [
        marker,
        "",
        f"**Preregistration commit:** `{data['meta']['prereg_addendum2_commit']}` ([PREREG_ADDENDUM_2.md](PREREG_ADDENDUM_2.md)). "
        f"Evaluated on fresh untouched seeds `8000..{8000 + n - 1}` ($N = {n}$, Scenario A, $A/m_0 = 0$).",
        "",
        "### 13.1 Baseline Comparison Summary",
        "",
        "| Method | Strict (<1e-2) [95% Wilson CI] | Med. Time All (s) [IQR] | Med. Time Strict (s) | Med. $e_{\\text{reprop}}$ | Stage / Restart Breakdown |",
        "|:---|:---:|---:|---:|---:|:---|",
        f"| Deterministic Energy-Opt. Shoot | {'1/1 (det.)' if det['success_strict'] else '0/1 (det.)'} | {dt_med:.2f} [{det['runtime_stats']['q25']:.2f}, {det['runtime_stats']['q75']:.2f}] | {dt_med:.2f} | {det['reprop_arrival_residual']:.2e} | Fixed prograde $\\boldsymbol{{\\lambda}}_0^{{\\text{{fix}}}}$ ({det['n_propagations']} prop., SD {dt_std:.2f} s) |",
    ]
    for key, label, note in [
        ("single_shoot", "Single-Start Random Shooting (`k=0`)", "1 random costate"),
        (
            "multistart_shoot_5",
            "Multi-Start Random Shooting (`K<=5`)",
            f"Med. {arms['multistart_shoot_5']['attempts_strict']['median']:.0f} att. (mean {arms['multistart_shoot_5']['attempts_strict']['mean']:.2f})"
            if arms["multistart_shoot_5"].get("attempts_strict") else "—",
        ),
        (
            "hybrid_rho1",
            "Hybrid (`PoNN rho_ext=1 -> Shoot`)",
            f"PoNN {arms['hybrid_rho1']['ponn_time_s']['median']:.2f} s + Shoot {arms['hybrid_rho1']['shoot_time_s']['median']:.2f} s",
        ),
        (
            "split_domain",
            "Split-Domain PoNN Pipeline (5-arc)",
            f"Coarse {arms['split_domain']['ponn_time_s']['median']:.2f} s + Split {arms['split_domain']['split_time_s']['median']:.2f} s"
            if arms["split_domain"].get("split_time_s") else "—",
        ),
    ]:
        a = arms[key]
        ks = a["n_strict"]
        ci = a["ci_strict"]
        ta = a["time_all_s"]
        ts = a["time_strict_s"]
        rp = a["reprop_strict"]
        ts_str = f"{ts['median']:.2f}" if ts else "—"
        rp_str = f"{rp['median']:.2e}" if rp else "—"
        lines.append(
            f"| {label} | {ks}/{n} ({100*ks/n:.1f}% [{100*ci[0]:.1f}%, {100*ci[1]:.1f}%]) | "
            f"{ta['median']:.2f} [{ta['q25']:.2f}, {ta['q75']:.2f}] | "
            f"{ts_str} | {rp_str} | {note} |"
        )

    lines += [
        "",
        f"### 13.2 Holm–Bonferroni Corrected Pairwise Tests ($M = {len(summ['holm_tests'])}$, FWER $\\alpha = 0.05$)",
        "",
        "| ID | Comparison | Test | Effect | $p_{\\text{raw}}$ | $p_{\\text{Holm}}$ | Verdict |",
        "|:---:|:---|:---|:---|---:|---:|:---:|",
    ]
    for t in summ["holm_tests"]:
        if t["type"] == "mcnemar":
            eff = f"`{t['delta_pp']:+.1f} pp` (A-only={t['a_only']}, B-only={t['b_only']})"
        else:
            eff = f"`{t['median_diff_s']:+.2f} s` (n={t['n_paired']})"
        lines.append(
            f"| `{t['id']}` | {t['comparison']} | {t['type']} | {eff} | {_fmt_p(t['p_raw'])} | {_fmt_p(t['p_holm'])} | **`{t['verdict']}`** |"
        )
    lines += [
        "",
        "- Deliverables: `results/tables/tab9_addendum2_baselines.tex` and `results/figures/fig_addendum2_baselines.pdf`.",
        "",
    ]
    report_path.write_text(txt.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")

    # Update README.md Addendum 2 summary
    readme_path = Path("README.md")
    rtxt = readme_path.read_text(encoding="utf-8")
    rmarker = "\n## Addendum 2 Baselines & Split-Domain Comparison (`seeds 8000..8099`)"
    if rmarker in rtxt:
        rtxt = rtxt.split(rmarker)[0]
    a_ss = arms["single_shoot"]
    a_ms = arms["multistart_shoot_5"]
    a_hy = arms["hybrid_rho1"]
    a_sd = arms["split_domain"]
    rblock = f"""{rmarker}

- **Preregistration:** [PREREG_ADDENDUM_2.md](PREREG_ADDENDUM_2.md) (commit `{data['meta']['prereg_addendum2_commit']}`).
- **Strict convergence ($e_{{\\text{{reprop}}}} < 10^{{-2}}$, $N={n}$):** Single-start Shooting `{a_ss['n_strict']}/{n}`, Multi-start Shooting ($K\\le 5$) `{a_ms['n_strict']}/{n}` (med. `{a_ms['time_strict_s']['median']:.2f} s`), Deterministic Energy-Opt. Shooting `1/1` (`{dt_med:.2f} s`), Hybrid (`rho_ext=1`) `{a_hy['n_strict']}/{n}` (med. `{a_hy['time_strict_s']['median']:.2f} s`), Split-Domain PoNN `{a_sd['n_strict']}/{n}` (med. `{a_sd['time_strict_s']['median']:.2f} s`).
- **Deliverables:** `results/tables/tab9_addendum2_baselines.tex`, `results/figures/fig_addendum2_baselines.pdf`, and [REPORT.md](REPORT.md) Section 13.
"""
    readme_path.write_text(rtxt.rstrip() + "\n" + rblock, encoding="utf-8")


def main() -> None:
    path = RESULTS / "addendum2_baselines_8000_8099.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    write_tab9_baselines(data)
    plot_fig_addendum2(data)
    update_report_and_readme(data)
    print(f"[Addendum 2 Deliverables] Generated tab9, fig_addendum2_baselines.pdf, updated REPORT.md §13 & README.md (N={data['summary']['n_seeds']}).", flush=True)


if __name__ == "__main__":
    main()
