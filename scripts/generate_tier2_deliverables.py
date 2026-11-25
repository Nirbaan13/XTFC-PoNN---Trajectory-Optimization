"""Generate Tier 2 LaTeX booktabs tables, PDF plots, and REPORT.md Section 11 progress note
strictly from stored JSONs:
  - results/tier2_plateau_6000_6029.json
  - results/tier2_collocation_mesh_6000_6029.json

Applies the preregistered numeric wording rule from PREREG_ADDENDUM.md:
  State "observed plateau" ONLY if the median open-loop error across converged seeds
  stays within a factor of 2.0 across the overdetermined (L, n) sweep; otherwise state
  that it decreases and report the exact reduction factor.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def _f(v, fmt="{:.2e}"):
    if v is None or not np.isfinite(v):
        return "--"
    return fmt.format(v)


def analyze_plateau_verdict(pl: dict) -> dict:
    """Apply preregistered numeric rule: ratio max_med / min_med across overdetermined (L, n) CGL cells."""
    meds = {}
    for cid, c in pl["cells"].items():
        m = c["meta"]
        if m["grid_type"] == "cgl" and m["tol"] == 1e-9 and not m["underdetermined"]:
            s = c["summary"].get("reprop_smooth_loose")
            if s and s["n"] >= 3:
                meds[cid] = float(s["median"])
    if not meds:
        return {"factor": None, "verdict_phrase": "CANNOT TEST", "min_cell": None, "max_cell": None}
    min_k = min(meds, key=meds.get)
    max_k = max(meds, key=meds.get)
    factor = meds[max_k] / max(meds[min_k], 1e-15)
    is_plateau = factor <= 2.0
    phrase = (
        f"observed plateau (factor {factor:.2f}x <= 2.0 across overdetermined sweep)"
        if is_plateau
        else f"decreases by a factor of {factor:.2f}x (from {meds[max_k]:.2e} at {max_k} to {meds[min_k]:.2e} at {min_k})"
    )
    return {
        "factor": factor,
        "is_plateau": is_plateau,
        "min_cell": min_k,
        "min_val": meds[min_k],
        "max_cell": max_k,
        "max_val": meds[max_k],
        "verdict_phrase": phrase,
    }


def write_tab6_plateau(pl: dict, verdict: dict) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Tier~2.7 PoNN $(L, n)$ sweep on Scenario A (fresh seeds 6000--6029, $N=30$ per cell). Condition numbers $\kappa_2(J_s)$ (column-scaled QR/SVD Jacobian) and $\kappa_2(J)$ (unscaled) are medians at terminal $\rho$. Finding: " + verdict["verdict_phrase"].replace("_", r"\_") + r".}",
        r"\label{tab:tier2_plateau}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Cell (Grid, $L$, $n$, tol) & Loose / Strict & $e_{\mathrm{reprop}}$ (smooth) & $e_{\mathrm{reprop}}$ (hard) & State RMS & $\kappa_2(J_s)$ & $\kappa_2(J)$ & Time (s) \\",
        r"\midrule",
    ]
    for cid, c in pl["cells"].items():
        m = c["meta"]
        s = c["summary"]
        n_tot = s.get("n_seeds", 0)
        if n_tot == 0:
            continue
        nl, ns = s["n_loose"], s["n_strict"]
        rp_s = s["reprop_smooth_loose"]["median"] if s.get("reprop_smooth_loose") else None
        rp_h = s["reprop_hard_loose"]["median"] if s.get("reprop_hard_loose") else None
        st_r = s["state_approx_rms_loose"]["median"] if s.get("state_approx_rms_loose") else None
        ks = s["cond_J_scaled"]["median"] if s.get("cond_J_scaled") else None
        ku = s["cond_J_unscaled"]["median"] if s.get("cond_J_unscaled") else None
        rt = s["runtime_s"]["median"] if s.get("runtime_s") else None
        tag = f"{m['grid_type'].upper()}, $L={m['L']}$, $n={m['n']}$, {m['tol']:.0e}" + ("$^*$" if m["underdetermined"] else "")
        lines.append(
            f"{tag} & {nl}/{n_tot} / {ns}/{n_tot} & {_f(rp_s)} & {_f(rp_h)} & {_f(st_r)} & {_f(ks)} & {_f(ku)} & {_f(rt, '{:.2f}')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab6_ponn_plateau.tex").write_text("\n".join(lines))


def write_tab7_collocation(cm: dict) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Tier~2.8 LGL collocation mesh study on Scenario A (fresh seeds 6000--6029, $N=" + str(cm["summary"]["n_seeds"]) + r"$ per mesh), including two-stage mesh refinement ($N=120 \to 240$).}",
        r"\label{tab:tier2_collocation_mesh}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Mesh & Loose [95\% CI] & Strict [95\% CI] & Med. $e_{\mathrm{reprop}}$ (loose) & Med. $|\Delta m_f|$ vs. Shoot (kg) & Med. Time (s) \\",
        r"\midrule",
    ]
    labels = {
        "N60": "$N=60$",
        "N120": "$N=120$",
        "N180": "$N=180$",
        "N240": "$N=240$",
        "N120_to_N240_refined": r"$N=120 \to 240$ (refined)",
    }
    n = cm["summary"]["n_seeds"]
    for mk, lbl in labels.items():
        ms = cm["summary"]["meshes"][mk]
        nl, ns = ms["n_loose"], ms["n_strict"]
        cil, cis = ms["ci_loose"], ms["ci_strict"]
        rp = ms["reprop_loose"]["median"] if ms.get("reprop_loose") else None
        fe = ms["fuel_err_loose"]["median"] if ms.get("fuel_err_loose") else None
        rt = ms["runtime_s"]["median"] if ms.get("runtime_s") else None
        lines.append(
            f"{lbl} & {nl}/{n} ({100*nl/n:.1f}\\% [{100*cil[0]:.1f}, {100*cil[1]:.1f}]) & "
            f"{ns}/{n} ({100*ns/n:.1f}\\% [{100*cis[0]:.1f}, {100*cis[1]:.1f}]) & "
            f"{_f(rp)} & {_f(fe, '{:.3f}')} & {_f(rt, '{:.2f}')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab7_collocation_mesh.tex").write_text("\n".join(lines))


def plot_fig_tier2_plateau(pl: dict) -> None:
    ns_list = [140, 280, 540, 1080]
    Ls_list = [40, 60, 80, 120, 160]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, L in enumerate(Ls_list):
        xs, ys = [], []
        for n in ns_list:
            if n < L:
                continue
            cid = f"cgl_L{L}_n{n}_t1e-9"
            s = pl["cells"][cid]["summary"].get("reprop_smooth_loose")
            if s and s["n"] > 0:
                xs.append(n); ys.append(s["median"])
        if xs:
            ax1.loglog(xs, ys, "o-", color=colors[idx], lw=1.6, label=f"CGL, $L={L}$")

    # Control-discretization baseline (shooting control sampled on CGL grid)
    base_grids = pl["control_discretization_baseline"]["grids"]
    bx = [n for n in ns_list]
    by_cgl = [base_grids[f"cgl_n{n}"]["ctrl_disc_err_smooth"] for n in ns_list]
    ax1.loglog(bx, by_cgl, "k--", lw=1.5, label="Shooting ctrl on CGL grid")
    ax1.axhline(1e-2, color="gray", ls=":", lw=1.0, label="Strict ($10^{-2}$)")
    ax1.set_xlabel("Collocation points $n$")
    ax1.set_ylabel(r"Median open-loop $e_{\mathrm{reprop}}$")
    ax1.set_title("(a) PoNN Open-Loop Error vs. $(L, n)$ (CGL)")
    ax1.set_xticks(ns_list); ax1.set_xticklabels([str(x) for x in ns_list])
    ax1.legend(loc="best", frameon=True, fontsize=7.5)
    ax1.grid(True, which="both", alpha=0.25)

    # Panel (b): Uniform vs CGL at L=80 + smooth vs hard + tol=1e-12
    for g, col, lbl in [("cgl", "#2ca02c", "CGL ($L=80$, smooth)"), ("uniform", "#d62728", "Uniform ($L=80$, smooth)")]:
        xs, ys_s, ys_h = [], [], []
        for n in ns_list:
            cid = f"{'cgl' if g=='cgl' else 'uni'}_L80_n{n}_t1e-9"
            ss = pl["cells"][cid]["summary"].get("reprop_smooth_loose")
            sh = pl["cells"][cid]["summary"].get("reprop_hard_loose")
            if ss and ss["n"] > 0:
                xs.append(n); ys_s.append(ss["median"])
                ys_h.append(sh["median"] if sh and sh["n"] > 0 else np.nan)
        ax2.loglog(xs, ys_s, "o-", color=col, lw=1.6, label=lbl)
        ax2.loglog(xs, ys_h, "s:", color=col, lw=1.1, alpha=0.75, label=lbl.replace("smooth", "hard"))

    for cid, mk, lbl in [("cgl_L80_n540_t1e-12", "*", "CGL tol=1e-12"), ("uni_L80_n540_t1e-12", "D", "Uni tol=1e-12")]:
        ss = pl["cells"][cid]["summary"].get("reprop_smooth_loose")
        if ss and ss["n"] > 0:
            ax2.loglog([540], [ss["median"]], marker=mk, ms=9, ls="none", color="k", label=lbl)

    ax2.set_xlabel("Collocation points $n$")
    ax2.set_ylabel(r"Median open-loop $e_{\mathrm{reprop}}$")
    ax2.set_title("(b) Grid Type, Hard vs. Smooth Re-Prop, & Tol ($L=80$)")
    ax2.set_xticks(ns_list); ax2.set_xticklabels([str(x) for x in ns_list])
    ax2.legend(loc="best", frameon=True, fontsize=7.5)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_tier2_ponn_plateau.pdf")
    plt.close(fig)


def update_report_tier2(pl: dict, cm: dict, verdict: dict) -> None:
    report_path = Path("REPORT.md")
    txt = report_path.read_text(encoding="utf-8")
    marker = "\n---\n\n## 11. Tier 2 Progress Note: PoNN Sweep & Collocation Mesh Study (`seeds = 6000..6029`)"
    if marker in txt:
        txt = txt.split(marker)[0]

    base = pl["control_discretization_baseline"]["grids"]
    lines = [
        marker,
        "",
        "### 11.1 Tier 2.7 PoNN $(L, n)$ Sweep, Error Decomposition, & Condition Numbers",
        "",
        f"- **Numeric sweep finding (PREREG_ADDENDUM.md rule):** Across all overdetermined CGL $(L, n)$ cells ($L \\in \\{{40,60,80,120,160\\}}, n \\in \\{{140,280,540,1080\\}}, n \\ge L$), median open-loop error **{verdict['verdict_phrase']}**.",
        f"- **Control-discretization vs. state-approximation error:** Re-propagating the exact shooting control sampled on the CGL grid yields $e_{{\\text{{reprop}}}} = {base['cgl_n140']['ctrl_disc_err_smooth']:.2e}$ ($n=140$), ${base['cgl_n280']['ctrl_disc_err_smooth']:.2e}$ ($n=280$), ${base['cgl_n540']['ctrl_disc_err_smooth']:.2e}$ ($n=540$), and ${base['cgl_n1080']['ctrl_disc_err_smooth']:.2e}$ ($n=1080$).",
        "- **Condition number definition:** $\\kappa_2(J_s) = \\sigma_{\\max}(J_s)/\\sigma_{\\min}(J_s)$ for the column-scaled Jacobian $J_s = J\\,\\operatorname{diag}(\\|J_{:,j}\\|_2)^{-1}$ factorized inside `lm_svd`, reported alongside unscaled $\\kappa_2(J)$ in `results/tables/tab6_ponn_plateau.tex`.",
        "- **Deliverables:** `results/tables/tab6_ponn_plateau.tex` and `results/figures/fig_tier2_ponn_plateau.pdf`.",
        "",
        "### 11.2 Tier 2.8 Collocation Mesh Study (`seeds = 6000..6029`)",
        "",
        "| Mesh | Loose [95% CI] | Strict [95% CI] | Med. $e_{\\text{reprop}}$ (loose) | Med. $|\\Delta m_f|$ vs. Shoot (kg) | Med. Time (s) |",
        "|:---|:---:|:---:|---:|---:|---:|",
    ]
    n = cm["summary"]["n_seeds"]
    for mk in ("N60", "N120", "N180", "N240", "N120_to_N240_refined"):
        ms = cm["summary"]["meshes"][mk]
        nl, ns = ms["n_loose"], ms["n_strict"]
        cil, cis = ms["ci_loose"], ms["ci_strict"]
        rp = ms["reprop_loose"]["median"] if ms.get("reprop_loose") else None
        fe = ms["fuel_err_loose"]["median"] if ms.get("fuel_err_loose") else None
        rt = ms["runtime_s"]["median"] if ms.get("runtime_s") else None
        lines.append(
            f"| `{mk}` | {nl}/{n} ({100*nl/n:.1f}% [{100*cil[0]:.1f}%, {100*cil[1]:.1f}%]) | "
            f"{ns}/{n} ({100*ns/n:.1f}% [{100*cis[0]:.1f}%, {100*cis[1]:.1f}%]) | "
            f"{_f(rp)} | {_f(fe, '{:.3f}')} | {_f(rt, '{:.2f}')} |"
        )
    lines += ["", "- Full LaTeX table: `results/tables/tab7_collocation_mesh.tex`.", ""]
    report_path.write_text(txt.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    pl = json.loads((RESULTS / "tier2_plateau_6000_6029.json").read_text())
    cm = json.loads((RESULTS / "tier2_collocation_mesh_6000_6029.json").read_text())
    verdict = analyze_plateau_verdict(pl)
    write_tab6_plateau(pl, verdict)
    write_tab7_collocation(cm)
    plot_fig_tier2_plateau(pl)
    update_report_tier2(pl, cm, verdict)
    print(f"[Tier 2 Deliverables] Generated tab6, tab7, fig_tier2_ponn_plateau.pdf. Verdict: {verdict['verdict_phrase']}", flush=True)


if __name__ == "__main__":
    main()
