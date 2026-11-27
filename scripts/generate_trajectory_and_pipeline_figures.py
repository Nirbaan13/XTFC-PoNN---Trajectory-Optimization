"""Generate two publication-grade PDF figures from stored JSON data:

1. ``results/figures/fig_scenario_a_trajectory.pdf``:
   Scenario A heliocentric Earth-Mars transfer from the stored indirect
   shooting solution (``results/scenario_a_phase_a3.json``), showing both
   ecliptic top view (x-y) and 3-D heliocentric view (x-y-z), with ON
   (thrust) and OFF (coast) arcs coloured differently and all 4 optimal
   switches explicitly marked.

2. ``results/figures/fig_pipeline_schematic.pdf``:
   Schematic of the optimization pipelines:
   ``Random Init -> PoNN -> {Hybrid Shooting, Split-Domain Refinement}``
   versus ``Multi-Start Indirect Shooting``, annotated with measured
   Scenario A outcomes loaded dynamically from
   ``results/addendum2_baselines_8000_8099.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ON_COLOR = "#C0392B"     # crimson red for thrust (ON) arcs
OFF_COLOR = "#2471A3"    # steel blue for coast (OFF) arcs
SW_COLOR = "#F4D03F"     # gold fill for switch markers
EARTH_COLOR = "#1E8449"  # green for Earth departure
MARS_COLOR = "#8E44AD"   # purple for Mars arrival
SUN_COLOR = "#F39C12"    # amber for Sun


def _interp_pos(t_days: np.ndarray, r: np.ndarray, t_query: float) -> np.ndarray:
    return np.array([np.interp(t_query, t_days, r[:, k]) for k in range(3)], dtype=float)


def _extract_arc_segments(
    t_days: np.ndarray, r: np.ndarray, switch_times: list[float], sequence: str
) -> list[tuple[str, np.ndarray]]:
    """Split (t_days, r) cleanly at the exact switch times into contiguous arcs."""
    arc_types = [tok.strip().upper() for tok in sequence.split("-")]
    bounds = [float(t_days[0])] + [float(ts) for ts in switch_times] + [float(t_days[-1])]
    segments: list[tuple[str, np.ndarray]] = []
    for i, arc_type in enumerate(arc_types):
        t_a, t_b = bounds[i], bounds[i + 1]
        mask = (t_days > t_a) & (t_days < t_b)
        pts = np.vstack([
            _interp_pos(t_days, r, t_a)[None, :],
            r[mask],
            _interp_pos(t_days, r, t_b)[None, :],
        ])
        segments.append((arc_type, pts))
    return segments


def plot_scenario_a_trajectory() -> Path:
    a3 = json.loads((RESULTS / "scenario_a_phase_a3.json").read_text(encoding="utf-8"))
    sh = a3["single_seed"]["shooting"]
    t_days = np.asarray(sh["series"]["t_days"], dtype=float)
    r = np.asarray(sh["series"]["r"], dtype=float)
    alpha = np.asarray(sh["series"]["alpha"], dtype=float)
    delta = np.asarray(sh["series"]["delta"], dtype=float)
    sw_times = [float(x) for x in sh["switch_times_days"]]
    seq = str(sh["thrust_sequence"])
    fuel_kg = float(sh["fuel_kg"])
    tf_days = float(t_days[-1])

    segments = _extract_arc_segments(t_days, r, sw_times, seq)
    sw_pts = np.array([_interp_pos(t_days, r, ts) for ts in sw_times])

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(11.4, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.16)

    # ------------------------------------------------------------------
    # Panel (a): Ecliptic Top View (x-y)
    # ------------------------------------------------------------------
    ax_xy = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    r_earth = float(np.linalg.norm(r[0, :2]))
    r_mars = float(np.linalg.norm(r[-1, :2]))
    ax_xy.plot(r_earth * np.cos(theta), r_earth * np.sin(theta), color="#95A5A6", lw=0.9, ls=":", zorder=1)
    ax_xy.plot(r_mars * np.cos(theta), r_mars * np.sin(theta), color="#BDC3C7", lw=0.9, ls=":", zorder=1)

    for arc_type, pts in segments:
        if arc_type == "ON":
            ax_xy.plot(pts[:, 0], pts[:, 1], color=ON_COLOR, lw=2.5, ls="-", zorder=3)
        else:
            ax_xy.plot(pts[:, 0], pts[:, 1], color=OFF_COLOR, lw=2.1, ls="--", zorder=3)

    # Subtle thrust direction arrows on ON arcs
    step_q = max(1, len(r) // 26)
    for idx in range(step_q // 2, len(r) - 1, step_q):
        if delta[idx] > 0.5:
            ax_xy.quiver(
                r[idx, 0], r[idx, 1],
                0.17 * alpha[idx, 0], 0.17 * alpha[idx, 1],
                angles="xy", scale_units="xy", scale=1.0,
                color=ON_COLOR, alpha=0.55, width=0.004, headwidth=3.8, headlength=4.5, zorder=2,
            )

    # Sun, Earth departure, Mars arrival
    ax_xy.scatter([0.0], [0.0], color=SUN_COLOR, edgecolor="k", lw=0.7, s=95, zorder=5)
    ax_xy.annotate("Sun", (0.0, 0.0), xytext=(-10, 10), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax_xy.scatter([r[0, 0]], [r[0, 1]], color=EARTH_COLOR, edgecolor="k", lw=0.7, s=55, zorder=5)
    ax_xy.annotate(
        r"Earth ($t_0=0$ d)", (r[0, 0], r[0, 1]),
        xytext=(-48, 12), textcoords="offset points", fontsize=8.2, color=EARTH_COLOR, fontweight="bold",
    )
    ax_xy.scatter([r[-1, 0]], [r[-1, 1]], color=MARS_COLOR, edgecolor="k", lw=0.7, s=55, zorder=5)
    ax_xy.annotate(
        f"Mars ($t_f={tf_days:.1f}$ d)", (r[-1, 0], r[-1, 1]),
        xytext=(-24, -22), textcoords="offset points", fontsize=8.2, color=MARS_COLOR, fontweight="bold",
    )

    # 4 Switch markers + non-overlapping callout offsets
    sw_offsets_xy = [
        (-86, -34),   # s1 (~46.6 d, lower-left)
        (8, -36),     # s2 (~68.0 d, lower-center)
        (14, -24),    # s3 (~142.7 d, right outside orbit)
        (16, -18),    # s4 (~290.3 d, top-center)
    ]
    sw_labels = ["ON→OFF", "OFF→ON", "ON→OFF", "OFF→ON"]
    for i, (pt, ts, off, tr) in enumerate(zip(sw_pts, sw_times, sw_offsets_xy, sw_labels), start=1):
        ax_xy.scatter([pt[0]], [pt[1]], marker="D", s=46, facecolor=SW_COLOR, edgecolor="k", lw=0.9, zorder=6)
        ax_xy.annotate(
            f"$s_{i}$ ({ts:.1f} d)\n{tr}",
            (pt[0], pt[1]),
            xytext=off,
            textcoords="offset points",
            fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#7F8C8D", alpha=0.94, lw=0.6),
            arrowprops=dict(arrowstyle="-", color="#566573", lw=0.8, shrinkA=0, shrinkB=3),
            zorder=7,
        )

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlim(-1.75, 1.85)
    ax_xy.set_ylim(-1.75, 1.75)
    ax_xy.set_xlabel("$x$ [AU]")
    ax_xy.set_ylabel("$y$ [AU]")
    ax_xy.set_title(f"(a) Ecliptic Top View ($x$–$y$)\nShooting: {fuel_kg:.2f} kg, {seq}")
    ax_xy.grid(True, alpha=0.28)

    # ------------------------------------------------------------------
    # Panel (b): 3-D Heliocentric View (x-y-z)
    # ------------------------------------------------------------------
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_3d.plot(r_earth * np.cos(theta), r_earth * np.sin(theta), np.zeros_like(theta), color="#95A5A6", lw=0.8, ls=":")
    ax_3d.plot(r_mars * np.cos(theta), r_mars * np.sin(theta), np.zeros_like(theta), color="#BDC3C7", lw=0.8, ls=":")

    # Drop faint projection onto z_min plane to highlight 3-D inclination
    z_floor = -0.10
    ax_3d.plot(r[:, 0], r[:, 1], np.full_like(r[:, 2], z_floor), color="#D5D8DC", lw=1.0, ls="-", alpha=0.8)

    for arc_type, pts in segments:
        if arc_type == "ON":
            ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=ON_COLOR, lw=2.5, ls="-")
        else:
            ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=OFF_COLOR, lw=2.1, ls="--")

    ax_3d.scatter([0.0], [0.0], [0.0], color=SUN_COLOR, edgecolor="k", lw=0.6, s=85, depthshade=False)
    ax_3d.scatter([r[0, 0]], [r[0, 1]], [r[0, 2]], color=EARTH_COLOR, edgecolor="k", lw=0.6, s=50, depthshade=False)
    ax_3d.scatter([r[-1, 0]], [r[-1, 1]], [r[-1, 2]], color=MARS_COLOR, edgecolor="k", lw=0.6, s=50, depthshade=False)

    sw_offsets_3d = [
        (-0.34, -0.05, 0.012),   # s1
        (0.08, -0.22, -0.022),   # s2
        (0.14, 0.08, 0.012),     # s3
        (0.12, -0.10, -0.018),   # s4
    ]
    for i, (pt, off) in enumerate(zip(sw_pts, sw_offsets_3d), start=1):
        ax_3d.scatter([pt[0]], [pt[1]], [pt[2]], marker="D", s=44, facecolor=SW_COLOR, edgecolor="k", lw=0.8, depthshade=False)
        ax_3d.plot([pt[0], pt[0]], [pt[1], pt[1]], [z_floor, pt[2]], color="#7F8C8D", lw=0.6, ls=":")
        ax_3d.text(pt[0] + off[0], pt[1] + off[1], pt[2] + off[2], f"$s_{i}$", fontsize=8.5, fontweight="bold")

    ax_3d.set_xlim(-1.65, 1.65)
    ax_3d.set_ylim(-1.65, 1.65)
    ax_3d.set_zlim(z_floor, 0.10)
    ax_3d.set_xlabel("$x$ [AU]", labelpad=6)
    ax_3d.set_ylabel("$y$ [AU]", labelpad=6)
    ax_3d.set_zlabel("$z$ [AU]", labelpad=6)
    ax_3d.view_init(elev=25, azim=-58)
    ax_3d.set_title("(b) 3-D Heliocentric View\n(with Ecliptic Shadow on $z = -0.10$ AU)")

    # Shared figure legend along the bottom
    handles = [
        Line2D([0], [0], color=ON_COLOR, lw=2.5, ls="-", label="Thrust arc (ON, $\\delta=1$)"),
        Line2D([0], [0], color=OFF_COLOR, lw=2.1, ls="--", label="Coast arc (OFF, $\\delta=0$)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=SW_COLOR, markeredgecolor="k", markersize=6.5, label="Optimal switch ($s_1$–$s_4$)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=EARTH_COLOR, markeredgecolor="k", markersize=6.5, label="Earth ($t_0$)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MARS_COLOR, markeredgecolor="k", markersize=6.5, label="Mars ($t_f$)"),
        Line2D([0], [0], color="#95A5A6", lw=1.0, ls=":", label="Ref. circular orbits (1.0 / 1.52 AU)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.19)

    out_path = FIGURES / "fig_scenario_a_trajectory.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#1B2631",
    lw: float = 1.4,
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.022",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x + w / 2.0, y + h * 0.65, title,
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_color, zorder=4,
        )
        ax.text(
            x + w / 2.0, y + h * 0.28, subtitle,
            ha="center", va="center", fontsize=7.6, color="#2C3E50", zorder=4,
        )
    else:
        ax.text(
            x + w / 2.0, y + h / 2.0, title,
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_color, zorder=4,
        )


def _draw_arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str = "#34495E",
    rad: float = 0.0,
    ls: str = "-",
    lw: float = 1.6,
) -> None:
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>,head_length=6.5,head_width=3.8",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linestyle=ls, linewidth=lw, zorder=5,
    )
    ax.add_patch(arrow)


def plot_pipeline_schematic() -> Path:
    a2 = json.loads((RESULTS / "addendum2_baselines_8000_8099.json").read_text(encoding="utf-8"))
    arms = a2["summary"]["arms"]
    n_seeds = int(a2["summary"]["n_seeds"])

    hyb_n = arms["hybrid_rho1"]["n_strict"]
    hyb_t = arms["hybrid_rho1"]["time_strict_s"]["median"]
    spl_n = arms["split_domain"]["n_strict"]
    spl_t = arms["split_domain"]["time_strict_s"]["median"]
    ms5_n = arms["multistart_shoot_5"]["n_strict"]
    ms5_t = arms["multistart_shoot_5"]["time_strict_s"]["median"]
    sgl_n = arms["single_shoot"]["n_strict"]
    sgl_t = arms["single_shoot"]["time_strict_s"]["median"]

    plt.rcParams.update({
        "font.size": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Background group banners
    top_group = FancyBboxPatch(
        (0.015, 0.40), 0.97, 0.535,
        boxstyle="round,pad=0.0,rounding_size=0.022",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.1, zorder=0,
    )
    ax.add_patch(top_group)
    ax.text(
        0.03, 0.905,
        "PoNN-Initialized Pipelines (Coarse Spectral Physics-Informed Warm-Start)",
        ha="left", va="center", fontsize=9.6, fontweight="bold", color="#1A5276", zorder=1,
    )

    bot_group = FancyBboxPatch(
        (0.015, 0.065), 0.97, 0.275,
        boxstyle="round,pad=0.0,rounding_size=0.022",
        facecolor="#FDFEFE", edgecolor="#D5D8DC", linewidth=1.1, zorder=0,
    )
    ax.add_patch(bot_group)
    ax.text(
        0.03, 0.312,
        "Pure Indirect Shooting Baseline (No PoNN Stage)",
        ha="left", va="center", fontsize=9.6, fontweight="bold", color="#5D6D7E", zorder=1,
    )

    # Divider badge: "VERSUS"
    ax.axhline(0.37, xmin=0.03, xmax=0.97, color="#AEB6BF", lw=1.0, ls="--", zorder=1)
    ax.text(
        0.50, 0.37, "  VERSUS  ", ha="center", va="center", fontsize=8.4,
        fontweight="bold", color="#566573",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#AEB6BF", lw=0.9),
        zorder=5,
    )

    # --- Top track: Random Init -> PoNN -> {Hybrid Shooting, Split-Domain Refinement} ---
    _draw_box(
        ax, 0.035, 0.565, 0.145, 0.17,
        "Random Init",
        r"$\beta \sim$ scaled $U(0,1)$" + "\n(1 seed)",
        facecolor="#EBF5FB", edgecolor="#2E86C1",
    )
    _draw_box(
        ax, 0.225, 0.555, 0.205, 0.19,
        "PoNN Stage (XTFC)",
        "Chebyshev $L=70$, $n=140$ CGL\nExact BCs via CE; $\\rho$ homotopy",
        facecolor="#D6EAF8", edgecolor="#1B4F72",
    )
    _draw_arrow(ax, (0.180, 0.650), (0.225, 0.650), color="#1B4F72")

    # Upper branch: Hybrid Shooting
    _draw_box(
        ax, 0.505, 0.695, 0.240, 0.17,
        "Hybrid Indirect Shooting",
        r"Extract $\lambda(t_0)$ at $\rho_{\mathrm{ext}}=1$" + "\nContinue $\\rho$ schedule to $10^{-5}$",
        facecolor="#E8F8F5", edgecolor="#117864",
    )
    _draw_box(
        ax, 0.790, 0.695, 0.175, 0.17,
        f"Strict: {hyb_n}/{n_seeds}",
        f"Med. time: {hyb_t:.1f} s\n(incl. PoNN stage)",
        facecolor="#D1F2EB", edgecolor="#0E6251", title_color="#0E6251",
    )

    # Lower branch: Split-Domain Refinement
    _draw_box(
        ax, 0.505, 0.440, 0.240, 0.17,
        "Split-Domain Refinement",
        "Detect switches $S(t)=0$\n5-arc PoNN, exact $\\delta\\in\\{0,1\\}$",
        facecolor="#FEF9E7", edgecolor="#B7950B",
    )
    _draw_box(
        ax, 0.790, 0.440, 0.175, 0.17,
        f"Strict: {spl_n}/{n_seeds}",
        f"Med. time: {spl_t:.1f} s\n(incl. PoNN stage)",
        facecolor="#FCF3CF", edgecolor="#9A7D0A", title_color="#7D6608",
    )

    # Fork arrows from PoNN to both branches
    _draw_arrow(ax, (0.430, 0.680), (0.505, 0.780), color="#117864", rad=-0.10)
    ax.text(0.464, 0.768, r"$\lambda(t_0)$", ha="center", va="bottom", fontsize=8.2, color="#117864", fontweight="bold")
    _draw_arrow(ax, (0.430, 0.620), (0.505, 0.525), color="#9A7D0A", rad=0.10)
    ax.text(0.464, 0.525, r"$\hat{t}_{s,1..4}$", ha="center", va="top", fontsize=8.2, color="#9A7D0A", fontweight="bold")

    _draw_arrow(ax, (0.745, 0.780), (0.790, 0.780), color="#117864")
    _draw_arrow(ax, (0.745, 0.525), (0.790, 0.525), color="#9A7D0A")

    # --- Bottom track: Multi-Start Indirect Shooting ---
    _draw_box(
        ax, 0.035, 0.135, 0.235, 0.145,
        "Random Costate Init",
        r"$\lambda_0 \sim U(-1,1)^7$ ($K \leq 5$ restarts)" + f"\nSingle-start: {sgl_n}/{n_seeds} ({sgl_t:.1f} s)",
        facecolor="#F4F6F7", edgecolor="#7F8C8D",
    )
    _draw_box(
        ax, 0.340, 0.135, 0.375, 0.145,
        "Multi-Start Indirect Shooting",
        "20-step $\\rho$ continuation per restart ($\\rho: 1 \\to 10^{-5}$)\nStop at first strict success ($e_{\\mathrm{reprop}} < 10^{-2}$)",
        facecolor="#FDEDEC", edgecolor="#922B21",
    )
    _draw_box(
        ax, 0.790, 0.135, 0.175, 0.145,
        f"Strict: {ms5_n}/{n_seeds}",
        f"Med. time: {ms5_t:.1f} s\n(to 1st strict success)",
        facecolor="#FADBD8", edgecolor="#78281F", title_color="#78281F",
    )

    _draw_arrow(ax, (0.270, 0.207), (0.340, 0.207), color="#78281F")
    _draw_arrow(ax, (0.715, 0.207), (0.790, 0.207), color="#78281F")
    # Clean retry loop routed underneath the bottom boxes
    _draw_arrow(ax, (0.525, 0.135), (0.152, 0.135), color="#922B21", rad=-0.16, ls="--", lw=1.3)
    ax.text(
        0.338, 0.088, "Retry if failed ($k < 5$)",
        ha="center", va="center", fontsize=7.8, color="#922B21", style="italic",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#FDFEFE", edgecolor="none"),
        zorder=6,
    )

    # Overall title and footnote
    ax.text(
        0.50, 0.968,
        "Trajectory Optimization Pipelines: PoNN-Initialized Refinement vs. Multi-Start Indirect Shooting",
        ha="center", va="center", fontsize=11.0, fontweight="bold", color="#17202A",
    )
    ax.text(
        0.50, 0.025,
        f"Measured outcomes from stored Scenario A benchmark (fresh seeds 8000–8099, N = {n_seeds}; strict threshold: DOP853 $e_{{\\mathrm{{reprop}}}} < 10^{{-2}}$).",
        ha="center", va="center", fontsize=8.0, color="#566573",
    )

    out_path = FIGURES / "fig_pipeline_schematic.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p1 = plot_scenario_a_trajectory()
    p2 = plot_pipeline_schematic()
    print(f"[Trajectory & Pipeline Figures] Generated {p1} and {p2}", flush=True)


if __name__ == "__main__":
    main()
