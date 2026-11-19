"""Publication-grade plotting module for Earth-Mars low-thrust trajectory optimization.

Generates:
- Figure 4.1: Multi-panel heliocentric results (Loss, Mass/Thrust, Trajectories,
  Thrust Direction, Energy Error, Terminal & Re-propagated Residuals).
- Figure 4.2: 3D Schematic of the Earth-Mars low-thrust transfer (explicitly labeled as schematic).
- Figure 4.3: 2D Schematic top view (X-Y plane) of the transfer (explicitly labeled as schematic).
- Perturbation Comparison: Accelerations of J2 and SRP vs thrust and central gravity (Table 2).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .constants import R_MARS_CANON
from .metrics import compute_table_2_perturbations


COLORS = {
    "X-TFC PoNN": "#1f77b4",
    "Indirect shooting": "#d62728",
    "Pseudospectral collocation": "#2ca02c",
}


def plot_figure_4_1(
    rep_bundle: dict,
    output_path: str | Path = "results/figures/figure_4_1_heliocentric_results.png",
    scenario_title: str = "",
) -> Path:
    """Generates Figure 4.1 from a run_three_methods() bundle."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evals = rep_bundle["eval"]
    raws = rep_bundle["raw"]
    prob = rep_bundle["problem"]

    fig = plt.figure(figsize=(16, 12), dpi=300)
    if scenario_title:
        fig.suptitle(scenario_title, fontsize=13, fontweight="bold", y=0.995)

    # (a) X-TFC PoNN Training Loss Convergence
    ax1 = fig.add_subplot(2, 3, 1)
    xt_raw = raws["X-TFC PoNN"]
    if xt_raw.loss_history:
        iters = np.arange(1, len(xt_raw.loss_history) + 1)
        ax1.semilogy(iters, xt_raw.loss_history, color=COLORS["X-TFC PoNN"], linewidth=2, marker="o", markersize=3)
        ax1.annotate(
            f"Iter 1: {xt_raw.loss_history[0]:.2e}",
            (1, xt_raw.loss_history[0]),
            textcoords="offset points", xytext=(8, -6), fontsize=8,
        )
        ax1.annotate(
            f"Iter {len(xt_raw.loss_history)}: {xt_raw.loss_history[-1]:.2e}",
            (len(xt_raw.loss_history), xt_raw.loss_history[-1]),
            textcoords="offset points", xytext=(-75, 8), fontsize=8,
        )
    ax1.set_title(f"(a) X-TFC PoNN Recorded Loss ({len(xt_raw.loss_history)} LM Iters)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Recorded LM Iteration", fontsize=10)
    ax1.set_ylabel("Mean Squared 14-ODE Residual", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # (b) Spacecraft Mass & Thrust Magnitude History
    ax2 = fig.add_subplot(2, 3, 2)
    ax2_t = ax2.twinx()
    for name, ev in evals.items():
        c = COLORS[name]
        ax2.plot(ev["t_days"], ev["m_kg"], label=f"{name} Mass", color=c, linewidth=2)
        ax2_t.plot(ev["t_days"], ev["T_N"], linestyle="--", color=c, alpha=0.75, linewidth=1.4)
    ax2.set_title("(b) Spacecraft Mass & Thrust Magnitude", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time of Flight (days)", fontsize=10)
    ax2.set_ylabel("Spacecraft Mass (kg)", fontsize=10)
    ax2_t.set_ylabel("Thrust Magnitude (N) [Dashed]", fontsize=10)
    ax2_t.set_ylim([-0.02, prob.t_max_newton * 1.15])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    # (c) Heliocentric Trajectories (X-Y)
    ax3 = fig.add_subplot(2, 3, 3)
    th_c = np.linspace(0, 2 * np.pi, 240)
    ax3.plot(np.cos(th_c), np.sin(th_c), "k--", alpha=0.4, label="Earth Orbit (1.0 AU)")
    ax3.plot(R_MARS_CANON * np.cos(th_c), R_MARS_CANON * np.sin(th_c), "m--", alpha=0.4, label="Mars Orbit (1.524 AU)")
    ax3.plot(0, 0, "yo", markersize=10, markeredgecolor="orange", label="Sun")
    ax3.plot(prob.r0[0], prob.r0[1], "bo", markersize=6, label="Earth Departure")
    ax3.plot(prob.rf[0], prob.rf[1], "rs", markersize=7, label=f"Mars Target ({np.degrees(prob.theta_f):.1f}°)")
    for name, ev in evals.items():
        r = ev["r_au"]
        ax3.plot(r[:, 0], r[:, 1], label=name, color=COLORS[name], linewidth=2)
        ax3.plot(r[-1, 0], r[-1, 1], "o", color=COLORS[name], markersize=4)
    ax3.set_title(f"(c) Heliocentric Trajectories (tf = {prob.tf_days:.1f} d)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (AU)", fontsize=10)
    ax3.set_ylabel("Y (AU)", fontsize=10)
    ax3.set_aspect("equal")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="lower left", fontsize=7.5)

    # (d) Thrust Pointing Direction
    ax4 = fig.add_subplot(2, 3, 4)
    for name, ev in evals.items():
        u = ev["U"]
        pitch = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        ax4.plot(ev["t_days"], pitch, label=name, color=COLORS[name], linewidth=1.8)
    ax4.set_title("(d) Thrust Direction Angle (In-Plane)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time of Flight (days)", fontsize=10)
    ax4.set_ylabel("atan2(u_y, u_x) (deg)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="best", fontsize=8)

    # (e) Relative Energy Conservation Error Profile
    ax5 = fig.add_subplot(2, 3, 5)
    for name, ev in evals.items():
        prof = np.maximum(ev["eps_E_profile"], 1e-16)
        ax5.semilogy(ev["t_days"], prof, label=f"{name} (max {ev['energy_error']:.1e})", color=COLORS[name], linewidth=1.8)
    ax5.set_title("(e) Relative Energy Balance Error $\\varepsilon_E(t)$", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Time of Flight (days)", fontsize=10)
    ax5.set_ylabel("$|E(t) - E(t_0) - W(t)| / |E(t_0)|$", fontsize=10)
    ax5.grid(True, linestyle="--", alpha=0.6)
    ax5.legend(loc="best", fontsize=8)

    # (f) Terminal & DOP853 Re-propagated Mars Arrival Residual
    ax6 = fig.add_subplot(2, 3, 6)
    names = list(evals.keys())
    x = np.arange(len(names))
    w = 0.35
    direct_res = [max(evals[m]["arrival_residual"], 1e-16) for m in names]
    reprop_res = [max(evals[m]["reprop_arrival_residual"], 1e-16) for m in names]
    b1 = ax6.bar(x - w / 2, direct_res, w, label="Solver Terminal $\\varepsilon_M$", color="#4c78a8", edgecolor="black")
    b2 = ax6.bar(x + w / 2, reprop_res, w, label="DOP853 Re-propagated $\\varepsilon_M$", color="#f58518", edgecolor="black")
    ax6.set_yscale("log")
    ax6.set_xticks(x)
    ax6.set_xticklabels(names, rotation=12, fontsize=8.5)
    ax6.set_title("(f) Mars Arrival Residual $\\varepsilon_M$", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Dimensionless Residual Norm", fontsize=10)
    ymin = min(min(direct_res), min(reprop_res)) * 0.1
    ymax = max(max(direct_res), max(reprop_res)) * 30.0
    ax6.set_ylim([ymin, ymax])
    ax6.grid(True, which="both", linestyle="--", alpha=0.5, axis="y")
    ax6.legend(loc="best", fontsize=8)
    for bar, raw_val in zip(b1, [evals[m]["arrival_residual"] for m in names]):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3, f"{raw_val:.1e}", ha="center", va="bottom", fontsize=7.5)
    for bar, raw_val in zip(b2, [evals[m]["reprop_arrival_residual"] for m in names]):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3, f"{raw_val:.1e}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure_4_2_schematic_3d(
    rep_bundle: dict,
    output_path: str | Path = "results/figures/figure_4_2_schematic_3d.png",
) -> Path:
    """Generates Figure 4.2: 3D Schematic of the Earth-Mars transfer from real solution data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evals = rep_bundle["eval"]
    prob = rep_bundle["problem"]

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    th = np.linspace(0, 2 * np.pi, 220)
    ax.plot(np.cos(th), np.sin(th), np.zeros_like(th), "b--", alpha=0.45, label="Earth Orbit (1.0 AU)")
    ax.plot(R_MARS_CANON * np.cos(th), R_MARS_CANON * np.sin(th), np.zeros_like(th), "r--", alpha=0.45, label="Mars Orbit (1.524 AU)")
    ax.scatter([0], [0], [0], color="gold", s=240, edgecolors="orange", label="Sun")

    for name, ev in evals.items():
        r = ev["r_au"]
        u = ev["U"]
        T = ev["T_N"]
        c = COLORS[name]
        ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f"{name} (tf={prob.tf_days:.1f} d)", color=c, linewidth=2.3)
        step = max(1, len(r) // 7)
        for idx in range(step, len(r), step):
            tv = (T[idx] / max(prob.t_max_newton, 1e-9)) * u[idx] * 0.14
            ax.quiver(r[idx, 0], r[idx, 1], r[idx, 2], tv[0], tv[1], tv[2], color=c, arrow_length_ratio=0.35, linewidth=1.3)

    ax.scatter([prob.r0[0]], [prob.r0[1]], [prob.r0[2]], color="blue", s=80, label="Earth Departure (t = 0)")
    ax.scatter([prob.rf[0]], [prob.rf[1]], [prob.rf[2]], color="red", s=100, marker="s",
               label=f"Mars Target (tf = {prob.tf_days:.1f} d, θ_f = {np.degrees(prob.theta_f):.1f}°)")

    ax.set_title(
        f"Figure 4.2 (SCHEMATIC): 3D Heliocentric Transfer Geometry from Computed Solutions (tf = {prob.tf_days:.1f} d)",
        fontsize=11.5, fontweight="bold", pad=18,
    )
    ax.set_xlabel("X (AU)", fontsize=10)
    ax.set_ylabel("Y (AU)", fontsize=10)
    ax.set_zlabel("Z (AU)", fontsize=10)
    ax.set_zlim([-0.4, 0.4])
    ax.legend(loc="upper right", fontsize=8.5)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure_4_3_top_view(
    rep_bundle: dict,
    output_path: str | Path = "results/figures/figure_4_3_top_view_2d.png",
) -> Path:
    """Generates Figure 4.3: 2D Schematic Top View (X-Y plane) from real solution data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evals = rep_bundle["eval"]
    prob = rep_bundle["problem"]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    th = np.linspace(0, 2 * np.pi, 250)
    ax.plot(np.cos(th), np.sin(th), "b--", alpha=0.5, linewidth=1.5, label="Earth Orbit (1.0 AU)")
    ax.plot(R_MARS_CANON * np.cos(th), R_MARS_CANON * np.sin(th), "r--", alpha=0.5, linewidth=1.5, label="Mars Orbit (1.524 AU)")
    ax.plot(0, 0, "yo", markersize=14, markeredgecolor="orange", label="Sun")

    for name, ev in evals.items():
        r = ev["r_au"]
        u = ev["U"]
        T = ev["T_N"]
        c = COLORS[name]
        ax.plot(r[:, 0], r[:, 1], label=name, color=c, linewidth=2.3)
        step = max(1, len(r) // 8)
        for idx in range(step, len(r) - step // 2, step):
            pos = r[idx, :2]
            f_vec = (T[idx] / max(prob.t_max_newton, 1e-9)) * u[idx, :2] * 0.14
            ax.arrow(pos[0], pos[1], f_vec[0], f_vec[1], head_width=0.03, head_length=0.03, fc=c, ec=c, alpha=0.85)

    ax.plot(prob.r0[0], prob.r0[1], "bo", markersize=10, label="Earth Departure (t = 0)")
    ax.plot(prob.rf[0], prob.rf[1], "rs", markersize=10,
            label=f"Mars Target (tf = {prob.tf_days:.1f} d, θ_f = {np.degrees(prob.theta_f):.1f}°)")

    ax.set_title(
        f"Figure 4.3 (SCHEMATIC): Top View (X-Y Plane) of Computed Transfer Trajectories (tf = {prob.tf_days:.1f} d)",
        fontsize=11.5, fontweight="bold",
    )
    ax.set_xlabel("X (AU)", fontsize=11)
    ax.set_ylabel("Y (AU)", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_perturbation_magnitudes(
    output_path: str | Path = "results/figures/perturbation_accelerations.png",
) -> Path:
    """Visualizes Table 2: Perturbation Accelerations vs Distance."""
    rows = compute_table_2_perturbations()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    labels = [r.case_description.split(" (m =")[0] for r in rows]
    a_pert = [r.a_pert_m_s2 for r in rows]
    a_thrust = [r.a_thrust_max_m_s2 for r in rows]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, a_pert, width, label="Perturbing Acceleration ($a_{pert}$)", color="#ff7f0e", edgecolor="black")
    ax.bar(x + width / 2, a_thrust, width, label="Max Thrust Acceleration ($a_T$)", color="#1f77b4", edgecolor="black")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Acceleration (m/s²)", fontsize=11)
    ax.set_title("Perturbation Accelerations ($J_2$ & SRP) vs Maximum Thrust Across Distances (Table 2)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5, axis="y")
    ax.legend(fontsize=10)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
