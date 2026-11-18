"""Publication-grade plotting module for Earth-Mars low-thrust trajectory optimization.

Generates:
- Figure 4.1: Multi-panel heliocentric results (Loss, Mass/Thrust, Trajectories, Thrust Direction, Energy Error, Residuals).
- Figure 4.2: 3D Schematic of the Earth-Mars low-thrust transfer.
- Figure 4.3: 2D Schematic top view (X-Y plane) of the transfer.
- Perturbation Comparison: Accelerations of J2 and SRP vs thrust and central gravity.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .indirect_shooting import TrajectorySolution
from .constants import R_MARS_CANON


def plot_figure_4_1(
    solutions: dict[str, TrajectorySolution],
    output_path: str | Path = "results/figures/figure_4_1_heliocentric_results.png"
) -> Path:
    """Generates Figure 4.1 matching the multi-panel description in the paper."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), dpi=300)
    colors = {
        "X-TFC PoNN": "#1f77b4",             # Blue
        "Indirect Shooting": "#d62728",       # Red
        "Pseudospectral Collocation": "#2ca02c"  # Green
    }

    # --------------------------------------------------------------------------
    # Subplot 1: (a) X-TFC PoNN Training Loss Convergence (Log Scale)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    xtfc_sol = solutions.get("X-TFC PoNN")
    if xtfc_sol and xtfc_sol.loss_history:
        epochs = np.linspace(0, xtfc_sol.iterations, len(xtfc_sol.loss_history))
        ax1.semilogy(epochs, xtfc_sol.loss_history, color=colors["X-TFC PoNN"], linewidth=2)
        ax1.scatter([0, xtfc_sol.iterations], [xtfc_sol.loss_history[0], xtfc_sol.loss_history[-1]],
                    color=colors["X-TFC PoNN"], s=40, zorder=5)
        ax1.annotate(f"Initial: {xtfc_sol.loss_history[0]:.1e}",
                     (0, xtfc_sol.loss_history[0]), textcoords="offset points", xytext=(10, -5), fontsize=8)
        ax1.annotate(f"Final: {xtfc_sol.loss_history[-1]:.1e}",
                     (xtfc_sol.iterations, xtfc_sol.loss_history[-1]), textcoords="offset points", xytext=(-65, 10), fontsize=8)
    ax1.set_title("(a) X-TFC PoNN Physics Loss Convergence", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Training Epochs", fontsize=10)
    ax1.set_ylabel("Physics Residual Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --------------------------------------------------------------------------
    # Subplot 2: (b) Mass and Thrust Magnitude History
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2)
    ax2_thrust = ax2.twinx()
    for name, sol in solutions.items():
        if len(sol.t_days) > 0:
            ax2.plot(sol.t_days, sol.m_kg, label=f"{name} Mass", color=colors[name], linewidth=2)
            ax2_thrust.plot(sol.t_days, sol.thrust_n, linestyle="--", color=colors[name], alpha=0.75, linewidth=1.5)
            
    ax2.set_title("(b) Spacecraft Mass & Thrust Profiles", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time of Flight (days)", fontsize=10)
    ax2.set_ylabel("Spacecraft Mass (kg)", fontsize=10)
    ax2_thrust.set_ylabel("Thrust Magnitude (N) [Dashed]", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    # --------------------------------------------------------------------------
    # Subplot 3: (c) Heliocentric Trajectories (X-Y Plane)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    # Reference orbits
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(theta_circ), np.sin(theta_circ), "k--", alpha=0.4, label="Earth Orbit (1.0 AU)")
    ax3.plot(R_MARS_CANON * np.cos(theta_circ), R_MARS_CANON * np.sin(theta_circ), "m--", alpha=0.4, label="Mars Orbit (1.524 AU)")
    ax3.plot(0, 0, "yo", markersize=10, markeredgecolor="orange", label="Sun")
    
    for name, sol in solutions.items():
        if len(sol.r_au) > 0:
            ax3.plot(sol.r_au[:, 0], sol.r_au[:, 1], label=name, color=colors[name], linewidth=2)
            ax3.plot(sol.r_au[0, 0], sol.r_au[0, 1], "bo", markersize=5)
            ax3.plot(sol.r_au[-1, 0], sol.r_au[-1, 1], "rs", markersize=5)

    ax3.set_title("(c) Heliocentric Trajectories (X-Y)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (AU)", fontsize=10)
    ax3.set_ylabel("Y (AU)", fontsize=10)
    ax3.set_aspect("equal")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="lower left", fontsize=8)

    # --------------------------------------------------------------------------
    # Subplot 4: (d) Thrust In-Plane Pitch Angle
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    for name, sol in solutions.items():
        if len(sol.thrust_vec) > 0:
            pitch_deg = np.degrees(np.arctan2(sol.thrust_vec[:, 1], sol.thrust_vec[:, 0]))
            ax4.plot(sol.t_days, pitch_deg, label=name, color=colors[name], linewidth=2)
            
    ax4.set_title("(d) Thrust Pointing Direction (In-Plane Pitch)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time of Flight (days)", fontsize=10)
    ax4.set_ylabel("Thrust Vector Angle (deg)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=8)

    # --------------------------------------------------------------------------
    # Subplot 5: (e) Relative Energy Conservation Error Profile
    # --------------------------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)
    for name, sol in solutions.items():
        if len(sol.energy_error_hist) > 0:
            ax5.semilogy(sol.t_days, np.maximum(sol.energy_error_hist, 1e-12), label=f"{name} (max {sol.energy_error:.1e})",
                         color=colors[name], linewidth=2)
            
    ax5.set_title("(e) Relative Energy Conservation Error $\\varepsilon_E(t)$", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Time of Flight (days)", fontsize=10)
    ax5.set_ylabel("Relative Error $|E(t) - E(0) - W(t)| / |E(0)|$", fontsize=10)
    ax5.grid(True, linestyle="--", alpha=0.6)
    ax5.legend(loc="lower right", fontsize=8)

    # --------------------------------------------------------------------------
    # Subplot 6: (f) Mars Arrival Terminal Residual Comparison
    # --------------------------------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)
    method_labels = list(solutions.keys())
    # Display at least 1e-16 for visualization on log scale if 0.0
    residuals_plot = [max(solutions[m].arrival_residual, 1e-16) for m in method_labels]
    residuals_raw = [solutions[m].arrival_residual for m in method_labels]
    bar_colors = [colors[m] for m in method_labels]
    
    bars = ax6.bar(method_labels, residuals_plot, color=bar_colors, width=0.5, edgecolor="black")
    ax6.set_yscale("log")
    ax6.set_title("(f) Mars Arrival Residual $\\varepsilon_M$ (Dimensionless)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Terminal Constraint Violation", fontsize=10)
    min_res = min(residuals_plot)
    ax6.set_ylim([min_res * 0.1, 1e-3])
    ax6.grid(True, which="both", linestyle="--", alpha=0.5, axis="y")
    
    for idx, bar in enumerate(bars):
        yval = residuals_raw[idx]
        y_text = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2.0, y_text * 1.5, f"{yval:.1e}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4.1 saved to {output_path}")
    return output_path


def plot_figure_4_2_schematic_3d(
    solutions: dict[str, TrajectorySolution],
    output_path: str | Path = "results/figures/figure_4_2_schematic_3d.png"
) -> Path:
    """Generates Figure 4.2: 3D Schematic of the Earth-Mars Low-Thrust Transfer."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Orbits
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), "b--", alpha=0.5, label="Earth Orbit (1.0 AU)")
    ax.plot(R_MARS_CANON * np.cos(theta), R_MARS_CANON * np.sin(theta), np.zeros_like(theta), "r--", alpha=0.5, label="Mars Orbit (1.524 AU)")

    # Sun
    ax.scatter([0], [0], [0], color="gold", s=250, edgecolors="orange", label="Sun")

    # Plot trajectories and thrust arrows
    colors = {"X-TFC PoNN": "#1f77b4", "Indirect Shooting": "#d62728", "Pseudospectral Collocation": "#2ca02c"}
    for name, sol in solutions.items():
        if len(sol.r_au) > 0:
            ax.plot(sol.r_au[:, 0], sol.r_au[:, 1], sol.r_au[:, 2], label=f"{name} Trajectory", color=colors[name], linewidth=2.5)
            # Arrow markers
            step = len(sol.r_au) // 6
            for idx in range(step, len(sol.r_au), step):
                r_pt = sol.r_au[idx]
                t_vec = sol.thrust_vec[idx] / 0.5 * 0.12
                ax.quiver(r_pt[0], r_pt[1], r_pt[2], t_vec[0], t_vec[1], t_vec[2],
                          color=colors[name], arrow_length_ratio=0.35, linewidth=1.5)

    ax.scatter([1.0], [0.0], [0.0], color="blue", s=80, label="Earth Departure (t=0)")
    # Target point
    xtfc = solutions.get("X-TFC PoNN")
    if xtfc and len(xtfc.r_au) > 0:
        ax.scatter([xtfc.r_au[-1, 0]], [xtfc.r_au[-1, 1]], [0.0], color="red", s=100, marker="s", label="Mars Arrival ($t_f \\approx 248$ d)")

    ax.set_title("Figure 4.2: 3D Schematic of Earth-Mars Low-Thrust Heliocentric Transfer", fontsize=12, fontweight="bold", pad=20)
    ax.set_xlabel("X (AU)", fontsize=10)
    ax.set_ylabel("Y (AU)", fontsize=10)
    ax.set_zlabel("Z (AU)", fontsize=10)
    ax.set_zlim([-0.4, 0.4])
    ax.legend(loc="upper right", fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4.2 saved to {output_path}")
    return output_path


def plot_figure_4_3_top_view(
    solutions: dict[str, TrajectorySolution],
    output_path: str | Path = "results/figures/figure_4_3_top_view_2d.png"
) -> Path:
    """Generates Figure 4.3: Schematic Top View (X-Y plane) of Earth-Mars Transfer."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Orbits
    theta = np.linspace(0, 2 * np.pi, 250)
    ax.plot(np.cos(theta), np.sin(theta), "b--", alpha=0.5, linewidth=1.5, label="Earth Orbit (1.0 AU)")
    ax.plot(R_MARS_CANON * np.cos(theta), R_MARS_CANON * np.sin(theta), "r--", alpha=0.5, linewidth=1.5, label="Mars Orbit (1.524 AU)")
    ax.plot(0, 0, "yo", markersize=14, markeredgecolor="orange", label="Sun")

    colors = {"X-TFC PoNN": "#1f77b4", "Indirect Shooting": "#d62728", "Pseudospectral Collocation": "#2ca02c"}
    for name, sol in solutions.items():
        if len(sol.r_au) > 0:
            ax.plot(sol.r_au[:, 0], sol.r_au[:, 1], label=name, color=colors[name], linewidth=2.5)
            # Add thrust vector arrows along trajectory
            step = len(sol.r_au) // 8
            for idx in range(step, len(sol.r_au) - step // 2, step):
                pos = sol.r_au[idx, :2]
                f_vec = sol.thrust_vec[idx, :2] / 0.5 * 0.15
                ax.arrow(pos[0], pos[1], f_vec[0], f_vec[1], head_width=0.03, head_length=0.03,
                         fc=colors[name], ec=colors[name], alpha=0.85)

    ax.plot(1.0, 0.0, "bo", markersize=10, label="Earth Departure (t0 = 0)")
    ref_sol = list(solutions.values())[0]
    if len(ref_sol.r_au) > 0:
        ax.plot(ref_sol.r_au[-1, 0], ref_sol.r_au[-1, 1], "rs", markersize=10, label="Mars Arrival (tf ~ 248 days)")

    ax.set_title("Figure 4.3: Schematic Top View (X-Y Plane) of Earth–Mars Low-Thrust Transfer", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (AU)", fontsize=11)
    ax.set_ylabel("Y (AU)", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4.3 saved to {output_path}")
    return output_path


def plot_perturbation_magnitudes(
    output_path: str | Path = "results/figures/perturbation_accelerations.png"
) -> Path:
    """Visualizes Table 2: Perturbation Accelerations vs Distance."""
    from .metrics import compute_table_2_perturbations
    rows = compute_table_2_perturbations()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    labels = [r.case_name.replace(" (m = 1,538.6 kg)", "").replace(" (m = 1,218 kg)", "").replace(" (m = 943 kg)", "") for r in rows]
    a_pert = [r.a_pert_m_s2 for r in rows]
    a_thrust = [r.a_thrust_m_s2 for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, a_pert, width, label="Perturbing Acceleration ($a_{pert}$)", color="#ff7f0e", edgecolor="black")
    ax.bar(x + width / 2, a_thrust, width, label="Max Thrust Acceleration ($a_T$)", color="#1f77b4", edgecolor="black")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Acceleration (m/s²)", fontsize=11)
    ax.set_title("Perturbation Accelerations ($J_2$ & SRP) vs Maximum Thrust Across Distances (Table 2)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5, axis="y")
    ax.legend(fontsize=10)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Perturbation plot saved to {output_path}")
    return output_path
