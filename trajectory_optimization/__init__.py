"""Earth-Mars Low-Thrust Trajectory Optimization package.

Benchmarking X-TFC PoNN, Indirect Shooting, and Pseudospectral Collocation
under J2 and Solar Radiation Pressure perturbations.
"""

from .constants import *
from .dynamics import *
from .indirect_shooting import IndirectShootingSolver, TrajectorySolution
from .xtfc_ponn import XTFC_PoNN_Solver
from .pseudospectral import PseudospectralCollocationSolver
from .escape_phase import propagate_escape_phase, generate_table_e1, generate_table_e2
from .metrics import compute_table_2_perturbations, format_table_2_markdown
from .benchmarks import run_representative_benchmarks, run_robustness_protocol
from .plotting import (
    plot_figure_4_1,
    plot_figure_4_2_schematic_3d,
    plot_figure_4_3_top_view,
    plot_perturbation_magnitudes
)

__version__ = "1.0.0"
