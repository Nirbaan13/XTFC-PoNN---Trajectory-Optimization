"""Script to propagate and display Earth parking orbit escape phase results (Table E1)."""

from trajectory_optimization.escape_phase import generate_table_e1


def main():
    print("=" * 88)
    print("TABLE E1: NUMERICAL ESCAPE-PHASE PROPAGATION WITH AND WITHOUT J2")
    print("(Prescribed tangential thrust, 0.50 N, initial mass 1,538.6 kg)")
    print("=" * 88)
    print(f"{'Case':36s} | {'Time (days)':12s} | {'Xenon (kg)':11s} | {'Mass at E=0 (kg)':16s} | {'RHS Evals':9s}")
    print("-" * 88)
    for c in generate_table_e1():
        print(
            f"{c.case_name:36s} | {c.escape_time_days:12.4f} | "
            f"{c.xenon_used_kg:11.4f} | {c.final_mass_kg:16.4f} | {c.n_rhs_evals:9d}"
        )


if __name__ == "__main__":
    main()
