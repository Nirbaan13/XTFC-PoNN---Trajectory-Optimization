"""Script to propagate and display Earth parking orbit escape phase results."""

from trajectory_optimization.escape_phase import generate_table_e1, generate_table_e2


def main():
    print("=" * 75)
    print("TABLE E1: NUMERICAL ESCAPE-PHASE PROPAGATION WITH AND WITHOUT J2")
    print("(Prescribed tangential thrust, 0.5 N, initial mass 1,538.6 kg)")
    print("=" * 75)
    print(f"{'Case':32s} | {'Time (days)':12s} | {'Xenon (kg)':10s} | {'Mass at E=0 (kg)':16s}")
    print("-" * 75)
    for c in generate_table_e1():
        print(f"{c.case_name:32s} | {c.escape_time_days:12.1f} | {c.xenon_used_kg:10.1f} | {c.final_mass_kg:16,.1f}")

    print("\n" + "=" * 85)
    print("TABLE E2: ESCAPE PHASE WITH J2 SOLVED BY THE THREE METHODS")
    print("=" * 85)
    print(f"{'Method':30s} | {'Time (days)':12s} | {'Xenon (kg)':10s} | {'Final Mass':12s} | {'End-State Residual':18s}")
    print("-" * 85)
    for r in generate_table_e2():
        print(f"{r['method']:30s} | {r['escape_time_days']:12.2f} | {r['xenon_used_kg']:10.1f} | {r['final_mass_kg']:12,.1f} | {r['end_state_residual']:18s}")


if __name__ == "__main__":
    main()
