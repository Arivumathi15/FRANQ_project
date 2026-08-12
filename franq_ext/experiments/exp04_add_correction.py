"""04 - Full system: add the budget-bounded targeted correction loop (Pillar 3).
Improvement over 03 isolates Pillar 3 (factual accuracy rises; correction-regret reported)."""
from franq_ext.experiments._run_one import run_rung
from franq_ext.experiments.conditions import full


def main() -> None:
    run_rung(4, "A3_full", "+ Correction + regret (Pillar 3)", full)


if __name__ == "__main__":
    main()
