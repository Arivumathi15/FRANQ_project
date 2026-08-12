"""01 - Baseline: reproduce FRANQ (fixed IF-ELSE router, detection only, no correction)."""
from franq_ext.experiments._run_one import run_rung
from franq_ext.experiments.conditions import franq_baseline


def main() -> None:
    run_rung(1, "B0_franq", "FRANQ (fixed rule, no correction)", franq_baseline)


if __name__ == "__main__":
    main()
