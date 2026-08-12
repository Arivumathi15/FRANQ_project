"""02 - Add the learned calibrated router (Pillar 2), still no graph features, no correction.
Improvement over 01 isolates Pillar 2 (better AUROC/PRR/ECE)."""
from franq_ext.experiments._run_one import run_rung
from franq_ext.experiments.conditions import learned_router_no_graph


def main() -> None:
    run_rung(2, "A1_router", "+ Learned router (Pillar 2)", learned_router_no_graph)


if __name__ == "__main__":
    main()
