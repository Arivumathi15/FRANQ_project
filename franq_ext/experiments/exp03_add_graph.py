"""03 - Add dependency-aware graph features (Pillar 1) to the learned router, no correction.
Improvement over 02 isolates Pillar 1 (entity_suspicion + dependency_strength help)."""
from franq_ext.experiments._run_one import run_rung
from franq_ext.experiments.conditions import learned_router_with_graph


def main() -> None:
    run_rung(3, "A2_graph", "+ Graph features (Pillar 1)", learned_router_with_graph)


if __name__ == "__main__":
    main()
