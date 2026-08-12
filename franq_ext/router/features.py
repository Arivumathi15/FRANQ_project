"""Feature assembly + labelling for the learned router (Pillar 2).

The router looks at several signals together (paper Section 4, Pillar 2):
faithfulness, retrieval confidence, semantic entropy, parametric UQ, and — crucially —
two graph-derived signals from Pillar 1 (dependency_strength and entity_suspicion),
which is what lets the router route *dependency-aware*.
"""
from __future__ import annotations

from franq_ext.schema import Fact

FEATURE_NAMES = [
    "faithfulness",
    "retrieval_confidence",
    "semantic_entropy",
    "parametric_uq",
    "dependency_strength",
    "entity_suspicion",
]


def fact_features(fact: Fact, graph=None, use_graph: bool = True) -> list[float]:
    # Pillar 1 ablation: when use_graph is False, the two graph-derived signals are zeroed
    # so the learned router sees only flat (non-dependency-aware) features.
    if use_graph:
        dependency = fact.dependency_strength if fact.dependency_strength is not None else 0.0
        entity_suspicion = graph.entity_suspicion(fact.entity) if graph is not None else 0.0
    else:
        dependency = 0.0
        entity_suspicion = 0.0
    return [
        fact.faithfulness if fact.faithfulness is not None else 0.0,
        fact.retrieval_confidence if fact.retrieval_confidence is not None else 0.0,
        fact.semantic_entropy if fact.semantic_entropy is not None else 0.0,
        fact.parametric_uq if fact.parametric_uq is not None else 0.0,
        dependency,
        float(entity_suspicion),
    ]


def label_fact(fact: Fact, gold_facts: list[tuple[str, str, str]]) -> int | None:
    """1 if the fact's value is wrong, 0 if correct, None if no gold to judge it.

    Uses the same lenient QA matching as the metrics, so labels and reported accuracy agree.
    """
    from franq_ext.metrics.regret import is_correct

    gold = {(e.strip().lower(), a.strip().lower()): val for (e, a, val) in gold_facts}
    key = fact.key()
    if key not in gold:
        return None
    return 0 if is_correct(fact.value, gold[key]) else 1


def build_training_matrix(rows: list[tuple[Fact, list[tuple[str, str, str]]]], graph_of=None):
    """Turn (fact, gold_facts) rows into (X, y), dropping facts with no gold label.

    `graph_of` optionally maps a fact's id() -> its EntityGraph for entity_suspicion.
    """
    X, y = [], []
    for fact, gold in rows:
        label = label_fact(fact, gold)
        if label is None:
            continue
        graph = graph_of(fact) if graph_of is not None else None
        X.append(fact_features(fact, graph))
        y.append(label)
    return X, y
