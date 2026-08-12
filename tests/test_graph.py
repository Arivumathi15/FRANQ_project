from franq_ext.graph import EntityGraph
from franq_ext.schema import Fact


def test_dependency_strength_for_two_siblings():
    facts = [
        Fact("Einstein", "birth year", "1879"),
        Fact("Einstein", "birth city", "Munich"),
    ]
    EntityGraph().build(facts)
    # Two attributes on one entity -> 1 - 1/2 = 0.5 each.
    assert all(abs(f.dependency_strength - 0.5) < 1e-9 for f in facts)


def test_isolated_fact_has_zero_dependency():
    facts = [Fact("Newton", "birth city", "Woolsthorpe")]
    EntityGraph().build(facts)
    assert facts[0].dependency_strength == 0.0


def test_entity_suspicion_uses_worst_attribute():
    facts = [
        Fact("Einstein", "birth year", "1879", faithfulness=0.9),
        Fact("Einstein", "birth city", "Munich", faithfulness=0.2),
    ]
    g = EntityGraph().build(facts)
    # 1 - min(0.9, 0.2) = 0.8
    assert abs(g.entity_suspicion("Einstein") - 0.8) < 1e-9


def test_groups_partition_by_entity():
    facts = [
        Fact("Einstein", "birth year", "1879"),
        Fact("Curie", "birth city", "Warsaw"),
    ]
    groups = EntityGraph().build(facts).groups()
    assert set(groups.keys()) == {"einstein", "curie"}
