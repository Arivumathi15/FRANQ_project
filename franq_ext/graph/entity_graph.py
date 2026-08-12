"""Pillar 1 - Dependency-Aware Entity-Attribute Verification Graph.

Instead of a flat list of independent facts, we link every fact to its entity. Two
signals come out of this structure and feed the router:

  * dependency_strength(fact): how entangled a fact is with its siblings. A fact about an
    entity that has several attributes is riskier to trust in isolation (if the entity is
    wrong, all its attributes are shaky) than a lone fact. Defined as 1 - 1/deg(entity),
    so an isolated fact scores 0 and highly-connected entities approach 1.

  * entity_suspicion(entity): if ANY attribute of an entity looks unfaithful, the whole
    entity is suspect, which raises the error probability of its other attributes too.
    This is the 'doubt the grandfather -> re-check the grandchildren' effect that flat
    fact-checking cannot see.
"""
from __future__ import annotations

import networkx as nx

from franq_ext.schema import Fact


class EntityGraph:
    def __init__(self) -> None:
        self.g = nx.Graph()
        self.entity_facts: dict[str, list[Fact]] = {}

    def build(self, facts: list[Fact]) -> "EntityGraph":
        self.g.clear()
        self.entity_facts.clear()
        for f in facts:
            ent = f.entity.strip()
            ent_node = ("entity", ent.lower())
            attr_node = ("attr", f.key())
            self.g.add_node(ent_node, kind="entity", label=ent)
            self.g.add_node(attr_node, kind="attribute", label=f"{f.attribute}={f.value}")
            self.g.add_edge(ent_node, attr_node)
            self.entity_facts.setdefault(ent.lower(), []).append(f)
        self._annotate_dependency(facts)
        return self

    def _annotate_dependency(self, facts: list[Fact]) -> None:
        for f in facts:
            deg = len(self.entity_facts.get(f.entity.strip().lower(), []))
            f.dependency_strength = 0.0 if deg <= 1 else 1.0 - 1.0 / deg

    def groups(self) -> dict[str, list[Fact]]:
        """entity (lowercased) -> its facts. The unit the router reasons over."""
        return self.entity_facts

    def entity_suspicion(self, entity: str) -> float:
        """Max unfaithfulness among an entity's attributes, in [0,1].

        1 - min(faithfulness over siblings). Facts with no faithfulness yet are ignored.
        """
        facts = self.entity_facts.get(entity.strip().lower(), [])
        scored = [f.faithfulness for f in facts if f.faithfulness is not None]
        if not scored:
            return 0.0
        return 1.0 - min(scored)

    def num_edges(self) -> int:
        return self.g.number_of_edges()
