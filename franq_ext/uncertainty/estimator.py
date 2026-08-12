"""Uncertainty quantification module (paper Section 6).

Annotates each fact with its two UQ signals, sampling the LLM once per fact.
"""
from __future__ import annotations

from franq_ext.llm.client import LLMClient
from franq_ext.schema import Fact
from franq_ext.uncertainty.semantic_entropy import semantic_entropy
from franq_ext.uncertainty.parametric_uq import parametric_confidence


class UncertaintyEstimator:
    def __init__(self, llm: LLMClient, cfg) -> None:
        self.llm = llm
        self.cfg = cfg

    def _question(self, fact: Fact) -> str:
        return f"What is the {fact.attribute} of {fact.entity}?"

    def annotate(self, fact: Fact, question: str = "") -> Fact:
        q = self._question(fact)
        samples = self.llm.sample(q, self.cfg.n_samples)
        fact.semantic_entropy = semantic_entropy(samples)
        fact.parametric_uq = parametric_confidence(
            self.llm, fact.entity, fact.attribute, fact.value, question or q
        )
        return fact

    def annotate_all(self, facts: list[Fact], question: str = "") -> list[Fact]:
        return [self.annotate(f, question) for f in facts]
