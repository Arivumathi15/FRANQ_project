"""The LLM interface every module depends on.

Modules never import a concrete backend directly — they take an `LLMClient`. That is
what lets the whole pipeline run offline (MockLLM) in tests/CI and on a real model
(LocalHFLLM) for the reported experiments, with zero code changes elsewhere.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from franq_ext.schema import Fact


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, temperature: float | None = None) -> str:
        """Single deterministic-ish completion (used for correction proposals, etc.)."""

    @abstractmethod
    def sample(self, prompt: str, n: int) -> list[str]:
        """n stochastic samples of the same prompt — the basis of semantic entropy."""

    @abstractmethod
    def sequence_confidence(self, context: str, statement: str) -> float:
        """Token/sequence-probability confidence in [0,1] that `statement` follows.

        This is FRANQ's 'token-probability UQ' signal (our `parametric_uq`).
        """

    @abstractmethod
    def extract_facts(self, answer: str, question: str = "") -> list[Fact]:
        """Turn a free-text answer into (entity, attribute, value) facts."""

    @abstractmethod
    def answer_attribute(self, entity: str, attribute: str, evidence: str) -> str:
        """Given retrieved evidence, produce the value for one entity-attribute.

        Used by the targeted correction loop (Pillar 3).
        """
