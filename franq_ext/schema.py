"""Shared data structures passed between modules.

Keeping these in one place means the extraction module, graph module, router,
corrector and evaluator all speak the same language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fact:
    """A single (entity, attribute, value) unit extracted from an answer."""
    entity: str
    attribute: str
    value: str

    # --- filled in as the fact flows through the pipeline ---
    faithfulness: Optional[float] = None       # AlignScore support (0..1)
    retrieval_confidence: Optional[float] = None  # top retrieval score for this fact (0..1)
    semantic_entropy: Optional[float] = None    # higher = model less sure (0..)
    parametric_uq: Optional[float] = None       # token-probability confidence (0..1)
    dependency_strength: Optional[float] = None  # graph signal from sibling attributes

    error_prob: Optional[float] = None          # router output: P(fact is wrong)
    route: Optional[str] = None                 # which UQ strategy the router chose
    flagged: bool = False                       # marked for correction?
    corrected: bool = False                     # value was replaced?
    original_value: Optional[str] = None        # value before correction

    def key(self) -> tuple[str, str]:
        """Identity of a fact = (entity, attribute); the value is what may change."""
        return (self.entity.strip().lower(), self.attribute.strip().lower())

    def triple(self) -> tuple[str, str, str]:
        return (self.entity, self.attribute, self.value)


@dataclass
class Example:
    """One evaluation instance."""
    qid: str
    question: str
    gold_answer: str
    contexts: list[str] = field(default_factory=list)   # retrieved evidence passages
    # A SEPARATE, richer evidence pool used ONLY by the targeted-correction loop (Pillar 3).
    # The initial answer is generated from `contexts` (a short RAG snippet); correction is
    # allowed a deeper, focused retrieval over `correction_contexts` (e.g. the full article).
    # Keeping them distinct is what gives correction something new to find — reusing the same
    # snippet would just regenerate the same (possibly wrong) value. Empty -> falls back to
    # `contexts`.
    correction_contexts: list[str] = field(default_factory=list)
    # Ground-truth facts, as (entity, attribute, value). Used to score correctness
    # and to compute correction-regret.
    gold_facts: list[tuple[str, str, str]] = field(default_factory=list)
    # The (possibly hallucinated) answer the generator produced; filled at runtime.
    generated_answer: Optional[str] = None


@dataclass
class PipelineResult:
    """Everything one run of the pipeline produced for one example."""
    example: Example
    facts: list[Fact]
    correction_regret: Optional[float] = None
    n_correct_before: int = 0
    n_correct_after: int = 0
    n_flagged: int = 0
    n_corrected: int = 0
