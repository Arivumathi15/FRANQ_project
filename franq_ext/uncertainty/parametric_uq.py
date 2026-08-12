"""Parametric-knowledge UQ (paper Section 5) — FRANQ's 'token-probability' signal.

Estimates how confident the model is *from its own parameters*, separate from retrieval.
We ask the model, closed-book (no evidence in context), how probable the claim statement
is. High probability => the model itself is confident; low => it doesn't really know.
"""
from __future__ import annotations

from franq_ext.llm.client import LLMClient


def parametric_confidence(llm: LLMClient, entity: str, attribute: str, value: str,
                          question: str = "") -> float:
    """Token-probability confidence in [0,1] that the claim holds, closed-book."""
    statement = f"The {attribute} of {entity} is {value}."
    # Closed-book: condition only on the question (or nothing), never on evidence, so this
    # measures parametric knowledge rather than retrieval support.
    return llm.sequence_confidence(question, statement)
