"""Extraction module (paper Section 6: 'Extraction module').

Turns a generated answer into structured (entity, attribute, value) facts by delegating
to the active LLM backend. Thin on purpose: the backend decides *how* extraction happens
(regex for the mock, an LLM prompt for the real model), the pipeline stays identical.
"""
from __future__ import annotations

from franq_ext.llm.client import LLMClient
from franq_ext.schema import Example, Fact


class FactExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, example: Example) -> list[Fact]:
        answer = example.generated_answer or example.gold_answer
        facts = self.llm.extract_facts(answer, example.question)
        # De-duplicate on (entity, attribute), keeping the first value seen.
        seen: set[tuple[str, str]] = set()
        unique: list[Fact] = []
        for f in facts:
            if f.key() not in seen:
                seen.add(f.key())
                unique.append(f)
        return unique
