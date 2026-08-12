"""RAG answer generation — the missing front-end that makes the real datasets produce
genuine, labelable hallucinations.

For an entity-attribute benchmark (e.g. PopQA grouped by subject), each attribute is a
sub-question. We let the LLM answer it *from the retrieved evidence* (RAG). The model's
answer becomes the fact's value — which may be right or a hallucination — and is labelled
against the gold value. This is what the earlier real-data path was missing: without a
real generation step, facts were read from the gold answer and were therefore always
correct (AUROC undefined, accuracy 1.0 -> 1.0).
"""
from __future__ import annotations

from franq_ext.llm.client import LLMClient
from franq_ext.schema import Example, Fact


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def answer(self, entity: str, attribute: str, contexts: list[str]) -> str:
        evidence = "\n".join(contexts) if contexts else ""
        return self.llm.answer_attribute(entity, attribute, evidence).strip()


class StructuredFactSource:
    """Builds one fact per gold (entity, attribute) by GENERATING its value via RAG.

    Keys come from the data (so labelling against gold is exact); the value is the model's
    generated answer (so it can hallucinate). Grouping several attributes under one entity
    is what gives Pillar 1's graph real structure on real data.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.generator = AnswerGenerator(llm)

    def __call__(self, example: Example) -> list[Fact]:
        facts = []
        for (entity, attribute, _gold) in example.gold_facts:
            value = self.generator.answer(entity, attribute, example.contexts)
            facts.append(Fact(entity, attribute, value))
        return facts


def structured_fact_source_factory(llm: LLMClient) -> StructuredFactSource:
    return StructuredFactSource(llm)
