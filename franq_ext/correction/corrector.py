"""Pillar 3 (part 1) - Budget-Bounded Targeted Correction Loop.

When the router flags a fact as likely wrong, we do NOT regenerate the whole answer.
Instead we run a *focused* search for just that entity-attribute (e.g. "Einstein birth
city"), retrieve fresh evidence, propose a value, and replace ONLY that value — and only
if the new value is better supported than the old one. All of this happens inside a
retrieval budget so the loop can never search forever (paper Section 4, Pillar 3).
"""
from __future__ import annotations

from dataclasses import dataclass

from franq_ext.llm.client import LLMClient
from franq_ext.retrieval.dense_retriever import Retriever
from franq_ext.scoring.align_scorer import AlignScorer
from franq_ext.schema import Fact


@dataclass
class CorrectionRecord:
    fact_key: tuple[str, str]
    old_value: str
    new_value: str
    old_faithfulness: float
    new_faithfulness: float
    attempts: int
    accepted: bool


class Corrector:
    def __init__(self, llm: LLMClient, retriever: Retriever, scorer: AlignScorer, cfg) -> None:
        self.llm = llm
        self.retriever = retriever   # already indexed on the correction corpus
        self.scorer = scorer
        self.cfg = cfg               # CorrectionConfig

    def _targeted_query(self, fact: Fact) -> str:
        return f"{fact.entity} {fact.attribute}"

    def correct(self, fact: Fact) -> CorrectionRecord:
        query = self._targeted_query(fact)
        old_value = fact.value
        old_faith = fact.faithfulness if fact.faithfulness is not None else 0.0
        best_value, best_faith = old_value, old_faith
        attempts = 0

        # Budget-bounded: attempt i inspects the top-(i+1) retrieved passages.
        hits = self.retriever.retrieve(query, top_k=self.cfg.budget)
        for i in range(min(self.cfg.budget, max(len(hits), 1))):
            attempts += 1
            evidence_passages = [h[0] for h in hits[: i + 1]]
            evidence = "\n".join(evidence_passages)
            candidate = self.llm.answer_attribute(fact.entity, fact.attribute, evidence).strip()
            if not candidate:
                continue
            cand_faith = self.scorer.score_claim(
                fact.entity, fact.attribute, candidate, evidence_passages
            )
            if self.cfg.safety_checked:
                # Accept only a strictly better-supported, above-threshold value.
                improved = cand_faith > best_faith and cand_faith >= self.cfg.flag_threshold
            else:
                # Reckless: accept any different candidate (no safety guard).
                improved = candidate.lower() != old_value.lower()
            if improved:
                best_value, best_faith = candidate, cand_faith
                if candidate.lower() != old_value.lower():
                    break  # found a replacement; stop early to respect budget

        if self.cfg.safety_checked:
            accepted = best_value.lower() != old_value.lower() and best_faith > old_faith
        else:
            accepted = best_value.lower() != old_value.lower()
        if accepted:
            fact.original_value = old_value
            fact.value = best_value
            fact.faithfulness = best_faith
            fact.corrected = True

        return CorrectionRecord(
            fact_key=fact.key(),
            old_value=old_value,
            new_value=fact.value,
            old_faithfulness=old_faith,
            new_faithfulness=best_faith,
            attempts=attempts,
            accepted=accepted,
        )

    def correct_flagged(self, facts: list[Fact]) -> list[CorrectionRecord]:
        records = []
        for f in facts:
            if f.flagged and self.cfg.enabled:
                records.append(self.correct(f))
        return records
