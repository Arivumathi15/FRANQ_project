"""Pillar 3 (part 2) - Re-verification + Correction-Regret.

After the correction loop runs, re-check the whole answer and measure whether the fix
damaged anything that was already correct. Correctness here is factual (value vs gold);
the safety guarantee is the correction-regret number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from franq_ext.scoring.align_scorer import AlignScorer
from franq_ext.schema import Fact
from franq_ext.metrics.regret import correction_regret, is_correct


@dataclass
class ReverifyResult:
    correction_regret: float
    n_correct_before: int
    n_correct_after: int
    per_fact: list[dict] = field(default_factory=list)


class ReVerifier:
    def __init__(self, scorer: AlignScorer) -> None:
        self.scorer = scorer

    def recompute_faithfulness(self, facts: list[Fact], contexts: list[str]) -> None:
        for f in facts:
            f.faithfulness = self.scorer.score_fact(f, contexts)

    def evaluate(
        self,
        facts: list[Fact],
        gold_facts: list[tuple[str, str, str]],
        before_values: dict[tuple[str, str], str],
    ) -> ReverifyResult:
        """`before_values` maps fact key -> its value BEFORE correction."""
        gold = {(e.strip().lower(), a.strip().lower()): v for (e, a, v) in gold_facts}
        before_after: list[tuple[bool, bool]] = []
        per_fact: list[dict] = []
        for f in facts:
            key = f.key()
            if key not in gold:
                continue  # can only judge facts we have gold for
            gold_v = gold[key]
            before_v = before_values.get(key, f.value)
            correct_before = is_correct(before_v, gold_v)
            correct_after = is_correct(f.value, gold_v)
            before_after.append((correct_before, correct_after))
            per_fact.append({
                "fact": key,
                "before": before_v,
                "after": f.value,
                "gold": gold_v,
                "correct_before": correct_before,
                "correct_after": correct_after,
            })
        return ReverifyResult(
            correction_regret=correction_regret(before_after),
            n_correct_before=sum(1 for b, _ in before_after if b),
            n_correct_after=sum(1 for _, a in before_after if a),
            per_fact=per_fact,
        )
