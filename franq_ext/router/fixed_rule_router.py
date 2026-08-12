"""FRANQ baseline router — the fixed IF-ELSE rule that Pillar 2 replaces.

FRANQ switches on faithfulness (AlignScore): if a fact is NOT faithful to the retrieved
context, trust the semantic-entropy signal; otherwise trust the token-probability signal.
There is no learning, no calibration, and no use of the entity graph — exactly the
rigidity Problem 2 describes.
"""
from __future__ import annotations

from franq_ext.schema import Fact


class FixedRuleRouter:
    def __init__(self, faithful_threshold: float = 0.5, flag_threshold: float = 0.5) -> None:
        self.faithful_threshold = faithful_threshold
        self.flag_threshold = flag_threshold

    def fit(self, X=None, y=None) -> "FixedRuleRouter":
        return self  # nothing to learn — that's the whole point of the baseline

    def route_facts(self, facts: list[Fact], graph=None, flag_threshold: float | None = None) -> list[Fact]:
        thr = self.flag_threshold if flag_threshold is None else flag_threshold
        for f in facts:
            faithful = (f.faithfulness or 0.0) >= self.faithful_threshold
            if not faithful:
                # Unfaithful branch -> rely on semantic entropy as the error signal.
                f.error_prob = float(f.semantic_entropy or 0.0)
                f.route = "semantic_entropy"
            else:
                # Faithful branch -> rely on token-probability (1 - confidence).
                f.error_prob = float(1.0 - (f.parametric_uq or 0.0))
                f.route = "token_prob"
            f.flagged = bool(f.error_prob >= thr)
        return facts
