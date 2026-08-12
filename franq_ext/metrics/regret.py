"""Correction-Regret — the project's headline novelty (paper Section 4, Pillar 3).

    correction_regret = (facts correct BEFORE that became wrong AFTER)
                        / (facts correct BEFORE)

It answers the safety question no compared baseline asks: did fixing one fact quietly
break a fact that was already right? 0 = no collateral damage; higher = more damage.
"""
from __future__ import annotations


import re

_ARTICLES = {"a", "an", "the"}


def _norm(v: str) -> str:
    """Lowercase, drop punctuation and leading articles, collapse whitespace."""
    s = re.sub(r"[^a-z0-9\s]", " ", str(v).lower())
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


def is_correct(value: str, gold_value: str) -> bool:
    """Lenient QA correctness.

    Exact match is far too strict for free-form generation ('Einstein was born in Ulm' vs
    gold 'Ulm' would count as wrong). We accept a match if any acceptable gold answer is
    contained in the prediction (or vice-versa), after normalisation. `gold_value` may hold
    several acceptable answers separated by '||' (e.g. PopQA aliases).
    """
    pred = _norm(value)
    if not pred:
        return False
    for gold in str(gold_value).split("||"):
        g = _norm(gold)
        if not g:
            continue
        if g == pred or g in pred or pred in g:
            return True
    return False


def correction_regret(before_after: list[tuple[bool, bool]]) -> float:
    """`before_after` is a list of (correct_before, correct_after) per fact."""
    n_correct_before = sum(1 for b, _ in before_after if b)
    if n_correct_before == 0:
        return 0.0  # nothing was correct to damage
    n_broken = sum(1 for b, a in before_after if b and not a)
    return n_broken / n_correct_before
