"""Semantic entropy (paper Section 5) — one of the two UQ signals FRANQ switches between.

Idea (Kuhn et al., 2023): ask the model the same question several times; if its answers
*mean* different things, the model is unsure. We sample k answers, cluster them by
meaning, and take the (normalised) entropy of the cluster distribution.

  * Offline default: cluster by normalised surface form — reliable for short factual
    answers (a birth city / year is a few tokens).
  * `nli_equivalence` hook: on GPU runs the cluster test can be swapped for bidirectional
    NLI entailment without changing callers.
"""
from __future__ import annotations

import math
import re


def _normalise(ans: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", ans.lower()).strip()


def _cluster(samples: list[str], equivalent=None) -> list[list[str]]:
    equivalent = equivalent or (lambda a, b: _normalise(a) == _normalise(b))
    clusters: list[list[str]] = []
    for s in samples:
        for c in clusters:
            if equivalent(s, c[0]):
                c.append(s)
                break
        else:
            clusters.append([s])
    return clusters


def semantic_entropy(samples: list[str], equivalent=None, normalise: bool = True) -> float:
    """Entropy over meaning-clusters of `samples`.

    Returns 0 when all samples agree (confident) and approaches 1 (if `normalise`) when
    every sample means something different (maximally unsure).
    """
    if not samples:
        return 0.0
    clusters = _cluster(samples, equivalent)
    n = len(samples)
    h = -sum((len(c) / n) * math.log(len(c) / n) for c in clusters)
    if normalise and n > 1:
        h /= math.log(n)  # divide by max possible entropy -> [0,1]
    return float(h)
