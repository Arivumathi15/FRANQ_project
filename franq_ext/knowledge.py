"""A tiny curated 'world knowledge' base for the OFFLINE MockLLM path only.

Why this exists: the deterministic demo/tests must run with no model download and no
network, yet still show the correction loop finding a *better* value than the
hallucinated one. This KB simulates 'external knowledge the model retrieves during a
targeted search'.

IMPORTANT: this is used ONLY by the mock backend. The real (GPU) experiment runs never
touch this file — there, correction uses the dense retriever + LLM over the corpus, so
gold answers are never leaked into the correction process. See README for the split.
"""
from __future__ import annotations

# (entity_lower, attribute_lower) -> canonical value
WORLD: dict[tuple[str, str], str] = {
    ("albert einstein", "birth year"): "1879",
    ("albert einstein", "birth city"): "Ulm",
    ("albert einstein", "field"): "physics",
    ("marie curie", "birth year"): "1867",
    ("marie curie", "birth city"): "Warsaw",
    ("marie curie", "field"): "physics and chemistry",
    ("isaac newton", "birth year"): "1643",
    ("isaac newton", "birth city"): "Woolsthorpe",
    ("ada lovelace", "birth year"): "1815",
    ("ada lovelace", "birth city"): "London",
    ("charles darwin", "birth year"): "1809",
    ("charles darwin", "birth city"): "Shrewsbury",
}

# How confident the model itself is about each key, in [0, 1]. Low confidence -> the
# mock produces disagreeing samples (high semantic entropy), mimicking an unsure model.
CONFIDENCE: dict[tuple[str, str], float] = {
    ("albert einstein", "birth year"): 0.95,
    ("albert einstein", "birth city"): 0.55,
    ("marie curie", "birth year"): 0.9,
    ("marie curie", "birth city"): 0.8,
    ("isaac newton", "birth year"): 0.7,
    ("isaac newton", "birth city"): 0.5,
}


def _norm(entity: str, attribute: str) -> tuple[str, str]:
    return (entity.strip().lower(), attribute.strip().lower())


def lookup(entity: str, attribute: str) -> str | None:
    return WORLD.get(_norm(entity, attribute))


def confidence(entity: str, attribute: str) -> float:
    return CONFIDENCE.get(_norm(entity, attribute), 0.6)
