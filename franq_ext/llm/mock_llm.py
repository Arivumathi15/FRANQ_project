"""Deterministic offline LLM backend.

Drives the whole pipeline in tests, CI, and the demo with NO model download and NO
network. It simulates a model that:
  * knows some facts confidently and others shakily (see franq_ext.knowledge),
  * produces agreeing samples when confident (low semantic entropy) and disagreeing
    samples when unsure (high semantic entropy),
  * can 'retrieve' a better value during targeted correction.

Determinism comes from seeding a local RNG with a hash of each prompt, so the same
input always yields the same output.
"""
from __future__ import annotations

import hashlib
import random
import re

from franq_ext.llm.client import LLMClient
from franq_ext.schema import Fact
from franq_ext import knowledge


_DISTRACTOR_CITIES = ["Munich", "Berlin", "Vienna", "Paris", "Zurich", "Prague"]
_DISTRACTOR_YEARS = ["1880", "1875", "1882", "1878", "1870"]


def _seeded(prompt: str, salt: int = 0) -> random.Random:
    # Stable hash (NOT Python's salted hash()) so samples are reproducible across runs.
    digest = hashlib.md5(f"{salt}:{prompt}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:8], 16))


class MockLLM(LLMClient):
    def __init__(self) -> None:
        # Build an alias index: last name / full name -> canonical entity string.
        self._alias: dict[str, str] = {}
        for (ent, _attr) in knowledge.WORLD:
            self._alias[ent] = ent
            self._alias[ent.split()[-1]] = ent  # e.g. "einstein" -> "albert einstein"

    # ------------------------------------------------------------------ helpers
    def _resolve_entity(self, text: str) -> str | None:
        t = text.lower()
        for alias, canonical in self._alias.items():
            if alias in t:
                return canonical
        return None

    def _match_key(self, text: str) -> tuple[str, str] | None:
        canonical = self._resolve_entity(text)
        if canonical is None:
            return None
        t = text.lower()
        for (ent, attr) in knowledge.WORLD:
            if ent == canonical and attr in t:
                return (ent, attr)
        return None

    # ------------------------------------------------------------------ interface
    def complete(self, prompt: str, temperature: float | None = None) -> str:
        key = self._match_key(prompt)
        if key is not None:
            return knowledge.WORLD[key]
        return ""  # nothing sensible to say in the offline world

    def sample(self, prompt: str, n: int) -> list[str]:
        key = self._match_key(prompt)
        if key is None:
            # Unknown -> maximally uncertain: all-different tokens.
            return [f"opt{i}" for i in range(n)]
        truth = knowledge.WORLD[key]
        conf = knowledge.confidence(*key)
        # Deterministic & monotonic: lower confidence -> more (distinct) distractors ->
        # more meaning-clusters -> higher semantic entropy.
        n_distractors = min(round((1.0 - conf) * n), n)
        _, attr = key
        pool = [c for c in (_DISTRACTOR_YEARS if "year" in attr else _DISTRACTOR_CITIES)
                if c != truth]
        offset = _seeded(prompt).randrange(len(pool)) if pool else 0
        out = [truth] * (n - n_distractors)
        for j in range(n_distractors):
            out.append(pool[(offset + j) % len(pool)] if pool else f"alt{j}")
        return out

    def sequence_confidence(self, context: str, statement: str) -> float:
        key = self._match_key(statement) or self._match_key(context)
        if key is None:
            return 0.5
        return knowledge.confidence(*key)

    def extract_facts(self, answer: str, question: str = "") -> list[Fact]:
        facts: list[Fact] = []
        # Pattern A: "<Entity> was born in <year> in <city>"
        m = re.search(
            r"(?P<ent>[A-Z][\w.]*(?:\s[A-Z][\w.]*)*)\s+was born in\s+"
            r"(?P<year>\d{3,4})\s+in\s+(?P<city>[A-Z][\w-]*)",
            answer,
        )
        if m:
            ent = m.group("ent")
            facts.append(Fact(ent, "birth year", m.group("year")))
            facts.append(Fact(ent, "birth city", m.group("city")))
            return facts
        # Pattern B: "<Entity> was born in <city> in <year>"
        m = re.search(
            r"(?P<ent>[A-Z][\w.]*(?:\s[A-Z][\w.]*)*)\s+was born in\s+"
            r"(?P<city>[A-Z][\w-]*)\s+in\s+(?P<year>\d{3,4})",
            answer,
        )
        if m:
            ent = m.group("ent")
            facts.append(Fact(ent, "birth city", m.group("city")))
            facts.append(Fact(ent, "birth year", m.group("year")))
            return facts
        # Pattern C: "<Entity> was born in <city>."  (single attribute)
        m = re.search(
            r"(?P<ent>[A-Z][\w.]*(?:\s[A-Z][\w.]*)*)\s+was born in\s+"
            r"(?P<city>[A-Z][\w-]*)",
            answer,
        )
        if m:
            facts.append(Fact(m.group("ent"), "birth city", m.group("city")))
        return facts

    def answer_attribute(self, entity: str, attribute: str, evidence: str) -> str:
        canonical = self._alias.get(entity.strip().lower()) or self._resolve_entity(entity)
        if canonical is not None:
            val = knowledge.lookup(canonical, attribute)
            if val is not None:
                return val
        # Fall back to whatever the evidence text offers.
        key = self._match_key(evidence)
        return knowledge.WORLD[key] if key else ""
