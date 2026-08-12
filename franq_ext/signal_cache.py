"""On-disk cache of per-example fact signals (generated value + faithfulness + UQ).

The 4 ablation conditions differ only in the ROUTER — the expensive LLM work (RAG answer
generation, semantic-entropy sampling, faithfulness scoring) is identical across them. This
cache computes those once and reuses them, cutting GPU time ~4x. It also persists to disk,
so a re-run (or a second condition) skips the work entirely.

Disabled by setting FRANQ_SIGNAL_CACHE=0.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os

from franq_ext.schema import Fact

_FIELDS = ["faithfulness", "retrieval_confidence", "semantic_entropy", "parametric_uq"]


class SignalCache:
    def __init__(self, signature: str = "", path: str | None = None) -> None:
        self.enabled = os.environ.get("FRANQ_SIGNAL_CACHE", "1") != "0"
        results = os.environ.get("FRANQ_RESULTS", "results")
        self.path = path or os.path.join(results, "signal_cache.json")
        self.signature = signature
        self.data: dict = {}
        self._dirty = 0
        if self.enabled and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self.data = {}
        if self.enabled:
            atexit.register(self.save)

    def _key(self, example) -> str:
        raw = f"{self.signature}|{example.qid}|{chr(10).join(example.contexts)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_facts(self, example) -> list[Fact] | None:
        if not self.enabled:
            return None
        rec = self.data.get(self._key(example))
        if not rec:
            return None
        facts = []
        for d in rec:
            f = Fact(d["entity"], d["attribute"], d["value"])
            for k in _FIELDS:
                setattr(f, k, d.get(k))
            facts.append(f)
        return facts

    def put_facts(self, example, facts: list[Fact]) -> None:
        if not self.enabled:
            return
        rec = []
        for f in facts:
            d = {"entity": f.entity, "attribute": f.attribute, "value": f.value}
            for k in _FIELDS:
                d[k] = getattr(f, k)
            rec.append(d)
        self.data[self._key(example)] = rec
        self._dirty += 1
        if self._dirty >= 10:
            self.save()

    def save(self) -> None:
        if not self.enabled or self._dirty == 0:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh)
        self._dirty = 0
