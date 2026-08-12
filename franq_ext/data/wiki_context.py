"""Fetch + cache short Wikipedia contexts for entities (used by the structured PopQA path).

PopQA ships no evidence passages, but faithfulness scoring and correction need retrieved
context. We fetch the subject's Wikipedia summary from the public REST API and cache it to
disk, so a re-run is instant and reproducible. Network failures degrade gracefully to an
empty context (that example simply gets weak evidence).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

_CACHE_PATH = os.environ.get("FRANQ_WIKI_CACHE", os.path.join(".cache", "wiki_context.json"))
_API = "https://en.wikipedia.org/w/api.php"


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)


_CACHE = _load_cache()


def get_context(title: str, sentences: int = 12, sleep: float = 0.0) -> list[str]:
    """Return context passages for `title` from Wikipedia's intro (cached).

    Uses the query API's `extracts` (up to `sentences` sentences of the lead section) —
    a fuller intro than the one-line REST summary, so targeted correction actually has the
    specific fact to retrieve.
    """
    key = title.strip()
    if not key:
        return []
    if key in _CACHE:
        return _CACHE[key]
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "exintro": "1", "explaintext": "1", "redirects": "1",
        "exsentences": str(sentences), "titles": key,
    }
    url = _API + "?" + urllib.parse.urlencode(params)
    passages: list[str] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "franq_ext/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        extract = next(iter(pages.values()), {}).get("extract", "") if pages else ""
        if extract:
            parts = [p.strip() for p in extract.replace("\n", " ").split(". ") if p.strip()]
            passages = [p if p.endswith(".") else p + "." for p in parts]
    except Exception:
        passages = []  # graceful: no context for this entity
    if sleep:
        time.sleep(sleep)
    _CACHE[key] = passages
    _save_cache(_CACHE)
    return passages
