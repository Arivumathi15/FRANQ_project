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
# Full-article extracts are large, so they live in their own cache file.
_FULL_CACHE_PATH = os.environ.get(
    "FRANQ_WIKI_FULL_CACHE", os.path.join(".cache", "wiki_context_full.json")
)
_API = "https://en.wikipedia.org/w/api.php"


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)


_CACHE = _load_cache(_CACHE_PATH)
_FULL_CACHE = _load_cache(_FULL_CACHE_PATH)


def _split_sentences(extract: str) -> list[str]:
    parts = [p.strip() for p in extract.replace("\n", " ").split(". ") if p.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


def _fetch_extract(title: str, params: dict, timeout: int = 15) -> str:
    """Call the MediaWiki extracts API and return the raw plaintext extract ('' on failure)."""
    url = _API + "?" + urllib.parse.urlencode({**params, "titles": title})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "franq_ext/0.1 (research)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        return next(iter(pages.values()), {}).get("extract", "") if pages else ""
    except Exception:
        return ""  # graceful: caller degrades to empty context


def get_context(title: str, sentences: int = 12, sleep: float = 0.0) -> list[str]:
    """Return the LEAD-section passages for `title` (cached).

    This is the short RAG snippet the initial answer is generated from. `sentences` caps the
    lead length; override with FRANQ_GEN_SENTENCES to make the initial evidence tighter (which
    surfaces more tail hallucinations for correction to fix).
    """
    key = title.strip()
    if not key:
        return []
    sentences = int(os.environ.get("FRANQ_GEN_SENTENCES", sentences))
    cache_key = f"{key}::{sentences}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    extract = _fetch_extract(key, {
        "action": "query", "format": "json", "prop": "extracts",
        "exintro": "1", "explaintext": "1", "redirects": "1",
        "exsentences": str(sentences),
    })
    passages = _split_sentences(extract)
    if sleep:
        time.sleep(sleep)
    _CACHE[cache_key] = passages
    _save_cache(_CACHE, _CACHE_PATH)
    return passages


def get_full_context(title: str, max_sentences: int = 80, sleep: float = 0.0) -> list[str]:
    """Return the FULL-article passages for `title` (cached) — the deeper evidence pool the
    targeted-correction loop (Pillar 3) is allowed to search.

    Uses the whole plaintext article (no `exintro`), capped to `max_sentences` so the index
    stays small. This is what lets correction find a fact the short lead omitted — the gap
    the initial generation could not close.
    """
    key = title.strip()
    if not key:
        return []
    max_sentences = int(os.environ.get("FRANQ_CORR_SENTENCES", max_sentences))
    cache_key = f"{key}::{max_sentences}"
    if cache_key in _FULL_CACHE:
        return _FULL_CACHE[cache_key]
    extract = _fetch_extract(key, {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1",
    })
    passages = _split_sentences(extract)[:max_sentences]
    if sleep:
        time.sleep(sleep)
    _FULL_CACHE[cache_key] = passages
    _save_cache(_FULL_CACHE, _FULL_CACHE_PATH)
    return passages
