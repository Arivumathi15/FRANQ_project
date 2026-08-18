"""Fetch + cache short Wikipedia contexts for entities (used by the structured PopQA path).

PopQA ships no evidence passages, but faithfulness scoring and correction need retrieved
context. We fetch the subject's Wikipedia extract from the public API and cache it to disk.

IMPORTANT (Kaggle/Colab): shared-IP notebooks get rate-limited by Wikipedia, so hammering
the API live drops ~95% of requests and leaves most examples context-less (accuracy collapses
to the model's blind guessing). Two defences:
  1. A BUNDLED cache (`franq_ext/data/wiki_cache/*.json`) shipped with the package — prebuilt
     off-Kaggle, it means a normal run needs NO Wikipedia access at all.
  2. Live fetches (only for anything not in the bundle) are polite: a real User-Agent, a small
     inter-request delay, and retry-with-backoff that honours HTTP 429. Failures still degrade
     gracefully to empty context.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

_CACHE_PATH = os.environ.get("FRANQ_WIKI_CACHE", os.path.join(".cache", "wiki_context.json"))

# Prebuilt, read-only cache shipped inside the package (see scripts/build_wiki_cache.py).
_BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "wiki_cache")
_BUNDLE_LEAD = os.path.join(_BUNDLE_DIR, "wiki_context.json")
_BUNDLE_FULL = os.path.join(_BUNDLE_DIR, "wiki_context_full.json")


def _default_full_cache_path() -> str:
    """Full-article extracts get their own file, kept NEXT TO the lead cache so a single
    FRANQ_WIKI_CACHE override (e.g. /kaggle/working/.cache/...) makes BOTH writable. Only
    falls back to the explicit FRANQ_WIKI_FULL_CACHE env when it is set."""
    explicit = os.environ.get("FRANQ_WIKI_FULL_CACHE")
    if explicit:
        return explicit
    base_dir = os.path.dirname(_CACHE_PATH) or "."
    return os.path.join(base_dir, "wiki_context_full.json")


# Full-article extracts are large, so they live in their own cache file.
_FULL_CACHE_PATH = _default_full_cache_path()
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
    # Best-effort: a read-only working dir must not crash the run — just skip persisting
    # (fetches still work this session, only cross-session caching is lost).
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    except OSError as e:
        print(f"[wiki_context] cache save skipped ({path}): {e}", flush=True)


# Writable cache overrides the bundled seed; the bundle covers the standard entity set.
_CACHE = {**_load_cache(_BUNDLE_LEAD), **_load_cache(_CACHE_PATH)}
_FULL_CACHE = {**_load_cache(_BUNDLE_FULL), **_load_cache(_FULL_CACHE_PATH)}

# Polite live-fetch knobs (only used for cache misses). Tunable via env for cache-building.
_FETCH_SLEEP = float(os.environ.get("FRANQ_WIKI_SLEEP", "0.1"))
_FETCH_RETRIES = int(os.environ.get("FRANQ_WIKI_RETRIES", "4"))
_UA = os.environ.get(
    "FRANQ_WIKI_UA",
    "franq_ext/0.2 (academic research; https://github.com/; mailto:research@example.com)",
)


def _split_sentences(extract: str) -> list[str]:
    parts = [p.strip() for p in extract.replace("\n", " ").split(". ") if p.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


def _fetch_extract(title: str, params: dict, timeout: int = 20) -> str:
    """Call the MediaWiki extracts API and return the raw plaintext extract ('' on failure).

    Retries with exponential backoff and honours HTTP 429/503 Retry-After, so a rate-limited
    shared IP (Kaggle) recovers instead of silently returning empty context.
    """
    url = _API + "?" + urllib.parse.urlencode({**params, "titles": title})
    backoff = 1.0
    for attempt in range(_FETCH_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            return next(iter(pages.values()), {}).get("extract", "") if pages else ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < _FETCH_RETRIES - 1:
                wait = float(e.headers.get("Retry-After", "") or backoff)
                time.sleep(min(wait, 30.0))
                backoff *= 2
                continue
            return ""
        except Exception:
            if attempt < _FETCH_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            return ""  # graceful: caller degrades to empty context
    return ""


# Defaults: a SHORT lead (realistic RAG snippet -> surfaces tail hallucinations) vs a DEEP
# full article (what targeted correction is allowed to search). The gap between them is the
# room Pillar 3 has to improve accuracy. Override via FRANQ_GEN_SENTENCES / FRANQ_CORR_SENTENCES
# (must match the bundled cache keys, so change both the env AND rebuild the bundle).
_LEAD_SENTENCES = int(os.environ.get("FRANQ_GEN_SENTENCES", "3"))
_FULL_SENTENCES = int(os.environ.get("FRANQ_CORR_SENTENCES", "100"))


def get_context(title: str, sentences: int | None = None) -> list[str]:
    """Return the LEAD-section passages for `title` (bundled cache first, then polite fetch).

    This is the short RAG snippet the initial answer is generated from.
    """
    key = title.strip()
    if not key:
        return []
    sentences = _LEAD_SENTENCES if sentences is None else sentences
    cache_key = f"{key}::{sentences}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    extract = _fetch_extract(key, {
        "action": "query", "format": "json", "prop": "extracts",
        "exintro": "1", "explaintext": "1", "redirects": "1",
        "exsentences": str(sentences),
    })
    passages = _split_sentences(extract)
    time.sleep(_FETCH_SLEEP)  # be polite to the shared endpoint
    _CACHE[cache_key] = passages
    _save_cache(_CACHE, _CACHE_PATH)
    return passages


def get_full_context(title: str, max_sentences: int | None = None) -> list[str]:
    """Return the FULL-article passages for `title` (bundled cache first, then polite fetch) —
    the deeper evidence pool the targeted-correction loop (Pillar 3) is allowed to search.

    Uses the whole plaintext article (no `exintro`), capped to `max_sentences` so the index
    stays small. This is what lets correction find a fact the short lead omitted — the gap
    the initial generation could not close.
    """
    key = title.strip()
    if not key:
        return []
    max_sentences = _FULL_SENTENCES if max_sentences is None else max_sentences
    cache_key = f"{key}::{max_sentences}"
    if cache_key in _FULL_CACHE:
        return _FULL_CACHE[cache_key]
    extract = _fetch_extract(key, {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1",
    })
    passages = _split_sentences(extract)[:max_sentences]
    time.sleep(_FETCH_SLEEP)
    _FULL_CACHE[cache_key] = passages
    _save_cache(_FULL_CACHE, _FULL_CACHE_PATH)
    return passages
