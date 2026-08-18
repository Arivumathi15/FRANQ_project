"""Prebuild the bundled Wikipedia context cache shipped with the package.

Run this OFF-Kaggle (where Wikipedia is not rate-limited). It fetches the lead + full-article
extracts for every PopQA subject with >= min_attributes attributes and writes them into
franq_ext/data/wiki_cache/. Those files are then committed / zipped so Kaggle runs need NO
live Wikipedia access.

    python scripts/build_wiki_cache.py            # all >=3-attribute subjects (default)
    FRANQ_MIN_ATTRS=2 python scripts/build_wiki_cache.py

Cache keys embed the sentence budgets, so they must match the runtime defaults
(FRANQ_GEN_SENTENCES / FRANQ_CORR_SENTENCES). We import the real getters so they always agree.
"""
from __future__ import annotations

import os
import sys

# Point the writable cache at the bundle dir BEFORE importing wiki_context, so its getters
# persist straight into the shipped files with exactly the runtime cache keys.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
_BUNDLE_DIR = os.path.join(_ROOT, "franq_ext", "data", "wiki_cache")
os.makedirs(_BUNDLE_DIR, exist_ok=True)
os.environ["FRANQ_WIKI_CACHE"] = os.path.join(_BUNDLE_DIR, "wiki_context.json")
os.environ["FRANQ_WIKI_FULL_CACHE"] = os.path.join(_BUNDLE_DIR, "wiki_context_full.json")
os.environ.setdefault("FRANQ_WIKI_SLEEP", "0.15")

from datasets import load_dataset  # noqa: E402
from franq_ext.data import wiki_context as wc  # noqa: E402


def main() -> None:
    # Cache all titles down to 2 attributes so BOTH FRANQ_MIN_ATTRS=2 (default) and =3
    # (the notebook) are fully covered regardless of which n the run selects.
    min_attrs = int(os.environ.get("FRANQ_MIN_ATTRS", "2"))
    ds = load_dataset("akariasai/PopQA", split="test")

    # Group by disambiguated Wikipedia title — the SAME key the loader now uses.
    groups: dict[str, dict] = {}
    for r in ds:
        subj = str(r.get("subj", "")).strip()
        prop = str(r.get("prop", "")).strip()
        obj = str(r.get("obj", "")).strip()
        title = str(r.get("s_wiki_title", subj) or subj).strip()
        if not (subj and prop and obj and title):
            continue
        groups.setdefault(title, set()).add(prop)

    titles = [t for t, props in groups.items() if len(props) >= min_attrs]
    titles = list(dict.fromkeys(titles))  # de-dup, keep order
    total = len(titles)
    print(f"building cache for {total} subjects (>= {min_attrs} attributes)")

    empty_lead = empty_full = 0
    for i, title in enumerate(titles, 1):
        lead = wc.get_context(title)
        full = wc.get_full_context(title)
        empty_lead += 0 if lead else 1
        empty_full += 0 if full else 1
        if i % 25 == 0 or i == total:
            print(f"  {i}/{total}  (empty lead={empty_lead}, empty full={empty_full})", flush=True)

    wc._save_cache(wc._CACHE, os.environ["FRANQ_WIKI_CACHE"])
    wc._save_cache(wc._FULL_CACHE, os.environ["FRANQ_WIKI_FULL_CACHE"])
    print(f"\nDONE. lead cache entries={len(wc._CACHE)}, full cache entries={len(wc._FULL_CACHE)}")
    print(f"empty lead={empty_lead}/{total}  empty full={empty_full}/{total}")
    print(f"wrote -> {_BUNDLE_DIR}")


if __name__ == "__main__":
    main()
