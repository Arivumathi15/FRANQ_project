"""Dataset loaders -> a unified list[Example].

  * load_sample()  : the bundled offline set (tests + demo, no network).
  * load_popqa()   : PopQA, entity-attribute triples (primary real benchmark, Pillar 1).
  * load_triviaqa(): TriviaQA, from the FRANQ base paper (for comparability).

The two real loaders need the `datasets` library (installed only for GPU runs) and
return Examples with empty `contexts`; the pipeline's retriever fills contexts in.
"""
from __future__ import annotations

import json
import os

from franq_ext.schema import Example

_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample", "sample_examples.json")


def load_sample() -> list[Example]:
    with open(_SAMPLE_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    examples = []
    for r in raw:
        examples.append(
            Example(
                qid=r["qid"],
                question=r["question"],
                gold_answer=r["gold_answer"],
                contexts=r.get("contexts", []),
                gold_facts=[tuple(t) for t in r.get("gold_facts", [])],
                generated_answer=r.get("generated_answer"),
            )
        )
    return examples


def load_popqa(n: int | None = 800, seed: int = 13) -> list[Example]:
    """PopQA: fields include `question`, `subj`, `prop`, `obj`, `possible_answers`.

    We build one gold fact per example from (subj, prop, obj) — a natural entity-attribute
    triple, which is exactly what Pillar 1's graph consumes.
    """
    from datasets import load_dataset  # lazy

    ds = load_dataset("akariasai/PopQA", split="test")
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    examples = []
    for i, r in enumerate(ds):
        examples.append(
            Example(
                qid=f"popqa-{i}",
                question=r["question"],
                gold_answer=str(r.get("obj", "")),
                contexts=[],
                gold_facts=[(str(r.get("subj", "")), str(r.get("prop", "")), str(r.get("obj", "")))],
            )
        )
    return examples


def load_popqa_structured(n: int | None = 300, seed: int = 13,
                          with_contexts: bool = True, min_attributes: int = 2) -> list[Example]:
    """PopQA grouped by SUBJECT -> one Example per entity with MULTIPLE attributes.

    This is what gives Pillar 1's graph real structure on real data: an entity like
    'Albert Einstein' ends up with several (attribute, value) facts checked together.
    `min_attributes` keeps only subjects with at least that many distinct properties —
    essential, because single-attribute entities give the graph nothing to connect (and
    then its features are just noise). Contexts come from the subject's Wikipedia intro
    (cached). `generated_answer` is left None — the StructuredFactSource generates each
    attribute's value via RAG at run time.
    """
    from datasets import load_dataset  # lazy

    ds = load_dataset("akariasai/PopQA", split="test")
    ds = ds.shuffle(seed=seed)

    # Group by the DISAMBIGUATED Wikipedia title, not the bare surface name. PopQA reuses a
    # surface name (e.g. "Baby") across several distinct entities (a novel, a song, a film),
    # each with its own s_wiki_title. Grouping by `subj` conflated them into one incoherent
    # entity AND made the assigned title order-dependent (so the bundled context cache missed).
    # Grouping by title gives one real article per group, coherent facts, and a stable key.
    groups: dict[str, dict] = {}
    for r in ds:
        subj = str(r.get("subj", "")).strip()
        prop = str(r.get("prop", "")).strip()
        obj = str(r.get("obj", "")).strip()
        title = str(r.get("s_wiki_title", subj) or subj).strip()
        if not (subj and prop and obj and title):
            continue
        # PopQA ships `possible_answers` (a JSON list of acceptable aliases). Fold them into
        # the gold so lenient matching can accept any valid surface form.
        aliases = [obj]
        raw_alt = r.get("possible_answers")
        if raw_alt:
            try:
                aliases += [str(a) for a in json.loads(raw_alt)]
            except (json.JSONDecodeError, TypeError):
                pass
        gold_value = "||".join(dict.fromkeys(a.strip() for a in aliases if a.strip()))
        g = groups.setdefault(title, {"subj": subj, "facts": {}})
        g["facts"][prop] = gold_value  # one value (with aliases) per (title, property)

    # Keep only multi-attribute entities, then take the first n.
    multi = [(t, g) for t, g in groups.items() if len(g["facts"]) >= min_attributes]
    if n is not None:
        multi = multi[:n]

    examples = []
    n_with_ctx = n_with_corr = 0
    for i, (title, g) in enumerate(multi):
        subj = g["subj"]
        gold_facts = [(subj, prop, obj) for prop, obj in g["facts"].items()]
        contexts: list[str] = []
        correction_contexts: list[str] = []
        if with_contexts:
            from franq_ext.data.wiki_context import get_context, get_full_context
            # Generation sees the short lead; correction (Pillar 3) may search the full article.
            contexts = get_context(title)
            correction_contexts = get_full_context(title)
        n_with_ctx += 1 if contexts else 0
        n_with_corr += 1 if correction_contexts else 0
        examples.append(
            Example(
                qid=f"popqa-struct-{i}",
                question=f"Facts about {subj}",
                gold_answer="; ".join(f"{p}={o}" for p, o in g["facts"].items()),
                contexts=contexts,
                correction_contexts=correction_contexts,
                gold_facts=gold_facts,
            )
        )
    if with_contexts and examples:
        # Coverage diagnostic: if these are ~0, the Wikipedia fetch is being blocked and the
        # low factual accuracy is a data problem, not a model problem.
        avg_ctx = sum(len(e.contexts) for e in examples) / len(examples)
        avg_corr = sum(len(e.correction_contexts) for e in examples) / len(examples)
        print(
            f"[popqa_structured] context coverage: "
            f"lead {n_with_ctx}/{len(examples)} (avg {avg_ctx:.1f} passages), "
            f"full-article {n_with_corr}/{len(examples)} (avg {avg_corr:.1f} passages)",
            flush=True,
        )
    return examples


def load_triviaqa(n: int | None = 800, seed: int = 13) -> list[Example]:
    """TriviaQA (rc.nocontext) — used in the FRANQ base paper. Gold facts are left
    empty (open-domain answers, not clean triples); it is used for the detection
    metrics (AUROC/PRR), not for the graph-dependent regret metric."""
    from datasets import load_dataset  # lazy

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    examples = []
    for i, r in enumerate(ds):
        answer = r.get("answer", {}).get("value", "")
        examples.append(
            Example(
                qid=f"triviaqa-{i}",
                question=r["question"],
                gold_answer=str(answer),
                contexts=[],
                gold_facts=[],
            )
        )
    return examples
