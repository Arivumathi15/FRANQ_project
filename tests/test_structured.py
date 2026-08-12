"""Structured (RAG generate-then-check) path: the front-end used for the real datasets.

We simulate the LLM hallucinating one attribute and confirm the same pipeline detects,
corrects, and reports zero regret — i.e. the real-data path is no longer degenerate.
"""
from franq_ext.config import Config
from franq_ext.data import load_sample
from franq_ext.pipeline import Pipeline
from franq_ext.generation import StructuredFactSource
from franq_ext.evaluation.harness import evaluate_with_pipeline
from franq_ext.train_router import train_router


class _CorruptingSource:
    """Like StructuredFactSource, but flips one attribute to a wrong value to mimic a
    model hallucination (the offline MockLLM otherwise answers everything correctly)."""

    def __init__(self, llm):
        self.inner = StructuredFactSource(llm)

    def __call__(self, example):
        facts = self.inner(example)
        for f in facts:
            if f.entity.lower() == "einstein" and f.attribute == "birth city":
                f.value = "Munich"  # gold is Ulm -> injected hallucination
        return facts


def _factory(llm):
    return _CorruptingSource(llm)


def test_structured_facts_are_generated_and_grouped():
    cfg = Config()
    ex = next(e for e in load_sample() if e.qid == "einstein")
    pipe = Pipeline(cfg, fact_source_factory=_factory)
    facts, graph, _ = pipe.compute_signals(ex)
    # Two attributes for Einstein, grouped under one entity (Pillar 1 has real structure).
    assert {f.attribute for f in facts} == {"birth year", "birth city"}
    assert list(graph.groups().keys()) == ["einstein"]


def test_structured_pipeline_detects_and_corrects_hallucination():
    cfg = Config()
    examples = load_sample()
    pipe = Pipeline(cfg, fact_source_factory=_factory)
    pipe.router = train_router(cfg, examples, pipeline=pipe, augment=40)
    res = evaluate_with_pipeline(pipe, examples, condition="structured")

    assert res.n_facts == 6                    # 3 entities x 2 attributes
    assert res.factual_acc_after == 1.0        # the hallucinated city was fixed
    assert res.correction_success == 1.0       # the one wrong fact corrected
    assert res.correction_regret == 0.0        # nothing correct was broken
    assert res.n_corrected >= 1
