from franq_ext.config import default_config
from franq_ext.data import load_sample
from franq_ext.llm import MockLLM
from franq_ext.retrieval import LexicalRetriever
from franq_ext.scoring import LexicalAlignScorer
from franq_ext.correction import Corrector
from franq_ext.schema import Fact


def _corrector(contexts):
    cfg = default_config()
    retr = LexicalRetriever()
    retr.index(contexts)
    return Corrector(MockLLM(), retr, LexicalAlignScorer(), cfg.correction), cfg


def test_corrects_hallucinated_city():
    ctx = next(e for e in load_sample() if e.qid == "einstein").contexts
    corrector, _ = _corrector(ctx)
    fact = Fact("Einstein", "birth city", "Munich", faithfulness=0.15, flagged=True)
    rec = corrector.correct(fact)
    assert rec.accepted
    assert fact.value == "Ulm"
    assert fact.corrected and fact.original_value == "Munich"
    assert rec.new_faithfulness > rec.old_faithfulness


def test_respects_budget():
    ctx = next(e for e in load_sample() if e.qid == "einstein").contexts
    corrector, cfg = _corrector(ctx)
    fact = Fact("Einstein", "birth city", "Munich", faithfulness=0.15, flagged=True)
    rec = corrector.correct(fact)
    assert rec.attempts <= cfg.correction.budget


def test_correct_value_is_not_changed():
    ctx = next(e for e in load_sample() if e.qid == "einstein").contexts
    corrector, _ = _corrector(ctx)
    # Already correct; even if flagged, no better value exists -> unchanged.
    fact = Fact("Einstein", "birth year", "1879", faithfulness=0.9, flagged=True)
    rec = corrector.correct(fact)
    assert not rec.accepted
    assert fact.value == "1879"
    assert not fact.corrected


def test_only_flagged_facts_are_corrected():
    ctx = next(e for e in load_sample() if e.qid == "einstein").contexts
    corrector, _ = _corrector(ctx)
    flagged = Fact("Einstein", "birth city", "Munich", faithfulness=0.15, flagged=True)
    unflagged = Fact("Einstein", "birth city", "Munich", faithfulness=0.15, flagged=False)
    recs = corrector.correct_flagged([flagged, unflagged])
    assert len(recs) == 1  # only the flagged one was touched
