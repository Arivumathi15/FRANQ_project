"""The signal cache must round-trip facts and persist to disk (so conditions/re-runs reuse
the expensive LLM work instead of recomputing it)."""
import os

from franq_ext.schema import Example, Fact
from franq_ext.signal_cache import SignalCache


def _example():
    return Example(qid="e1", question="q", gold_answer="", contexts=["ctx a", "ctx b"],
                   gold_facts=[("E", "attr", "v")])


def _fact():
    return Fact("E", "attr", "v", faithfulness=0.8, retrieval_confidence=0.3,
                semantic_entropy=0.5, parametric_uq=0.7)


def test_roundtrip_and_persistence(tmp_path):
    os.environ["FRANQ_SIGNAL_CACHE"] = "1"
    try:
        path = str(tmp_path / "sig.json")
        ex = _example()
        c1 = SignalCache(signature="sigA", path=path)
        assert c1.get_facts(ex) is None            # cold miss
        c1.put_facts(ex, [_fact()])
        c1.save()
        assert os.path.exists(path)

        # A fresh cache loads from disk and returns the stored signals.
        c2 = SignalCache(signature="sigA", path=path)
        got = c2.get_facts(ex)
        assert got is not None and len(got) == 1
        assert got[0].value == "v" and abs(got[0].faithfulness - 0.8) < 1e-9

        # A different signature (e.g. different model) must NOT hit the same entry.
        c3 = SignalCache(signature="sigB", path=path)
        assert c3.get_facts(ex) is None
    finally:
        os.environ["FRANQ_SIGNAL_CACHE"] = "0"


def test_disabled_cache_is_a_noop(tmp_path):
    os.environ["FRANQ_SIGNAL_CACHE"] = "0"
    path = str(tmp_path / "sig.json")
    c = SignalCache(signature="x", path=path)
    c.put_facts(_example(), [_fact()])
    assert c.get_facts(_example()) is None
    assert not os.path.exists(path)
