from franq_ext.data import load_sample
from franq_ext.extraction import FactExtractor
from franq_ext.llm import MockLLM


def test_extracts_two_facts_from_einstein():
    ex = next(e for e in load_sample() if e.qid == "einstein")
    facts = FactExtractor(MockLLM()).extract(ex)
    keys = {f.key() for f in facts}
    assert ("einstein", "birth year") in keys
    assert ("einstein", "birth city") in keys
    # The generated answer carries the hallucinated city.
    city = next(f for f in facts if f.attribute == "birth city")
    assert city.value == "Munich"


def test_no_duplicate_facts():
    ex = next(e for e in load_sample() if e.qid == "curie")
    facts = FactExtractor(MockLLM()).extract(ex)
    keys = [f.key() for f in facts]
    assert len(keys) == len(set(keys))
