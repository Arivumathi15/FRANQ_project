from franq_ext.config import default_config
from franq_ext.llm import MockLLM
from franq_ext.schema import Fact
from franq_ext.uncertainty import UncertaintyEstimator, semantic_entropy


def test_semantic_entropy_zero_when_all_agree():
    assert semantic_entropy(["Ulm", "Ulm", "Ulm"]) == 0.0


def test_semantic_entropy_high_when_all_differ():
    assert semantic_entropy(["a", "b", "c"]) > 0.9  # normalised, near 1


def test_unsure_attribute_has_higher_entropy_than_confident_one():
    est = UncertaintyEstimator(MockLLM(), default_config().uq)
    city = est.annotate(Fact("Einstein", "birth city", "Munich"))   # mock confidence 0.55
    year = est.annotate(Fact("Einstein", "birth year", "1879"))     # mock confidence 0.95
    assert city.semantic_entropy > year.semantic_entropy
    assert city.parametric_uq < year.parametric_uq
