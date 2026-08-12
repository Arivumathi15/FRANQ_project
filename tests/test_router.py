import numpy as np

from franq_ext.config import default_config
from franq_ext.router import (
    CalibratedRouter,
    FixedRuleRouter,
    fact_features,
    label_fact,
)
from franq_ext.metrics import expected_calibration_error
from franq_ext.schema import Fact


def test_feature_vector_length():
    f = Fact("Einstein", "birth city", "Munich", faithfulness=0.2,
             retrieval_confidence=0.3, semantic_entropy=0.8, parametric_uq=0.4,
             dependency_strength=0.5)
    assert len(fact_features(f)) == 6


def test_label_fact_detects_wrong_and_right():
    gold = [("Einstein", "birth city", "Ulm")]
    wrong = Fact("Einstein", "birth city", "Munich")
    right = Fact("Einstein", "birth city", "Ulm")
    assert label_fact(wrong, gold) == 1
    assert label_fact(right, gold) == 0
    assert label_fact(Fact("X", "y", "z"), gold) is None


def test_fixed_rule_flags_unfaithful_high_entropy_fact():
    r = FixedRuleRouter(faithful_threshold=0.5, flag_threshold=0.5)
    bad = Fact("Einstein", "birth city", "Munich", faithfulness=0.15, semantic_entropy=0.8)
    good = Fact("Einstein", "birth year", "1879", faithfulness=0.9, parametric_uq=0.95)
    r.route_facts([bad, good])
    assert bad.flagged and bad.route == "semantic_entropy"
    assert not good.flagged and good.route == "token_prob"


def _synthetic_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        wrong = rng.random() < 0.5
        # Wrong facts: low faithfulness, high entropy, low parametric confidence.
        faith = rng.uniform(0.0, 0.4) if wrong else rng.uniform(0.6, 1.0)
        ent = rng.uniform(0.6, 1.0) if wrong else rng.uniform(0.0, 0.4)
        par = rng.uniform(0.0, 0.4) if wrong else rng.uniform(0.6, 1.0)
        X.append([faith, rng.random(), ent, par, rng.random(), rng.random()])
        y.append(int(wrong))
    return X, y


def test_calibrated_router_learns_and_is_reasonably_calibrated():
    cfg = default_config()
    Xtr, ytr = _synthetic_data(400, seed=1)
    Xte, yte = _synthetic_data(200, seed=2)
    router = CalibratedRouter(cfg).fit(Xtr, ytr)
    probs = router.predict_error_prob(Xte)
    # Discriminative: mean predicted error is higher for truly-wrong facts.
    yte = np.asarray(yte)
    assert probs[yte == 1].mean() > probs[yte == 0].mean() + 0.2
    # Calibrated: ECE should be small on held-out data.
    assert expected_calibration_error(probs, yte) < 0.15


def test_calibrated_router_handles_single_class():
    cfg = default_config()
    router = CalibratedRouter(cfg).fit([[0.1] * 6, [0.2] * 6], [0, 0])
    # Falls back to prior without crashing.
    assert 0.0 <= router.predict_error_prob([[0.5] * 6])[0] <= 1.0
