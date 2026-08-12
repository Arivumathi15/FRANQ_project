from franq_ext.metrics import auroc, prr


def test_auroc_perfect_and_inverted():
    wrong = [0, 0, 1, 1]
    assert auroc([0.1, 0.2, 0.8, 0.9], wrong) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], wrong) == 0.0


def test_auroc_single_class_is_nan():
    import math
    assert math.isnan(auroc([0.1, 0.2], [0, 0]))


def test_prr_oracle_alignment_is_one():
    risk = [1, 1, 0, 0, 0, 0]
    # Uncertainty equals the true risk ordering -> as good as the oracle -> PRR = 1.
    assert abs(prr(risk, risk) - 1.0) < 1e-9


def test_prr_anticorrelated_is_low():
    risk = [1, 1, 0, 0, 0, 0]
    anti = [0, 0, 1, 1, 1, 1]  # most 'uncertain' about the correct ones
    assert prr(anti, risk) < 0.5
