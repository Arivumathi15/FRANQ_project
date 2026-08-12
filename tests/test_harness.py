from franq_ext.config import Config
from franq_ext.data import load_sample
from franq_ext.evaluation import evaluate_config
from franq_ext.train_router import train_router


def test_full_condition_fixes_errors_with_zero_regret():
    cfg = Config()
    examples = load_sample()
    router = train_router(cfg, examples, augment=40)
    res = evaluate_config(cfg, examples, condition="full", router=router)

    assert res.n_facts == 6
    assert res.factual_acc_after >= res.factual_acc_before
    assert res.factual_acc_after == 1.0        # all facts correct after correction
    assert res.correction_success == 1.0       # both wrong facts fixed
    assert res.correction_regret == 0.0        # nothing good was broken
    assert res.n_corrected >= 2


def test_correction_off_leaves_accuracy_unchanged():
    cfg = Config()
    cfg.correction.enabled = False
    examples = load_sample()
    router = train_router(cfg, examples, augment=40)
    res = evaluate_config(cfg, examples, condition="detect-only", router=router)

    assert res.factual_acc_after == res.factual_acc_before
    assert res.n_corrected == 0
