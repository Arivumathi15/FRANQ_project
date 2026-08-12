from franq_ext.config import Config
from franq_ext.data import load_sample
from franq_ext.pipeline import Pipeline
from franq_ext.train_router import train_router


def _einstein():
    return next(e for e in load_sample() if e.qid == "einstein")


def test_full_pipeline_corrects_and_has_zero_regret():
    cfg = Config()
    examples = load_sample()
    router = train_router(cfg, examples, augment=40)
    out = Pipeline(cfg, router=router).run(_einstein())

    city = next(f for f in out.facts if f.attribute == "birth city")
    assert city.value == "Ulm" and city.corrected          # Munich -> Ulm
    assert out.result.correction_regret == 0.0             # nothing good was broken
    assert out.result.n_correct_after == 2                 # both facts correct now
    assert out.result.n_correct_before == 1


def test_disabling_correction_leaves_hallucination_in_place():
    cfg = Config()
    cfg.correction.enabled = False
    examples = load_sample()
    router = train_router(cfg, examples, augment=40)
    out = Pipeline(cfg, router=router).run(_einstein())

    city = next(f for f in out.facts if f.attribute == "birth city")
    assert city.value == "Munich" and not city.corrected   # left uncorrected
    assert out.result.n_corrected == 0


def test_pipeline_is_deterministic():
    cfg = Config()
    examples = load_sample()
    r1 = Pipeline(cfg, router=train_router(cfg, examples, augment=40)).run(_einstein())
    r2 = Pipeline(cfg, router=train_router(cfg, examples, augment=40)).run(_einstein())
    assert r1.result.correction_regret == r2.result.correction_regret
    assert [f.value for f in r1.facts] == [f.value for f in r2.facts]
