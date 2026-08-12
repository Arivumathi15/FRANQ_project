from franq_ext.data import load_sample
from franq_ext.scoring import LexicalAlignScorer


def _ctx(qid):
    return next(e for e in load_sample() if e.qid == qid).contexts


def test_hallucinated_city_scores_low():
    s = LexicalAlignScorer()
    # Evidence explicitly denies Munich as the birthplace.
    assert s.score_claim("Einstein", "birth city", "Munich", _ctx("einstein")) < 0.5


def test_correct_year_scores_high():
    s = LexicalAlignScorer()
    assert s.score_claim("Einstein", "birth year", "1879", _ctx("einstein")) >= 0.5


def test_correct_city_scores_high():
    s = LexicalAlignScorer()
    assert s.score_claim("Curie", "birth city", "Warsaw", _ctx("curie")) >= 0.5


def test_absent_value_is_not_confidently_supported():
    s = LexicalAlignScorer()
    # 1642 is wrong (evidence says 1643) and never appears -> low support.
    assert s.score_claim("Newton", "birth year", "1642", _ctx("newton")) < 0.5
