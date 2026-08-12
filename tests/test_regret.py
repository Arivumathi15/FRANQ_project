from franq_ext.metrics import correction_regret, is_correct
from franq_ext.scoring import LexicalAlignScorer
from franq_ext.verification import ReVerifier
from franq_ext.schema import Fact


def test_regret_zero_when_nothing_broken():
    # 1 correct before, still correct after; plus a wrong->right fix.
    assert correction_regret([(True, True), (False, True)]) == 0.0


def test_regret_one_when_the_only_correct_fact_is_broken():
    assert correction_regret([(True, False)]) == 1.0


def test_regret_zero_when_no_correct_before():
    assert correction_regret([(False, False), (False, True)]) == 0.0


def test_regret_half():
    assert correction_regret([(True, True), (True, False)]) == 0.5


def test_is_correct_normalises():
    assert is_correct(" ULM ", "Ulm")
    assert not is_correct("Munich", "Ulm")


def test_reverifier_flags_collateral_damage():
    # Simulate a bad correction: a previously-correct fact was overwritten with a wrong value.
    gold = [("Einstein", "birth year", "1879"), ("Einstein", "birth city", "Ulm")]
    before = {("einstein", "birth year"): "1879", ("einstein", "birth city"): "Munich"}
    facts = [
        Fact("Einstein", "birth year", "1899"),   # was correct (1879), now broken
        Fact("Einstein", "birth city", "Ulm"),     # was wrong (Munich), now fixed
    ]
    ev = ReVerifier(LexicalAlignScorer()).evaluate(facts, gold, before)
    assert ev.n_correct_before == 1
    assert ev.correction_regret == 1.0  # the one correct fact got damaged
