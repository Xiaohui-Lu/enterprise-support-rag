import pytest

from support_rag.agreement import cohen_kappa, percent_agreement

# Ten answers: the labels say 7 correct, 2 partial, 1 incorrect. The judge swaps one
# correct and one partial, and agrees on the other eight.
LABELS = ["correct"] * 7 + ["partial", "partial", "incorrect"]
JUDGE = ["correct"] * 6 + ["partial", "correct", "partial", "incorrect"]


def test_the_worked_example():
    # p_o = 8/10. p_e = (7*7 + 2*2 + 1*1) / 100 = 0.54. kappa = 0.26 / 0.46.
    assert percent_agreement(LABELS, JUDGE) == pytest.approx(0.8)
    assert cohen_kappa(LABELS, JUDGE) == pytest.approx(0.26 / 0.46)


def test_a_judge_that_always_says_correct_scores_zero_despite_high_agreement():
    always = ["correct"] * 10
    assert percent_agreement(LABELS, always) == pytest.approx(0.7)
    assert cohen_kappa(LABELS, always) == pytest.approx(0.0)


def test_perfect_agreement_is_one():
    assert cohen_kappa(LABELS, LABELS) == pytest.approx(1.0)


def test_systematic_disagreement_is_below_zero():
    a = ["correct", "partial"] * 5
    b = ["partial", "correct"] * 5
    assert cohen_kappa(a, b) < 0


def test_kappa_is_undefined_when_both_raters_use_one_grade():
    assert cohen_kappa(["correct"] * 3, ["correct"] * 3) is None


def test_grades_for_different_items_are_rejected():
    with pytest.raises(ValueError):
        cohen_kappa(["correct"], ["correct", "partial"])
    with pytest.raises(ValueError):
        percent_agreement([], [])
