import pytest

from support_rag.tuning import Arm, clears_bar, format_report
from support_rag.tuning import compare_to_default as _compare_to_default

QUESTIONS = [f"q{i}" for i in range(10)]
SUMMARY = {
    "n": 10,
    "mean_input_tokens": 100.0,
    "mean_output_tokens": 500.0,
    "cost_of_answers_usd": 0.01,
    "latency_p50_ms": 3000.0,
}
JUDGE = ("sonnet", "high", "judge-v1")


def compare_to_default(arms):
    return _compare_to_default(arms, default_effort="medium", default_k=5)


def _arm(strategy, effort="medium", k=None, right=5, judge=JUDGE, summary=SUMMARY, **extra):
    """The first `right` questions are correct."""
    correct = {q: i < right for i, q in enumerate(QUESTIONS)}
    return Arm(strategy, effort, k, correct, summary, judge, f"{strategy}-{effort}-{k}", **extra)


def _s1_report_inputs(**candidate):
    """A dense k=10 default and one candidate, both right on the first 6 questions."""
    cheap = {**SUMMARY, "mean_input_tokens": 50.0}
    default = _arm("s1", k=10, right=6, retriever="dense")
    return [default, _arm("s1", k=5, right=6, retriever="dense", summary=cheap, **candidate)]


def _compare_s1(arms, margin=10.0):
    return _compare_to_default(
        arms, default_effort="medium", default_k=10, default_retriever="dense", margin=margin
    )


def test_each_candidate_is_compared_with_its_own_strategys_default():
    arms = [_arm("s0", right=2), _arm("s0", "high", right=4), _arm("s1", k=5, right=6)]
    result = compare_to_default(arms)
    (comparison,) = result["comparisons"]
    assert (comparison["strategy"], comparison["setting"]) == ("s0", "effort high")
    assert comparison["difference"] == 20.0
    assert (comparison["only_a"], comparison["only_b"]) == (2, 0)


def test_k_is_part_of_s1s_setting():
    arms = [_arm("s1", k=5, right=6), _arm("s1", k=3, right=5)]
    (comparison,) = compare_to_default(arms)["comparisons"]
    assert comparison["setting"] == "effort medium, k=3"
    assert comparison["difference"] == -10.0


def test_an_effort_tried_on_several_strategies_is_pooled_per_question():
    arms = [
        _arm("s0", right=2),
        _arm("s1", k=5, right=6),
        _arm("s0", "low", right=4),  # +2 questions
        _arm("s1", "low", k=5, right=6),  # no change
    ]
    (pooled,) = compare_to_default(arms)["pooled"]
    assert pooled["setting"] == "effort low"
    assert pooled["n"] == 10  # questions, not question-strategy pairs
    assert pooled["difference"] == pytest.approx(10.0)  # (2 + 0) / 2 strategies / 10


def test_a_strategy_without_its_default_is_rejected():
    with pytest.raises(ValueError):
        compare_to_default([_arm("s0", "high")])


def test_different_judge_settings_are_rejected():
    arms = [_arm("s0"), _arm("s0", "high", judge=("sonnet", "medium", "judge-v1"))]
    with pytest.raises(ValueError):
        compare_to_default(arms)


def test_the_report_marks_the_default():
    report = format_report("effort", compare_to_default([_arm("s0"), _arm("s0", "low")]))
    assert "effort medium (default)" in report
    assert "| S0, closed book | effort low | 10 |" in report


# clears_bar()


@pytest.mark.parametrize(
    ("difference", "low", "passes"),
    [
        (0.0, -8.0, True),  # 2 fixed, 2 broken
        (4.0, -4.0, True),  # 3 fixed, 1 broken
        (0.0, -10.0, False),  # on the line isn't above it
        (-4.0, -10.0, False),  # 0 fixed, 2 broken
        (-4.0, -12.0, False),  # 1 fixed, 3 broken
        (-2.0, -6.0, False),  # a narrow interval, but it measured lower
    ],
)
def test_a_cheaper_candidate_must_match_the_default_and_stay_within_the_margin(
    difference, low, passes
):
    assert clears_bar(difference, low, cheaper=True, margin=10.0) is passes


def test_floating_point_dust_cannot_lift_a_bound_over_the_margin():
    assert not clears_bar(0.0, -9.9999999999, cheaper=True, margin=10.0)


def test_a_candidate_that_isnt_cheaper_must_be_clearly_better():
    assert not clears_bar(4.0, -4.0, cheaper=False, margin=10.0)
    assert not clears_bar(6.0, 0.0, cheaper=False, margin=10.0)
    assert clears_bar(8.0, 2.0, cheaper=False, margin=10.0)


def test_without_a_margin_a_cheaper_candidate_must_be_clearly_better_too():
    assert not clears_bar(4.0, -4.0, cheaper=True, margin=None)
    assert clears_bar(8.0, 2.0, cheaper=True, margin=None)


# S1 arms that differ in retriever or in what they read


def test_two_s1_arms_at_the_same_k_are_told_apart_by_retriever():
    arms = [
        _arm("s1", k=10, right=6, retriever="dense"),
        _arm("s1", k=10, right=5, retriever="bm25"),
    ]
    (comparison,) = _compare_s1(arms)["comparisons"]
    assert comparison["setting"] == "effort medium, k=10, BM25"
    assert comparison["difference"] == -10.0


def test_a_section_arm_is_a_candidate_even_at_the_default_k():
    arms = [
        _arm("s1", k=10, right=6, retriever="dense"),
        _arm("s1", k=10, right=6, retriever="dense", unit="h2"),
    ]
    (comparison,) = _compare_s1(arms)["comparisons"]
    assert comparison["setting"] == "effort medium, k=10, dense search, best h2 section"


def test_arms_at_the_default_effort_are_never_pooled():
    arms = [
        _arm("s1", k=10, right=6, retriever="dense"),
        _arm("s1", k=10, right=5, retriever="bm25"),
        _arm("s1", k=10, right=5, retriever="dense", unit="h2"),
    ]
    assert _compare_s1(arms)["pooled"] == []


def test_the_margin_applies_only_to_a_cheaper_candidate():
    (cheaper,) = _compare_s1(_s1_report_inputs())["comparisons"]
    assert cheaper["clears_bar"] is True
    same_cost = [
        _arm("s1", k=10, right=6, retriever="dense"),
        _arm("s1", k=5, right=6, retriever="dense"),
    ]
    (not_cheaper,) = _compare_s1(same_cost)["comparisons"]
    assert not_cheaper["clears_bar"] is False


def test_the_report_states_the_margin_and_each_verdict():
    report = format_report("s1", _compare_s1(_s1_report_inputs()))
    assert "entirely above -10 points" in report
    assert "(D-042)" in report
    assert "Clears its bar" in report
    assert report.count("| yes |") == 1


def test_a_report_without_a_margin_keeps_d033s_rule_and_no_verdict_column():
    report = format_report("effort", compare_to_default([_arm("s0"), _arm("s0", "low")]))
    assert "(D-033)" in report
    assert "Clears its bar" not in report


# Answers citing a feature request that isn't gold


def test_only_not_correct_answers_count_as_misled_by_a_feature_request():
    # q0 to q5 are correct, q6 to q9 aren't.
    cites = {"q0": True, "q6": True, "q7": True, "q8": False}
    arm = _arm("s1", k=10, right=6, retriever="dense", cites_feature_request=cites)
    (row,) = _compare_s1([arm])["feature_requests"]
    assert (row["not_correct"], row["not_correct_citing"], row["correct_citing"]) == (4, 2, 1)


def test_a_failed_judge_call_counts_as_neither():
    correct = {"q0": None, "q1": False}
    cites = {"q0": True, "q1": True}
    arm = Arm("s1", "medium", 10, correct, SUMMARY, JUDGE, "r", "dense", "article", cites)
    (row,) = _compare_s1([arm])["feature_requests"]
    assert (row["not_correct"], row["not_correct_citing"], row["correct_citing"]) == (1, 1, 0)


def test_the_feature_request_section_appears_only_when_it_was_counted():
    without = format_report("s1", _compare_s1(_s1_report_inputs()))
    assert "feature request" not in without
    cites = {"q6": True}
    arms = [_arm("s1", k=10, right=6, retriever="dense", cites_feature_request=cites)]
    assert "isn't gold" in format_report("s1", _compare_s1(arms))
