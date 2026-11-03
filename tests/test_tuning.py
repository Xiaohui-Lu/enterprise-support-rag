import pytest

from support_rag.tuning import Arm, compare_to_default, format_report

QUESTIONS = [f"q{i}" for i in range(10)]
SUMMARY = {
    "n": 10,
    "mean_input_tokens": 100.0,
    "mean_output_tokens": 500.0,
    "cost_of_answers_usd": 0.01,
    "latency_p50_ms": 3000.0,
}
JUDGE = ("sonnet", "high", "judge-v1")


def _arm(strategy, effort="medium", k=None, right=5, judge=JUDGE):
    """The first `right` questions are correct."""
    correct = {q: i < right for i, q in enumerate(QUESTIONS)}
    return Arm(strategy, effort, k, correct, SUMMARY, judge, f"{strategy}-{effort}-{k}")


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
