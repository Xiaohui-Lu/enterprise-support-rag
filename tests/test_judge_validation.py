import pytest

from support_rag.judge_validation import (
    agreement_row,
    final_labels,
    format_report,
    validate_judge,
)
from support_rag.runs import append_result, start_run

SAMPLE = {
    "answer_runs": {"s0": {"run_id": "answers-s0"}, "s1": {"run_id": "answers-s1"}},
    "items": [
        {"item": 1, "question_id": "q1", "strategy": "s0"},
        {"item": 2, "question_id": "q1", "strategy": "s1"},
        {"item": 3, "question_id": "q2", "strategy": "s0"},
        {"item": 4, "question_id": "q2", "strategy": "s1"},
    ],
}
PANEL = {
    "items": [
        {"item": 1, "correctness": "correct"},
        {"item": 2, "correctness": None},
        {"item": 3, "correctness": "partial"},
        {"item": 4, "correctness": None},
    ]
}
HAND = {2: {"correctness": "incorrect"}}


def _judge_run(tmp_path, strategy, effort, grades, answers_run=None):
    run_config = {
        "kind": "judge",
        "strategy": strategy,
        "effort": effort,
        "answers_run": answers_run or f"answers-{strategy}",
    }
    run_dir = start_run(f"judge-{strategy}-{effort}", run_config, tmp_path)
    for question_id, grade in grades.items():
        record = {"question_id": question_id, "correctness": grade, "judged": True}
        append_result(run_dir, record | {"cost_usd": 0.01, "usage": {"output_tokens": 200}})
    return run_dir


def test_a_hand_label_wins_and_unsettled_items_without_one_are_left_out():
    labels = final_labels(PANEL, HAND)
    assert labels == {1: ("correct", "panel"), 2: ("incorrect", "hand"), 3: ("partial", "panel")}


def test_correct_vs_not_merges_partial_with_incorrect():
    row = agreement_row([("partial", "incorrect"), ("correct", "correct")])
    assert row["agreement"] == 0.5
    assert row["kappa_correct_vs_not"] == pytest.approx(1.0)


def test_each_item_is_matched_to_the_judges_verdict_on_its_own_answer(tmp_path):
    runs = [
        _judge_run(tmp_path, "s0", "high", {"q1": "correct", "q2": "correct"}),
        _judge_run(tmp_path, "s1", "high", {"q1": "incorrect", "q2": "correct"}),
    ]
    result = validate_judge(runs, SAMPLE, final_labels(PANEL, HAND))["high"]
    assert result["groups"]["all"]["n"] == 3
    assert result["groups"]["s0"]["agreement"] == 0.5  # item 3: partial vs correct
    assert result["groups"]["hand"]["agreement"] == 1.0  # item 2
    assert result["confusion"] == {"correct -> correct": 1, "incorrect -> incorrect": 1,
                                   "partial -> correct": 1}  # fmt: skip


def test_a_judge_run_over_other_answers_is_rejected(tmp_path):
    runs = [
        _judge_run(tmp_path, "s0", "high", {"q1": "correct"}, answers_run="someone-else"),
        _judge_run(tmp_path, "s1", "high", {"q1": "correct"}),
    ]
    with pytest.raises(ValueError):
        validate_judge(runs, SAMPLE, final_labels(PANEL, HAND))


def test_every_effort_needs_every_strategy(tmp_path):
    runs = [_judge_run(tmp_path, "s0", "medium", {"q1": "correct", "q2": "correct"})]
    with pytest.raises(ValueError):
        validate_judge(runs, SAMPLE, final_labels(PANEL, HAND))


def test_a_judge_failure_is_counted_not_graded(tmp_path):
    runs = [
        _judge_run(tmp_path, "s0", "high", {"q1": None, "q2": "partial"}),
        _judge_run(tmp_path, "s1", "high", {"q1": "incorrect", "q2": "correct"}),
    ]
    result = validate_judge(runs, SAMPLE, final_labels(PANEL, HAND))["high"]
    assert result["judge_failed"] == 1
    assert result["groups"]["all"]["n"] == 2


def test_the_report_has_a_row_per_effort(tmp_path):
    grades = {"q1": "correct", "q2": "partial"}
    runs = [
        _judge_run(tmp_path, strategy, effort, grades)
        for effort in ("high", "medium")  # given out of order on purpose
        for strategy in ("s0", "s1")
    ]
    labels = final_labels(PANEL, HAND)
    report = format_report(validate_judge(runs, SAMPLE, labels), labels)
    assert "| medium | 3 |" in report and "| high | 3 |" in report
    assert report.index("| medium |") < report.index("| high |")
