import pytest

from support_rag.rot_report import compare_rot, format_rot_report
from support_rag.runs import append_result, start_run

QUESTIONS = [f"q{i}" for i in range(10)]  # q8 and q9 are multi-article


def _arm(tmp_path, name, answers_config, n_right, depth=0.5, questions=QUESTIONS):
    """An answer run and its judge run; the first `n_right` questions are graded correct."""
    answers = start_run(f"answers-{name}", {"split": "dev_real", **answers_config}, tmp_path)
    judge = start_run(
        f"judge-{name}",
        {"kind": "judge", "answers_run": answers.name, "model": "sonnet", "effort": "high",
         "prompt_version": "judge-v1"},
        tmp_path,
    )  # fmt: skip
    for i, question_id in enumerate(questions):
        multi = question_id in ("q8", "q9")
        rot = None
        if answers_config["strategy"] == "rot":
            rot = {"n_distractors": 40, "gold_depths": [depth] * (2 if multi else 1)}
        append_result(
            answers,
            {"question_id": question_id, "multi_article": multi, "gold_ids": ["G"],
             "cited_ids": ["G", "D"] if i == 0 else ["G"], "usage": {"input_tokens": 40_000},
             "cost_usd": 0.002, "rot": rot},
        )  # fmt: skip
        append_result(
            judge,
            {"question_id": question_id, "correctness": "correct" if i < n_right else "partial"},
        )
    return judge


def _rot(size, distractors="hard"):
    return {"strategy": "rot", "distractors": distractors, "size": size}


def test_each_arm_is_paired_with_the_gold_only(tmp_path):
    runs = [
        _arm(tmp_path, "gold", {"strategy": "s_star"}, 9),
        _arm(tmp_path, "hard-90k", _rot(90_000), 6),
    ]
    result = compare_rot(runs, runs_dir=tmp_path)
    (comparison,) = result["against_gold"]
    assert comparison["difference"] == -30.0
    assert (comparison["only_a"], comparison["only_b"]) == (0, 3)
    assert [row["arm"] for row in result["rows"]] == [("gold only", 0), ("hard", 90_000)]


def test_hard_and_random_are_compared_at_the_same_size(tmp_path):
    runs = [
        _arm(tmp_path, "gold", {"strategy": "s_star"}, 9),
        _arm(tmp_path, "hard-40k", _rot(40_000), 5),
        _arm(tmp_path, "random-40k", _rot(40_000, "random"), 8),
        _arm(tmp_path, "random-10k", _rot(10_000, "random"), 8),
    ]
    result = compare_rot(runs, runs_dir=tmp_path)
    (comparison,) = result["hard_vs_random"]  # 10K has no hard arm
    assert comparison["size"] == 40_000
    assert comparison["difference"] == -30.0


def test_the_gold_only_run_is_cut_to_the_pilots_questions(tmp_path):
    runs = [
        _arm(tmp_path, "gold", {"strategy": "s_star"}, 9),
        _arm(tmp_path, "hard-10k", _rot(10_000), 4, questions=QUESTIONS[:5]),
    ]
    result = compare_rot(runs, runs_dir=tmp_path)
    assert result["n_questions"] == 5
    assert result["rows"][0]["pct"] == 100.0


def test_position_is_read_from_single_article_questions_only(tmp_path):
    runs = [
        _arm(tmp_path, "gold", {"strategy": "s_star"}, 9),
        _arm(tmp_path, "hard-90k", _rot(90_000), 6, depth=0.9),
    ]
    (row,) = compare_rot(runs, runs_dir=tmp_path)["by_depth"]
    assert row["thirds"][:2] == [None, None]
    assert row["thirds"][2] == (75.0, 8)  # 6 of the 8 single-article questions


def test_a_report_needs_the_gold_only_run(tmp_path):
    with pytest.raises(ValueError, match="gold-only"):
        compare_rot([_arm(tmp_path, "hard-10k", _rot(10_000), 4)], runs_dir=tmp_path)


def test_the_report_lists_every_table(tmp_path):
    runs = [
        _arm(tmp_path, "gold", {"strategy": "s_star"}, 9),
        _arm(tmp_path, "hard-90k", _rot(90_000), 6),
        _arm(tmp_path, "random-90k", _rot(90_000, "random"), 7),
    ]
    report = format_rot_report(compare_rot(runs, runs_dir=tmp_path))
    assert "| hard, 90K vs gold only | 10 |" in report
    assert "## Hard against random, same size" in report
    assert "| random, 90K | 87.5% (8) | 0.0% (2) |" in report  # 7 right, all single-article
    assert "| hard, 90K | n/a | 75.0% (8) | n/a |" in report  # every gold at depth 0.5
