import pytest

from support_rag.answer_report import compare_strategies, format_report, strategy_name
from support_rag.runs import append_result, load_config, load_summary, start_run, write_summary

QUESTIONS = [f"q{i}" for i in range(10)]


def test_s1_is_named_after_what_it_searched_and_read():
    bm25 = {"k": 10, "retriever": {"name": "bm25"}}
    dense = {"k": 5, "retriever": {"name": "dense"}, "unit": "article"}
    sections = {"k": 10, "retriever": {"name": "dense"}, "unit": "h2"}
    assert strategy_name("s1", bm25) == "S1, BM25 top 10"
    assert strategy_name("s1", dense) == "S1, dense search top 5"
    assert strategy_name("s1", sections) == "S1, dense search top 10, best h2 section"
    assert strategy_name("s0", {}) == "S0, closed book"


def _runs(tmp_path, right_by_strategy, effort="high", failed=(), split="dev_real"):
    """An answer run and a judge run per strategy; right_by_strategy: how many of the 10
    questions each gets right (the first N)."""
    judge_runs = []
    for strategy, n_right in right_by_strategy.items():
        answers = start_run(
            f"answers-{strategy}",
            {
                "strategy": strategy,
                "split": split,
                "model": "haiku",
                "effort": "medium",
                "milestone": "M1" if split == "test" else None,
            },
            tmp_path,
        )
        write_summary(
            answers,
            {"n": 10, "declined": 1, "mean_input_tokens": 100.0, "mean_output_tokens": 500.0,
             "cost_of_answers_usd": 0.01, "latency_p50_ms": 3000.0},
        )  # fmt: skip
        judge = start_run(
            f"judge-{strategy}",
            {"kind": "judge", "strategy": strategy, "answers_run": answers.name,
             "model": "sonnet", "effort": effort, "prompt_version": "judge-v1"},
            tmp_path,
        )  # fmt: skip
        for i, question_id in enumerate(QUESTIONS):
            grade = "correct" if i < n_right else "partial"
            if (strategy, question_id) in failed:
                grade = None
            append_result(judge, {"question_id": question_id, "correctness": grade})
        judge_runs.append(judge)
    return judge_runs


def test_s2_is_compared_with_s1(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s1": 6, "s2": 8}), runs_dir=tmp_path)
    (comparison,) = result["comparisons"]
    assert (comparison["better"], comparison["base"]) == ("s2", "s1")
    assert comparison["difference"] == 20.0


def test_the_report_shows_how_s2s_loops_went(tmp_path):
    runs = _runs(tmp_path, {"s1": 6, "s2": 8})
    answers = tmp_path / load_config(runs[1])["answers_run"]
    write_summary(
        answers,
        load_summary(answers)
        | {
            "agent": {
                "steps": {"2": 7, "6": 3},
                "stopped_by": {"steps": 3},
                "mean_searches": 1.5,
                "mean_reads": 2.5,
                "tool_errors": 1,
                "mean_largest_prompt_tokens": 6000.0,
                "mean_cached_input_tokens": 80.0,
                "cost_without_caching_usd": 0.02,
            }
        },
    )
    report = format_report(compare_strategies(runs, runs_dir=tmp_path))
    assert "## S2's loop" in report
    assert "7 took 2, 3 took 6" in report
    assert "steps: 3" in report
    assert "without caching" in report


def test_a_report_without_s2_has_no_loop_section(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6})
    assert "S2's loop" not in format_report(compare_strategies(runs, runs_dir=tmp_path))


def test_each_strategy_gets_its_rate_and_an_interval(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s0": 2, "s1": 6}), runs_dir=tmp_path)
    s1 = result["strategies"]["s1"]
    assert s1["pct_correct"] == 60.0
    assert s1["ci"][0] < 60.0 < s1["ci"][1]
    assert s1["cost_per_answer_usd"] == pytest.approx(0.001)


def test_comparisons_are_paired_on_the_same_questions(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s0": 2, "s1": 6}), runs_dir=tmp_path)
    (comparison,) = result["comparisons"]  # only S1 vs S0 has both strategies
    assert (comparison["better"], comparison["base"]) == ("s1", "s0")
    assert comparison["difference"] == 40.0
    assert (comparison["only_a"], comparison["only_b"]) == (4, 0)
    assert comparison["p_holm"] == comparison["p_value"]  # one comparison: nothing to adjust


def test_a_judge_failure_drops_the_question_from_its_comparisons(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6}, failed={("s1", "q0")})
    result = compare_strategies(runs, runs_dir=tmp_path)
    assert result["strategies"]["s1"]["judge_failed"] == 1
    assert result["comparisons"][0]["n"] == 9


def test_runs_with_different_judge_settings_are_rejected(tmp_path):
    runs = _runs(tmp_path / "a", {"s0": 2}) + _runs(tmp_path / "b", {"s1": 6}, effort="medium")
    with pytest.raises(ValueError):
        compare_strategies(runs, runs_dir=tmp_path / "a")


def test_the_report_names_the_judge_and_every_comparison(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6, "s_star": 8})
    report = format_report(compare_strategies(runs, runs_dir=tmp_path))
    assert "rubric judge-v1, effort high" in report
    assert report.count(" vs ") == 3


def test_a_dev_report_says_it_is_for_tuning(tmp_path):
    report = format_report(compare_strategies(_runs(tmp_path, {"s0": 2}), runs_dir=tmp_path))
    assert "These are dev results" in report


def test_a_test_report_names_its_milestone(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6}, split="test")
    report = format_report(compare_strategies(runs, runs_dir=tmp_path))
    assert "milestone M1" in report
    assert "dev results" not in report
