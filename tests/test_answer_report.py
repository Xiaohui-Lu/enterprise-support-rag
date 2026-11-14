import pytest

from support_rag.answer_report import compare_strategies, format_report, strategy_name
from support_rag.runs import append_result, load_config, load_summary, start_run, write_summary

QUESTIONS = [f"q{i}" for i in range(10)]  # q0-q4 expert-written; q8 and q9 multi-article


def test_s1_is_named_after_what_it_searched_and_read():
    bm25 = {"k": 10, "retriever": {"name": "bm25"}}
    dense = {"k": 5, "retriever": {"name": "dense"}, "unit": "article"}
    sections = {"k": 10, "retriever": {"name": "dense"}, "unit": "h2"}
    assert strategy_name("s1", bm25) == "S1, BM25 top 10"
    assert strategy_name("s1", dense) == "S1, dense search top 5"
    assert strategy_name("s1", sections) == "S1, dense search top 10, best h2 section"
    assert strategy_name("s0", {}) == "S0, closed book"


def _runs(
    tmp_path, right_by_strategy, effort="high", failed=(), split="dev_real", cost=None, cached=()
):
    """An answer run and a judge run per strategy; right_by_strategy: how many of the 10
    questions each gets right (the first N). "s1" is S1 on dense search, "s1-bm25" on BM25,
    and "s3-k20" is S3 with K=20. Question i takes (i + 1) seconds."""
    judge_runs = []
    for key, n_right in right_by_strategy.items():
        strategy, _, variant = key.partition("-")
        settings = {}
        if strategy == "s1":
            settings = {"k": 10, "retriever": {"name": variant or "dense"}}
        elif strategy == "s3":
            settings = {"k": int(variant.removeprefix("k")), "retriever": {"name": "dense"}}
        answers = start_run(
            f"answers-{key}",
            {
                "strategy": strategy,
                "split": split,
                "model": "haiku",
                "effort": "medium",
                "milestone": "M1" if split == "test" else None,
                **settings,
            },
            tmp_path,
        )
        per_answer = (cost or {}).get(key, 0.001)
        write_summary(
            answers,
            {"n": 10, "declined": 1, "mean_input_tokens": 100.0, "mean_output_tokens": 500.0,
             "cost_of_answers_usd": 10 * per_answer},
        )  # fmt: skip
        for i, question_id in enumerate(QUESTIONS):
            append_result(
                answers,
                {"question_id": question_id, "subset": "expertwritten" if i < 5 else "simulated",
                 "multi_article": i >= 8, "cost_usd": per_answer, "latency_ms": 1000.0 * (i + 1),
                 "cached": (key, question_id) in cached},
            )  # fmt: skip
        judge = start_run(
            f"judge-{key}",
            {"kind": "judge", "strategy": strategy, "answers_run": answers.name,
             "model": "sonnet", "effort": effort, "prompt_version": "judge-v1"},
            tmp_path,
        )  # fmt: skip
        for i, question_id in enumerate(QUESTIONS):
            grade = "correct" if i < n_right else "partial"
            if (key, question_id) in failed:
                grade = None
            append_result(judge, {"question_id": question_id, "correctness": grade})
        judge_runs.append(judge)
    return judge_runs


def test_s2_is_compared_with_the_default_s1(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s1": 6, "s2": 8}), runs_dir=tmp_path)
    (comparison,) = result["against_default"]
    assert (comparison["better"], comparison["base"]) == ("s2", "s1-dense")
    assert comparison["difference"] == 20.0


def test_s1_runs_are_kept_apart_by_retriever(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s1": 6, "s1-bm25": 4}), runs_dir=tmp_path)
    assert list(result["strategies"]) == ["s1-bm25", "s1-dense"]
    assert result["default_s1"] == "s1-dense"
    (comparison,) = result["against_default"]
    assert (comparison["better"], comparison["difference"]) == ("s1-bm25", -20.0)


def test_without_dense_search_the_only_s1_is_the_default(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s0": 2, "s1-bm25": 6}), runs_dir=tmp_path)
    assert result["default_s1"] == "s1-bm25"
    assert result["against_default"] == []
    assert [(c["better"], c["base"]) for c in result["retrieval"]] == [("s1-bm25", "s0")]


def test_replacing_the_default_needs_a_clear_win(tmp_path):
    runs = _runs(tmp_path, {"s1": 1, "s2": 10, "s3-k20": 2})
    by_strategy = {
        c["better"]: c for c in compare_strategies(runs, runs_dir=tmp_path)["against_default"]
    }
    assert by_strategy["s2"]["replaces_default"]  # 9 won, 0 lost
    assert not by_strategy["s3-k20"]["replaces_default"]  # 1 won, 0 lost: the interval touches 0


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


def test_each_s3_k_is_its_own_row_compared_with_s1(tmp_path):
    runs = _runs(tmp_path, {"s3-k100": 5, "s1": 6, "s3-k20": 7})
    result = compare_strategies(runs, runs_dir=tmp_path)
    assert list(result["strategies"]) == ["s1-dense", "s3-k20", "s3-k100"]
    assert result["strategies"]["s3-k20"]["name"] == "S3, long context, dense search up to 20"
    pairs = [(c["better"], c["base"], c["difference"]) for c in result["against_default"]]
    assert pairs == [("s3-k20", "s1-dense", 10.0), ("s3-k100", "s1-dense", -10.0)]


def test_two_runs_of_s3_at_the_same_k_are_rejected(tmp_path):
    runs = _runs(tmp_path, {"s3-k20": 5})
    with pytest.raises(ValueError, match="per K"):
        compare_strategies(runs + runs, runs_dir=tmp_path)


def test_the_report_shows_what_s3s_token_cap_did(tmp_path):
    runs = _runs(tmp_path, {"s1": 6, "s3-k100": 5})
    answers = tmp_path / load_config(runs[1])["answers_run"]
    write_summary(
        answers,
        load_summary(answers)
        | {
            "long_context": {
                "k": 100,
                "capped": 7,
                "articles_sent": {"min": 71, "median": 88.5, "max": 100},
                "max_prompt_tokens": 91_234,
                "over_price_line": 0,
                "estimate_error": {"min": -0.05, "median": 0.0, "max": 0.031},
            }
        },
    )
    report = format_report(compare_strategies(runs, runs_dir=tmp_path))
    assert "## S3's token cap" in report
    assert "| 7 of 10 | 71 / 88.5 / 100 | 91,234 | 0 | -5.0% / +0.0% / +3.1% |" in report


def test_a_report_without_s2_has_no_loop_section(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6})
    assert "S2's loop" not in format_report(compare_strategies(runs, runs_dir=tmp_path))


def test_each_strategy_gets_its_rate_and_an_interval(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s0": 2, "s1": 6}), runs_dir=tmp_path)
    s1 = result["strategies"]["s1-dense"]
    assert s1["pct_correct"] == 60.0
    assert s1["ci"][0] < 60.0 < s1["ci"][1]
    assert s1["cost_per_answer_usd"] == pytest.approx(0.001)


def test_latency_comes_from_the_answers_paid_for(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6}, cached={("s1", "q9")})
    strategies = compare_strategies(runs, runs_dir=tmp_path)["strategies"]
    assert strategies["s0"]["latency_p50_ms"] == 5500.0
    assert strategies["s0"]["latency_p95_ms"] == pytest.approx(9550.0)
    assert strategies["s1-dense"]["latency_p50_ms"] == 5000.0  # q9's 10 s was a cache hit


def test_comparisons_are_paired_on_the_same_questions(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s0": 2, "s1": 6}), runs_dir=tmp_path)
    (comparison,) = result["retrieval"]  # only S1 vs S0 has both strategies
    assert (comparison["better"], comparison["base"]) == ("s1-dense", "s0")
    assert comparison["difference"] == 40.0
    assert (comparison["only_a"], comparison["only_b"]) == (4, 0)
    assert comparison["p_holm"] == comparison["p_value"]  # one comparison: nothing to adjust


def test_a_judge_failure_drops_the_question_from_its_comparisons(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6}, failed={("s1", "q0")})
    result = compare_strategies(runs, runs_dir=tmp_path)
    assert result["strategies"]["s1-dense"]["judge_failed"] == 1
    assert result["retrieval"][0]["n"] == 9


def test_the_frontier_leaves_out_s_star_and_prices_each_step(tmp_path):
    cost = {"s0": 0.0004, "s1": 0.0012, "s_star": 0.0004}
    runs = _runs(tmp_path, {"s0": 2, "s1": 9, "s_star": 10}, cost=cost)
    frontier = compare_strategies(runs, runs_dir=tmp_path)["frontier"]["cost"]
    assert frontier["keys"] == ["s0", "s1-dense"]
    (step,) = frontier["steps"]
    assert step["extra"] == pytest.approx(0.0008)
    assert step["difference"] == 70.0
    assert step["per_extra_correct"] == pytest.approx(0.0008 / 0.7)


def test_a_step_whose_gain_may_be_zero_gets_no_price(tmp_path):
    runs = _runs(tmp_path, {"s0": 5, "s1": 6}, cost={"s0": 0.0004, "s1": 0.0012})
    result = compare_strategies(runs, runs_dir=tmp_path)
    (step,) = result["frontier"]["cost"]["steps"]
    assert step["per_extra_correct"] is None  # 1 won, 0 lost
    assert "not shown to add correct answers" in format_report(result)


def test_question_types_split_the_questions(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s0": 2, "s1": 9}), runs_dir=tmp_path)
    by_type = {t["type"]: t for t in result["by_type"]}
    assert {name: t["n"] for name, t in by_type.items()} == {
        "Single-article": 8, "Multi-article": 2, "Expert-written": 5, "Simulated": 5,
    }  # fmt: skip
    assert by_type["Single-article"]["rows"]["s1-dense"]["pct_correct"] == 100.0  # q0-q7
    assert by_type["Multi-article"]["rows"]["s1-dense"]["pct_correct"] == 50.0  # q8 of q8, q9
    assert by_type["Simulated"]["rows"]["s0"]["pct_correct"] == 0.0
    assert by_type["Simulated"]["frontier"] == ["s1-dense"]  # same cost, more accurate


def test_runs_with_different_judge_settings_are_rejected(tmp_path):
    runs = _runs(tmp_path / "a", {"s0": 2}) + _runs(tmp_path / "b", {"s1": 6}, effort="medium")
    with pytest.raises(ValueError):
        compare_strategies(runs, runs_dir=tmp_path / "a")


def test_the_report_names_the_judge_and_every_comparison(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6, "s_star": 8})
    report = format_report(compare_strategies(runs, runs_dir=tmp_path))
    assert "rubric judge-v1, effort high" in report
    assert report.count(" vs ") == 3


def test_the_report_shows_its_chart(tmp_path):
    result = compare_strategies(_runs(tmp_path, {"s0": 2, "s1": 6}), runs_dir=tmp_path)
    assert "](figures/answers_dev.svg)" in format_report(result, figure="figures/answers_dev.svg")


def test_a_dev_report_says_it_is_for_tuning(tmp_path):
    report = format_report(compare_strategies(_runs(tmp_path, {"s0": 2}), runs_dir=tmp_path))
    assert "These are dev results" in report


def test_a_test_report_names_its_milestone(tmp_path):
    runs = _runs(tmp_path, {"s0": 2, "s1": 6}, split="test")
    report = format_report(compare_strategies(runs, runs_dir=tmp_path))
    assert "milestone M1" in report
    assert "dev results" not in report
