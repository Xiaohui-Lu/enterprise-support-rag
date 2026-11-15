import pytest

from support_rag.router_report import compare_router, format_router_report
from support_rag.runs import append_result, start_run, write_summary

QUESTIONS = [f"q{i}" for i in range(10)]
# S1 gets q0-q5 right and declines q9. S3 gets q0-q4, q6, q7 and q9 right, so it gains q6, q7
# and q9, and loses q5. R2 sends q6 and q7 to S3 on a weak top score and q9 on S1's decline.
S1_RIGHT = {"q0", "q1", "q2", "q3", "q4", "q5"}
S3_RIGHT = {"q0", "q1", "q2", "q3", "q4", "q6", "q7", "q9"}
ROUTES = {"q6": "score", "q7": "score", "q9": "decline"}


def _run(tmp_path, key, strategy, settings, right, records):
    answers = start_run(
        f"answers-{key}",
        {"strategy": strategy, "split": "test", "model": "haiku", "effort": "medium",
         "milestone": "M3", **settings},
        tmp_path,
    )  # fmt: skip
    write_summary(
        answers,
        {"n": 10, "declined": 1, "mean_input_tokens": 100.0, "mean_output_tokens": 500.0,
         "cost_of_answers_usd": sum(r["cost_usd"] for r in records.values())},
    )  # fmt: skip
    for i, q in enumerate(QUESTIONS):
        append_result(
            answers,
            {"question_id": q, "subset": "expertwritten", "multi_article": i >= 8,
             "latency_ms": 1000.0, "cached": False, **records[q]},
        )  # fmt: skip
    judge = start_run(
        f"judge-{key}",
        {"kind": "judge", "strategy": strategy, "answers_run": answers.name, "model": "sonnet",
         "effort": "high", "prompt_version": "judge-v1"},
        tmp_path,
    )  # fmt: skip
    for q in QUESTIONS:
        append_result(
            judge, {"question_id": q, "correctness": "correct" if q in right else "partial"}
        )
    return judge


def _runs(tmp_path):
    s1 = {q: {"cost_usd": 0.001, "declined": q == "q9"} for q in QUESTIONS}
    s3 = {q: {"cost_usd": 0.008, "declined": False} for q in QUESTIONS}
    r2 = {}
    for q in QUESTIONS:
        taken = ROUTES.get(q, "s1")
        cost = {"s1": 0.001, "score": 0.008, "decline": 0.009}[taken]
        top = 0.5 if taken == "score" else 0.7
        r2[q] = {"cost_usd": cost, "declined": False, "cached": True,
                 "router": {"route": taken, "top_score": top, "threshold": 0.5615}}  # fmt: skip
    r2_right = (S1_RIGHT - set(ROUTES)) | (S3_RIGHT & set(ROUTES))
    dense = {"k": 10, "retriever": {"name": "dense"}}
    return [
        _run(tmp_path, "s1", "s1", dense, S1_RIGHT, s1),
        _run(tmp_path, "s3", "s3", {**dense, "k": 100}, S3_RIGHT, s3),
        _run(tmp_path, "r2", "r2",
             {**dense, "escalation_k": 100, "threshold": 0.5615, "budget": 20}, r2_right, r2),
    ]  # fmt: skip


def test_the_router_is_judged_against_s1_and_random_escalation(tmp_path):
    result = compare_router(_runs(tmp_path), runs_dir=tmp_path)
    random_check, against_s1 = result["verdicts"]
    assert against_s1["difference"] == pytest.approx(30.0)  # 9 right against 6
    # R2 sent 3 of 10 and gained 3; a random 3 gains 0.3 of the total, +2, on average.
    assert random_check["router_gain"] == 3
    assert random_check["random_gain"] == pytest.approx(0.6)
    assert random_check["difference"] == pytest.approx(24.0)
    assert result["control"]["pct_correct"] == pytest.approx(60 + 0.3 * (80 - 60))


def test_the_run_matches_a_replay_of_its_rule(tmp_path):
    result = compare_router(_runs(tmp_path), runs_dir=tmp_path)
    assert result["replay"] == {"same_route": 10, "same_grade": 10}
    assert result["routes"] == {"s1": 7, "score": 2, "decline": 1}


def test_each_signal_alone_is_replayed_from_the_same_runs(tmp_path):
    variants = {
        v["label"]: v for v in compare_router(_runs(tmp_path), runs_dir=tmp_path)["variants"]
    }
    assert variants["S1's decline only"]["escalated"] == 1
    assert variants["Weak top score only, 20% budget"]["escalated"] == 2
    assert variants["Both signals, 20% budget (R2)"]["pct_correct"] == pytest.approx(90.0)


def test_the_router_stays_out_of_m2s_verdict_family(tmp_path):
    comparison = compare_router(_runs(tmp_path), runs_dir=tmp_path)["comparison"]
    assert [c["better"] for c in comparison["against_default"]] == ["s3-k100"]


def test_the_report_says_whether_the_router_becomes_the_default(tmp_path):
    report = format_router_report(compare_router(_runs(tmp_path), runs_dir=tmp_path))
    assert "R2 becomes the default:" in report
    assert "Random escalation at R2's rate, expected" in report
