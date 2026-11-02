from support_rag.retrieval_report import paired_changes, write_report
from support_rag.runs import append_result, start_run, write_summary


def _result(question_id, subset, recall_at_5, mrr):
    return {
        "question_id": question_id,
        "subset": subset,
        "metrics": {"recall@5": recall_at_5, "mrr@20": mrr},
    }


def test_paired_changes_compares_the_same_questions():
    base = [_result("q1", "synthetic", 1.0, 1.0), _result("q2", "synthetic", 0.0, 0.5)]
    other = [_result("q2", "synthetic", 0.0, 1.0), _result("q1", "synthetic", 1.0, 0.25)]
    assert paired_changes(base, other, "mrr@20") == {"better": 1, "worse": 1, "same": 0}
    assert paired_changes(base, other, "recall@5") == {"better": 0, "worse": 0, "same": 2}


def _run(tmp_path, name, stem, results):
    run_dir = start_run(
        name,
        {"split": "dev", "n_questions": len(results), "tokenizer": {"stem": stem}},
        runs_dir=tmp_path / "runs",
    )
    for result in results:
        append_result(run_dir, result)
    write_summary(run_dir, {"all": {"n": len(results), "recall@5": 0.5, "mrr@20": 0.5}})
    return run_dir


def test_report_lists_runs_and_compares_later_runs_with_the_first(tmp_path):
    base = _run(tmp_path, "stem", True, [_result("ew-000", "expertwritten", 1.0, 1.0)])
    other = _run(tmp_path, "nostem", False, [_result("ew-000", "expertwritten", 0.0, 0.5)])
    text = write_report([base, other], path=tmp_path / "report.md").read_text(encoding="utf-8")
    assert base.name in text and other.name in text
    assert "| real | mrr@20 | 0 | 1 | 0 |" in text
