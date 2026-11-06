import pytest

from support_rag.retrieval_report import (
    describe,
    paired_changes,
    paired_scores,
    verdict,
    write_report,
)
from support_rag.runs import append_result, start_run, write_summary

METRICS = ("recall@10", "all_gold@10", "recall@5", "mrr@20")


def _result(question_id, subset, score, mrr=None):
    metrics = dict.fromkeys(METRICS, score)
    metrics["mrr@20"] = score if mrr is None else mrr
    return {"question_id": question_id, "subset": subset, "metrics": metrics}


def test_paired_changes_compares_the_same_questions():
    base = [_result("q1", "synthetic", 1.0, 1.0), _result("q2", "synthetic", 0.0, 0.5)]
    other = [_result("q2", "synthetic", 0.0, 1.0), _result("q1", "synthetic", 1.0, 0.25)]
    assert paired_changes(base, other, "mrr@20") == {"better": 1, "worse": 1, "same": 0}
    assert paired_changes(base, other, "recall@5") == {"better": 0, "worse": 0, "same": 2}


def test_paired_scores_refuses_runs_over_different_questions():
    one, other = [_result("q1", "simulated", 1.0)], [_result("q2", "simulated", 1.0)]
    with pytest.raises(ValueError):
        paired_scores(one, other, "mrr@20")


def test_only_an_interval_that_excludes_zero_counts():
    assert verdict(0.02, 0.16) == "clearly better"
    assert verdict(-0.04, 0.24) == "within noise"
    assert verdict(-0.20, -0.01) == "clearly worse"


def test_describe_names_the_retriever_and_its_main_setting():
    bm25 = {"retriever": {"name": "bm25"}, "tokenizer": {"stem": False}}
    dense = {"retriever": {"name": "dense", "model": "m", "dimensions": 8, "dtype": "float16"}}
    hybrid = {
        "retriever": {
            "name": "hybrid",
            "fusion": "rrf",
            "candidate_depth": 100,
            "dense": {"model": "m"},
        }
    }
    assert describe(bm25) == "BM25, stemming off"
    assert describe(dense) == "dense: m, 8 dims, float16"
    assert describe(hybrid) == "hybrid: rrf, BM25 + m, top 100 of each"


@pytest.mark.parametrize(
    ("candidates", "shortlist"),
    [("dense", "m, top 50"), ("both", "BM25 + m, top 50 of each")],
)
def test_describe_says_what_a_reranker_read(candidates, shortlist):
    rerank = {
        "retriever": {
            "name": "rerank",
            "reranker": {"model": "r", "dtype": "float16"},
            "candidates": candidates,
            "candidate_depth": 50,
            "dense": {"model": "m"},
        }
    }
    assert describe(rerank) == f"rerank: r, float16, over {shortlist}"


@pytest.mark.parametrize(
    ("title", "ending"),
    [(True, "h2 (1,234 sections)"), (False, "h2 (1,234 sections, no titles)")],
)
def test_describe_says_how_a_dense_run_was_cut(title, ending):
    retriever = {"name": "dense", "model": "m", "dimensions": 8, "dtype": "fp", "title": title}
    run = {"retriever": retriever, "unit": "h2", "n_sections": 1234}
    assert describe(run) == f"dense: m, 8 dims, fp, {ending}"


def test_describe_says_a_reranker_read_sections():
    rerank = {
        "retriever": {
            "name": "rerank",
            "reranker": {"model": "r", "dtype": "fp"},
            "candidates": "dense",
            "candidate_depth": 50,
            "dense": {"model": "m"},
        },
        "unit": "h2",
    }
    assert describe(rerank) == "rerank: r, fp, over m, best h2 section of the top 50 articles"


def _run(tmp_path, name, results):
    run_dir = start_run(
        name,
        {"split": "dev", "n_questions": len(results), "retriever": {"name": name}},
        runs_dir=tmp_path / "runs",
    )
    for result in results:
        append_result(run_dir, result)
    write_summary(run_dir, {"all": {"n": len(results), **dict.fromkeys(METRICS, 0.5)}})
    return run_dir


def _real(scores):
    return [_result(f"ew-{i:03}", "expertwritten", s) for i, s in enumerate(scores)]


def test_report_compares_each_run_with_every_earlier_run(tmp_path):
    runs = [_run(tmp_path, name, _real([1.0, 0.0])) for name in ("a", "b", "c")]
    text = write_report(runs, name="x", path=tmp_path / "report.md").read_text(encoding="utf-8")
    for later, earlier in ((1, 0), (2, 0), (2, 1)):
        assert f"### `{runs[later].name}` vs `{runs[earlier].name}`" in text


def test_report_applies_the_decision_rule_to_real_recall_at_10(tmp_path):
    base = _run(tmp_path, "base", _real([0.0] * 5 + [1.0] * 5))
    fixes_all = _run(tmp_path, "fixes", _real([1.0] * 10))
    swaps = _run(tmp_path, "swaps", _real([1.0] * 5 + [0.0] * 5))
    path = tmp_path / "report.md"
    text = write_report([base, fixes_all, swaps], name="x", path=path).read_text(encoding="utf-8")
    row = "| real | recall@10 | 0.500 | 1.000 | +0.500 [+0.200, +0.800] | 5 | 0 | 5 | 0.062 |"
    assert row in text
    assert "**clearly better**" in text
    assert "**within noise**" in text
