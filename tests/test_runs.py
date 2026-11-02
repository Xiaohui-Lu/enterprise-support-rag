import pytest

from support_rag import config
from support_rag.runs import (
    append_result,
    git_state,
    load_config,
    load_results,
    load_summary,
    start_run,
    write_summary,
)


def test_start_run_records_provenance(tmp_path):
    run_dir = start_run("bm25-stem_dev", {"split": "dev"}, runs_dir=tmp_path)
    saved = load_config(run_dir)
    assert run_dir.name.endswith("_bm25-stem_dev")
    assert saved["run_id"] == run_dir.name
    assert saved["split"] == "dev"
    assert saved["dataset_revision"] == config.DATASET_REVISION
    assert set(saved["git"]) == {"commit", "dirty"}
    assert {"python", "bm25s", "pystemmer"} <= set(saved["versions"])


def test_an_existing_run_is_never_overwritten(tmp_path, monkeypatch):
    (tmp_path / "20261031-120000_same").mkdir()

    class FrozenClock:
        @staticmethod
        def now():
            from datetime import datetime

            return datetime(2026, 10, 31, 12, 0, 0)

    monkeypatch.setattr("support_rag.runs.datetime", FrozenClock)
    with pytest.raises(FileExistsError):
        start_run("same", {}, runs_dir=tmp_path)


def test_results_are_appended_one_line_per_question(tmp_path):
    append_result(tmp_path, {"question_id": "ew-000", "metrics": {"recall@5": 1.0}})
    append_result(tmp_path, {"question_id": "ew-007", "metrics": {"recall@5": 0.0}})
    lines = (tmp_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [r["question_id"] for r in load_results(tmp_path)] == ["ew-000", "ew-007"]


def test_summary_round_trips(tmp_path):
    write_summary(tmp_path, {"all": {"n": 2, "recall@5": 0.5}})
    assert load_summary(tmp_path) == {"all": {"n": 2, "recall@5": 0.5}}


def test_git_state_is_empty_outside_a_repository(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None}
