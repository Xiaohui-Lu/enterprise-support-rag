import json
from pathlib import Path

import pytest

from support_rag import config
from support_rag.wixqa import load_articles, load_questions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_question_ids_follow_row_order(tmp_path):
    rows = [
        {"question": "q0", "answer": "a", "article_ids": ["k1"]},
        {"question": "q1", "answer": "a", "article_ids": ["k1", "k2"]},
    ]
    write_jsonl(tmp_path / config.DATASET_FILES["expertwritten"], rows)

    questions = load_questions("expertwritten", tmp_path)

    assert [q.id for q in questions] == ["ew-000", "ew-001"]
    assert questions[1].gold_ids == ("k1", "k2")
    assert questions[1].is_multi_article
    assert not questions[0].is_multi_article


def test_synthetic_ids_have_four_digits(tmp_path):
    row = {"question": "q", "answer": "a", "article_ids": ["k1"]}
    write_jsonl(tmp_path / config.DATASET_FILES["synthetic"], [row])

    assert load_questions("synthetic", tmp_path)[0].id == "syn-0000"


def test_articles_keep_their_html(tmp_path):
    row = {
        "id": "k1",
        "title": "Title",
        "url": "https://example.com/k1",
        "article_type": "article",
        "contents": "plain text",
        "html_content": "<h2>Title</h2>",
    }
    write_jsonl(tmp_path / config.DATASET_FILES["kb"], [row])

    (article,) = load_articles(tmp_path)

    assert article.html == "<h2>Title</h2>"
    assert article.contents == "plain text"


def test_missing_file_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="srag download"):
        load_questions("simulated", tmp_path)
