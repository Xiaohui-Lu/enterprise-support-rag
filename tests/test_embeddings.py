import re
from pathlib import Path

import numpy as np
import pytest

from support_rag.embeddings import MODELS, article_vectors, corpus_fingerprint, unit_rows
from support_rag.records import Article

MODEL = MODELS["qwen3-0.6b"]


def _article(article_id, contents):
    return Article(article_id, contents, "", "article", contents, "")


class CountingEncoder:
    """Embeds each text as (length, 1), and counts how often it's asked to."""

    def __init__(self):
        self.calls = 0

    def encode_documents(self, texts):
        self.calls += 1
        return unit_rows(np.array([[len(t), 1.0] for t in texts]))

    def encode_queries(self, texts):
        return self.encode_documents(texts)


def test_every_model_is_pinned_to_a_full_commit():
    for model in MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision), model.name


def test_unit_rows_have_length_one():
    rows = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)
    assert rows.dtype == np.float32
    assert rows[0] == pytest.approx([0.6, 0.8])


def test_article_vectors_are_computed_once_and_then_loaded(tmp_path):
    articles = [_article("a", "short"), _article("b", "a longer text")]
    encoder = CountingEncoder()
    first, stored_first = article_vectors(articles, MODEL, encoder, vectors_dir=tmp_path)
    second, stored_second = article_vectors(articles, MODEL, encoder, vectors_dir=tmp_path)
    assert encoder.calls == 1
    assert (stored_first, stored_second) == (False, True)
    assert np.array_equal(first, second)
    assert first.shape == (2, 2)


def test_changed_articles_get_new_vectors(tmp_path):
    encoder = CountingEncoder()
    article_vectors([_article("a", "old text")], MODEL, encoder, vectors_dir=tmp_path)
    article_vectors([_article("a", "new text")], MODEL, encoder, vectors_dir=tmp_path)
    assert encoder.calls == 2


def test_the_fingerprint_depends_on_article_order():
    a, b = _article("a", "one"), _article("b", "two")
    assert corpus_fingerprint([a, b]) != corpus_fingerprint([b, a])


def test_the_cli_offers_every_embedding_model():
    from support_rag import cli

    source = Path(cli.__file__).read_text(encoding="utf-8")
    for name in MODELS:
        assert f'"{name}"' in source
