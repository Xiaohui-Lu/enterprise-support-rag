"""BM25 keyword retrieval over whole articles: the baseline retriever (D-012, D-015, D-016).

The tokenizer is ours and the scoring is bm25s's. The same tokenizer must process the
articles and the questions, or "connecting" never matches "connect".
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import bm25s
import Stemmer
from bm25s.stopwords import STOPWORDS_EN

from support_rag.records import Article, Hit

# Lucene's classic 33-word English list, the one Elasticsearch's English analyzer uses.
STOPWORDS = frozenset(STOPWORDS_EN)
_STEMMER = Stemmer.Stemmer("english")  # Snowball

Tokenizer = Callable[[str], list[str]]


def tokenize(text: str, *, stem: bool = True) -> list[str]:
    """Turn text into BM25 tokens, in order.

    1. Lowercase.
    2. Split on anything that isn't a letter or a digit, so "ADI.If" gives "adi", "if"
       and "error 404" keeps "404".
    3. Drop words in STOPWORDS.
    4. If stem is True, stem each word with _STEMMER ("connecting" -> "connect").

    Only ASCII letters count, so "café" gives "caf": fine for an English help center.
    """
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)]

    filtered_tokens = [t for t in tokens if t not in STOPWORDS]

    if stem:
        filtered_tokens = _STEMMER.stemWords(filtered_tokens)

    return filtered_tokens


class BM25Retriever:
    """Ranks whole articles against a query with BM25.

    bm25s precomputes every (token, article) score when the index is built, so a query
    only adds up a few rows. Building takes seconds; the help-center snapshot never
    changes, so it's built once per run.
    """

    def __init__(self, articles: Sequence[Article], tokenizer: Tokenizer = tokenize) -> None:
        self._ids = [article.id for article in articles]
        self._tokenizer = tokenizer
        # Defaults: the Lucene BM25 variant with k1=1.5, b=0.75. Not tuned (D-016).
        self._index = bm25s.BM25()
        # `contents` already starts with the title, so the title isn't added again.
        corpus_tokens = [tokenizer(article.contents) for article in articles]
        self._index.index(corpus_tokens, show_progress=False)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The top k articles for a query, best first. Articles scoring 0 are left out."""
        k = min(k, len(self._ids))  # bm25s refuses a k larger than the corpus
        docs, scores = self._index.retrieve([self._tokenizer(query)], k=k, show_progress=False)
        # With no matching token, every score is 0 and the order is arbitrary. Keeping
        # those would let a gold article count as "found" by luck.
        return [
            Hit(self._ids[doc], float(score))
            for doc, score in zip(docs[0], scores[0], strict=True)
            if score > 0
        ]
