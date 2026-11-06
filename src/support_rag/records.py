"""Plain records shared by every stage: loading, retrieval, answering and evaluation.

Nothing here knows about WixQA. Downstream code depends on these records, never on the
loader, so another knowledge base only needs a new loader (D-013).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    url: str
    article_type: str  # "article", "feature_request" or "known_issue"
    contents: str  # plain text; starts with the title
    html: str  # original HTML; its headings mark where sections start (D-039)


@dataclass(frozen=True)
class Question:
    id: str
    subset: str  # "expertwritten", "simulated" or "synthetic"
    question: str
    answer: str  # reference answer (Markdown)
    gold_ids: tuple[str, ...]  # the KB articles needed to answer it

    @property
    def is_multi_article(self) -> bool:
        return len(self.gold_ids) > 1


@dataclass(frozen=True)
class Section:
    """A piece of an article with its own vector (D-039)."""

    article_id: str
    text: str  # what gets embedded and reranked: the piece, with the article's title in front


@dataclass(frozen=True)
class Hit:
    """One retrieved article. A retriever returns hits best first."""

    article_id: str
    score: float
    section: str | None = None  # the best piece's text, when the retriever searches sections
