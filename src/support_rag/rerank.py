"""Reranking: a cross-encoder rereads a shortlist and reorders it (D-038).

Dense vectors are made without seeing the question. A cross-encoder reads the question and
one article together, so it can weigh a single sentence against that question. Nothing can
be computed ahead, though: every question takes one pass per article. So it reads only a
shortlist, and it can only reorder what's on that list.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from support_rag.records import Article, Hit

if TYPE_CHECKING:
    from support_rag.dense import SectionRetriever
    from support_rag.retrieval_eval import Retriever

CANDIDATE_DEPTH = 50  # how many articles each retriever supplies


@dataclass(frozen=True)
class RerankerModel:
    name: str  # short name for run records
    repo_id: str  # on Hugging Face
    revision: str  # full commit hash: the weights never change under us
    gpu_dtype: str = "float16"  # on the CPU it's always float32


RERANKERS = {
    model.name: model
    for model in (
        # fp16 on the GPU. On 50 articles it matched fp32 (correlation 0.99999, the same top
        # 10) and ran 3 times faster. The 8 longest articles take 7.2 GB in fp16 and 13.5 GB
        # in fp32, more than the RTX 2080 Ti's 11 GB.
        RerankerModel(
            "qwen3-reranker-0.6b",
            "Qwen/Qwen3-Reranker-0.6B",
            "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        ),
        # Small enough that fp32 is no burden. It cuts each question and article at 512 tokens.
        RerankerModel(
            "minilm-l6",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            gpu_dtype="float32",
        ),
    )
}


class PairScorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def checked_scores(scores: Iterable[float]) -> list[float]:
    """Plain Python floats, which serialize to JSON. Raises ValueError on NaN or infinity,
    which is what an fp16 overflow inside a model would produce."""
    plain = [float(score) for score in scores]
    if not all(math.isfinite(score) for score in plain):
        raise ValueError("some scores are not finite; check the model's dtype")
    return plain


class CrossEncoderScorer:
    """A local cross-encoder with its authors' default settings, untuned (D-038)."""

    def __init__(self, model: RerankerModel, *, device: str | None = None) -> None:
        # Imported here so commands that don't rerank never load PyTorch.
        import torch
        from sentence_transformers import CrossEncoder

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, model.gpu_dtype) if self.device == "cuda" else torch.float32
        self._model = CrossEncoder(
            model.repo_id,
            revision=model.revision,
            device=self.device,
            model_kwargs={"dtype": dtype},
        )
        self.params: dict[str, Any] = {
            "model": model.name,
            "repo_id": model.repo_id,
            "revision": model.revision,
            "dtype": str(dtype).removeprefix("torch."),
            "device": self.device,
            "max_tokens": self._model.max_seq_length,
            "prompts": dict(self._model.prompts),
        }

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        # Small batches: the longest articles run to about 10K tokens.
        pairs = [(query, document) for document in documents]
        return checked_scores(self._model.predict(pairs, batch_size=8, show_progress_bar=False))


def merged_candidates(rankings: Iterable[Sequence[Hit]]) -> list[str]:
    """Each article once, in the order the lists give them: the first list, then whatever
    the next one adds. A dict keeps the order its keys were first added in."""
    return list(dict.fromkeys(hit.article_id for ranking in rankings for hit in ranking))


def best_first(article_ids: Sequence[str], scores: Sequence[float]) -> list[Hit]:
    """The articles sorted by score, highest first.

    Python's sort is stable, so equal scores keep the order they came in, which is the
    retrievers' ranking. That also gives the same result on every run.

    a, b, c with scores 1, 3, 1  ->  b (3), a (1), c (1)
    """
    ranked = sorted(zip(article_ids, scores, strict=True), key=lambda pair: -pair[1])
    return [Hit(article_id, score) for article_id, score in ranked]


class RerankingRetriever:
    """Asks its retrievers for a shortlist, then has the cross-encoder reorder it."""

    def __init__(
        self,
        articles: Sequence[Article],
        retrievers: Sequence[Retriever],
        scorer: PairScorer,
        *,
        depth: int = CANDIDATE_DEPTH,
    ) -> None:
        self._texts = {article.id: article.contents for article in articles}
        self._retrievers = retrievers
        self._scorer = scorer
        self._depth = depth

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The top k articles for a query, best first."""
        rankings = [retriever.search(query, k=self._depth) for retriever in self._retrievers]
        candidates = merged_candidates(rankings)
        scores = self._scorer.score(query, [self._texts[c] for c in candidates])
        return best_first(candidates, scores)[:k]


class SectionRerankingRetriever:
    """Has the cross-encoder reorder a section retriever's top articles, reading each
    article's best section instead of the whole article (D-039)."""

    def __init__(
        self, retriever: SectionRetriever, scorer: PairScorer, *, depth: int = CANDIDATE_DEPTH
    ) -> None:
        self._retriever = retriever
        self._scorer = scorer
        self._depth = depth

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The top k articles for a query, best first."""
        hits = self._retriever.search(query, k=self._depth)
        scores = self._scorer.score(query, [hit.section for hit in hits])
        return best_first([hit.article_id for hit in hits], scores)[:k]
