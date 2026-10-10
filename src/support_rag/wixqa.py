"""Download and load the WixQA benchmark.

The raw files stay exactly as published (JSONL) in data/raw/ and are not committed:
anyone can re-download identical files from the revision pinned in config.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from support_rag import config

QA_SUBSETS = ("expertwritten", "simulated", "synthetic")

# The dataset has no question IDs, so an ID is a prefix plus the row number in the
# pinned file, e.g. "ew-017". Stable because the revision is pinned.
_ID_FORMAT = {"expertwritten": ("ew", 3), "simulated": ("sim", 3), "synthetic": ("syn", 4)}


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    url: str
    article_type: str  # "article", "feature_request" or "known_issue"
    contents: str  # plain text
    html: str  # original HTML; keeps the headings for section-aware chunking later


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


def download(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    """Fetch the four dataset files at the pinned revision. Safe to re-run."""
    from huggingface_hub import hf_hub_download  # imported here so tests don't need it

    return [
        Path(
            hf_hub_download(
                repo_id=config.DATASET_REPO,
                repo_type="dataset",
                filename=filename,
                revision=config.DATASET_REVISION,
                local_dir=raw_dir,
            )
        )
        for filename in config.DATASET_FILES.values()
    ]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `srag download` first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_articles(raw_dir: Path = config.RAW_DIR) -> list[Article]:
    rows = _read_jsonl(raw_dir / config.DATASET_FILES["kb"])
    return [
        Article(
            id=row["id"],
            title=row.get("title", ""),
            url=row["url"],
            article_type=row["article_type"],
            contents=row["contents"],
            html=row.get("html_content", ""),
        )
        for row in rows
    ]


def load_questions(subset: str, raw_dir: Path = config.RAW_DIR) -> list[Question]:
    prefix, width = _ID_FORMAT[subset]
    rows = _read_jsonl(raw_dir / config.DATASET_FILES[subset])
    return [
        Question(
            id=f"{prefix}-{i:0{width}d}",
            subset=subset,
            question=row["question"],
            answer=row["answer"],
            gold_ids=tuple(row["article_ids"]),
        )
        for i, row in enumerate(rows)
    ]
