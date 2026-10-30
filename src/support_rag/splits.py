"""Frozen dev/test splits (docs/experiment-design.md, section 4).

The split is generated once, committed to git and never regenerated, so the test set
stays the same even if this code or Python's random module changes later.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from support_rag import config
from support_rag.records import Question
from support_rag.wixqa import load_questions

SEED = 20261010
DEV_REAL_PER_SUBSET = 25
DEV_SYNTHETIC = 500


def stratified_sample(questions: list[Question], n: int, rng: random.Random) -> list[Question]:
    """Pick n questions, keeping the same share of multi-article questions as the input."""
    multi = [q for q in questions if q.is_multi_article]
    single = [q for q in questions if not q.is_multi_article]
    n_multi = round(n * len(multi) / len(questions))
    return rng.sample(multi, n_multi) + rng.sample(single, n - n_multi)


def make_splits(
    expertwritten: list[Question],
    simulated: list[Question],
    synthetic: list[Question],
    seed: int = SEED,
) -> dict:
    rng = random.Random(seed)

    dev_real = stratified_sample(expertwritten, DEV_REAL_PER_SUBSET, rng)
    dev_real += stratified_sample(simulated, DEV_REAL_PER_SUBSET, rng)
    dev_real_ids = {q.id for q in dev_real}
    test = [q for q in expertwritten + simulated if q.id not in dev_real_ids]

    dev_synthetic_ids = {q.id for q in rng.sample(synthetic, DEV_SYNTHETIC)}
    reference = [q for q in synthetic if q.id not in dev_synthetic_ids]

    return {
        "version": 1,
        "seed": seed,
        "dataset_revision": config.DATASET_REVISION,
        "counts": {
            "dev_real": len(dev_real_ids),
            "dev_real_multi_article": sum(q.is_multi_article for q in dev_real),
            "dev_synthetic": len(dev_synthetic_ids),
            "test": len(test),
            "test_multi_article": sum(q.is_multi_article for q in test),
            "synthetic_reference": len(reference),
        },
        "dev_real": sorted(dev_real_ids),
        "dev_synthetic": sorted(dev_synthetic_ids),
        "test": sorted(q.id for q in test),
        "synthetic_reference": sorted(q.id for q in reference),
    }


def write_splits(
    path: Path = config.SPLITS_PATH, raw_dir: Path = config.RAW_DIR, force: bool = False
) -> dict:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. The test set is locked; use --force only if you mean "
            "to re-split, and record why in the results notes."
        )
    splits = make_splits(
        load_questions("expertwritten", raw_dir),
        load_questions("simulated", raw_dir),
        load_questions("synthetic", raw_dir),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return splits


def load_splits(path: Path = config.SPLITS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
