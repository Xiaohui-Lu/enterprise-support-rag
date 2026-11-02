"""Project-wide constants: where the data comes from and where files go."""

import os
from pathlib import Path

# The dataset is pinned to one Hugging Face commit, so every result in this repo comes
# from exactly these files. Change it only on purpose, and re-run everything if you do.
DATASET_REPO = "Wix/WixQA"
DATASET_REVISION = "d662dc42479c14e202eccd832f8c4b66a035c4cc"

# File for each subset inside the dataset repo. Upstream names the QA files test.jsonl.
DATASET_FILES = {
    "kb": "wix_kb_corpus/wix_kb_corpus.jsonl",
    "expertwritten": "wixqa_expertwritten/test.jsonl",
    "simulated": "wixqa_simulated/test.jsonl",
    "synthetic": "wixqa_synthetic/test.jsonl",
}

# Repo root: two levels above src/support_rag/. SRAG_ROOT overrides it (e.g. in Docker).
ROOT = Path(os.environ.get("SRAG_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = ROOT / "data" / "raw"
SPLITS_PATH = ROOT / "data" / "splits" / "splits.json"
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"  # run records (D-018); git-ignored
CACHE_DIR = ROOT / "cache"  # LLM response cache (D-020); git-ignored
ENV_PATH = ROOT / ".env"  # secrets such as ANTHROPIC_API_KEY; git-ignored


def load_env(path: Path = ENV_PATH) -> None:
    """Copy KEY=VALUE lines from .env into the environment. Variables already set win.

    Never prints or logs the values.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
