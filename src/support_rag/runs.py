"""Run records (D-018): what produced a result, and what happened question by question.

Each run is a directory under runs/ with config.json, questions.jsonl (one line per
question, appended as each finishes) and summary.json.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from support_rag import config

CONFIG_FILE = "config.json"
QUESTIONS_FILE = "questions.jsonl"
SUMMARY_FILE = "summary.json"

# Libraries whose version can change a result.
_TRACKED_PACKAGES = ("bm25s", "pystemmer", "sentence-transformers", "torch", "transformers")


def git_state(repo: Path = config.ROOT) -> dict[str, Any]:
    """The current commit and whether the working tree has uncommitted changes.

    Both are None outside a git checkout (for example, in a Docker image).
    """
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        dirty = _git(repo, "status", "--porcelain") != ""
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True
    )
    return result.stdout.strip()


def start_run(name: str, run_config: dict[str, Any], runs_dir: Path = config.RUNS_DIR) -> Path:
    """Create runs/<timestamp>_<name>/ and write config.json, with provenance added."""
    now = datetime.now().astimezone()
    run_dir = runs_dir / f"{now:%Y%m%d-%H%M%S}_{name}"
    run_dir.mkdir(parents=True)  # fails if it exists: a run is never overwritten
    full_config = {
        "run_id": run_dir.name,
        "created_at": now.isoformat(timespec="seconds"),
        **run_config,
        "dataset_revision": config.DATASET_REVISION,
        "git": git_state(),
        "versions": {"python": platform.python_version()}
        | {package: metadata.version(package) for package in _TRACKED_PACKAGES},
    }
    _write_json(run_dir / CONFIG_FILE, full_config)
    return run_dir


def append_result(run_dir: Path, record: dict[str, Any]) -> None:
    """Add one question's result as a line of questions.jsonl."""
    with (run_dir / QUESTIONS_FILE).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    _write_json(run_dir / SUMMARY_FILE, summary)


def load_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))


def load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / SUMMARY_FILE).read_text(encoding="utf-8"))


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / QUESTIONS_FILE).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
