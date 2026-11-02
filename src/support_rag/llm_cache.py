"""On-disk cache of LLM responses for evaluation runs (D-020).

The key is a hash of everything that can change the output: model, prompts, messages
and parameters. Change any of them and it's a new call; repeat them exactly and the
saved response comes back, free and identical. Re-running a report never pays twice.

Each entry keeps the original call's token usage and latency, so cost reports use what
the call really cost, and a cache hit is never mistaken for a fast, free call.

Only successful, complete responses belong here: the caller must not store errors,
timeouts or cut-off answers, or the failure would be replayed forever.

This is not the provider's prompt caching, which reuses a processed prompt prefix on the
provider's side but still runs the model.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from support_rag import config

CACHE_PATH = config.CACHE_DIR / "llm_calls.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key        TEXT PRIMARY KEY,
    request    TEXT NOT NULL,
    response   TEXT NOT NULL,
    usage      TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
)
"""


def cache_key(request: dict[str, Any]) -> str:
    """SHA-256 of the request as canonical JSON, so key order in the dict doesn't matter."""
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedCall:
    response: dict[str, Any]  # the provider's response, as JSON
    usage: dict[str, int]  # tokens of the original call (input, output, cached input, ...)
    latency_ms: float  # of the original call; a cache hit takes ~0 ms
    created_at: str


class LLMCache:
    def __init__(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(_SCHEMA)
        self._db.commit()

    def get(self, request: dict[str, Any]) -> CachedCall | None:
        row = self._db.execute(
            "SELECT response, usage, latency_ms, created_at FROM calls WHERE key = ?",
            (cache_key(request),),
        ).fetchone()
        if row is None:
            return None
        response, usage, latency_ms, created_at = row
        return CachedCall(json.loads(response), json.loads(usage), latency_ms, created_at)

    def put(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        usage: dict[str, int],
        latency_ms: float,
    ) -> None:
        """Save a successful call. If the key is already saved, the first response stays,
        so a replay always returns the same answer."""
        self._db.execute(
            "INSERT OR IGNORE INTO calls VALUES (?, ?, ?, ?, ?, ?)",
            (
                cache_key(request),
                json.dumps(request, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False),
                json.dumps(usage),
                latency_ms,
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> LLMCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
