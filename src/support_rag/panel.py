"""The labeling panel (D-028): four Claude configurations pre-label the hand-label sample.

Each member grades every sample item through the judge's own request, with only the model
and effort changed, so it sees exactly what the judge sees and never the strategy. Each
member's verdicts are a run record. The printed summary is aggregates only (how often the
members agree), never an item's grade, so the items they split on can still be labeled
blind (D-027).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.judge import JUDGE_PROMPT_VERSION
from support_rag.judge_eval import judge_answer, judge_request, summarize_verdicts
from support_rag.records import Article, Question
from support_rag.runs import append_result, load_config, load_results, start_run, write_summary

# Two models at two efforts each. Opus vs Fable gives two opinions; high vs max shows
# whether a model's grade holds when it thinks longer. Sonnet (the judge) and Haiku (the
# generator) stay off the panel (D-028).
PANEL = (
    ("claude-opus-5-5", "high"),
    ("claude-opus-5-5", "max"),
    ("claude-fable-5-1", "high"),
    ("claude-fable-5-1", "max"),
)
# Opus at max effort thought for up to 14,140 tokens per item in the pilot, close to the
# judge's 16,000. 20,000 leaves room and stays under the SDK's non-streaming limit (about
# 21,300), which matters when a batch miss is retried in real time.
PANEL_MAX_TOKENS = 20_000


def member_name(model: str, effort: str) -> str:
    return f"{model.removeprefix('claude-')}-{effort}"


def panel_requests(
    items: Sequence[Mapping[str, Any]],
    questions: Mapping[str, Question],
    articles_by_id: Mapping[str, Article],
) -> list[dict[str, Any]]:
    """Every member's request for every item, built exactly as the judge builds its own."""
    return [
        judge_request(
            _as_answer(item),
            questions[item["question_id"]],
            articles_by_id,
            effort=effort,
            model=model,
            max_tokens=PANEL_MAX_TOKENS,
        )
        for model, effort in PANEL
        for item in items
    ]


def _as_answer(item: Mapping[str, Any]) -> dict[str, Any]:
    return {**item, "declined": False}  # declines are never sampled (D-027)


def run_panel(
    *,
    limit: int | None = None,
    batch: bool = False,
    batch_id: str | None = None,
    runs_dir: Path = config.RUNS_DIR,
) -> tuple[list[Path], dict[str, Any] | None]:
    """Grade the sample with every member and save one run per member.

    With `batch` (or `batch_id`), the calls not cached yet go through the batch API first,
    and the runs then replay them from the cache.
    """
    from support_rag.labels import load_sample
    from support_rag.llm import LLMClient, run_batch
    from support_rag.llm_cache import LLMCache
    from support_rag.wixqa import load_articles, load_questions

    items = load_sample()["items"][:limit]  # the sample is shuffled: a prefix is random
    questions = {
        q.id: q for subset in ("expertwritten", "simulated") for q in load_questions(subset)
    }
    articles_by_id = {a.id: a for a in load_articles()}

    run_dirs = []
    batch_summary = None
    with LLMCache() as cache:
        if batch or batch_id:
            requests = panel_requests(items, questions, articles_by_id)
            batch_summary = run_batch(requests, cache, batch_id=batch_id)
        llm = LLMClient(cache)
        for model, effort in PANEL:
            run_config = {
                "kind": "panel",
                "model": model,
                "effort": effort,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "max_tokens": PANEL_MAX_TOKENS,
                "sample": "data/labels/sample.json",
                "n_items": len(items),
                "limit": limit,
                "batch_id": batch_summary["batch_id"] if batch_summary else None,
            }
            run_dir = start_run(f"panel-{member_name(model, effort)}", run_config, runs_dir)
            results = []
            for item in items:
                question = questions[item["question_id"]]
                result = judge_answer(
                    llm,
                    _as_answer(item),
                    question,
                    articles_by_id,
                    effort=effort,
                    model=model,
                    max_tokens=PANEL_MAX_TOKENS,
                )
                append_result(run_dir, {"item": item["item"], **asdict(result)})
                results.append(result)
            write_summary(run_dir, summarize_verdicts(results))
            run_dirs.append(run_dir)
    return run_dirs, batch_summary


def panel_agreement(run_dirs: Sequence[Path], sample: Mapping[str, Any]) -> dict[str, Any]:
    """How often the members agree, in counts only: no item's grade, no grade totals."""
    members = [member_name(c["model"], c["effort"]) for c in map(load_config, run_dirs)]
    verdicts = [{r["item"]: r for r in load_results(run_dir)} for run_dir in run_dirs]
    strategy = {item["item"]: item["strategy"] for item in sample["items"]}
    items = sorted(set.intersection(*(set(v) for v in verdicts)))

    correctness = {i: [v[i]["correctness"] for v in verdicts] for i in items}
    graded = {i: grades for i, grades in correctness.items() if None not in grades}
    split = {i for i, grades in graded.items() if len(set(grades)) > 1}
    grounded = [
        [v[i]["grounded"] for v in verdicts]
        for i in graded
        if all(v[i]["grounded"] not in (None, "n/a") for v in verdicts)
    ]
    return {
        "members": members,
        "items": len(items),
        "judge_failed": {
            m: sum(v[i]["correctness"] is None for i in items)
            for m, v in zip(members, verdicts, strict=True)
        },  # fmt: skip
        "correctness": {
            "graded_by_all": len(graded),
            "unanimous": len(graded) - len(split),
            "split": len(split),
            "split_shapes": dict(Counter(_shape(graded[i]) for i in split).most_common()),
            "odd_one_out_in_3_1": dict(
                Counter(
                    members[_odd_one_out(graded[i])] for i in split if _shape(graded[i]) == "3-1"
                ).most_common()
            ),
            "pairwise_agreement_pct": {
                f"{members[a]} / {members[b]}": _pct(
                    sum(g[a] == g[b] for g in graded.values()), len(graded)
                )
                for a, b in combinations(range(len(members)), 2)
            },
            "split_by_strategy": {
                s: f"{sum(strategy[i] == s for i in split)} of "
                f"{sum(strategy[i] == s for i in graded)}"
                for s in sorted(set(strategy.values()))
            },
        },
        "groundedness": {
            "graded_by_all": len(grounded),
            "unanimous": sum(len(set(g)) == 1 for g in grounded),
        },
        "mean_output_tokens": {
            m: round(sum(v[i]["usage"].get("output_tokens", 0) for i in items) / len(items))
            for m, v in zip(members, verdicts, strict=True)
        },
        "cost_usd": {
            m: round(sum(v[i]["cost_usd"] for i in items), 4)
            for m, v in zip(members, verdicts, strict=True)
        },
    }


def _shape(grades: Sequence[str]) -> str:
    """How the votes split: "3-1", "2-2", "2-1-1" (or "4" when unanimous)."""
    return "-".join(str(n) for n in sorted(Counter(grades).values(), reverse=True))


def _odd_one_out(grades: Sequence[str]) -> int:
    counts = Counter(grades)
    return next(n for n, grade in enumerate(grades) if counts[grade] == 1)


def _pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def format_agreement(agreement: Mapping[str, Any]) -> str:
    c, g = agreement["correctness"], agreement["groundedness"]
    lines = [
        f"Panel: {', '.join(agreement['members'])}",
        f"Items: {agreement['items']}; judge failures: {agreement['judge_failed']}",
        "",
        f"Correctness, graded by all four: {c['graded_by_all']}",
        f"  unanimous: {c['unanimous']}   split: {c['split']}   shapes: {c['split_shapes']}",
        f"  odd one out in 3-1 splits: {c['odd_one_out_in_3_1']}",
        f"  split by strategy: {c['split_by_strategy']}",
        "  pairwise agreement:",
        *(f"    {pair}: {pct}%" for pair, pct in c["pairwise_agreement_pct"].items()),
        "",
        f"Groundedness, graded by all four: {g['graded_by_all']}; unanimous: {g['unanimous']}",
        "",
        f"Mean output tokens: {agreement['mean_output_tokens']}",
        f"Cost (USD): {agreement['cost_usd']}; total {sum(agreement['cost_usd'].values()):.4f}",
    ]
    return "\n".join(lines)
