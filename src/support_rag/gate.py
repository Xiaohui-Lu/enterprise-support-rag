"""The eval gate (D-053): code that changes what S1 sends the model can't merge until its
answers have been checked against the baseline.

Two layers, over the 50 real dev questions:

1. The drift check rebuilds S1's request for each question with the service's own code and
   compares its hash with the baseline's. An identical request gets the identical answer and
   grade from the call cache (D-020), so if no request changed, nothing is answered: the gate
   passes for free and without noise.
2. Only the questions whose requests changed are answered and graded, through the evaluation's
   code and the validated judge (D-031). The change passes if the paired difference against the
   baseline keeps its 95% interval above -10 points (D-042's bar).

`srag gate` runs it locally, from a clean commit, and on a pass writes reports/gate.json: the
git tree of src/ it ran on, and each question's request hash and grade, the next baseline. CI
runs `srag gate --check`, which fails unless that file passed for exactly the src/ being built.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.stats import paired_comparison

GATE_PATH = config.REPORTS_DIR / "gate.json"
SPLIT = "dev_real"
MARGIN = -10.0  # points: D-042's non-inferiority bar
JUDGE_EFFORT = "high"  # the judge's validated setting (D-031)
DEFAULT_MAX_USD = 1.0
# An S1 answer (about $0.0013) and a real-time grade at high effort (about $0.0103), at M2's
# per-answer costs. Used only to stop before an expensive run.
COST_PER_QUESTION_USD = 0.012


def src_tree(repo: Path = config.ROOT) -> str:
    """The git tree hash of the committed src/: it changes when any code file does."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD:src"],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True,
    )  # fmt: skip
    return result.stdout.strip()


def changed_questions(
    request_keys: Mapping[str, str], baseline: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """The questions whose request differs from the baseline's, or that it doesn't have."""
    return sorted(
        q for q, key in request_keys.items() if baseline.get(q, {}).get("request_key") != key
    )


def decide(
    items: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]] | None,
    changed: Sequence[str],
    *,
    accept: bool = False,
) -> dict[str, Any]:
    """Pass or fail. With no baseline, this run becomes it; with no change, it passes as is.
    Otherwise the paired difference must keep its interval above MARGIN, or be `accept`ed."""
    if baseline is None:
        return {"passed": True, "reason": "first run: these results are the baseline"}
    if not changed:
        return {"passed": True, "reason": "no request changed"}
    comparison = paired_comparison(
        {q: item["correct"] for q, item in items.items()},
        {q: item["correct"] for q, item in baseline.items()},
    )
    if comparison["ci"][0] > MARGIN:
        reason = f"within the margin: the interval stays above {MARGIN:+.0f} points"
        return {"passed": True, "reason": reason, "comparison": comparison}
    if accept:
        reason = "accepted despite missing the margin: record why in a decision entry"
        return {"passed": True, "reason": reason, "comparison": comparison, "accepted": True}
    reason = f"below the margin: the interval reaches under {MARGIN:+.0f} points"
    return {"passed": False, "reason": reason, "comparison": comparison}


def check(gate: Mapping[str, Any] | None, tree: str) -> str | None:
    """CI's check: None if the gate passed for this src/ tree, or what's wrong."""
    rerun = "run `uv run srag gate` from a clean commit and commit reports/gate.json"
    if gate is None:
        return f"there's no gate result: {rerun}"
    if gate["src_tree"] != tree:
        return f"src/ changed since the gate last ran (at {gate['commit'][:7]}): {rerun}"
    if not gate["decision"]["passed"]:
        return f"the gate failed: {gate['decision']['reason']}"
    return None


def load_gate(path: Path = GATE_PATH) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_gate(
    *, max_usd: float = DEFAULT_MAX_USD, accept: bool = False, path: Path = GATE_PATH
) -> dict[str, Any]:
    """Run both layers and, on a pass, write the new baseline. Returns the result."""
    from dataclasses import asdict

    from support_rag.answer import PROMPT_VERSION, build_request
    from support_rag.answer_eval import (
        DEFAULT_EFFORT,
        DEFAULT_K,
        DEFAULT_RETRIEVER,
        answer_question,
        context_for,
    )
    from support_rag.judge import JUDGE_PROMPT_VERSION
    from support_rag.judge_eval import judge_answer
    from support_rag.llm import GENERATOR_MODEL, JUDGE_MODEL, LLMClient
    from support_rag.llm_cache import LLMCache, cache_key
    from support_rag.retrieval_eval import answer_retriever, questions_for_split
    from support_rag.runs import git_state
    from support_rag.wixqa import load_articles

    git = git_state()
    if git["dirty"] is not False:
        raise ValueError("commit everything first: the gate records the code it checked")
    previous = load_gate(path)
    baseline = None if previous is None else previous["questions"]

    questions = questions_for_split(SPLIT)
    articles = load_articles()
    articles_by_id = {a.id: a for a in articles}
    retriever, _ = answer_retriever(DEFAULT_RETRIEVER, "article", articles)
    with LLMCache() as cache:
        llm = LLMClient(cache)
        contexts = {
            q.id: context_for(
                "s1", q, articles_by_id=articles_by_id, retriever=retriever, k=DEFAULT_K
            )
            for q in questions
        }
        requests = {
            q.id: build_request(q, contexts[q.id], effort=DEFAULT_EFFORT) for q in questions
        }
        keys = {q: cache_key(request) for q, request in requests.items()}
        changed = changed_questions(keys, baseline or {})

        uncached = [q for q in changed if cache.get(requests[q]) is None]
        estimate = len(uncached) * COST_PER_QUESTION_USD
        if estimate > max_usd:
            raise ValueError(
                f"{len(uncached)} changed questions aren't cached: about ${estimate:.2f}, over "
                f"the ${max_usd:.2f} cap. Raise it with --max-usd if that's intended."
            )

        items = {q: dict(baseline[q]) for q in keys if baseline and q not in changed}
        paid = 0.0
        for question in (q for q in questions if q.id in changed):
            answer = answer_question(llm, question, contexts[question.id], effort=DEFAULT_EFFORT)
            graded = judge_answer(
                llm, asdict(answer), question, articles_by_id, effort=JUDGE_EFFORT
            )
            paid += (0 if answer.cached else answer.cost_usd) + (
                0 if graded.cached or not graded.judged else graded.cost_usd
            )
            items[question.id] = {
                "request_key": keys[question.id],
                "correct": None if graded.correctness is None else graded.correctness == "correct",
                "gold_in_context": any(g in answer.context_ids for g in question.gold_ids),
            }

    decision = decide(items, baseline, changed, accept=accept)
    graded_items = [item["correct"] for item in items.values() if item["correct"] is not None]
    result = {
        "src_tree": src_tree(),
        "commit": git["commit"],
        "split": SPLIT,
        "n": len(items),
        "generator": {"model": GENERATOR_MODEL, "effort": DEFAULT_EFFORT,
                      "prompt_version": PROMPT_VERSION, "k": DEFAULT_K},
        "judge": {"model": JUDGE_MODEL, "effort": JUDGE_EFFORT,
                  "prompt_version": JUDGE_PROMPT_VERSION},
        "changed": changed,
        "decision": decision,
        "pct_correct": round(100 * sum(graded_items) / len(graded_items), 1),
        "gold_in_context": sum(item["gold_in_context"] for item in items.values()),
        "paid_usd": round(paid, 4),
        "questions": dict(sorted(items.items())),
    }  # fmt: skip
    if decision["passed"]:
        path.write_text(
            json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
        )
    return result
