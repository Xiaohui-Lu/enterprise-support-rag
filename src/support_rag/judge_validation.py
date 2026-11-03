"""Validating the judge against the labels (D-005, D-024, D-030).

Each sample item's label is the hand label where there is one, otherwise the panel's
unanimous label (D-028). The judge's verdict on an item comes from the judge run over the
answer run the item was drawn from. Agreement is reported overall, per strategy and per
label source, since the panel's labels aren't audited yet. The report holds aggregates
only, so the later blind audit of panel labels isn't anchored on the judge's grades.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.agreement import cohen_kappa, percent_agreement
from support_rag.judge import CORRECTNESS
from support_rag.runs import load_config, load_results

REPORT_PATH = config.REPORTS_DIR / "judge_validation.md"
KAPPA_BAR = 0.6  # D-005
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
GROUP_NAMES = {
    "all": "All items",
    "s0": "S0, closed book",
    "s1": "S1, BM25 top 5",
    "s_star": "S*, gold articles",
    "hand": "Hand labels (the panel's splits)",
    "panel": "Panel labels (unanimous, not audited yet)",
}


def final_labels(
    panel: Mapping[str, Any], hand_labels: Mapping[int, Mapping[str, Any]]
) -> dict[int, tuple[str, str]]:
    """Each item's correctness label and its source, "hand" or "panel". A hand label wins."""
    labels = {}
    for entry in panel["items"]:
        item = entry["item"]
        if item in hand_labels:
            labels[item] = (hand_labels[item]["correctness"], "hand")
        elif entry["correctness"] is not None:
            labels[item] = (entry["correctness"], "panel")
    return labels


def _binary(grade: str) -> str:
    return "correct" if grade == "correct" else "not correct"


def agreement_row(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Agreement on (label, judge) pairs, on the three grades and on correct vs not. The
    second is the decision behind the headline metric, % correct (D-030)."""
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None, "kappa_correct_vs_not": None}
    labels = [label for label, _ in pairs]
    judge = [grade for _, grade in pairs]
    return {
        "n": len(pairs),
        "agreement": percent_agreement(labels, judge),
        "kappa": cohen_kappa(labels, judge),
        "kappa_correct_vs_not": cohen_kappa(
            [_binary(g) for g in labels], [_binary(g) for g in judge]
        ),
    }


def validate_judge(
    judge_runs: Sequence[Path], sample: Mapping[str, Any], labels: Mapping[int, tuple[str, str]]
) -> dict[str, dict[str, Any]]:
    """Agreement with the labels for each judge effort. Each effort needs one judge run per
    strategy, over the same answer run the sample was drawn from."""
    runs_by_effort: dict[str, dict[str, Path]] = defaultdict(dict)
    for run_dir in judge_runs:
        run_config = load_config(run_dir)
        strategy, effort = run_config.get("strategy"), run_config.get("effort")
        if run_config.get("kind") != "judge":
            raise ValueError(f"{run_dir} is not a judge run")
        if run_config["answers_run"] != sample["answer_runs"][strategy]["run_id"]:
            raise ValueError(f"{run_dir} grades a different answer run than the sample's")
        if strategy in runs_by_effort[effort]:
            raise ValueError(f"two judge runs for {strategy} at effort {effort}")
        runs_by_effort[effort][strategy] = run_dir

    strategies = sorted({item["strategy"] for item in sample["items"]})
    results = {}
    for effort in sorted(runs_by_effort, key=EFFORT_ORDER.index):
        runs = runs_by_effort[effort]
        if sorted(runs) != strategies:
            raise ValueError(f"effort {effort}: need one judge run for each of {strategies}")
        verdicts = {
            (strategy, r["question_id"]): r
            for strategy, run_dir in runs.items()
            for r in load_results(run_dir)
        }
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        confusion: Counter[tuple[str, str]] = Counter()
        failed = 0
        for item in sample["items"]:
            if item["item"] not in labels:
                continue
            label, source = labels[item["item"]]
            grade = verdicts[(item["strategy"], item["question_id"])]["correctness"]
            if grade is None:  # cut off or refused: a judge failure, counted (D-026)
                failed += 1
                continue
            for group in ("all", item["strategy"], source):
                groups[group].append((label, grade))
            confusion[(label, grade)] += 1
        calls = [r for r in verdicts.values() if r["judged"]]
        results[effort] = {
            "runs": sorted(load_config(run_dir)["run_id"] for run_dir in runs.values()),
            "judge_failed": failed,
            "groups": {g: agreement_row(groups[g]) for g in ["all", *strategies, "hand", "panel"]},
            "confusion": {f"{label} -> {grade}": n for (label, grade), n in confusion.items()},
            "cost_per_call_usd": sum(r["cost_usd"] for r in calls) / len(calls) if calls else None,
            "mean_output_tokens": (
                sum(r["usage"].get("output_tokens", 0) for r in calls) / len(calls)
                if calls
                else None
            ),
        }
    return results


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _kappa(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_report(results: Mapping[str, Mapping[str, Any]], labels: Mapping[int, Any]) -> str:
    sources = Counter(source for _, source in labels.values())
    lines = [
        "# Judge validation",
        "",
        "Generated by `srag validate-judge`. The judge (Claude Sonnet 5.5, rubric judge-v1) "
        "graded the dev answers in the label sample. Each item's label is the hand label "
        "where there is one, otherwise the four-model panel's unanimous label (D-027, D-028).",
        "",
        f"- Labels: {len(labels)} items, {sources['hand']} labeled by hand (the items the panel "
        f"split on) and {sources['panel']} by the panel (unanimous, not audited by hand yet).",
        f"- The bar is a Cohen's kappa of {KAPPA_BAR} or more (D-005). Kappa on correct vs not "
        "is the decision behind the headline metric, % correct (D-030).",
        "- The groups are small, so differences of a few points are noise. Confidence "
        "intervals come with the statistics step (D-006).",
        "",
        "## Agreement by judge effort",
        "",
        "| Effort | Items | Agreement | Kappa, 3 grades | Kappa, correct vs not "
        "| Cost per call | Mean output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for effort, result in results.items():
        row = result["groups"]["all"]
        lines.append(
            f"| {effort} | {row['n']} | {_pct(row['agreement'])} | {_kappa(row['kappa'])} "
            f"| {_kappa(row['kappa_correct_vs_not'])} | ${result['cost_per_call_usd']:.4f} "
            f"| {result['mean_output_tokens']:.0f} |"
        )
    for effort, result in results.items():
        lines += [
            "",
            f"## Effort {effort}",
            "",
            f"Judge runs: {', '.join(f'`{run}`' for run in result['runs'])}. "
            f"Judge failures on labeled items: {result['judge_failed']}.",
            "",
            "| Group | Items | Agreement | Kappa, 3 grades | Kappa, correct vs not |",
            "|---|---:|---:|---:|---:|",
        ]
        for group, row in result["groups"].items():
            lines.append(
                f"| {GROUP_NAMES.get(group, group)} | {row['n']} | {_pct(row['agreement'])} "
                f"| {_kappa(row['kappa'])} | {_kappa(row['kappa_correct_vs_not'])} |"
            )
        lines += [
            "",
            "Labels in rows, the judge's grades in columns:",
            "",
            "| Label | " + " | ".join(CORRECTNESS) + " |",
            "|---|" + "---:|" * len(CORRECTNESS),
        ]
        for label in CORRECTNESS:
            counts = [result["confusion"].get(f"{label} -> {grade}", 0) for grade in CORRECTNESS]
            lines.append(f"| {label} | " + " | ".join(str(n) for n in counts) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    results: Mapping[str, Mapping[str, Any]],
    labels: Mapping[int, Any],
    path: Path = REPORT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(results, labels), encoding="utf-8", newline="\n")
    return path
