"""The context-rot report (RQ5, D-045): % correct as distractors fill the context.

Input: the gold-only judge run (S*) and one judge run per arm, all with the same judge
settings. Within each type the largest size is paired with the smallest; every arm is also
paired with the gold only, and hard with random at each size, all on the same questions. A
judge failure leaves the question out of that arm and its comparisons.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.llm import LONG_PROMPT_TOKENS, prompt_tokens
from support_rag.runs import load_config, load_results
from support_rag.stats import bootstrap_ci, paired_comparison, with_holm

GOLD_ONLY = ("gold only", 0)
TYPE_ORDER = {"gold only": 0, "hard": 1, "random": 2}


def _arm(answers_config: Mapping[str, Any]) -> tuple[str, int]:
    if answers_config["strategy"] == "s_star":
        return GOLD_ONLY
    if answers_config["strategy"] != "rot":
        raise ValueError("give the gold-only run (S*) and context-rot runs only")
    return answers_config["distractors"], answers_config["size"]


def _name(arm: tuple[str, int]) -> str:
    return arm[0] if arm == GOLD_ONLY else f"{arm[0]}, {arm[1] // 1000}K"


def _rate(values: Sequence[bool]) -> dict[str, Any]:
    floats = [float(v) for v in values]
    low, high = bootstrap_ci(floats)
    return {"n": len(floats), "pct": 100 * statistics.fmean(floats), "ci": (100 * low, 100 * high)}


def compare_rot(judge_runs: Sequence[Path], *, runs_dir: Path = config.RUNS_DIR) -> dict[str, Any]:
    judge_configs = [load_config(run_dir) for run_dir in judge_runs]
    if any(c.get("kind") != "judge" for c in judge_configs):
        raise ValueError("give judge runs")
    judges = {(c["model"], c["effort"], c["prompt_version"]) for c in judge_configs}
    if len(judges) != 1:
        raise ValueError("the judge runs use different judge settings")

    correct: dict[tuple[str, int], dict[str, bool | None]] = {}
    answers: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    answers_runs = {}
    splits = set()
    for run_dir, judge_config in zip(judge_runs, judge_configs, strict=True):
        answers_config = load_config(runs_dir / judge_config["answers_run"])
        arm = _arm(answers_config)
        if arm in correct:
            raise ValueError(f"two runs for {_name(arm)}")
        correct[arm] = {
            r["question_id"]: None if r["correctness"] is None else r["correctness"] == "correct"
            for r in load_results(run_dir)
        }
        answers[arm] = {
            r["question_id"]: r for r in load_results(runs_dir / judge_config["answers_run"])
        }
        answers_runs[arm] = answers_config["run_id"]
        splits.add(answers_config["split"])
    if GOLD_ONLY not in correct or len(correct) < 2:
        raise ValueError("give the gold-only run (S*) and at least one context-rot run")
    if len(splits) != 1:
        raise ValueError("the runs come from different splits")
    (split,) = splits

    rot_arms = sorted((a for a in correct if a != GOLD_ONLY), key=lambda a: (a[1], a[0]))
    questions = set(correct[rot_arms[0]])
    if any(set(correct[a]) != questions for a in rot_arms):
        raise ValueError("the context-rot runs cover different questions")
    if not questions <= set(correct[GOLD_ONLY]):
        raise ValueError("the gold-only run is missing some of the questions")
    arms = [GOLD_ONLY, *rot_arms]
    for arm in arms:  # the gold-only run may cover more questions: keep the same ones
        correct[arm] = {q: correct[arm][q] for q in sorted(questions)}

    rows = []
    for arm in arms:
        graded = {q: c for q, c in correct[arm].items() if c is not None}
        records = [answers[arm][q] for q in questions]
        rot = [r["rot"] for r in records if r.get("rot")]
        rows.append(
            {
                "arm": arm,
                **_rate(list(graded.values())),
                "judge_failed": len(questions) - len(graded),
                "mean_prompt_tokens": statistics.fmean(prompt_tokens(r["usage"]) for r in records),
                "mean_distractors": statistics.fmean(r["n_distractors"] for r in rot) if rot else 0,
                "cited_a_distractor": sum(
                    any(c not in r["gold_ids"] for c in r["cited_ids"]) for r in records
                ),
                "over_price_line": sum(
                    prompt_tokens(r["usage"]) > LONG_PROMPT_TOKENS for r in records
                ),
                "cost_per_answer_usd": statistics.fmean(r["cost_usd"] for r in records),
            }
        )

    sizes = sorted({size for _, size in rot_arms})
    # The main measure of rot: within each type, the largest size against the smallest. The
    # two arms hold the same gold at the same depth, and the smaller one's distractors are a
    # subset of the larger one's, so the only difference is the extra distractors. Unlike the
    # comparisons with the gold only, the types don't share one sample (D-045).
    within_type = with_holm(
        [
            {
                "distractors": distractors,
                "sizes": (sizes[-1], sizes[0]),
                **paired_comparison(
                    correct[(distractors, sizes[-1])], correct[(distractors, sizes[0])]
                ),
            }
            for distractors in ("hard", "random")
            if len(sizes) > 1
            and (distractors, sizes[-1]) in correct
            and (distractors, sizes[0]) in correct
        ]
    )
    against_gold = with_holm(
        [{"arm": arm, **paired_comparison(correct[arm], correct[GOLD_ONLY])} for arm in rot_arms]
    )
    hard_vs_random = with_holm(
        [
            {"size": size, **paired_comparison(correct[("hard", size)], correct[("random", size)])}
            for size in sizes
            if ("hard", size) in correct and ("random", size) in correct
        ]
    )

    multi = {q: answers[GOLD_ONLY][q]["multi_article"] for q in questions}
    by_kind = []
    for arm in arms:
        cells = {}
        for kind, wanted in (("single", False), ("multi", True)):
            values = [c for q, c in correct[arm].items() if multi[q] == wanted and c is not None]
            cells[kind] = (100 * statistics.fmean(values), len(values)) if values else None
        by_kind.append({"arm": arm, **cells})

    # Position, at the largest size, on single-article questions: one gold, one depth.
    largest = sizes[-1]
    by_depth = []
    for distractors in ("hard", "random"):
        arm = (distractors, largest)
        if arm not in correct:
            continue
        thirds: dict[int, list[bool]] = {0: [], 1: [], 2: []}
        for q, c in correct[arm].items():
            record = answers[arm][q]
            if c is None or record["multi_article"]:
                continue
            thirds[min(int(record["rot"]["gold_depths"][0] * 3), 2)].append(c)
        by_depth.append(
            {
                "arm": arm,
                "thirds": [
                    (100 * statistics.fmean(v), len(v)) if v else None for v in thirds.values()
                ],
            }
        )

    model, effort, prompt_version = judges.pop()
    return {
        "split": split,
        "judge": {"model": model, "effort": effort, "prompt_version": prompt_version},
        "n_questions": len(questions),
        "rows": rows,
        "within_type": within_type,
        "against_gold": against_gold,
        "hard_vs_random": hard_vs_random,
        "by_kind": by_kind,
        "by_depth": by_depth,
        "runs": {
            "judge": sorted(c["run_id"] for c in judge_configs),
            "answers": [answers_runs[arm] for arm in arms],
        },
    }


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _comparison_line(label: str, c: Mapping[str, Any]) -> str:
    low, high = c["ci"]
    return (
        f"| {label} | {c['n']} | {c['difference']:+.1f} [{low:+.1f}, {high:+.1f}] "
        f"| {c['only_a']} | {c['only_b']} | {_p(c['p_value'])} | {_p(c['p_holm'])} |"
    )


def format_rot_report(result: Mapping[str, Any], *, figure: str | None = None) -> str:
    """The report as Markdown; `figure` is the chart's path relative to the report."""
    judge = result["judge"]
    lines = [
        f"# Context rot on {result['split']}",
        "",
        "Generated by `srag report-rot`.",
        "",
        f"- {result['n_questions']} questions. Each question's gold articles were buried among "
        "distractors up to an estimated prompt size: hard distractors are dense search's "
        "top-ranked non-gold articles, random ones are drawn from the rest. Each gold article "
        "sits at one random depth, the same in every arm. The gold-only arm is S\\* (D-045).",
        f"- Judge: {judge['model']}, rubric {judge['prompt_version']}, effort {judge['effort']}. "
        "% correct counts only answers graded correct; a decline is incorrect (D-025).",
        "- Comparisons are paired on the same questions, with a paired-bootstrap interval and "
        "McNemar's exact test. Holm's adjustment runs separately over each table.",
        "",
        "## % correct by context size",
        "",
        *(
            [f"![% correct against prompt tokens, hard and random distractors]({figure})", ""]
            if figure
            else []
        ),
        "| Arm | Questions | % correct [95% CI] | Prompt tokens | Distractors | Cited a distractor "
        "| Over 100K | Cost per answer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        low, high = row["ci"]
        lines.append(
            f"| {_name(row['arm'])} | {row['n']} | {row['pct']:.1f}% [{low:.1f}, {high:.1f}] "
            f"| {row['mean_prompt_tokens']:,.0f} | {row['mean_distractors']:.1f} "
            f"| {row['cited_a_distractor']} | {row['over_price_line']} "
            f"| ${row['cost_per_answer_usd']:.5f} |"
        )
    header = (
        "| Comparison | Questions | Difference, points [95% CI] | Only the first right "
        "| Only the second right | McNemar p | Holm p |"
    )
    if result["within_type"]:
        lines += [
            "",
            "## Largest against smallest, same type",
            "",
            "The main measure of rot: the same gold at the same depth, with more distractors.",
            "",
            header,
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            _comparison_line(
                f"{c['distractors']}, {c['sizes'][0] // 1000}K vs {c['sizes'][1] // 1000}K", c
            )
            for c in result["within_type"]
        ]
    lines += [
        "",
        "## Against the gold only",
        "",
        "All of these share one gold-only sample, so their differences move together.",
        "",
        header,
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        _comparison_line(f"{_name(c['arm'])} vs gold only", c) for c in result["against_gold"]
    ]
    if result["hard_vs_random"]:
        lines += ["", "## Hard against random, same size", "", header]
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        lines += [
            _comparison_line(f"hard vs random, {c['size'] // 1000}K", c)
            for c in result["hard_vs_random"]
        ]

    def cell(value: tuple[float, int] | None) -> str:
        return "n/a" if value is None else f"{value[0]:.1f}% ({value[1]})"

    lines += [
        "",
        "## Single- and multi-article questions",
        "",
        "Directional: % correct, with the number of questions in brackets.",
        "",
        "| Arm | Single-article | Multi-article |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {_name(r['arm'])} | {cell(r['single'])} | {cell(r['multi'])} |"
        for r in result["by_kind"]
    ]
    if result["by_depth"]:
        lines += [
            "",
            "## Where the gold article sat, at the largest size",
            "",
            "Directional, single-article questions only: % correct by the gold article's depth, "
            "with the number of questions in brackets.",
            "",
            "| Arm | First third | Middle third | Last third |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| {_name(r['arm'])} | " + " | ".join(cell(t) for t in r["thirds"]) + " |"
            for r in result["by_depth"]
        ]
    runs = result["runs"]
    lines += [
        "",
        "Runs: judge "
        + ", ".join(f"`{r}`" for r in runs["judge"])
        + "; answers "
        + ", ".join(f"`{r}`" for r in runs["answers"])
        + ".",
    ]
    return "\n".join(lines) + "\n"


def write_rot_report(
    judge_runs: Sequence[Path], reports_dir: Path = config.REPORTS_DIR, name: str | None = None
) -> Path:
    """Write reports/context_rot_<name>.md and its chart, reports/figures/context_rot_<name>.svg;
    `name` defaults to the split."""
    from support_rag.charts import rot_chart  # matplotlib is a dev dependency

    result = compare_rot(judge_runs)
    stem = f"context_rot_{name or result['split']}"
    rot_chart(result, reports_dir / "figures" / f"{stem}.svg")
    path = reports_dir / f"{stem}.md"
    path.write_text(
        format_rot_report(result, figure=f"figures/{stem}.svg"), encoding="utf-8", newline="\n"
    )
    return path
