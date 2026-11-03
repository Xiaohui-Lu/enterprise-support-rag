"""Answer quality per strategy, with confidence intervals and paired comparisons (D-032).

Input: one judge run per strategy, all with the same judge settings and over the same
questions. Each judge run points to its answer run, which supplies tokens, cost and
latency. A decline counts as incorrect (D-025); a judge failure leaves the question out of
that strategy's numbers and out of any comparison that includes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.runs import load_config, load_results, load_summary
from support_rag.stats import N_RESAMPLES, bootstrap_ci, holm, mcnemar_exact, paired_bootstrap_ci

STRATEGY_NAMES = {"s0": "S0, closed book", "s1": "S1, BM25", "s_star": "S*, gold articles"}
# (better-expected, baseline): what retrieval adds, what better retrieval would add, both.
COMPARISONS = (("s1", "s0"), ("s_star", "s1"), ("s_star", "s0"))


def compare_strategies(
    judge_runs: Sequence[Path], *, runs_dir: Path = config.RUNS_DIR
) -> dict[str, Any]:
    configs = {}
    correct: dict[str, dict[str, bool | None]] = {}
    for run_dir in judge_runs:
        run_config = load_config(run_dir)
        if run_config.get("kind") != "judge" or run_config["strategy"] in configs:
            raise ValueError("give one judge run per strategy")
        configs[run_config["strategy"]] = run_config
        correct[run_config["strategy"]] = {
            r["question_id"]: None if r["correctness"] is None else r["correctness"] == "correct"
            for r in load_results(run_dir)
        }
    judges = {(c["model"], c["effort"], c["prompt_version"]) for c in configs.values()}
    if len(judges) != 1:
        raise ValueError("the judge runs use different judge settings")
    if len({frozenset(questions) for questions in correct.values()}) != 1:
        raise ValueError("the judge runs cover different questions")

    strategies = {}
    for strategy, by_question in correct.items():
        graded = [float(c) for c in by_question.values() if c is not None]
        answers_run = runs_dir / configs[strategy]["answers_run"]
        answers, answers_config = load_summary(answers_run), load_config(answers_run)
        strategies[strategy] = {
            "n": len(graded),
            "judge_failed": len(by_question) - len(graded),
            "pct_correct": 100 * sum(graded) / len(graded),
            "ci": tuple(100 * x for x in bootstrap_ci(graded)),
            "declined": answers["declined"],
            "mean_input_tokens": answers["mean_input_tokens"],
            "mean_output_tokens": answers["mean_output_tokens"],
            "cost_per_answer_usd": answers["cost_of_answers_usd"] / answers["n"],
            "latency_p50_ms": answers["latency_p50_ms"],
            "answers_run": answers_config["run_id"],
            "split": answers_config["split"],
            "milestone": answers_config.get("milestone"),
            "commit": answers_config["git"]["commit"],
            "name": STRATEGY_NAMES[strategy]
            + (f" top {answers_config['k']}" if answers_config.get("k") else ""),
            "generator": (answers_config["model"], answers_config["effort"]),
        }

    comparisons = []
    for better, base in COMPARISONS:
        if better not in correct or base not in correct:
            continue
        questions = sorted(
            q for q in correct[better]
            if correct[better][q] is not None and correct[base][q] is not None
        )  # fmt: skip
        a = [correct[better][q] for q in questions]
        b = [correct[base][q] for q in questions]
        test = mcnemar_exact(a, b)
        low, high = paired_bootstrap_ci([float(x) for x in a], [float(x) for x in b])
        comparisons.append(
            {
                "better": better,
                "base": base,
                "n": len(questions),
                "difference": 100 * (sum(a) - sum(b)) / len(questions),
                "ci": (100 * low, 100 * high),
                **test,
            }
        )
    for comparison, adjusted in zip(
        comparisons, holm([c["p_value"] for c in comparisons]), strict=True
    ):
        comparison["p_holm"] = adjusted

    model, effort, prompt_version = judges.pop()
    return {
        "judge": {"model": model, "effort": effort, "prompt_version": prompt_version},
        "judge_runs": sorted(c["run_id"] for c in configs.values()),
        "strategies": {s: strategies[s] for s in STRATEGY_NAMES if s in strategies},
        "comparisons": comparisons,
    }


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _provenance(strategies: Mapping[str, Mapping[str, Any]]) -> str:
    """Dev results are for tuning; a test report says which milestone and commit it came from."""
    if {s["split"] for s in strategies.values()} != {"test"}:
        return (
            "- These are dev results, used for tuning. Claims come from the test set at a "
            "milestone (D-004)."
        )
    milestones = ", ".join(sorted({str(s["milestone"]) for s in strategies.values()}))
    commits = ", ".join(sorted({str(s["commit"])[:7] for s in strategies.values()}))
    return (
        f"- Test set, milestone {milestones}, from commit {commits}: run once with the settings "
        "tuned on dev (D-033). Nothing was changed after seeing these results (D-034)."
    )


def format_report(result: Mapping[str, Any]) -> str:
    judge = result["judge"]
    any_strategy = next(iter(result["strategies"].values()))
    generator_model, generator_effort = any_strategy["generator"]
    lines = [
        f"# Answer quality on {any_strategy['split']}",
        "",
        "Generated by `srag report-answers`.",
        "",
        f"- Generator: {generator_model}, effort {generator_effort}. Judge: {judge['model']}, "
        f"rubric {judge['prompt_version']}, effort {judge['effort']}, validated in D-031.",
        "- % correct counts only answers graded correct; a decline is incorrect (D-025).",
        f"- 95% intervals come from resampling the questions {N_RESAMPLES:,} times. Comparisons "
        "are paired on the same questions: the difference has a paired-bootstrap interval, "
        "and McNemar's exact test uses only the questions where the two strategies differ. "
        "The Holm p-value accounts for making three comparisons (D-032).",
        _provenance(result["strategies"]),
        "",
        "## Strategies",
        "",
        "| Strategy | Questions | % correct [95% CI] | Declined | Input tokens | Output tokens "
        "| Cost per answer | Latency p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["strategies"].values():
        low, high = row["ci"]
        latency = row["latency_p50_ms"]
        lines.append(
            f"| {row['name']} | {row['n']} | {row['pct_correct']:.1f}% "
            f"[{low:.1f}, {high:.1f}] | {row['declined']} | {row['mean_input_tokens']:,.0f} "
            f"| {row['mean_output_tokens']:,.0f} | ${row['cost_per_answer_usd']:.5f} "
            f"| {'n/a' if latency is None else f'{latency / 1000:.1f} s'} |"
        )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "| Comparison | Questions | Difference, points [95% CI] | Only the first right "
        "| Only the second right | McNemar p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {strategy: row["name"] for strategy, row in result["strategies"].items()}
    for c in result["comparisons"]:
        low, high = c["ci"]
        lines.append(
            f"| {names[c['better']]} vs {names[c['base']]} | {c['n']} "
            f"| {c['difference']:+.1f} [{low:+.1f}, {high:+.1f}] | {c['only_a']} "
            f"| {c['only_b']} | {_p(c['p_value'])} | {_p(c['p_holm'])} |"
        )
    lines += [
        "",
        "Runs: judge "
        + ", ".join(f"`{r}`" for r in result["judge_runs"])
        + "; answers "
        + ", ".join(f"`{row['answers_run']}`" for row in result["strategies"].values())
        + ".",
    ]
    return "\n".join(lines) + "\n"


def write_report(judge_runs: Sequence[Path], reports_dir: Path = config.REPORTS_DIR) -> Path:
    result = compare_strategies(judge_runs)
    split = next(iter(result["strategies"].values()))["split"]
    path = reports_dir / f"answers_{split}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(result), encoding="utf-8", newline="\n")
    return path
