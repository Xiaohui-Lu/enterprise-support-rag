"""Answer quality per strategy, with confidence intervals and paired comparisons (D-032).

Input: one judge run per strategy (S3: one per K), all with the same judge settings and
over the same questions. Each judge run points to its answer run, which supplies tokens,
cost and latency. A decline counts as incorrect (D-025); a judge failure leaves the question
out of that strategy's numbers and out of any comparison that includes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.runs import load_config, load_results, load_summary
from support_rag.stats import N_RESAMPLES, bootstrap_ci, holm, mcnemar_exact, paired_bootstrap_ci

STRATEGY_NAMES = {
    "s0": "S0, closed book",
    "s1": "S1",
    "s2": "S2, agentic search",
    "s3": "S3, long context",
    "s_star": "S*, gold articles",
}
RETRIEVER_NAMES = {"bm25": "BM25", "dense": "dense search"}
# (better-expected, baseline): what retrieval adds, what better retrieval would add, both,
# and what letting the model search adds. Each S3 arm is also compared with S1.
COMPARISONS = (("s1", "s0"), ("s_star", "s1"), ("s_star", "s0"), ("s2", "s1"))


def arm(answers_config: Mapping[str, Any]) -> str:
    """A run's place in the report: its strategy, or "s3-k50" for S3, which runs at several K."""
    strategy = answers_config["strategy"]
    return f"s3-k{answers_config['k']}" if strategy == "s3" else strategy


def _arm_order(key: str) -> tuple[int, int]:
    strategy, _, k = key.partition("-k")
    return list(STRATEGY_NAMES).index(strategy), int(k or 0)


def strategy_name(strategy: str, answers_config: Mapping[str, Any]) -> str:
    """How a report names a run: "S1, dense search top 10", or just the strategy's name.
    S3's K is a ceiling: the token cap can send fewer (D-044)."""
    name = STRATEGY_NAMES[strategy]
    k = answers_config.get("k")
    if not k:
        return name
    top = "up to" if strategy == "s3" else "top"
    name = f"{name}, {RETRIEVER_NAMES[answers_config['retriever']['name']]} {top} {k}"
    unit = answers_config.get("unit", "article")
    return name if unit == "article" else f"{name}, best {unit} section"


def compare_strategies(
    judge_runs: Sequence[Path], *, runs_dir: Path = config.RUNS_DIR
) -> dict[str, Any]:
    judge_configs = [load_config(run_dir) for run_dir in judge_runs]
    if any(c.get("kind") != "judge" for c in judge_configs):
        raise ValueError("give one judge run per strategy")
    judges = {(c["model"], c["effort"], c["prompt_version"]) for c in judge_configs}
    if len(judges) != 1:
        raise ValueError("the judge runs use different judge settings")

    configs = {}
    answers_configs = {}
    correct: dict[str, dict[str, bool | None]] = {}
    for run_dir, run_config in zip(judge_runs, judge_configs, strict=True):
        answers_config = load_config(runs_dir / run_config["answers_run"])
        key = arm(answers_config)
        if key in configs:
            raise ValueError("give one judge run per strategy (S3: one per K)")
        configs[key], answers_configs[key] = run_config, answers_config
        correct[key] = {
            r["question_id"]: None if r["correctness"] is None else r["correctness"] == "correct"
            for r in load_results(run_dir)
        }
    if len({frozenset(questions) for questions in correct.values()}) != 1:
        raise ValueError("the judge runs cover different questions")

    strategies = {}
    for key, by_question in correct.items():
        graded = [float(c) for c in by_question.values() if c is not None]
        answers_config = answers_configs[key]
        answers = load_summary(runs_dir / answers_config["run_id"])
        strategies[key] = {
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
            "name": strategy_name(answers_config["strategy"], answers_config),
            "generator": (answers_config["model"], answers_config["effort"]),
            "agent": answers.get("agent"),
            "long_context": answers.get("long_context"),
        }

    comparisons = []
    s3_arms = sorted((key for key in correct if key.startswith("s3-")), key=_arm_order)
    for better, base in (*COMPARISONS, *((key, "s1") for key in s3_arms)):
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
        "strategies": {key: strategies[key] for key in sorted(strategies, key=_arm_order)},
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


def _agent_section(row: Mapping[str, Any]) -> list[str]:
    """How S2's loops went (D-043). Costs are for the whole run."""
    agent = row["agent"]
    n = sum(agent["steps"].values())
    steps = ", ".join(f"{count} took {step}" for step, count in agent["steps"].items())
    limits = ", ".join(f"{name}: {count}" for name, count in agent["stopped_by"].items())
    paid = row["cost_per_answer_usd"] * n
    uncached = agent["cost_without_caching_usd"]
    return [
        "",
        "## S2's loop",
        "",
        f"- Tool steps per question: {steps}.",
        f"- Questions a limit ended: {limits or 'none'}.",
        f"- Per question: {agent['mean_searches']:.1f} searches and {agent['mean_reads']:.1f} "
        f"reads. The largest prompt averaged {agent['mean_largest_prompt_tokens']:,.0f} tokens.",
        f"- Tool calls that came back as errors: {agent['tool_errors']}.",
        f"- Prompt caching: {agent['mean_cached_input_tokens']:,.0f} of "
        f"{row['mean_input_tokens']:,.0f} input tokens per question were read from the cache. "
        f"The run cost ${paid:.4f}, and would have cost ${uncached:.4f} without caching.",
    ]


def _long_context_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """What S3's token cap did at each K, and the check on the local estimate (D-044)."""
    lines = [
        "",
        "## S3's token cap",
        "",
        "Articles go in best first, and the list stops before the estimated prompt passes "
        "90,000 tokens, 10% under the 100,000-token price line (D-044). The estimate error is "
        "(real − estimated) / real; above zero means the estimate was low.",
        "",
        "| Strategy | Cap cut the list | Articles sent, min / median / max | Largest prompt "
        "| Over 100K | Estimate error, min / median / max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lc = row["long_context"]
        sent, error = lc["articles_sent"], lc["estimate_error"]
        lines.append(
            f"| {row['name']} | {lc['capped']} of {row['n'] + row['judge_failed']} "
            f"| {sent['min']} / {sent['median']:g} / {sent['max']} "
            f"| {lc['max_prompt_tokens']:,} | {lc['over_price_line']} "
            f"| {error['min']:+.1%} / {error['median']:+.1%} / {error['max']:+.1%} |"
        )
    return lines


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
        f"The Holm p-value accounts for making {len(result['comparisons'])} comparisons (D-032).",
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
    agent_row = result["strategies"].get("s2")
    if agent_row is not None and agent_row["agent"] is not None:
        lines += _agent_section(agent_row)
    long_context_rows = [r for r in result["strategies"].values() if r["long_context"]]
    if long_context_rows:
        lines += _long_context_section(long_context_rows)
    lines += [
        "",
        "Runs: judge "
        + ", ".join(f"`{r}`" for r in result["judge_runs"])
        + "; answers "
        + ", ".join(f"`{row['answers_run']}`" for row in result["strategies"].values())
        + ".",
    ]
    return "\n".join(lines) + "\n"


def write_report(
    judge_runs: Sequence[Path],
    reports_dir: Path = config.REPORTS_DIR,
    name: str | None = None,
) -> Path:
    """Write reports/answers_<name>.md; `name` defaults to the split."""
    result = compare_strategies(judge_runs)
    split = next(iter(result["strategies"].values()))["split"]
    path = reports_dir / f"answers_{name or split}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(result), encoding="utf-8", newline="\n")
    return path
