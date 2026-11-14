"""Answer quality per strategy: rates, paired comparisons, the cost frontier and question types
(D-032, D-046).

Input: one judge run per strategy (S1: one per retriever; S3: one per K), all with the same
judge settings and over the same questions. Each judge run points to its answer run, which
supplies tokens, cost and latency for every question. A decline counts as incorrect (D-025);
a judge failure leaves the question out of that strategy's numbers and out of any
comparison that includes it.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.frontier import FrontierPoint, pareto_frontier
from support_rag.runs import load_config, load_results, load_summary
from support_rag.stats import N_RESAMPLES, bootstrap_ci, paired_comparison, with_holm

STRATEGY_NAMES = {
    "s0": "S0, closed book",
    "s1": "S1",
    "s2": "S2, agentic search",
    "s3": "S3, long context",
    "s_star": "S*, gold articles",
}
RETRIEVER_NAMES = {"bm25": "BM25", "dense": "dense search"}
SHORT_RETRIEVER_NAMES = {"bm25": "BM25", "dense": "dense"}
DEFAULT_S1 = "s1-dense"  # D-042
# What retrieval adds, what better retrieval would add, and both; "s1" is the default S1.
# These describe and decide nothing. Every other strategy is compared with the default S1,
# and those comparisons are M2's verdicts (D-046).
RETRIEVAL_COMPARISONS = (("s1", "s0"), ("s_star", "s1"), ("s_star", "s0"))
VERDICT_P = 0.05
QUESTION_TYPES: tuple[tuple[str, Callable[[Mapping[str, Any]], bool]], ...] = (
    ("Single-article", lambda r: not r["multi_article"]),
    ("Multi-article", lambda r: r["multi_article"]),
    ("Expert-written", lambda r: r["subset"] == "expertwritten"),
    ("Simulated", lambda r: r["subset"] == "simulated"),
)


def arm(answers_config: Mapping[str, Any]) -> str:
    """A run's place in the report: "s1-dense" or "s1-bm25" for S1, "s3-k50" for S3, which
    run with several retrievers and K, or just the strategy."""
    strategy = answers_config["strategy"]
    if strategy == "s1":
        return f"s1-{answers_config['retriever']['name']}"
    if strategy == "s3":
        return f"s3-k{answers_config['k']}"
    return strategy


def _arm_order(key: str) -> tuple[int, int]:
    strategy, _, variant = key.partition("-")
    order = list(STRATEGY_NAMES).index(strategy)
    if strategy == "s3":
        return order, int(variant.removeprefix("k"))
    return order, list(RETRIEVER_NAMES).index(variant) if variant else 0


def short_name(key: str) -> str:
    """A chart label: "S1 dense", "S3 K=20", "S*"."""
    strategy, _, variant = key.partition("-")
    if strategy == "s1":
        return f"S1 {SHORT_RETRIEVER_NAMES[variant]}"
    if strategy == "s3":
        return f"S3 K={variant.removeprefix('k')}"
    return STRATEGY_NAMES[strategy].split(",")[0]


def default_s1(keys: Iterable[str]) -> str | None:
    """The S1 the others are compared with: dense search (D-042), or the only S1 given."""
    s1 = sorted(key for key in keys if key.startswith("s1-"))
    if DEFAULT_S1 in s1:
        return DEFAULT_S1
    return s1[0] if len(s1) == 1 else None


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


def _latency(records: Iterable[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """Median and 95th percentile over the answers paid for in the run: a cache hit takes no
    time, so it says nothing about latency (D-020, D-022)."""
    values = sorted(r["latency_ms"] for r in records if not r["cached"])
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    return statistics.median(values), statistics.quantiles(values, n=20, method="inclusive")[-1]


def _frontier(
    strategies: Mapping[str, Mapping[str, Any]],
    correct: Mapping[str, Mapping[str, bool | None]],
    measure: str,
) -> dict[str, Any]:
    """The frontier on one measure, cheapest first, and what each step along it adds.

    S* stays off: it needs the answer key. A step's cost per extra correct answer is given
    only when its gain's interval lies entirely above zero; otherwise the ratio could be
    anything up to infinity (D-046).
    """
    points = [
        FrontierPoint(key, row[measure], row["pct_correct"])
        for key, row in strategies.items()
        if key != "s_star" and row[measure] is not None
    ]
    on = pareto_frontier(points)
    steps = []
    for cheaper, dearer in pairwise(on):
        comparison = paired_comparison(correct[dearer.name], correct[cheaper.name])
        extra = dearer.cost - cheaper.cost
        shown = comparison["ci"][0] > 0
        steps.append(
            {
                "from": cheaper.name,
                "to": dearer.name,
                "extra": extra,
                **comparison,
                "per_extra_correct": extra / (comparison["difference"] / 100) if shown else None,
            }
        )
    return {"keys": [point.name for point in on], "steps": steps}


def _by_type(
    strategies: Mapping[str, Mapping[str, Any]],
    correct: Mapping[str, Mapping[str, bool | None]],
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """% correct, cost and the cost frontier within each question type (RQ4). Directional."""
    any_run = next(iter(records.values()))
    result = []
    for name, wanted in QUESTION_TYPES:
        questions = {q for q in correct[next(iter(correct))] if wanted(any_run[q])}
        if not questions:
            continue
        rows = {}
        for key in strategies:
            graded = [c for q, c in correct[key].items() if q in questions and c is not None]
            rows[key] = {
                "n": len(graded),
                "pct_correct": 100 * statistics.fmean(graded) if graded else None,
                "cost_per_answer_usd": statistics.fmean(
                    records[key][q]["cost_usd"] for q in questions
                ),
            }
        points = [
            FrontierPoint(key, row["cost_per_answer_usd"], row["pct_correct"])
            for key, row in rows.items()
            if key != "s_star" and row["pct_correct"] is not None
        ]
        frontier = [point.name for point in pareto_frontier(points)]
        result.append({"type": name, "n": len(questions), "rows": rows, "frontier": frontier})
    return result


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
            raise ValueError("give one judge run per strategy (S1: one per retriever; S3: per K)")
        configs[key], answers_configs[key] = run_config, answers_config
        correct[key] = {
            r["question_id"]: None if r["correctness"] is None else r["correctness"] == "correct"
            for r in load_results(run_dir)
        }
    if len({frozenset(questions) for questions in correct.values()}) != 1:
        raise ValueError("the judge runs cover different questions")

    strategies = {}
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for key in sorted(correct, key=_arm_order):
        by_question = correct[key]
        graded = [float(c) for c in by_question.values() if c is not None]
        answers_config = answers_configs[key]
        run_dir = runs_dir / answers_config["run_id"]
        answers = load_summary(run_dir)
        records[key] = {r["question_id"]: r for r in load_results(run_dir)}
        latency_p50, latency_p95 = _latency(records[key].values())
        strategies[key] = {
            "n": len(graded),
            "judge_failed": len(by_question) - len(graded),
            "pct_correct": 100 * sum(graded) / len(graded),
            "ci": tuple(100 * x for x in bootstrap_ci(graded)),
            "declined": answers["declined"],
            "mean_input_tokens": answers["mean_input_tokens"],
            "mean_output_tokens": answers["mean_output_tokens"],
            "cost_per_answer_usd": answers["cost_of_answers_usd"] / answers["n"],
            "latency_p50_ms": latency_p50,
            "latency_p95_ms": latency_p95,
            "answers_run": answers_config["run_id"],
            "split": answers_config["split"],
            "milestone": answers_config.get("milestone"),
            "commit": answers_config["git"]["commit"],
            "name": strategy_name(answers_config["strategy"], answers_config),
            "short_name": short_name(key),
            "generator": (answers_config["model"], answers_config["effort"]),
            "agent": answers.get("agent"),
            "long_context": answers.get("long_context"),
        }

    base = default_s1(correct)

    def resolve(key: str) -> str | None:
        return base if key == "s1" else key

    retrieval = with_holm(
        [
            {
                "better": better,
                "base": against,
                **paired_comparison(correct[better], correct[against]),
            }
            for better, against in ((resolve(b), resolve(a)) for b, a in RETRIEVAL_COMPARISONS)
            if better in correct and against in correct
        ]
    )
    candidates = [key for key in strategies if base and key not in ("s0", "s_star", base)]
    against_default = with_holm(
        [
            {"better": key, "base": base, **paired_comparison(correct[key], correct[base])}
            for key in candidates
        ]
    )
    for comparison in against_default:
        comparison["replaces_default"] = (
            comparison["ci"][0] > 0 and comparison["p_holm"] < VERDICT_P
        )

    model, effort, prompt_version = judges.pop()
    return {
        "judge": {"model": model, "effort": effort, "prompt_version": prompt_version},
        "judge_runs": sorted(c["run_id"] for c in configs.values()),
        "strategies": strategies,
        "default_s1": base,
        "against_default": against_default,
        "retrieval": retrieval,
        "frontier": {
            "cost": _frontier(strategies, correct, "cost_per_answer_usd"),
            "latency": _frontier(strategies, correct, "latency_p50_ms"),
        },
        "by_type": _by_type(strategies, correct, records),
    }


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _seconds(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.1f} s"


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


def _comparison_table(
    comparisons: Sequence[Mapping[str, Any]], names: Mapping[str, str], *, verdict: bool
) -> list[str]:
    header = (
        "| Comparison | Questions | Difference, points [95% CI] | Only the first right "
        "| Only the second right | McNemar p | Holm p |"
    )
    rule = "|---|---:|---:|---:|---:|---:|---:|"
    if verdict:
        header, rule = header + " Replaces the default |", rule + "---:|"
    lines = [header, rule]
    for c in comparisons:
        low, high = c["ci"]
        line = (
            f"| {names[c['better']]} vs {names[c['base']]} | {c['n']} "
            f"| {c['difference']:+.1f} [{low:+.1f}, {high:+.1f}] | {c['only_a']} "
            f"| {c['only_b']} | {_p(c['p_value'])} | {_p(c['p_holm'])} |"
        )
        if verdict:
            line += f" {'yes' if c['replaces_default'] else 'no'} |"
        lines.append(line)
    return lines


def _frontier_section(result: Mapping[str, Any], figure: str | None) -> list[str]:
    strategies = result["strategies"]
    short = {key: row["short_name"] for key, row in strategies.items()}
    lines = ["", "## The cost frontier", ""]
    if figure:
        lines += [f"![% correct against cost per answer and against median latency]({figure})", ""]
    lines += [
        "A strategy is on the frontier when no other strategy is both as cheap and as accurate "
        "(D-046). S\\* stays off it, because it needs the answer key. A cost per extra correct "
        "answer is given only when the step's gain has an interval entirely above zero.",
    ]
    for measure, label in (("cost", "cost per answer"), ("latency", "median latency")):
        frontier = result["frontier"][measure]
        if not frontier["keys"]:
            continue
        lines += [
            "",
            f"On {label}, the frontier is " + " → ".join(short[k] for k in frontier["keys"]) + ".",
        ]
        if not frontier["steps"]:
            continue
        if measure == "cost":
            lines += [
                "",
                "| Step | Extra cost per answer | Extra % correct [95% CI] "
                "| Cost per extra correct answer |",
                "|---|---:|---:|---:|",
            ]
        else:
            lines += [
                "",
                "| Step | Extra median latency | Extra % correct [95% CI] |",
                "|---|---:|---:|",
            ]
        for step in frontier["steps"]:
            low, high = step["ci"]
            gain = f"{step['difference']:+.1f} [{low:+.1f}, {high:+.1f}]"
            name = f"{short[step['from']]} → {short[step['to']]}"
            if measure == "cost":
                ratio = step["per_extra_correct"]
                per = "not shown to add correct answers" if ratio is None else f"${ratio:.4f}"
                lines.append(f"| {name} | +${step['extra']:.5f} | {gain} | {per} |")
            else:
                lines.append(f"| {name} | +{step['extra'] / 1000:.1f} s | {gain} |")
    return lines


def _by_type_section(result: Mapping[str, Any]) -> list[str]:
    by_type = result["by_type"]
    names = {key: row["name"] for key, row in result["strategies"].items()}
    short = {key: row["short_name"] for key, row in result["strategies"].items()}
    columns = " | ".join(f"{t['type']} ({t['n']})" for t in by_type)
    rule = "|---|" + "---:|" * len(by_type)
    lines = [
        "",
        "## By question type",
        "",
        "Directional only, with no verdicts (D-046): the groups are small. The number of "
        "questions is in brackets.",
        "",
        f"| % correct | {columns} |",
        rule,
    ]
    for key in result["strategies"]:
        cells = []
        for t in by_type:
            pct = t["rows"][key]["pct_correct"]
            cells.append("n/a" if pct is None else f"{pct:.1f}%")
        lines.append(f"| {names[key]} | " + " | ".join(cells) + " |")
    lines += ["", f"| Cost per answer | {columns} |", rule]
    for key in result["strategies"]:
        cells = [f"${t['rows'][key]['cost_per_answer_usd']:.5f}" for t in by_type]
        lines.append(f"| {names[key]} | " + " | ".join(cells) + " |")
    lines += ["", "The cost frontier within each type:", ""]
    lines += [
        f"- {t['type']}: " + " → ".join(short[k] for k in t["frontier"]) + "." for t in by_type
    ]
    return lines


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


def format_report(result: Mapping[str, Any], *, figure: str | None = None) -> str:
    """The report as Markdown; `figure` is the chart's path relative to the report."""
    judge = result["judge"]
    strategies = result["strategies"]
    any_strategy = next(iter(strategies.values()))
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
        "Holm's adjustment runs separately over each table of comparisons (D-032, D-046).",
        "- Cost is what the generator was paid per answer, prompt caching included; judging "
        "isn't counted. Latency is over the answers paid for in the run (D-046).",
        _provenance(strategies),
        "",
        "## Strategies",
        "",
        "| Strategy | Questions | % correct [95% CI] | Declined | Input tokens | Output tokens "
        "| Cost per answer | Latency p50 | Latency p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strategies.values():
        low, high = row["ci"]
        lines.append(
            f"| {row['name']} | {row['n']} | {row['pct_correct']:.1f}% "
            f"[{low:.1f}, {high:.1f}] | {row['declined']} | {row['mean_input_tokens']:,.0f} "
            f"| {row['mean_output_tokens']:,.0f} | ${row['cost_per_answer_usd']:.5f} "
            f"| {_seconds(row['latency_p50_ms'])} | {_seconds(row['latency_p95_ms'])} |"
        )
    lines += _frontier_section(result, figure)
    names = {key: row["name"] for key, row in strategies.items()}
    if result["against_default"]:
        lines += [
            "",
            "## Against the default",
            "",
            f"Each strategy against {names[result['default_s1']]}, the default (D-042). One "
            "replaces the default only if its interval lies entirely above zero and its Holm "
            f"p, over this table, is below {VERDICT_P} (D-046).",
            "",
            *_comparison_table(result["against_default"], names, verdict=True),
        ]
    if result["retrieval"]:
        lines += [
            "",
            "## What retrieval adds",
            "",
            "These describe and decide nothing (D-046).",
            "",
            *_comparison_table(result["retrieval"], names, verdict=False),
        ]
    if result["by_type"]:
        lines += _by_type_section(result)
    agent_row = strategies.get("s2")
    if agent_row is not None and agent_row["agent"] is not None:
        lines += _agent_section(agent_row)
    long_context_rows = [r for r in strategies.values() if r["long_context"]]
    if long_context_rows:
        lines += _long_context_section(long_context_rows)
    lines += [
        "",
        "Runs: judge "
        + ", ".join(f"`{r}`" for r in result["judge_runs"])
        + "; answers "
        + ", ".join(f"`{row['answers_run']}`" for row in strategies.values())
        + ".",
    ]
    return "\n".join(lines) + "\n"


def write_report(
    judge_runs: Sequence[Path],
    reports_dir: Path = config.REPORTS_DIR,
    name: str | None = None,
) -> Path:
    """Write reports/answers_<name>.md and its chart, reports/figures/answers_<name>.svg;
    `name` defaults to the split."""
    from support_rag.charts import answers_chart  # matplotlib is a dev dependency

    result = compare_strategies(judge_runs)
    split = next(iter(result["strategies"].values()))["split"]
    stem = f"answers_{name or split}"
    answers_chart(result, reports_dir / "figures" / f"{stem}.svg")
    path = reports_dir / f"{stem}.md"
    path.write_text(
        format_report(result, figure=f"figures/{stem}.svg"), encoding="utf-8", newline="\n"
    )
    return path
