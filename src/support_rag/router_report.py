"""The router's report: R2 against S1 and against random escalation, and R2 on the cost
frontier, written to reports/router_<name>.md (D-048).

Everything here comes from finished runs: the router's own run, and the S1 and S3 runs it
repeats. The other rules and budgets are replays of the same `route` function on those runs.
"""

from __future__ import annotations

import statistics
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.answer_report import (
    DEFAULT_S1,
    QUESTION_TYPES,
    VERDICT_P,
    _frontier_section,
    _p,
    _seconds,
    compare_strategies,
    load_judged,
)
from support_rag.router import BUDGET, ESCALATION_K, SIGNALS, THRESHOLDS, against_random, oracle
from support_rag.router import route as route_for
from support_rag.stats import paired_comparison, with_holm

ROUTER = "r2"
TARGET = f"s3-k{ESCALATION_K}"
# The strategies a real router could choose from: not closed book, not the superseded BM25,
# and not S*, which needs the answer key.
ORACLE_POOL = ("s1-dense", "s2", "s3-k20", "s3-k50", "s3-k100")
VARIANTS: tuple[tuple[str, int, frozenset[str]], ...] = (
    ("Both signals, 10% budget", 10, SIGNALS),
    ("Both signals, 20% budget (R2)", 20, SIGNALS),
    ("Both signals, 30% budget", 30, SIGNALS),
    ("Both signals, 50% budget", 50, SIGNALS),
    ("Weak top score only, 20% budget", 20, frozenset({"score"})),
    ("S1's decline only", BUDGET, frozenset({"decline"})),
)
ROUTE_NAMES = {
    "s1": "S1's answer stands",
    "score": "Weak top score: straight to S3 at K=100",
    "decline": "S1 declined: S3 at K=100 answers again",
}


def _mean(values: Collection[float]) -> float:
    return sum(values) / len(values)


def compare_router(
    judge_runs: Sequence[Path], *, runs_dir: Path = config.RUNS_DIR
) -> dict[str, Any]:
    """R2's verdicts, its replays and its place on the frontier. Needs judge runs for R2, S1
    on dense search and S3 at K=100 on the same questions; any others join the frontier and
    the full oracle (D-048)."""
    judged = load_judged(judge_runs, runs_dir=runs_dir)
    missing = [key for key in (ROUTER, DEFAULT_S1, TARGET) if key not in judged]
    if missing:
        raise ValueError("give judge runs for R2, S1 on dense search and S3 at K=100")
    r2, s1, s3 = judged[ROUTER], judged[DEFAULT_S1], judged[TARGET]
    questions = sorted(
        q for q in r2["correct"] if all(arm["correct"][q] is not None for arm in (r2, s1, s3))
    )
    n = len(questions)
    threshold = r2["answers_config"]["threshold"]
    top = {q: r2["records"][q]["router"]["top_score"] for q in questions}
    declined = {q: s1["records"][q]["declined"] for q in questions}
    routes = {q: r2["records"][q]["router"]["route"] for q in questions}
    right = {
        key: {q: bool(arm["correct"][q]) for q in questions}
        for key, arm in judged.items()
        if all(arm["correct"][q] is not None for q in questions)
    }
    cost = {
        key: {q: arm["records"][q]["cost_usd"] for q in questions} for key, arm in judged.items()
    }

    # The live run must be what its rule gives on the runs it repeats (D-008).
    replayed = {q: route_for(top[q], declined[q], threshold=threshold) for q in questions}
    chosen = {q: right[DEFAULT_S1][q] if routes[q] == "s1" else right[TARGET][q] for q in questions}

    gains = {q: int(right[TARGET][q]) - int(right[DEFAULT_S1][q]) for q in questions}
    escalated = {q for q in questions if routes[q] != "s1"}
    verdicts = with_holm(
        [
            {
                "name": "R2 against random escalation at the same rate",
                **against_random(escalated, gains),
            },
            {"name": "R2 against S1", **paired_comparison(right[ROUTER], right[DEFAULT_S1])},
        ]
    )
    for verdict in verdicts:
        verdict["passes"] = verdict["ci"][0] > 0 and verdict["p_holm"] < VERDICT_P

    rate = len(escalated) / n
    pct = {key: 100 * _mean([float(c) for c in by_q.values()]) for key, by_q in right.items()}
    control = {
        "pct_correct": pct[DEFAULT_S1] + rate * (pct[TARGET] - pct[DEFAULT_S1]),
        "cost_per_answer_usd": _mean(
            [(1 - rate) * cost[DEFAULT_S1][q] + rate * cost[TARGET][q] for q in questions]
        ),
    }

    variants = []
    for label, budget, signals in VARIANTS:
        taken = {
            q: route_for(top[q], declined[q], threshold=THRESHOLDS[budget], signals=signals)
            for q in questions
        }
        up = {q for q in questions if taken[q] != "s1"}
        paid = {"s1": (DEFAULT_S1,), "score": (TARGET,), "decline": (DEFAULT_S1, TARGET)}
        variants.append(
            {
                "label": label,
                "threshold": THRESHOLDS[budget] if "score" in signals else None,
                "escalated": len(up),
                "pct_correct": 100
                * _mean([float(right[TARGET if q in up else DEFAULT_S1][q]) for q in questions]),
                "against_s1": 100 * sum(gains[q] for q in up) / n,
                "against_random": 100
                * (sum(gains[q] for q in up) - len(up) / n * sum(gains.values()))
                / n,
                "cost_per_answer_usd": _mean(
                    [sum(cost[key][q] for key in paid[taken[q]]) for q in questions]
                ),
            }
        )

    pool = [key for key in ORACLE_POOL if key in right]
    oracles = [
        oracle({k: right[k] for k in keys}, {k: cost[k] for k in keys})
        for keys in ((DEFAULT_S1, TARGET), pool)
        if len(keys) > 1
    ]

    by_type = []
    for label, belongs in QUESTION_TYPES:
        group = [q for q in questions if belongs(r2["records"][q])]
        if group:
            by_type.append(
                {
                    "label": label,
                    "n": len(group),
                    "escalated": sum(q in escalated for q in group),
                    "pct": {
                        key: 100 * _mean([float(right[key][q]) for q in group])
                        for key in (DEFAULT_S1, ROUTER, TARGET)
                    },
                }
            )

    comparison = compare_strategies(judge_runs, runs_dir=runs_dir)
    strategies = comparison["strategies"]
    gap = pct[TARGET] - pct[DEFAULT_S1]
    return {
        "comparison": comparison,
        "split": r2["answers_config"]["split"],
        "router": strategies[ROUTER],
        "threshold": threshold,
        "budget": r2["answers_config"]["budget"],
        "dirty": r2["answers_config"].get("git", {}).get("dirty"),
        "n": n,
        "routes": {name: sum(r == name for r in routes.values()) for name in ROUTE_NAMES},
        "replay": {
            "same_route": sum(replayed[q] == routes[q] for q in questions),
            "same_grade": sum(chosen[q] == right[ROUTER][q] for q in questions),
        },
        "verdicts": verdicts,
        "becomes_default": all(v["passes"] for v in verdicts),
        "rows": {key: strategies[key] for key in (DEFAULT_S1, ROUTER, TARGET)},
        "control": control,
        "share_of_gain": (pct[ROUTER] - pct[DEFAULT_S1]) / gap if gap > 0 else None,
        "share_of_cost": strategies[ROUTER]["cost_per_answer_usd"]
        / strategies[TARGET]["cost_per_answer_usd"],
        "variants": variants,
        "oracles": oracles,
        "by_type": by_type,
        "top_score": {
            "median": statistics.median(top.values()),
            "below_threshold": sum(score < threshold for score in top.values()),
        },
    }


def format_router_report(result: Mapping[str, Any], *, figure: str | None = None) -> str:
    comparison = result["comparison"]
    strategies = comparison["strategies"]
    short = {key: row["short_name"] for key, row in strategies.items()}
    router = result["router"]
    n = result["n"]
    judge = comparison["judge"]
    lines = [
        f"# The router on {result['split']}",
        "",
        "Generated by `srag report-router`.",
        "",
        f"- R2 (D-048): if dense search's top score is below {result['threshold']}, S3 at "
        f"K={ESCALATION_K} answers straight away; otherwise S1 answers from the top 10, and if "
        f"it declines, S3 at K={ESCALATION_K} answers again. The threshold is a "
        f"{result['budget']}% budget over the real dev questions' top scores, not a tuned "
        "value (D-047).",
        f"- Judge: {judge['model']}, rubric {judge['prompt_version']}, effort {judge['effort']}. "
        "% correct counts only answers graded correct; a decline is incorrect (D-025).",
        "- R2's calls repeat S1's and S3's, so the call cache answered them. Its cost adds up "
        "the calls it made, and its latency adds up their original real-time latencies, which "
        "the cache keeps.",
        f"- R2's run: `{router['answers_run']}`, from commit {str(router['commit'])[:7]}"
        + (" with uncommitted changes" if result["dirty"] else "")
        + (f", milestone {router['milestone']}." if router["milestone"] else ".")
        + (
            " These are dev results: nothing on dev decides anything (D-047)."
            if result["split"] != "test"
            else ""
        ),
        "",
        "## Where the questions went",
        "",
        "| Route | Questions | Share |",
        "|---|---:|---:|",
    ]
    for name, label in ROUTE_NAMES.items():
        count = result["routes"][name]
        lines.append(f"| {label} | {count} | {100 * count / n:.1f}% |")
    escalated = n - result["routes"]["s1"]
    lines += [
        f"| Sent to S3 in all | {escalated} | {100 * escalated / n:.1f}% |",
        "",
        f"The median top score was {result['top_score']['median']:.4f}.",
        "",
        "## The verdicts",
        "",
        "Each comparison needs a paired 95% interval entirely above zero and a Holm p below "
        "0.05 over these two (D-048). The first p is a one-sided permutation test over 10,000 "
        "random sets of the same size; the second is McNemar's exact test.",
        "",
        "| Comparison | Questions | Difference, points [95% CI] | Detail | p | Holm p | Passes |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for v in result["verdicts"]:
        low, high = v["ci"]
        if "router_gain" in v:
            detail = (
                f"R2's {v['escalated']} gained {v['router_gain']:+d}; "
                f"a random {v['escalated']} gains {v['random_gain']:+.1f} on average"
            )
        else:
            detail = f"only R2 right {v['only_a']}, only S1 right {v['only_b']}"
        lines.append(
            f"| {v['name']} | {v['n']} | {v['difference']:+.1f} [{low:+.1f}, {high:+.1f}] "
            f"| {detail} | {_p(v['p_value'])} | {_p(v['p_holm'])} "
            f"| {'yes' if v['passes'] else 'no'} |"
        )
    lines += [
        "",
        "R2 becomes the default: "
        + ("**yes**, both comparisons pass." if result["becomes_default"] else "**no**.")
        + ("" if result["becomes_default"] else " S1 on dense search stays the default (D-048)."),
        "",
        "## R2 beside the strategies it chooses between",
        "",
        "| Strategy | % correct [95% CI] | Cost per answer | Latency p50 | Latency p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in result["rows"].items():
        low, high = row["ci"]
        lines.append(
            f"| {row['name']} | {row['pct_correct']:.1f}% [{low:.1f}, {high:.1f}] "
            f"| ${row['cost_per_answer_usd']:.5f} | {_seconds(row['latency_p50_ms'])} "
            f"| {_seconds(row['latency_p95_ms'])} |"
        )
        if key == ROUTER:
            control = result["control"]
            lines.append(
                f"| Random escalation at R2's rate, expected | {control['pct_correct']:.1f}% "
                f"| ${control['cost_per_answer_usd']:.5f} | | |"
            )
    share = result["share_of_gain"]
    cost_share = f"{100 * result['share_of_cost']:.0f}%"
    lines += [
        "",
        f"R2 keeps {100 * share:.0f}% of S3 at K={ESCALATION_K}'s gain over S1, at {cost_share} "
        "of its cost per answer."
        if share is not None
        else f"S3 at K={ESCALATION_K} isn't ahead of S1 here, so there's no gain to keep. R2 "
        f"costs {cost_share} of what S3 does per answer.",
        "",
        "## Each signal alone, and other budgets",
        "",
        "Replays of the same rule on S1's and S3's runs, with thresholds fixed in D-048. "
        "These describe and decide nothing.",
        "",
        "| Rule | Threshold | Sent to S3 | % correct | Against S1, points "
        "| Against random, points | Cost per answer |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in result["variants"]:
        threshold = "n/a" if v["threshold"] is None else f"{v['threshold']}"
        lines.append(
            f"| {v['label']} | {threshold} | {v['escalated']} | {v['pct_correct']:.1f}% "
            f"| {v['against_s1']:+.1f} | {v['against_random']:+.1f} "
            f"| ${v['cost_per_answer_usd']:.5f} |"
        )
    lines += [
        "",
        "## The hindsight oracle",
        "",
        "For each question, the cheapest strategy that got it right: the ceiling for any "
        "router choosing among them.",
        "",
        "| Strategies it may choose | % correct | Cost per answer |",
        "|---|---:|---:|",
    ]
    for o in result["oracles"]:
        names = ", ".join(short[key] for key in sorted(o["strategies"], key=ORACLE_POOL.index))
        lines.append(f"| {names} | {o['pct_correct']:.1f}% | ${o['cost_per_answer_usd']:.5f} |")
    lines += [
        "",
        "## By question type",
        "",
        "Directional only: the groups are small.",
        "",
        "| Type | Questions | Sent to S3 | S1 | R2 | S3 K=100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in result["by_type"]:
        pct = t["pct"]
        lines.append(
            f"| {t['label']} | {t['n']} | {t['escalated']} | {pct[DEFAULT_S1]:.1f}% "
            f"| {pct[ROUTER]:.1f}% | {pct[TARGET]:.1f}% |"
        )
    replay = result["replay"]
    lines += [
        "",
        "## The run against a replay",
        "",
        "Replaying R2's rule on S1's and S3's runs gives the same route on "
        f"{replay['same_route']} of {n} questions and the same grade on {replay['same_grade']} "
        f"of {n}. A difference means a call missed the cache, or the runs differ.",
    ]
    lines += _frontier_section(comparison, figure)
    lines += ["", "Runs: judge " + ", ".join(f"`{r}`" for r in comparison["judge_runs"]) + "."]
    return "\n".join(lines) + "\n"


def write_router_report(
    judge_runs: Sequence[Path],
    reports_dir: Path = config.REPORTS_DIR,
    name: str | None = None,
) -> Path:
    """Write reports/router_<name>.md and its chart; `name` defaults to the split."""
    from support_rag.charts import answers_chart  # matplotlib is a dev dependency

    result = compare_router(judge_runs)
    stem = f"router_{name or result['split']}"
    answers_chart(result["comparison"], reports_dir / "figures" / f"{stem}.svg")
    path = reports_dir / f"{stem}.md"
    path.write_text(
        format_router_report(result, figure=f"figures/{stem}.svg"), encoding="utf-8", newline="\n"
    )
    return path
