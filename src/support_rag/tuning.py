"""Tuning on dev: candidate generator settings against the default, paired on the same
questions (D-033).

Each arm is one judge run (judge-v1 at high, D-031) over one answer run. Arms are grouped
by strategy and compared with that strategy's default arm: the baseline effort and k,
which are the tuned settings unless given (D-033's runs used effort medium and k=5). A
setting tried on several strategies, such as an effort level, is also compared pooled:
per question, the candidate's result minus the default's, averaged over the strategies,
so each question still counts once.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.answer_eval import DEFAULT_EFFORT, DEFAULT_K
from support_rag.answer_report import RETRIEVER_NAMES, STRATEGY_NAMES
from support_rag.runs import load_config, load_results, load_summary
from support_rag.stats import bootstrap_ci, holm, mcnemar_exact, paired_bootstrap_ci


@dataclass(frozen=True)
class Arm:
    strategy: str
    effort: str
    k: int | None  # S1 only
    correct: dict[str, bool | None]  # by question; None if the judge failed
    answers: dict[str, Any]  # the answer run's summary: tokens, cost, latency
    judge: tuple[str, str, str]  # model, effort, prompt version
    run_id: str
    retriever: str | None = None  # S1 only: "bm25" or "dense"
    unit: str = "article"  # what S1 reads: whole articles, or each one's best section (D-042)
    # By question: does the answer cite a feature request that isn't gold? (D-040)
    cites_feature_request: dict[str, bool] = field(default_factory=dict)

    @property
    def setting(self) -> str:
        parts = [f"effort {self.effort}"]
        if self.k is not None:
            parts.append(f"k={self.k}")
        if self.retriever is not None:
            parts.append(RETRIEVER_NAMES[self.retriever])
        if self.unit != "article":
            parts.append(f"best {self.unit} section")
        return ", ".join(parts)

    def is_default(self, effort: str, k: int, retriever: str | None = None) -> bool:
        if self.effort != effort or self.k not in (None, k) or self.unit != "article":
            return False
        return retriever is None or self.retriever in (None, retriever)


def load_arm(
    judge_run: Path,
    runs_dir: Path = config.RUNS_DIR,
    feature_requests: Collection[str] = frozenset(),
) -> Arm:
    judge_config = load_config(judge_run)
    if judge_config.get("kind") != "judge":
        raise ValueError(f"{judge_run} is not a judge run")
    answers_run = runs_dir / judge_config["answers_run"]
    answers_config = load_config(answers_run)
    cites: dict[str, bool] = {}
    if feature_requests:
        cites = {
            r["question_id"]: any(
                article_id in feature_requests and article_id not in r["gold_ids"]
                for article_id in r["cited_ids"]
            )
            for r in load_results(answers_run)
        }
    return Arm(
        strategy=answers_config["strategy"],
        effort=answers_config["effort"],
        k=answers_config.get("k"),
        correct={
            r["question_id"]: None if r["correctness"] is None else r["correctness"] == "correct"
            for r in load_results(judge_run)
        },
        answers=load_summary(answers_run),
        judge=(judge_config["model"], judge_config["effort"], judge_config["prompt_version"]),
        run_id=judge_config["run_id"],
        retriever=answers_config.get("retriever", {}).get("name"),
        unit=answers_config.get("unit", "article"),
        cites_feature_request=cites,
    )


def clears_bar(difference: float, low: float, *, cheaper: bool, margin: float | None) -> bool:
    """Whether a candidate may replace the default. In points, candidate minus default.

    A cheaper candidate has to show it isn't worse: it scores at least the default's rate,
    and its interval lies entirely above -margin (D-042). Any other candidate, or any
    candidate when no margin applies, has to be clearly better: its interval lies entirely
    above zero (D-033). Bounds are rounded, so floating-point dust can't decide a tie.
    """
    if cheaper and margin is not None:
        return difference >= 0 and round(low, 6) > -margin
    return round(low, 6) > 0


def _paired(candidate: Arm, default: Arm) -> tuple[list[bool], list[bool]]:
    questions = sorted(
        q for q, c in candidate.correct.items()
        if c is not None and default.correct.get(q) is not None
    )  # fmt: skip
    return [candidate.correct[q] for q in questions], [default.correct[q] for q in questions]


def compare_to_default(
    arms: Sequence[Arm],
    *,
    default_effort: str = DEFAULT_EFFORT,
    default_k: int = DEFAULT_K,
    default_retriever: str | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    """Every arm against its strategy's default. `default_retriever` tells S1 arms at the same
    k apart; `margin`, in points, applies D-042's bar to cheaper candidates."""
    if len({arm.judge for arm in arms}) != 1:
        raise ValueError("the judge runs use different judge settings")

    def is_default(arm: Arm) -> bool:
        return arm.is_default(default_effort, default_k, default_retriever)

    defaults = {}
    for arm in arms:
        if is_default(arm):
            if arm.strategy in defaults:
                raise ValueError(f"two default arms for {arm.strategy}")
            defaults[arm.strategy] = arm
    strategies = [s for s in STRATEGY_NAMES if any(arm.strategy == s for arm in arms)]
    if set(strategies) != set(defaults):
        raise ValueError(
            f"every strategy needs its default arm (effort {default_effort}, k={default_k})"
        )

    rows = []
    for arm in arms:
        graded = [float(c) for c in arm.correct.values() if c is not None]
        low, high = bootstrap_ci(graded)
        rows.append(
            {
                "strategy": arm.strategy,
                "setting": arm.setting,
                "default": is_default(arm),
                "n": len(graded),
                "pct_correct": 100 * sum(graded) / len(graded),
                "ci": (100 * low, 100 * high),
                "mean_input_tokens": arm.answers["mean_input_tokens"],
                "mean_output_tokens": arm.answers["mean_output_tokens"],
                "cost_per_answer_usd": arm.answers["cost_of_answers_usd"] / arm.answers["n"],
                "latency_p50_ms": arm.answers["latency_p50_ms"],
            }
        )
    rows.sort(key=lambda r: (strategies.index(r["strategy"]), not r["default"], r["setting"]))

    comparisons = []
    for arm in arms:
        if is_default(arm):
            continue
        default = defaults[arm.strategy]
        a, b = _paired(arm, default)
        low, high = paired_bootstrap_ci([float(x) for x in a], [float(x) for x in b])
        difference = 100 * (sum(a) - sum(b)) / len(a)
        cheaper = arm.answers["mean_input_tokens"] < default.answers["mean_input_tokens"]
        comparisons.append(
            {
                "strategy": arm.strategy,
                "setting": arm.setting,
                "n": len(a),
                "difference": difference,
                "ci": (100 * low, 100 * high),
                "clears_bar": clears_bar(difference, 100 * low, cheaper=cheaper, margin=margin),
                **mcnemar_exact(a, b),
            }
        )
    for comparison, adjusted in zip(
        comparisons, holm([c["p_value"] for c in comparisons]), strict=True
    ):
        comparison["p_holm"] = adjusted
    comparisons.sort(key=lambda c: (strategies.index(c["strategy"]), c["setting"]))

    return {
        "judge": arms[0].judge,
        "rows": rows,
        "comparisons": comparisons,
        "pooled": _pooled(arms, defaults, default_effort, default_k),
        "default": {"effort": default_effort, "k": default_k, "retriever": default_retriever},
        "margin": margin,
        "feature_requests": _feature_requests(arms),
        "runs": sorted(arm.run_id for arm in arms),
    }


def _feature_requests(arms: Sequence[Arm]) -> list[dict[str, Any]]:
    """Per arm, the answers that cite a feature request that isn't gold, split by whether the
    answer was correct (D-040, D-042). A failed judge call counts as neither."""
    rows = []
    for arm in arms:
        if not arm.cites_feature_request:
            continue
        cites = [(c, arm.cites_feature_request.get(q, False)) for q, c in arm.correct.items()]
        rows.append(
            {
                "setting": arm.setting,
                "not_correct": sum(c is False for c, _ in cites),
                "not_correct_citing": sum(c is False and cited for c, cited in cites),
                "correct_citing": sum(c is True and cited for c, cited in cites),
            }
        )
    return rows


def _pooled(
    arms: Sequence[Arm], defaults: Mapping[str, Arm], default_effort: str, default_k: int
) -> list[dict[str, Any]]:
    """Effort levels tried on more than one strategy, compared with the default per question
    averaged over those strategies. Questions stay the unit: three strategies' results on one
    question aren't independent, so they're averaged, not counted three times."""
    by_effort: dict[str, list[Arm]] = {}
    for arm in arms:
        if arm.effort != default_effort and arm.k in (None, default_k):
            by_effort.setdefault(arm.effort, []).append(arm)
    pooled = []
    for effort, group in sorted(by_effort.items()):
        if len(group) < 2:
            continue
        per_question: dict[str, list[float]] = {}
        for arm in group:
            default = defaults[arm.strategy]
            for q, c in arm.correct.items():
                if c is not None and default.correct.get(q) is not None:
                    per_question.setdefault(q, []).append(float(c) - float(default.correct[q]))
        differences = [sum(d) / len(d) for d in per_question.values()]
        low, high = bootstrap_ci(differences)
        pooled.append(
            {
                "setting": f"effort {effort}",
                "strategies": sorted(arm.strategy for arm in group),
                "n": len(differences),
                "difference": 100 * sum(differences) / len(differences),
                "ci": (100 * low, 100 * high),
            }
        )
    return pooled


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def format_report(name: str, result: Mapping[str, Any]) -> str:
    model, effort, prompt_version = result["judge"]
    default = result["default"]
    margin = result["margin"]
    baseline = f"effort {default['effort']}, k={default['k']} for S1"
    if default["retriever"] is not None:
        baseline += f", {RETRIEVER_NAMES[default['retriever']]}"
    rule = (
        "A candidate replaces the default only if its paired 95% interval is entirely above "
        "zero (D-033)."
    )
    if margin is not None:
        rule = (
            "A candidate with fewer input tokens than the default replaces it only if it scores "
            "at least the default's % correct and its paired 95% interval lies entirely above "
            f"-{margin:g} points. Any other candidate needs its interval entirely above zero "
            "(D-042)."
        )
    lines = [
        f"# Tuning on dev: {name}",
        "",
        "Generated by `srag report-tuning`. Candidates are compared with the default "
        f"({baseline}) on the same questions. Judge: {model}, rubric {prompt_version}, "
        f"effort {effort}. {rule}",
        "",
        "## Settings",
        "",
        "| Strategy | Setting | % correct [95% CI] | Input tokens | Output tokens "
        "| Cost per answer | Latency p50 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        low, high = r["ci"]
        latency = r["latency_p50_ms"]
        setting = r["setting"] + (" (default)" if r["default"] else "")
        lines.append(
            f"| {STRATEGY_NAMES[r['strategy']]} | {setting} | {r['pct_correct']:.1f}% "
            f"[{low:.1f}, {high:.1f}] | {r['mean_input_tokens']:,.0f} "
            f"| {r['mean_output_tokens']:,.0f} | ${r['cost_per_answer_usd']:.5f} "
            f"| {'n/a' if latency is None else f'{latency / 1000:.1f} s'} |"
        )
    lines += [
        "",
        "## Against the default, per strategy",
        "",
        "| Strategy | Candidate | Questions | Difference, points [95% CI] "
        "| Only the candidate right | Only the default right | McNemar p | Holm p |"
        + (" Clears its bar |" if margin is not None else ""),
        "|---|---|---:|---:|---:|---:|---:|---:|" + ("---|" if margin is not None else ""),
    ]
    for c in result["comparisons"]:
        low, high = c["ci"]
        verdict = "" if margin is None else f" {'yes' if c['clears_bar'] else 'no'} |"
        lines.append(
            f"| {STRATEGY_NAMES[c['strategy']]} | {c['setting']} | {c['n']} "
            f"| {c['difference']:+.1f} [{low:+.1f}, {high:+.1f}] | {c['only_a']} "
            f"| {c['only_b']} | {_p(c['p_value'])} | {_p(c['p_holm'])} |{verdict}"
        )
    if result["feature_requests"]:
        lines += [
            "",
            "## Answers citing a feature request that isn't gold",
            "",
            "Not correct means graded partial or incorrect, and a decline cites nothing (D-040).",
            "",
            "| Setting | Not correct | Of those, citing one | Correct answers citing one |",
            "|---|---:|---:|---:|",
        ]
        for row in result["feature_requests"]:
            lines.append(
                f"| {row['setting']} | {row['not_correct']} | {row['not_correct_citing']} "
                f"| {row['correct_citing']} |"
            )
    if result["pooled"]:
        lines += [
            "",
            "## Against the default, pooled over strategies",
            "",
            "Per question, the candidate's result minus the default's, averaged over the "
            "strategies; the interval resamples questions.",
            "",
            "| Candidate | Strategies | Questions | Mean difference, points [95% CI] |",
            "|---|---|---:|---:|",
        ]
        for p in result["pooled"]:
            low, high = p["ci"]
            lines.append(
                f"| {p['setting']} | {', '.join(p['strategies'])} | {p['n']} "
                f"| {p['difference']:+.1f} [{low:+.1f}, {high:+.1f}] |"
            )
    lines += ["", "Judge runs: " + ", ".join(f"`{r}`" for r in result["runs"]) + "."]
    return "\n".join(lines) + "\n"


def write_report(
    name: str,
    judge_runs: Sequence[Path],
    *,
    default_effort: str = DEFAULT_EFFORT,
    default_k: int = DEFAULT_K,
    default_retriever: str | None = None,
    margin: float | None = None,
    feature_requests: Collection[str] = frozenset(),
    reports_dir: Path = config.REPORTS_DIR,
) -> Path:
    arms = [load_arm(run, feature_requests=feature_requests) for run in judge_runs]
    result = compare_to_default(
        arms,
        default_effort=default_effort,
        default_k=default_k,
        default_retriever=default_retriever,
        margin=margin,
    )
    path = reports_dir / f"tuning_{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(name, result), encoding="utf-8", newline="\n")
    return path
