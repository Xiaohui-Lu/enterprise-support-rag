"""Confidence intervals and paired tests over questions (D-006, D-032).

The question is the unit: every strategy answers the same questions, so intervals come
from resampling questions, and comparisons pair each question's results across strategies.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from math import comb
from typing import Any

BOOTSTRAP_SEED = 20261102
N_RESAMPLES = 10_000


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values`, one value per question.

    Draw as many questions as there are, with replacement, take the mean, and repeat;
    the middle `confidence` share of those means is the interval. It asks how much the
    mean would move with a different draw of questions of the same kind.
    """
    if not values:
        raise ValueError("need at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    tail = (1 - confidence) / 2
    return means[int(tail * n_resamples)], means[int((1 - tail) * n_resamples) - 1]


def paired_bootstrap_ci(
    a: Sequence[float], b: Sequence[float], **kwargs: float
) -> tuple[float, float]:
    """Interval for mean(a) - mean(b), where a[i] and b[i] are the same question.

    Resampling questions and recomputing both means is the same as resampling the
    per-question differences, so that's what this does. Pairing removes the variation
    between easy and hard questions, which both strategies share, so the interval is
    narrower than comparing two separate intervals.
    """
    differences = [x - y for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(differences, **kwargs)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float | int]:
    """McNemar's exact test on paired correct/incorrect results (a[i], b[i]: one question).

    Questions both get right, or both get wrong, say nothing about which is better. Of the
    questions where exactly one is right, each is a fair coin flip if the two strategies
    are equally good, so the p-value is a two-sided exact binomial test: the sign test.
    """
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    return {"only_a": only_a, "only_b": only_b, "p_value": sign_test(only_a, only_b)}


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test: if neither side is truly better, each question that
    changed is a coin flip, so how unlikely is a split at least this uneven?"""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Running several tests raises the chance that one comes out "significant" by luck.
    Holm multiplies the smallest p-value by the number of tests, the next by one fewer,
    and so on, never letting an adjusted value drop below the one before it.
    """
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def paired_comparison(a: Mapping[str, bool | None], b: Mapping[str, bool | None]) -> dict[str, Any]:
    """`a` against `b`, question by question, on the questions both have a grade for.

    Each maps a question ID to correct (True), not correct (False) or no grade (None, a
    judge failure). Returns the difference in % correct, its paired-bootstrap interval and
    McNemar's exact test.
    """
    questions = sorted(q for q in a if a[q] is not None and b.get(q) is not None)
    first = [bool(a[q]) for q in questions]
    second = [bool(b[q]) for q in questions]
    low, high = paired_bootstrap_ci([float(x) for x in first], [float(x) for x in second])
    return {
        "n": len(questions),
        "difference": 100 * (sum(first) - sum(second)) / len(questions),
        "ci": (100 * low, 100 * high),
        **mcnemar_exact(first, second),
    }


def with_holm(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add "p_holm" to each comparison, adjusted over this list only."""
    for comparison, adjusted in zip(
        comparisons, holm([c["p_value"] for c in comparisons]), strict=True
    ):
        comparison["p_holm"] = adjusted
    return comparisons
