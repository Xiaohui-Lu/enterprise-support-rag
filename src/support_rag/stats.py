"""Confidence intervals and paired tests over questions (D-006, D-032).

The question is the unit: every strategy answers the same questions, so intervals come
from resampling questions, and comparisons pair each question's results across strategies.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from math import comb

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
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    p_value = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "p_value": p_value}


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
