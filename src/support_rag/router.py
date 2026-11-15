"""The router, R2: S1 unless retrieval looks weak or S1 declines, and then S3 (D-047, D-048).

Dense search ranks each question's articles once. A weak best match, a top score below the
threshold, goes straight to S3 at K=100. Otherwise S1 answers from the top 10, and if S1
declines, S3 at K=100 answers again and that answer stands.

The threshold is a budget, not a tuned value. 0.5615 is the 11th-lowest top score of the 50
real dev questions, so 10 of them (20%) fall below it, and no grade went into it (D-047).

The rest of this module scores a router against what it has to beat: S1, and random
escalation at the same rate (D-048).
"""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping
from typing import Any, Literal

from support_rag.stats import bootstrap_ci

ESCALATION_K = 100  # S3 at K=100, M2's most accurate strategy (D-044, D-046)
# Escalation budget -> top-score threshold. For a budget of b%, the threshold is the
# (b/2 + 1)th-lowest top score of the 50 real dev questions, so b% of them fall below it.
THRESHOLDS = {10: 0.5313, 20: 0.5615, 30: 0.5889, 50: 0.6415}
BUDGET = 20  # about the share of multi-article questions (D-048)
THRESHOLD = THRESHOLDS[BUDGET]
SIGNALS = frozenset({"score", "decline"})
PERMUTATIONS = 10_000
PERMUTATION_SEED = 20261114

Route = Literal["s1", "score", "decline"]


def skips_s1(top_score: float, threshold: float = THRESHOLD) -> bool:
    """True when dense search's best match is weak enough to go straight to S3."""
    return top_score < threshold


def route(
    top_score: float,
    s1_declined: bool,
    *,
    threshold: float = THRESHOLD,
    signals: Collection[str] = SIGNALS,
) -> Route:
    """Where R2 sends a question.

    - "score": the best match is weak, so S3 answers and S1 never runs.
    - "decline": S1 answered, declined, and S3 answered again.
    - "s1": S1's answer stands.

    `s1_declined` doesn't matter on the "score" route, where S1 doesn't run. `signals` turns
    a rule off, so each can be reported alone (D-048).

    top 0.50, threshold 0.5615                ->  "score"
    top 0.70, S1 declined                     ->  "decline"
    top 0.70, S1 answered                     ->  "s1"
    top 0.50, S1 declined, signals {decline}  ->  "decline"
    """
    if "score" in signals and skips_s1(top_score, threshold):
        return "score"
    if "decline" in signals and s1_declined:
        return "decline"
    return "s1"


def against_random(
    escalated: Collection[str],
    gains: Mapping[str, int],
    *,
    draws: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """The router against random escalation at the same rate (D-048).

    `gains[q]` is S3's grade minus S1's on question q: +1 if only S3 got it right, -1 if
    only S1 did, 0 otherwise. Sending a set of questions to S3 changes S1's number of correct
    answers by the sum of their gains. A random set of m of the n questions is expected to
    gain m/n of the total; the router's signal is worth something only if its own set gains
    more than that.

    Returns the difference in points of % correct, with a paired-bootstrap interval, and a
    one-sided permutation p: the share of random sets of m questions that gain at least as
    much as the router's. The router's own set counts as one of them, so p is never zero.

    gains {a: +1, b: 0, c: -1, d: 0}, escalated {a}  ->  router +1, random expected 0.25 x 0
    """
    questions = sorted(gains)
    if not set(escalated) <= set(questions):
        raise ValueError("every escalated question needs a gain")
    n, m = len(questions), len(escalated)
    rate = m / n
    total = sum(gains.values())
    router_gain = sum(gains[q] for q in escalated)
    # Per question: the gain the router's choice took, minus what a random choice takes on
    # average. Their mean is the whole difference, so a bootstrap over them gives its interval.
    differences = [((q in escalated) - rate) * gains[q] for q in questions]
    low, high = bootstrap_ci(differences)
    rng = random.Random(seed)
    values = [gains[q] for q in questions]
    as_good = sum(sum(rng.sample(values, m)) >= router_gain for _ in range(draws))
    return {
        "n": n,
        "escalated": m,
        "router_gain": router_gain,
        "random_gain": rate * total,
        "difference": 100 * (router_gain - rate * total) / n,
        "ci": (100 * low, 100 * high),
        "p_value": (1 + as_good) / (1 + draws),
    }


def oracle(
    correct: Mapping[str, Mapping[str, bool]], cost: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    """Hindsight: for each question, the cheapest strategy that got it right, or the cheapest
    one if none did. `correct` and `cost` map a strategy to its value on each question.

    S1 right at $1, S3 right at $5  ->  S1, $1
    S1 wrong at $1, S3 right at $5  ->  S3, $5
    both wrong                      ->  S1, $1, counted wrong
    """
    strategies = sorted(correct)
    questions = sorted(correct[strategies[0]])
    right = total_cost = 0.0
    for q in questions:
        winners = [s for s in strategies if correct[s][q]] or strategies
        right += any(correct[s][q] for s in strategies)
        total_cost += min(cost[s][q] for s in winners)
    return {
        "strategies": strategies,
        "n": len(questions),
        "pct_correct": 100 * right / len(questions),
        "cost_per_answer_usd": total_cost / len(questions),
    }
