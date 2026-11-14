"""The cost frontier (D-046): the strategies that no other strategy beats on both counts.

A single "accuracy per dollar" number ranks the cheapest strategy first, because it treats a
wrong answer as free. A frontier keeps every strategy worth considering instead: for each
one on it, nothing else is both cheaper and more accurate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierPoint:
    name: str
    cost: float  # dollars per answer, or seconds: anything where less is better
    pct_correct: float


def pareto_frontier(points: Sequence[FrontierPoint]) -> list[FrontierPoint]:
    """The points that nothing else beats, cheapest first.

    A point is beaten when another point costs no more and scores at least as well. So:
    - a dearer point stays only if it scores strictly higher than every cheaper point;
    - at equal cost, only the most accurate point stays;
    - of two identical points, the first one given stays.

    Along the result, cost and % correct both rise.
    """
    frontier: list[FrontierPoint] = []
    # Cheapest first and, at equal cost, most accurate first. sorted() is stable, so of two
    # identical points the first one given comes first.
    for point in sorted(points, key=lambda p: (p.cost, -p.pct_correct)):
        # Everything kept so far is no dearer than this point, and the last one kept is the
        # most accurate of them. This point stays only if it beats that.
        if not frontier or point.pct_correct > frontier[-1].pct_correct:
            frontier.append(point)
    return frontier
