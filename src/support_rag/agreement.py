"""Agreement between two raters grading the same items: the judge and the labels (D-030).

Percent agreement alone flatters a rater on a skewed scale. If most answers are correct, a
judge that always says "correct" agrees most of the time and has learned nothing. Cohen's
kappa subtracts the agreement the two raters would reach by chance, given how often each
one uses each grade.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """The share of items the two raters grade the same."""
    _check(a, b)
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """(p_o - p_e) / (1 - p_e): agreement beyond chance, as a share of the most possible.

    p_o is the observed agreement. p_e is the agreement expected if each rater kept its own
    grade frequencies but assigned them at random: for each grade, the chance both pick it,
    summed. 1 is perfect agreement, 0 is chance level, below 0 is worse than chance. None
    when chance agreement is already 1 (both raters give one and the same grade to
    everything), where kappa is undefined.
    """
    _check(a, b)
    n = len(a)
    count_a, count_b = Counter(a), Counter(b)
    p_e = sum(count_a[grade] * count_b[grade] for grade in count_a) / n**2
    if p_e == 1:
        return None
    return (percent_agreement(a, b) - p_e) / (1 - p_e)


def _check(a: Sequence[str], b: Sequence[str]) -> None:
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty lists of grades for the same items")
