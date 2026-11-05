import pytest

from support_rag.stats import bootstrap_ci, holm, mcnemar_exact, paired_bootstrap_ci, sign_test


def test_a_constant_has_no_uncertainty():
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0)


def test_half_right_on_50_questions_is_about_plus_or_minus_14_points():
    low, high = bootstrap_ci([1.0] * 25 + [0.0] * 25)
    assert 0.34 <= low <= 0.38
    assert 0.62 <= high <= 0.66


def test_the_same_seed_gives_the_same_interval():
    values = [i / 7 for i in range(20)]  # varied, so a different draw shows up
    assert bootstrap_ci(values) == bootstrap_ci(values)
    assert bootstrap_ci(values) != bootstrap_ci(values, seed=1)


def test_pairing_removes_what_both_strategies_share():
    # Both strategies get the same 25 questions right, and b also gets 5 more: every
    # question differs by 0 or by 1 in the same direction.
    a = [1.0] * 25 + [0.0] * 25
    b = [1.0] * 30 + [0.0] * 20
    low, high = paired_bootstrap_ci(b, a)
    assert 0 < low <= high
    unpaired_width = (
        bootstrap_ci(b)[1] - bootstrap_ci(b)[0] + bootstrap_ci(a)[1] - bootstrap_ci(a)[0]
    )
    assert high - low < unpaired_width / 2


def test_mcnemar_uses_only_the_questions_where_the_two_disagree():
    # 8 questions only a gets right, 1 only b: p = 2 * (C(9,0) + C(9,1)) / 2**9.
    a = [True] * 8 + [False] * 1 + [True] * 30 + [False] * 11
    b = [False] * 8 + [True] * 1 + [True] * 30 + [False] * 11
    result = mcnemar_exact(a, b)
    assert (result["only_a"], result["only_b"]) == (8, 1)
    assert result["p_value"] == pytest.approx(2 * 10 / 512)


def test_no_disagreement_means_no_evidence():
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_sign_test_depends_only_on_the_split_not_its_direction():
    assert sign_test(8, 1) == pytest.approx(2 * 10 / 512)
    assert sign_test(1, 8) == sign_test(8, 1)
    assert sign_test(0, 0) == 1.0


def test_holm_scales_the_smallest_p_value_most():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
