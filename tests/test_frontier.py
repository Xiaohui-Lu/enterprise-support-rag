from support_rag.frontier import FrontierPoint, pareto_frontier


def _names(points):
    return [p.name for p in pareto_frontier(points)]


def test_a_point_beaten_on_both_counts_is_dropped():
    points = [FrontierPoint("cheap", 1.0, 50.0), FrontierPoint("worse", 2.0, 40.0)]
    assert _names(points) == ["cheap"]


def test_the_frontier_runs_cheapest_first():
    points = [
        FrontierPoint("dear", 3.0, 90.0),
        FrontierPoint("cheap", 1.0, 50.0),
        FrontierPoint("middle", 2.0, 70.0),
    ]
    assert _names(points) == ["cheap", "middle", "dear"]


def test_a_dearer_point_must_score_strictly_higher():
    points = [FrontierPoint("cheap", 1.0, 50.0), FrontierPoint("same score", 2.0, 50.0)]
    assert _names(points) == ["cheap"]


def test_at_equal_cost_only_the_more_accurate_point_stays():
    points = [FrontierPoint("worse", 1.0, 50.0), FrontierPoint("better", 1.0, 60.0)]
    assert _names(points) == ["better"]


def test_of_two_identical_points_the_first_given_stays():
    points = [FrontierPoint("first", 1.0, 50.0), FrontierPoint("second", 1.0, 50.0)]
    assert _names(points) == ["first"]


def test_a_point_beaten_by_a_non_neighbour_is_dropped():
    # "dip" is dearer than "cheap" but scores lower, though it beats "middle" on cost.
    points = [
        FrontierPoint("cheap", 1.0, 60.0),
        FrontierPoint("dip", 1.5, 55.0),
        FrontierPoint("middle", 2.0, 70.0),
    ]
    assert _names(points) == ["cheap", "middle"]


def test_dev_numbers_keep_closed_book_and_dense_s1():
    # Dev's cost per answer and % correct (D-042, D-043, D-044); S* is never passed in.
    points = [
        FrontierPoint("S0", 0.00036, 24.0),
        FrontierPoint("S1 BM25", 0.00169, 76.0),
        FrontierPoint("S1 dense", 0.00122, 90.0),
        FrontierPoint("S2", 0.00098, 84.0),
        FrontierPoint("S3 K=20", 0.00219, 88.0),
        FrontierPoint("S3 K=100", 0.00808, 90.0),
    ]
    assert _names(points) == ["S0", "S2", "S1 dense"]


def test_no_points_make_an_empty_frontier():
    assert pareto_frontier([]) == []
