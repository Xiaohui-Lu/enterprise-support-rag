import pytest

from support_rag.gate import changed_questions, check, decide

QUESTIONS = [f"q{i:02}" for i in range(50)]


def _items(wrong=(), key="k"):
    """Every question right except `wrong`, each with request key `key` + its ID."""
    return {q: {"request_key": f"{key}{q}", "correct": q not in wrong} for q in QUESTIONS}


def test_identical_requests_change_nothing():
    baseline = _items()
    assert changed_questions({q: f"k{q}" for q in QUESTIONS}, baseline) == []


def test_a_new_or_different_request_counts_as_changed():
    baseline = _items()
    keys = {q: f"k{q}" for q in QUESTIONS} | {"q07": "other", "q99": "knew"}
    assert changed_questions(keys, baseline) == ["q07", "q99"]


def test_with_nothing_changed_the_gate_passes_without_comparing():
    decision = decide(_items(), _items(), [])
    assert decision["passed"] and "comparison" not in decision


def test_the_first_run_becomes_the_baseline():
    assert decide(_items(), None, QUESTIONS)["passed"]


def test_a_change_within_the_margin_passes():
    baseline = _items(wrong={"q01", "q02"})
    after = _items(wrong={"q02", "q03"})  # one fixed, one broken
    assert decide(after, baseline, ["q01", "q03"])["passed"]


def test_a_large_regression_fails():
    broken = {f"q{i:02}" for i in range(10)}
    decision = decide(_items(wrong=broken), _items(), sorted(broken))
    assert not decision["passed"]
    assert decision["comparison"]["difference"] == pytest.approx(-20.0)


def test_a_deliberate_change_can_be_accepted_and_says_so():
    broken = {f"q{i:02}" for i in range(10)}
    decision = decide(_items(wrong=broken), _items(), sorted(broken), accept=True)
    assert decision["passed"] and decision["accepted"]


def _gate(tree="t1", passed=True):
    return {"src_tree": tree, "commit": "abc1234def", "decision": {"passed": passed, "reason": "x"}}


def test_ci_passes_when_the_gate_passed_for_this_src():
    assert check(_gate(), "t1") is None


def test_ci_fails_when_src_changed_since_the_gate_ran():
    assert "src/ changed" in check(_gate(), "t2")


def test_ci_fails_without_a_gate_result_or_with_a_failed_one():
    assert "no gate result" in check(None, "t1")
    assert "failed" in check(_gate(passed=False), "t1")
