import pytest

from support_rag.answer import build_request
from support_rag.answer_eval import (
    answer_with_router,
    context_for,
    summarize_answers,
)
from support_rag.llm import LLMResult
from support_rag.records import Article, Hit, Question
from support_rag.router import against_random, oracle, route, skips_s1

ARTICLES = {
    article_id: Article(article_id, f"Title {article_id}", "", "article", f"Text {article_id}", "")
    for article_id in "ABCDEFGHIJKL"
}
QUESTION = Question("ew-000", "expertwritten", "How do I connect a domain?", "", ("C",))


def test_a_weak_top_score_skips_s1():
    assert skips_s1(0.50, 0.5615)
    assert not skips_s1(0.5615, 0.5615)  # strictly below


def test_each_route():
    assert route(0.50, False, threshold=0.5615) == "score"
    assert route(0.50, True, threshold=0.5615) == "score"  # S1 never ran
    assert route(0.70, True, threshold=0.5615) == "decline"
    assert route(0.70, False, threshold=0.5615) == "s1"


def test_either_signal_can_be_switched_off():
    assert route(0.50, True, threshold=0.5615, signals={"decline"}) == "decline"
    assert route(0.50, False, threshold=0.5615, signals={"decline"}) == "s1"
    assert route(0.70, True, threshold=0.5615, signals={"score"}) == "s1"


def test_random_escalation_is_worth_its_share_of_the_total_gain():
    gains = {f"q{i}": 1 if i < 2 else 0 for i in range(10)}  # S3 fixes q0 and q1
    result = against_random({"q0", "q5", "q6", "q7", "q8"}, gains)
    assert result["router_gain"] == 1
    assert result["random_gain"] == pytest.approx(1.0)  # half the questions, half the gain
    assert result["difference"] == pytest.approx(0.0)


def test_a_router_that_finds_every_gain_beats_random():
    gains = {f"q{i:03}": 1 if i < 5 else 0 for i in range(100)}
    result = against_random({f"q{i:03}" for i in range(5)}, gains)
    assert result["difference"] == pytest.approx(100 * (5 - 0.05 * 5) / 100)
    assert result["ci"][0] > 0
    assert result["p_value"] < 0.001


def test_the_permutation_p_is_never_zero():
    gains = {f"q{i}": 1 if i == 0 else 0 for i in range(20)}
    assert against_random({"q0"}, gains, draws=100)["p_value"] > 0


def test_an_escalated_question_needs_a_gain():
    with pytest.raises(ValueError):
        against_random({"q9"}, {"q0": 1})


def test_the_oracle_takes_the_cheapest_strategy_that_was_right():
    correct = {"s1": {"a": True, "b": False, "c": False}, "s3": {"a": True, "b": True, "c": False}}
    cost = {"s1": {"a": 1.0, "b": 1.0, "c": 1.0}, "s3": {"a": 5.0, "b": 5.0, "c": 5.0}}
    result = oracle(correct, cost)
    assert result["pct_correct"] == pytest.approx(200 / 3)
    assert result["cost_per_answer_usd"] == pytest.approx((1 + 5 + 1) / 3)


class ScoredRetriever:
    """Twelve articles, best first, the top one scoring `top`."""

    def __init__(self, top):
        self.top = top

    def search(self, query, k=10):
        return [Hit(a, self.top - 0.01 * i) for i, a in enumerate("ABCDEFGHIJKL")][:k]


class ScriptedLLM:
    """Replies in turn, each call taking 1 second and costing the same."""

    def __init__(self, *replies, cached=False):
        self.replies = list(replies)
        self.cached = cached
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        usage = {"input_tokens": 1000, "output_tokens": 200}
        return LLMResult(
            self.replies.pop(0), "end_turn", usage, 1000.0, self.cached, request["model"]
        )


def _routed(top, *replies, cached=False):
    llm = ScriptedLLM(*replies, cached=cached)
    result = answer_with_router(
        llm, QUESTION, retriever=ScoredRetriever(top), articles_by_id=ARTICLES, count=len,
        effort="medium", k=10, threshold=0.5615,
    )  # fmt: skip
    return llm, result


def test_a_weak_top_score_goes_straight_to_long_context():
    llm, result = _routed(0.50, "Do this [1].")
    assert len(llm.requests) == 1
    assert result.router["route"] == "score"
    assert result.long_context["k"] == 100
    assert len(result.context_ids) == 12  # S3 reads everything the ranking holds


def test_a_strong_top_score_keeps_s1s_answer():
    llm, result = _routed(0.70, "Do this [1].")
    assert len(llm.requests) == 1
    assert result.router["route"] == "s1"
    assert result.long_context is None
    assert len(result.context_ids) == 10


def test_a_decline_escalates_and_pays_for_both_calls():
    llm, result = _routed(0.70, "NOT_FOUND", "Do this [1].")
    assert len(llm.requests) == 2
    assert result.router["route"] == "decline"
    assert not result.declined  # S3's answer stands
    assert [c["strategy"] for c in result.router["calls"]] == ["s1", "s3-k100"]
    assert result.cost_usd == pytest.approx(sum(c["cost_usd"] for c in result.router["calls"]))
    assert result.latency_ms == 2000.0
    assert result.usage == {"input_tokens": 2000, "output_tokens": 400}


def test_the_router_sends_exactly_what_s1_and_s3_send_alone():
    """So the call cache answers every request it repeats (D-048)."""
    llm, _ = _routed(0.70, "NOT_FOUND", "Do this [1].")
    retriever = ScoredRetriever(0.70)
    alone = [
        build_request(
            QUESTION,
            context_for(strategy, QUESTION, articles_by_id=ARTICLES, retriever=retriever, k=k,
                        count=len),
            effort="medium",
        )
        for strategy, k in (("s1", 10), ("s3", 100))
    ]  # fmt: skip
    assert llm.requests == alone


def test_a_router_summary_counts_routes_and_paid_calls():
    _, kept = _routed(0.70, "Do this [1].")
    _, escalated = _routed(0.70, "NOT_FOUND", "Do this [1].", cached=True)
    _, skipped = _routed(0.50, "Do this [1].")
    summary = summarize_answers([kept, escalated, skipped])["router"]
    assert summary["routes"] == {"s1": 1, "score": 1, "decline": 1}
    assert summary["escalated"] == 2
    assert summary["model_calls"] == 4
    assert summary["calls_not_cached"] == 2
    assert summary["paid_this_run_usd"] == pytest.approx(kept.cost_usd + skipped.cost_usd, abs=1e-4)


def test_a_router_summary_checks_s3s_own_prompt_not_both_calls():
    _, escalated = _routed(0.70, "NOT_FOUND", "Do this [1].")
    summary = summarize_answers([escalated])
    assert escalated.usage["input_tokens"] == 2000  # S1's and S3's together
    assert summary["long_context"]["max_prompt_tokens"] == 1000  # S3's alone
