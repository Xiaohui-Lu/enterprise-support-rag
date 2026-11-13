from dataclasses import replace

import pytest

from support_rag.agent import AgentRun, ToolBox, snippet
from support_rag.answer_eval import (
    answer_long_context,
    answer_question,
    answer_with_agent,
    check_test_run,
    context_for,
    pick_evenly,
    reading_context,
    summarize_answers,
)
from support_rag.llm import LLMResult
from support_rag.records import Article, Hit, Question

ARTICLES = {
    article_id: Article(article_id, f"Title {article_id}", "", "article", f"Text {article_id}", "")
    for article_id in ("A", "B", "C")
}
QUESTION = Question("ew-000", "expertwritten", "How do I connect a domain?", "", ("C", "A"))


class FixedRetriever:
    def search(self, query, k=10):
        return [Hit("B", 2.0), Hit("C", 1.0)][:k]


class FakeLLM:
    def __init__(self, text, cached=False):
        self.text = text
        self.cached = cached
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        usage = {"input_tokens": 1000, "output_tokens": 200}
        return LLMResult(self.text, "end_turn", usage, 1500.0, self.cached, request["model"])


def _agent_answer(monkeypatch, text="Do this [2].", cached=False):
    """One S2 question whose loop made two model calls: a search, then the answer."""
    usage_1 = {"input_tokens": 1000, "output_tokens": 100}
    usage_2 = {"input_tokens": 200, "cache_read_input_tokens": 1000, "output_tokens": 300}
    calls = (
        LLMResult("", "tool_use", usage_1, 1000.0, cached, "claude-haiku-5-5"),
        LLMResult(text, "end_turn", usage_2, 2000.0, cached, "claude-haiku-5-5"),
    )

    def fake_loop(llm, toolbox, question, *, effort):
        toolbox.call("search", {"query": "domains"})
        return AgentRun(calls, None)

    monkeypatch.setattr("support_rag.answer_eval.run_agent", fake_loop)
    toolbox = ToolBox(FixedRetriever(), ARTICLES)
    return answer_with_agent(None, QUESTION, toolbox, effort="medium")


def test_an_agent_answer_adds_up_every_model_call(monkeypatch):
    result = _agent_answer(monkeypatch)
    assert result.usage == {
        "input_tokens": 1200,
        "cache_read_input_tokens": 1000,
        "output_tokens": 400,
    }
    assert result.latency_ms >= 3000.0
    assert result.agent["steps"] == 1
    assert result.agent["model_calls"] == 2
    assert result.agent["prompt_tokens"] == [1000, 1200]


def test_an_agent_answers_citations_map_to_the_articles_it_saw(monkeypatch):
    result = _agent_answer(monkeypatch)
    assert result.context_ids == ("B", "C")
    assert result.cited_ids == ("C",)
    assert result.context_sections == (snippet(ARTICLES["B"]), snippet(ARTICLES["C"]))


def test_an_agent_answer_records_what_it_would_have_cost_without_caching(monkeypatch):
    result = _agent_answer(monkeypatch)
    assert result.agent["cost_without_caching_usd"] > result.cost_usd


def test_input_tokens_in_a_summary_include_cached_ones(monkeypatch):
    summary = summarize_answers([_agent_answer(monkeypatch)])
    assert summary["mean_input_tokens"] == 2200.0
    assert summary["agent"]["steps"] == {"1": 1}
    assert summary["agent"]["mean_searches"] == 1.0


def test_a_summary_of_other_strategies_has_no_agent_section():
    result = answer_question(FakeLLM("Do this [1]."), QUESTION, [ARTICLES["A"]], effort="medium")
    assert "agent" not in summarize_answers([result])


def test_each_strategy_shows_the_right_articles():
    def ids(strategy):
        context = context_for(
            strategy, QUESTION, articles_by_id=ARTICLES, retriever=FixedRetriever(), k=2
        )
        return [article.id for article in context]

    assert ids("s0") == []
    assert ids("s1") == ["B", "C"]
    assert ids("s_star") == ["C", "A"]


def test_s1_without_a_retriever_is_an_error():
    with pytest.raises(ValueError):
        context_for("s1", QUESTION, articles_by_id=ARTICLES)


def _huge(text):
    """Each article estimated at 50,000 tokens, everything else at nothing."""
    return 50_000 if "Text" in text else 0


def test_s3_stops_before_the_token_budget():
    context = context_for(
        "s3", QUESTION, articles_by_id=ARTICLES, retriever=FixedRetriever(), k=2, count=_huge
    )
    assert [a.id for a in context] == ["B"]  # two would be 100,000, over the 90,000 budget


def test_s3_sends_every_article_that_fits():
    context = context_for(
        "s3", QUESTION, articles_by_id=ARTICLES, retriever=FixedRetriever(), k=2, count=len
    )
    assert [a.id for a in context] == ["B", "C"]


def test_s3_refuses_an_empty_context_rather_than_answer_closed_book():
    with pytest.raises(ValueError, match="over the budget"):
        context_for(
            "s3",
            QUESTION,
            articles_by_id=ARTICLES,
            retriever=FixedRetriever(),
            k=2,
            count=lambda text: 95_000 if "Text" in text else 0,
        )


def test_s3_without_a_token_counter_is_an_error():
    with pytest.raises(ValueError):
        context_for("s3", QUESTION, articles_by_id=ARTICLES, retriever=FixedRetriever(), k=2)


def _long_context_answer(context, k, usage=None):
    llm = FakeLLM("Do this [1].")
    if usage is not None:
        llm.complete = lambda request: LLMResult(
            "Do this [1].", "end_turn", usage, 9000.0, False, request["model"]
        )
    return answer_long_context(llm, QUESTION, context, k=k, count=len, effort="medium")


def test_an_s3_answer_records_what_the_cap_did():
    result = _long_context_answer([ARTICLES["B"]], k=100)
    assert result.long_context["sent"] == 1
    assert result.long_context["capped"] is True
    assert result.long_context["estimated_prompt_tokens"] > 0
    assert result.cited_ids == ("B",)


def test_an_s3_answer_that_sent_all_k_was_not_capped():
    result = _long_context_answer([ARTICLES["B"], ARTICLES["C"]], k=2)
    assert result.long_context["capped"] is False


def test_an_s3_summary_counts_prompts_over_the_price_line_from_real_usage():
    over = _long_context_answer([ARTICLES["B"]], k=100, usage={"input_tokens": 100_001})
    under = _long_context_answer([ARTICLES["B"]], k=100, usage={"input_tokens": 99_000})
    summary = summarize_answers([over, under])["long_context"]
    assert summary["over_price_line"] == 1
    assert summary["max_prompt_tokens"] == 100_001
    assert summary["capped"] == 2
    assert summary["articles_sent"] == {"min": 1, "median": 1, "max": 1}


def test_an_estimate_below_the_real_count_is_a_positive_error():
    # `len` counts characters, far below a real count of 100,000 tokens.
    result = _long_context_answer([ARTICLES["B"]], k=100, usage={"input_tokens": 100_000})
    assert summarize_answers([result])["long_context"]["estimate_error"]["max"] > 0.9


def test_a_summary_of_other_strategies_has_no_long_context_section():
    result = answer_question(FakeLLM("Do this [1]."), QUESTION, [ARTICLES["A"]], effort="medium")
    assert "long_context" not in summarize_answers([result])


def test_a_hit_with_a_section_shows_only_that_section():
    hits = [Hit("B", 2.0, section="Title B\nJust this part."), Hit("C", 1.0)]
    context = reading_context(hits, ARTICLES)
    assert [(a.id, a.contents) for a in context] == [
        ("B", "Title B\nJust this part."),
        ("C", "Text C"),
    ]


def test_the_prompt_holds_the_section_and_not_the_rest_of_the_article():
    llm = FakeLLM("Yes [1].")
    context = reading_context([Hit("B", 2.0, section="Title B\nJust this part.")], ARTICLES)
    answer_question(llm, QUESTION, context, effort="medium", sections=True)
    prompt = llm.requests[0]["messages"][0]["content"]
    assert "Just this part." in prompt
    assert "Text B" not in prompt


def test_a_section_answer_saves_what_the_model_read():
    context = [replace(ARTICLES["B"], contents="Title B\nPart.")]
    result = answer_question(FakeLLM("Yes [1]."), QUESTION, context, effort="medium", sections=True)
    assert result.context_ids == ("B",)
    assert result.context_sections == ("Title B\nPart.",)


def test_a_whole_article_answer_saves_no_sections():
    result = answer_question(FakeLLM("Yes [1]."), QUESTION, [ARTICLES["B"]], effort="medium")
    assert result.context_sections is None


@pytest.mark.parametrize(
    "flags",
    [
        ["--strategy", "s0", "--retriever", "dense"],
        ["--strategy", "s_star", "--unit", "h2"],
        ["--strategy", "s1", "--retriever", "bm25", "--unit", "h2"],
        ["--strategy", "s3"],  # S3 has no default K
        ["--strategy", "s3", "--retriever", "bm25", "--k", "20"],
        ["--strategy", "s0", "--k", "20"],
    ],
)
def test_the_cli_refuses_answer_options_that_would_be_ignored(flags):
    from support_rag.cli import main

    with pytest.raises(SystemExit, match="error"):
        main(["answer", *flags])


def test_answer_maps_citations_through_the_context_order():
    llm = FakeLLM("Point the DNS here [2], then wait [1].")
    context = [ARTICLES["B"], ARTICLES["C"]]
    result = answer_question(llm, QUESTION, context, effort="medium")
    assert result.context_ids == ("B", "C")
    assert result.cited_ids == ("C", "B")
    assert result.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1_000_000)
    assert llm.requests[0]["output_config"] == {"effort": "medium"}


def test_summary_counts_paid_calls_and_latency_from_real_calls_only():
    paid = answer_question(FakeLLM("Yes [1]."), QUESTION, [ARTICLES["A"]], effort="medium")
    replayed = answer_question(
        FakeLLM("NOT_FOUND", cached=True), QUESTION, [ARTICLES["A"]], effort="medium"
    )
    summary = summarize_answers([paid, replayed])
    assert summary["n"] == 2
    assert summary["declined"] == 1
    assert summary["cached"] == 1
    assert summary["paid_this_run_usd"] == pytest.approx(paid.cost_usd, abs=1e-4)
    assert summary["cost_of_answers_usd"] == pytest.approx(2 * paid.cost_usd, abs=1e-4)
    assert summary["latency_p50_ms"] == 1500.0


def test_a_pilot_spreads_its_questions_over_the_list():
    questions = [Question(f"q{i}", "s", "", "", ("A",)) for i in range(50)]
    assert [q.id for q in pick_evenly(questions, 5)] == ["q0", "q10", "q20", "q30", "q40"]
    assert len(pick_evenly(questions, None)) == 50


# check_test_run()


def test_the_test_set_needs_a_named_milestone():
    with pytest.raises(ValueError):
        check_test_run(None, git={"commit": "abc", "dirty": False})


def test_the_test_set_needs_a_clean_commit():
    with pytest.raises(ValueError):
        check_test_run("M1", git={"commit": "abc", "dirty": True})
    with pytest.raises(ValueError):
        check_test_run("M1", git={"commit": None, "dirty": None})  # outside git: can't tell


def test_a_milestone_run_from_a_clean_commit_is_allowed():
    check_test_run("M1", git={"commit": "abc", "dirty": False})
