from dataclasses import replace

import pytest

from support_rag.answer_eval import (
    answer_question,
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
