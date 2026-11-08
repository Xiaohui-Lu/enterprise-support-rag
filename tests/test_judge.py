import json

import pytest

from support_rag.judge import SYSTEM, build_judge_request, parse_verdict
from support_rag.judge_eval import judge_answer, judge_request, summarize_verdicts
from support_rag.llm import LLMResult
from support_rag.records import Article, Question

ARTICLES = {
    article_id: Article(article_id, f"Title {article_id}", "", "article", f"Text {article_id}", "")
    for article_id in ("A", "B", "C")
}
QUESTION = Question("ew-000", "expertwritten", "How do I unpublish?", "Click Unpublish.", ("B",))


def _answer(text, cited_ids=(), declined=False):
    return {
        "text": text,
        "context_ids": ["A", "B", "C"],
        "cited_ids": list(cited_ids),
        "declined": declined,
    }


class FakeLLM:
    def __init__(self, verdict=None, stop_reason="end_turn"):
        self.verdict = verdict or {"reasoning": "ok", "correctness": "correct", "grounded": "yes"}
        self.stop_reason = stop_reason
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        usage = {"input_tokens": 2500, "output_tokens": 600}
        text = json.dumps(self.verdict)
        return LLMResult(text, self.stop_reason, usage, 3000.0, False, request["model"])


# build_judge_request()


def test_the_judge_is_blind_to_strategy_and_model():
    request = build_judge_request("Q?", "Ref.", "Answer [2].", [(2, ARTICLES["B"])], effort="high")
    sent = json.dumps(request).lower()
    for hint in ("s0", "s1", "s_star", "strategy", "closed book", "haiku", "retriev"):
        assert hint not in sent


def test_cited_articles_keep_the_numbers_the_answer_uses():
    request = build_judge_request("Q?", "Ref.", "See [2].", [(2, ARTICLES["B"])], effort="high")
    assert "[2] Text B" in request["messages"][0]["content"]


def test_the_judge_request_pins_model_effort_and_schema():
    request = build_judge_request("Q?", "Ref.", "A.", [], effort="medium")
    assert request["model"] == "claude-sonnet-5-5"
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["system"] == SYSTEM
    assert "cites no articles" in request["messages"][0]["content"]


def test_a_verdict_outside_the_rubric_is_rejected():
    with pytest.raises(ValueError):
        parse_verdict(json.dumps({"reasoning": "", "correctness": "great", "grounded": "yes"}))


# judge_request()


def test_the_judge_reads_only_the_section_the_generator_read():
    answer = _answer("Answer [2].", ["B"]) | {"context_sections": ["sec A", "sec B", "sec C"]}
    request = judge_request(answer, QUESTION, ARTICLES, effort="high")
    prompt = request["messages"][0]["content"]
    assert "[2] sec B" in prompt
    assert "Text B" not in prompt
    assert "sec A" not in prompt


def test_the_judge_reads_the_whole_article_when_no_sections_were_saved():
    request = judge_request(_answer("Answer [2].", ["B"]), QUESTION, ARTICLES, effort="high")
    assert "[2] Text B" in request["messages"][0]["content"]


def test_an_old_answer_builds_the_same_request_as_one_saved_without_sections():
    old = _answer("Answer [1].", ["A"])
    new = old | {"context_sections": None}
    assert judge_request(old, QUESTION, ARTICLES, effort="high") == judge_request(
        new, QUESTION, ARTICLES, effort="high"
    )


# judge_answer()


def test_a_decline_is_incorrect_without_a_judge_call():
    llm = FakeLLM()
    result = judge_answer(
        llm, _answer("NOT_FOUND", declined=True), QUESTION, ARTICLES, effort="high"
    )
    assert result.correctness == "incorrect"
    assert result.judged is False
    assert llm.requests == []


def test_an_answer_is_graded_with_its_cited_articles():
    llm = FakeLLM()
    answer = _answer("Click Unpublish [2].", cited_ids=["B"])
    result = judge_answer(llm, answer, QUESTION, ARTICLES, effort="high")
    assert result.correctness == "correct"
    assert "[2] Text B" in llm.requests[0]["messages"][0]["content"]
    assert "Click Unpublish." in llm.requests[0]["messages"][0]["content"]  # the reference


def test_a_cut_off_verdict_is_a_judge_failure_not_a_grade():
    llm = FakeLLM(stop_reason="max_tokens")
    result = judge_answer(llm, _answer("Answer."), QUESTION, ARTICLES, effort="high")
    assert result.correctness is None
    assert summarize_verdicts([result])["judge_failed"] == 1


def test_summary_counts_grades_and_judge_calls():
    llm = FakeLLM({"reasoning": "", "correctness": "partial", "grounded": "partly"})
    graded = judge_answer(llm, _answer("Answer [1].", ["A"]), QUESTION, ARTICLES, effort="high")
    declined = judge_answer(
        llm, _answer("NOT_FOUND", declined=True), QUESTION, ARTICLES, effort="high"
    )
    summary = summarize_verdicts([graded, declined])
    assert (summary["correct"], summary["partial"], summary["incorrect"]) == (0, 1, 1)
    assert summary["judge_calls"] == 1
    assert summary["pct_correct"] == 0.0


def test_another_model_can_be_asked_the_same_question():
    judge = build_judge_request("Q?", "Ref.", "A [2].", [(2, ARTICLES["B"])], effort="high")
    other = build_judge_request(
        "Q?", "Ref.", "A [2].", [(2, ARTICLES["B"])], effort="high", model="claude-opus-5-5"
    )
    assert other["model"] == "claude-opus-5-5"
    assert {**other, "model": judge["model"]} == judge
