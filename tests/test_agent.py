import copy

from support_rag.agent import (
    FINAL_NOTE,
    TOOLS,
    AgentRun,
    ToolBox,
    build_agent_request,
    first_message,
    run_agent,
    snippet,
)
from support_rag.llm import LLMResult
from support_rag.records import Article, Hit, Question

LONG = " ".join(f"word{i}" for i in range(60))
ARTICLES = {
    "A": Article("A", "Title A", "", "article", f"Title A {LONG}", ""),
    "B": Article("B", "Title B", "", "article", "Title B is short.", ""),
    "C": Article("C", "Title C", "", "article", "Title C is short too.", ""),
}
QUESTION = Question("ew-000", "expertwritten", "How do I connect a domain?", "", ("A",))
THINKING = {"type": "thinking", "thinking": "", "signature": "abc"}


class FixedRetriever:
    """Returns A then B for the first query, then C then A for any other."""

    def __init__(self):
        self.queries = []

    def search(self, query, k=10):
        self.queries.append(query)
        first = len(self.queries) == 1
        return [Hit("A", 2.0), Hit("B", 1.0)] if first else [Hit("C", 2.0), Hit("A", 1.0)]


def _toolbox():
    return ToolBox(FixedRetriever(), ARTICLES)


def _search(query="domains", tool_id="t1"):
    return {"type": "tool_use", "id": tool_id, "name": "search", "input": {"query": query}}


def _read(number, tool_id="t2"):
    return {"type": "tool_use", "id": tool_id, "name": "read", "input": {"number": number}}


def _text(text):
    return {"type": "text", "text": text}


def _turn(*blocks, stop_reason="tool_use", usage=None):
    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    usage = usage or {"input_tokens": 1000, "output_tokens": 50}
    return LLMResult(text, stop_reason, usage, 100.0, False, "claude-haiku-5-5", content=blocks)


def _answer(text="Do this [1]."):
    return _turn(_text(text), stop_reason="end_turn")


class ScriptedLLM:
    """Plays back the given turns, and keeps a copy of every request it receives."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        return self.turns.pop(0)


def _run(llm, toolbox=None, **limits):
    return run_agent(llm, toolbox or _toolbox(), QUESTION, effort="medium", **limits)


# snippet() and ToolBox


def test_a_snippet_is_the_first_50_words_and_shows_it_was_cut():
    text = snippet(ARTICLES["A"])
    assert text.startswith("Title A word0")
    assert text.endswith(" ...")
    assert len(text.removesuffix(" ...").split()) == 50


def test_a_short_article_is_not_marked_as_cut():
    assert snippet(ARTICLES["B"]) == "Title B is short."


def test_numbers_are_given_on_first_sight_and_kept():
    toolbox = _toolbox()
    first = toolbox.call("search", {"query": "one"})
    second = toolbox.call("search", {"query": "two"})
    assert first.text.startswith("[1] Title A")
    assert "[2] Title B" in first.text
    assert second.text.startswith("[3] Title C")  # new article, next number
    assert "[1] Title A" in second.text  # A keeps its number
    assert toolbox.context_ids == ("A", "B", "C")


def test_read_returns_the_full_article():
    toolbox = _toolbox()
    toolbox.call("search", {"query": "one"})
    output = toolbox.call("read", {"number": 1})
    assert output.text == f"[1] {ARTICLES['A'].contents}"
    assert output.is_error is False
    assert toolbox.read_ids == ["A"]


def test_the_judge_sees_the_full_text_only_of_articles_the_model_read():
    toolbox = _toolbox()
    toolbox.call("search", {"query": "one"})
    toolbox.call("read", {"number": 1})
    toolbox.call("search", {"query": "two"})  # A comes up again, and stays read
    assert toolbox.context_sections == (
        ARTICLES["A"].contents,
        snippet(ARTICLES["B"]),
        snippet(ARTICLES["C"]),
    )


def test_a_number_never_shown_is_an_error():
    toolbox = _toolbox()
    toolbox.call("search", {"query": "one"})
    output = toolbox.call("read", {"number": 3})
    assert output.is_error is True
    assert toolbox.errors == 1
    assert toolbox.read_ids == []


def test_an_empty_query_is_an_error_and_never_searched():
    toolbox = _toolbox()
    assert toolbox.call("search", {"query": "  "}).is_error is True
    assert toolbox.queries == []


def test_an_unknown_tool_is_an_error():
    assert _toolbox().call("delete", {}).is_error is True


# build_agent_request()


def test_tools_stay_on_until_the_final_request():
    messages = [first_message(QUESTION)]
    assert "tool_choice" not in build_agent_request(messages, effort="medium")
    final = build_agent_request(messages, effort="medium", final=True)
    assert final["tool_choice"] == {"type": "none"}
    assert final["tools"] == TOOLS  # still listed, so the cached prefix doesn't change


def test_every_request_asks_for_prompt_caching():
    request = build_agent_request([first_message(QUESTION)], effort="medium")
    assert request["cache_control"] == {"type": "ephemeral"}


# run_agent()


def test_a_reply_with_no_tool_calls_ends_the_loop():
    llm = ScriptedLLM(_answer("Done [1]."))
    run = _run(llm)
    assert isinstance(run, AgentRun)
    assert run.final.text == "Done [1]."
    assert len(run.calls) == 1
    assert run.stopped_by is None


def test_the_first_request_holds_only_the_question():
    llm = ScriptedLLM(_answer())
    _run(llm)
    assert llm.requests[0]["messages"] == [first_message(QUESTION)]
    assert "tool_choice" not in llm.requests[0]


def test_tool_results_go_back_and_the_loop_continues():
    llm = ScriptedLLM(_turn(_search()), _answer())
    toolbox = _toolbox()
    run = _run(llm, toolbox)
    assert len(run.calls) == 2
    assert toolbox.queries == ["domains"]
    reply = llm.requests[1]["messages"][-1]
    assert reply["role"] == "user"
    (result,) = reply["content"]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "t1"
    assert result["content"].startswith("[1] Title A")


def test_the_whole_assistant_turn_goes_back_including_thinking():
    blocks = (THINKING, _text("Let me look."), _search())
    llm = ScriptedLLM(_turn(*blocks), _answer())
    _run(llm)
    turn = llm.requests[1]["messages"][1]
    assert turn["role"] == "assistant"
    assert list(turn["content"]) == list(blocks)


def test_several_tool_calls_in_one_turn_share_one_reply():
    llm = ScriptedLLM(_turn(_search()), _turn(_read(1, "r1"), _read(2, "r2")), _answer())
    run = _run(llm)
    assert len(run.calls) == 3  # the two reads are one step
    reply = llm.requests[2]["messages"][-1]
    assert [r["tool_use_id"] for r in reply["content"]] == ["r1", "r2"]


def test_a_bad_tool_call_goes_back_as_an_error_and_the_loop_continues():
    llm = ScriptedLLM(_turn(_read(9)), _answer())
    run = _run(llm)
    assert len(run.calls) == 2
    (result,) = llm.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True


def test_history_only_grows():
    llm = ScriptedLLM(_turn(_search("a", "t1")), _turn(_search("b", "t2")), _answer())
    _run(llm)
    for earlier, later in zip(llm.requests, llm.requests[1:], strict=False):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


def test_the_step_cap_forces_one_last_call_without_tools():
    llm = ScriptedLLM(_turn(_search("a", "t1")), _turn(_search("b", "t2")), _answer())
    run = _run(llm, max_steps=2)
    assert len(run.calls) == 3
    assert "tool_choice" not in llm.requests[1]
    assert llm.requests[2]["tool_choice"] == {"type": "none"}
    assert llm.requests[2]["messages"][-1]["content"][-1] == FINAL_NOTE
    assert run.stopped_by == "steps"


def test_the_loop_never_makes_more_than_the_cap_plus_one_calls():
    turns = [_turn(_search(f"q{i}", f"t{i}")) for i in range(5)]
    llm = ScriptedLLM(*turns)  # a model that won't stop asking, even with tools off
    run = _run(llm, max_steps=2)
    assert len(llm.requests) == 3
    assert len(run.calls) == 3


def test_a_large_prompt_forces_one_last_call_without_tools():
    big = {"input_tokens": 500, "cache_read_input_tokens": 29_600, "output_tokens": 50}
    llm = ScriptedLLM(_turn(_search(), usage=big), _answer())
    run = _run(llm, max_prompt_tokens=30_000)  # 500 alone wouldn't reach it: cached tokens count
    assert llm.requests[1]["tool_choice"] == {"type": "none"}
    assert llm.requests[1]["messages"][-1]["content"][-1] == FINAL_NOTE
    assert run.stopped_by == "tokens"


def test_a_cut_off_turn_ends_the_loop_and_its_tools_never_run():
    llm = ScriptedLLM(_turn(_search(), stop_reason="max_tokens"))
    toolbox = _toolbox()
    run = _run(llm, toolbox)
    assert run.final.stop_reason == "max_tokens"
    assert toolbox.queries == []
    assert len(llm.requests) == 1
