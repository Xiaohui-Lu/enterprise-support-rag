"""Agentic search, S2 (design section 5, D-043): the model searches and reads, then answers.

One question is one loop. The model asks for tools, the tools run, their results go back,
and that repeats until the model writes its answer or a limit makes it. Every turn is one
call through the same LLM client as the other strategies, so it's cached and counted (D-003,
D-020), and the answer has the same format as every other strategy's (D-023).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from support_rag.answer import CITE_RULE, NOT_FOUND, SHARED_RULES
from support_rag.llm import GENERATOR_MODEL, LLMResult, prompt_tokens
from support_rag.records import Article, Question

if TYPE_CHECKING:
    from support_rag.answer_eval import LLM
    from support_rag.retrieval_eval import Retriever

AGENT_PROMPT_VERSION = "agent-v1"  # bump on any change to the prompt or the tools below
MAX_STEPS = 6  # model turns that ask for tools (D-043)
MAX_PROMPT_TOKENS = 30_000  # a prompt this large ends the search (D-043)
SEARCH_RESULTS = 10
SNIPPET_WORDS = 50

SYSTEM_AGENT = f"""\
You are a customer support assistant for Wix, the website builder. Answer the customer's \
question using only help-center articles you find with your tools.

- Use search to find articles. Search results show only the start of each article, so use \
read to get an article's full text before you rely on it.
- If the results don't cover the question, search again with different words. A question \
about several things may need several searches.
- Each article has a number, shown in the search results.
{CITE_RULE}
{SHARED_RULES}"""

# Sent with the last tool results when a limit is reached (D-043).
FINAL_NOTE = {
    "type": "text",
    "text": (
        "You have no more tool calls. Answer now from what you have read, or reply with "
        f"exactly {NOT_FOUND} if you can't answer."
    ),
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search the Wix help center. Returns the best-matching articles, best first, each "
            "with its number and the first words of its text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for."}},
            "required": ["query"],
        },
    },
    {
        "name": "read",
        "description": "Get the full text of an article, by its number in the search results.",
        "input_schema": {
            "type": "object",
            "properties": {"number": {"type": "integer", "description": "The article's number."}},
            "required": ["number"],
        },
    },
]


def snippet(article: Article, words: int = SNIPPET_WORDS) -> str:
    """The first words of an article, which start with its title, marked if cut short."""
    parts = article.contents.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


@dataclass(frozen=True)
class ToolOutput:
    text: str
    is_error: bool = False


class ToolBox:
    """The two tools, and a record of what the model has seen, for one question."""

    def __init__(self, retriever: Retriever, articles_by_id: Mapping[str, Article]) -> None:
        self._retriever = retriever
        self._articles = articles_by_id
        self._seen: dict[str, str] = {}  # article ID to the text the model saw, in number order
        self.queries: list[str] = []
        self.read_ids: list[str] = []  # in the order read; a repeated read appears twice
        self.errors = 0
        self.tool_ms = 0.0

    @property
    def context_ids(self) -> tuple[str, ...]:
        """Every article the model has seen, by number: [1] is the first."""
        return tuple(self._seen)

    @property
    def context_sections(self) -> tuple[str, ...]:
        """What the model saw of each article, by number: the full text if it read the
        article, otherwise the snippet. The judge reads these (D-042)."""
        return tuple(self._seen.values())

    def call(self, name: str, tool_input: Mapping[str, Any]) -> ToolOutput:
        """Run one tool call. A bad call comes back as an error for the model to read, so a
        mistake doesn't end the question."""
        start = time.perf_counter()
        match name:
            case "search":
                output = self._search(tool_input)
            case "read":
                output = self._read(tool_input)
            case _:
                output = ToolOutput(f"There is no tool named {name!r}.", is_error=True)
        self.tool_ms += (time.perf_counter() - start) * 1000
        self.errors += output.is_error
        return output

    def _search(self, tool_input: Mapping[str, Any]) -> ToolOutput:
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolOutput("search needs a non-empty query.", is_error=True)
        self.queries.append(query)
        lines = []
        for hit in self._retriever.search(query, SEARCH_RESULTS):
            article = self._articles[hit.article_id]
            preview = snippet(article)
            self._seen.setdefault(article.id, preview)  # keeps the full text if already read
            lines.append(f"[{self.context_ids.index(article.id) + 1}] {preview}")
        return ToolOutput("\n\n".join(lines) if lines else "No articles found.")

    def _read(self, tool_input: Mapping[str, Any]) -> ToolOutput:
        number = tool_input.get("number")
        ids = self.context_ids
        if not isinstance(number, int) or not 1 <= number <= len(ids):
            return ToolOutput(
                f"There is no article numbered {number}. Use a number from the search results.",
                is_error=True,
            )
        article = self._articles[ids[number - 1]]
        self._seen[article.id] = article.contents
        self.read_ids.append(article.id)
        return ToolOutput(f"[{number}] {article.contents}")


def first_message(question: Question) -> dict[str, Any]:
    return {"role": "user", "content": f"Question: {question.question}"}


def build_agent_request(
    messages: Sequence[dict[str, Any]],
    *,
    effort: str,
    final: bool = False,
    max_tokens: int = 16_000,
    model: str = GENERATOR_MODEL,
) -> dict[str, Any]:
    """The Messages API request for one turn. `final` turns the tools off, so the model has
    to answer. Everything before `messages` stays byte-identical between turns, or the
    prompt cache would miss (D-043)."""
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "output_config": {"effort": effort},
        "cache_control": {"type": "ephemeral"},
        "system": SYSTEM_AGENT,
        "tools": TOOLS,
        "messages": list(messages),
    }
    if final:
        request["tool_choice"] = {"type": "none"}
    return request


@dataclass(frozen=True)
class AgentRun:
    calls: tuple[LLMResult, ...]  # every model call, in order
    stopped_by: str | None  # None if the model finished by itself; else "steps" or "tokens"

    @property
    def final(self) -> LLMResult:
        return self.calls[-1]


def run_agent(
    llm: LLM,
    toolbox: ToolBox,
    question: Question,
    *,
    effort: str,
    max_steps: int = MAX_STEPS,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> AgentRun:
    """Drive one question from the first model call to the answer (D-043).

    A turn that doesn't stop for tools ends the question, and its tools never run. After a
    turn that does, all its tool results go back in one user message. When that turn was
    step `max_steps`, or its prompt reached `max_prompt_tokens`, the message also carries
    FINAL_NOTE and the next call is the last, with tools off. So there are at most
    `max_steps` + 1 calls.
    """
    messages = [first_message(question)]
    calls: list[LLMResult] = []
    stopped_by: str | None = None
    while True:
        result = llm.complete(
            build_agent_request(messages, effort=effort, final=stopped_by is not None)
        )
        calls.append(result)
        if result.stop_reason != "tool_use" or stopped_by is not None:
            return AgentRun(tuple(calls), stopped_by)

        replies: list[dict[str, Any]] = []
        for block in result.content:
            if block["type"] == "tool_use":
                output = toolbox.call(block["name"], block["input"])
                replies.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": output.text,
                        "is_error": output.is_error,
                    }
                )

        if len(calls) >= max_steps:
            stopped_by = "steps"
        elif prompt_tokens(result.usage) >= max_prompt_tokens:
            stopped_by = "tokens"
        if stopped_by is not None:
            replies.append(FINAL_NOTE)

        messages.append({"role": "assistant", "content": list(result.content)})
        messages.append({"role": "user", "content": replies})
