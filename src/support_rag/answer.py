"""Answering strategies S0, S1 and S* (design section 5): build the request, parse the answer.

Every strategy uses the same model and the same answer format (D-023): plain text with
the articles it used cited inline as [1], [2], ..., or exactly NOT_FOUND. Articles are
numbered in the prompt and the numbers mapped back to article IDs here, because models
mangle 64-character IDs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from support_rag.llm import GENERATOR_MODEL
from support_rag.records import Article, Question

NOT_FOUND = "NOT_FOUND"
STRATEGIES = ("s0", "s1", "s_star")

# "[2]" or "[1, 3]": digits, then optionally ", digits" repeated. The group captures the
# inside of the brackets. Text in brackets, like a Markdown link label, never matches.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

PROMPT_VERSION = "answer-v1"  # bump on any change to the prompts below; it's in every run

_SHARED_RULES = f"""\
- Write for a customer: the steps they need, in the order they need them, and nothing \
they didn't ask about.
- If you can't answer the question, reply with exactly {NOT_FOUND} and nothing else."""

SYSTEM_WITH_ARTICLES = f"""\
You are a customer support assistant for Wix, the website builder. Answer the customer's \
question using only the help-center articles provided, numbered [1] to [n].

- Cite the articles you use with their numbers in square brackets, like [2], right after \
the sentence they support.
{_SHARED_RULES}"""

SYSTEM_CLOSED_BOOK = f"""\
You are a customer support assistant for Wix, the website builder. Answer the customer's \
question from what you know about Wix.

{_SHARED_RULES}"""


@dataclass(frozen=True)
class Answer:
    text: str
    cited_ids: tuple[str, ...]  # article IDs, each once, in the order first cited
    declined: bool  # the whole reply was NOT_FOUND
    invalid_citations: int  # citation numbers that point to no article in the prompt


def build_request(
    question: Question,
    context: Sequence[Article],
    *,
    effort: str,
    max_tokens: int = 16_000,
    model: str = GENERATOR_MODEL,
) -> dict[str, Any]:
    """The Messages API request for one question. An empty context means closed book (S0)."""
    if context:
        articles = "\n\n".join(
            f"[{number}] {article.contents}" for number, article in enumerate(context, start=1)
        )
        system = SYSTEM_WITH_ARTICLES
        content = f"<articles>\n{articles}\n</articles>\n\nQuestion: {question.question}"
    else:
        system = SYSTEM_CLOSED_BOOK
        content = f"Question: {question.question}"
    return {
        "model": model,
        "max_tokens": max_tokens,
        "output_config": {"effort": effort},
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }


def parse_answer(text: str, context_ids: Sequence[str]) -> Answer:
    """Turn the model's reply into an Answer.

    - Declined only if the whole reply, ignoring surrounding whitespace, is NOT_FOUND.
    - Citations are [n] or a comma-separated group like [1, 3]. Markdown links such as
      [Wix Payments](https://...) are not citations.
    - Number n maps to context_ids[n - 1]. A number outside 1..len(context_ids) is
      invalid; with no context (S0), every citation is invalid.

    "Connect it [2]. Then verify [1, 2]."  with ids (A, B)  ->  cited_ids (B, A), 0 invalid
    """
    if text.strip() == NOT_FOUND:
        return Answer(text, cited_ids=(), declined=True, invalid_citations=0)

    cited: list[str] = []
    invalid = 0
    for group in _CITATION.findall(text):  # "2", then "1, 2"
        for number in (int(part) for part in group.split(",")):
            if 1 <= number <= len(context_ids):
                article_id = context_ids[number - 1]  # citation [1] is index 0
                if article_id not in cited:
                    cited.append(article_id)
            else:
                invalid += 1
    return Answer(text, cited_ids=tuple(cited), declined=False, invalid_citations=invalid)
