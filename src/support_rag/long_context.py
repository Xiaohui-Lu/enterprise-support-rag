"""Long context, S3 (D-044): many retrieved articles in one prompt, kept under a token budget.

Haiku 5.5 bills a prompt over 100,000 tokens at the higher rate for the whole request, so S3
adds articles best first and stops before an estimated 90,000. The estimate is local:
tiktoken's count times Claude's measured ratio, so the cap adds no network call.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from support_rag.answer import numbered
from support_rag.records import Article

TokenCounter = Callable[[str], int]

# Claude's input tokens per tiktoken o200k token, fitted on 100 dev prompts (D-044).
CLAUDE_TOKENS_PER_TIKTOKEN = 1.54
# 10% under the 100K price line: the estimate was up to 8.4% low on a single prompt.
PROMPT_BUDGET = 90_000


def claude_token_estimator() -> TokenCounter:
    """Estimated Claude tokens for a piece of text, rounded up."""
    from support_rag.corpus_stats import tiktoken_counter

    count = tiktoken_counter()
    return lambda text: math.ceil(CLAUDE_TOKENS_PER_TIKTOKEN * count(text))


def capped_context(
    ranked: Sequence[Article], *, budget: int, count: TokenCounter, used: int = 0
) -> list[Article]:
    """The top of `ranked`, best first, while the estimated prompt stays within `budget`.

    `used` is what the prompt holds before any article: the instructions and the question.
    The list stops at the first article that doesn't fit, so the context is always the top n
    of the ranking, never a lower-ranked article in place of a better one.

    Articles of 30, 50 and 40 tokens, budget 100, used 10  ->  the first two (90 tokens)
    """
    context: list[Article] = []
    total = used
    for number, article in enumerate(ranked, start=1):
        tokens = count(numbered(number, article))
        if total + tokens > budget:
            break
        context.append(article)
        total += tokens
    return context


def estimate_request_tokens(request: dict[str, Any], count: TokenCounter) -> int:
    """The estimate for a whole request as sent: its system prompt and its one message."""
    return count(request["system"]) + count(request["messages"][0]["content"])
