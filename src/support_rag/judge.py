"""The LLM judge (D-005, D-025): grades an answer against the expert reference.

The judge is blind: it sees the question, the reference, the answer and the articles the
answer cites, never the strategy, model or run. Its verdict is JSON enforced by the API.
Declines are never sent to it: a NOT_FOUND is incorrect by definition (every WixQA
question is answerable).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from support_rag.llm import JUDGE_MODEL
from support_rag.records import Article

JUDGE_PROMPT_VERSION = "judge-v1"  # bump on any change to the rubric or the schema
CORRECTNESS = ("correct", "partial", "incorrect")
GROUNDED = ("yes", "partly", "no", "n/a")

# The rubric is also shown to the hand labeler (D-027), so both grade by the same words.
CORRECTNESS_RUBRIC = """\
Grade correctness against the reference answer:
- correct: following the answer resolves the customer's issue, consistent with the \
reference. One complete, valid method is enough; it doesn't have to list every \
alternative the reference gives. Extra accurate detail is fine.
- partial: the right direction, but a step the customer needs is missing or wrong, or \
they would need more help to finish.
- incorrect: wrong, contradicts the reference, or doesn't address the question."""

GROUNDED_RUBRIC = """\
Grade groundedness against the cited articles only:
- yes: every claim in the answer is supported by the cited articles.
- partly: some claims are supported and some are not.
- no: the cited articles don't support the answer.
- n/a: the answer cites no articles."""

SYSTEM = f"""\
You grade answers written by a customer support assistant for Wix, the website builder. \
You are given the customer's question, a reference answer written by a Wix support \
expert, the answer to grade, and the help-center articles that answer cites.

{CORRECTNESS_RUBRIC}

{GROUNDED_RUBRIC}

Write your reasoning first, briefly, then the two grades."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correctness": {"type": "string", "enum": list(CORRECTNESS)},
        "grounded": {"type": "string", "enum": list(GROUNDED)},
    },
    "required": ["reasoning", "correctness", "grounded"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Verdict:
    correctness: str
    grounded: str
    reasoning: str


def build_judge_request(
    question: str,
    reference: str,
    answer: str,
    cited: Sequence[tuple[int, Article]],
    *,
    effort: str,
    max_tokens: int = 16_000,
    model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    """`cited` pairs each cited article with the number the answer uses for it, so the
    judge can match "[2]" in the answer to the right article. Only the labeling panel
    (D-028) changes `model`: it sees exactly what the judge sees."""
    if cited:
        articles = "\n\n".join(f"[{number}] {article.contents}" for number, article in cited)
    else:
        articles = "The answer cites no articles."
    content = (
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_answer>\n{reference}\n</reference_answer>\n\n"
        f"<answer_to_grade>\n{answer}\n</answer_to_grade>\n\n"
        f"<cited_articles>\n{articles}\n</cited_articles>"
    )
    return {
        "model": model,
        "max_tokens": max_tokens,
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
        },
        "system": SYSTEM,
        "messages": [{"role": "user", "content": content}],
    }


def parse_verdict(text: str) -> Verdict:
    data = json.loads(text)
    if data["correctness"] not in CORRECTNESS or data["grounded"] not in GROUNDED:
        raise ValueError(f"verdict outside the rubric: {data}")
    return Verdict(data["correctness"], data["grounded"], data["reasoning"])
