from support_rag.answer import (
    NOT_FOUND,
    SYSTEM_CLOSED_BOOK,
    SYSTEM_WITH_ARTICLES,
    build_request,
    parse_answer,
)
from support_rag.records import Article, Question

QUESTION = Question("ew-000", "expertwritten", "Can I accept payments now?", "", ("B",))


def _article(article_id: str, contents: str) -> Article:
    return Article(article_id, contents.split("\n")[0], "", "article", contents, "")


# build_request()


def test_articles_are_numbered_in_the_order_given():
    context = [_article("B", "Accepting payments\n..."), _article("A", "Verifying\n...")]
    request = build_request(QUESTION, context, effort="medium")
    content = request["messages"][0]["content"]
    assert request["system"] == SYSTEM_WITH_ARTICLES
    assert content.index("[1] Accepting payments") < content.index("[2] Verifying")
    assert content.endswith("Question: Can I accept payments now?")


def test_closed_book_sends_no_articles():
    request = build_request(QUESTION, [], effort="medium")
    assert request["system"] == SYSTEM_CLOSED_BOOK
    assert request["messages"][0]["content"] == "Question: Can I accept payments now?"


def test_request_pins_the_model_and_effort_and_sends_no_sampling_parameters():
    request = build_request(QUESTION, [], effort="low")
    assert request["model"] == "claude-haiku-5-5"
    assert request["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & set(request)


# parse_answer()

IDS = ("A", "B", "C")


def test_citations_map_numbers_to_article_ids():
    answer = parse_answer("Go to Settings [2]. Then verify [3].", IDS)
    assert answer.cited_ids == ("B", "C")
    assert answer.invalid_citations == 0
    assert answer.declined is False


def test_each_article_is_listed_once_in_first_cited_order():
    answer = parse_answer("Connect it [2]. Then verify [1, 2]. Done [1].", IDS)
    assert answer.cited_ids == ("B", "A")


def test_grouped_citations_are_split():
    assert parse_answer("Both apply [1,3].", IDS).cited_ids == ("A", "C")


def test_numbers_outside_the_prompt_are_invalid():
    answer = parse_answer("See [4] and [0] and [2].", IDS)
    assert answer.cited_ids == ("B",)
    assert answer.invalid_citations == 2


def test_markdown_links_are_not_citations():
    answer = parse_answer("Use [Wix Payments](https://support.wix.com/x) [1].", IDS)
    assert answer.cited_ids == ("A",)
    assert answer.invalid_citations == 0


def test_closed_book_citations_are_all_invalid():
    answer = parse_answer("Go to Settings [1].", ())
    assert answer.cited_ids == ()
    assert answer.invalid_citations == 1


def test_an_exact_not_found_is_a_decline():
    answer = parse_answer(f"\n {NOT_FOUND}\n", IDS)
    assert answer.declined is True
    assert answer.cited_ids == ()


def test_not_found_inside_an_answer_is_not_a_decline():
    assert parse_answer(f"Partly: see [1]. {NOT_FOUND} for the rest.", IDS).declined is False
