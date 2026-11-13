from support_rag.long_context import capped_context, estimate_request_tokens
from support_rag.records import Article


def _article(article_id, words):
    return Article(article_id, f"Title {article_id}", "", "article", " ".join(["w"] * words), "")


def count_words(text):
    """A stand-in for the token estimate: one token per word, so "[1] w w w" is 4."""
    return len(text.split())


def test_articles_go_in_best_first_while_they_fit():
    ranked = [_article("A", 29), _article("B", 49), _article("C", 19)]  # 30, 50 and 20 tokens
    context = capped_context(ranked, budget=100, count=count_words, used=10)
    assert [a.id for a in context] == ["A", "B"]  # 10 + 30 + 50 = 90; C would make 110


def test_the_list_stops_at_the_first_article_that_does_not_fit():
    # B doesn't fit, and C would, but C never takes the place of a better-ranked article.
    ranked = [_article("A", 29), _article("B", 69), _article("C", 9)]  # 30, 70 and 10 tokens
    context = capped_context(ranked, budget=100, count=count_words, used=10)
    assert [a.id for a in context] == ["A"]


def test_an_article_that_lands_exactly_on_the_budget_fits():
    ranked = [_article("A", 89)]  # 90 tokens
    assert len(capped_context(ranked, budget=100, count=count_words, used=10)) == 1


def test_the_tokens_already_used_count_against_the_budget():
    ranked = [_article("A", 89)]
    assert capped_context(ranked, budget=100, count=count_words, used=11) == []


def test_everything_fits_under_a_large_budget():
    ranked = [_article(article_id, 10) for article_id in "ABCDE"]
    assert len(capped_context(ranked, budget=10_000, count=count_words)) == 5


def test_the_estimate_counts_the_system_prompt_and_the_message():
    request = {"system": "one two", "messages": [{"role": "user", "content": "three four five"}]}
    assert estimate_request_tokens(request, count_words) == 5
