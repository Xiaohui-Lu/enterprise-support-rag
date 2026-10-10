from support_rag.corpus_stats import build_report
from support_rag.wixqa import Article, Question


def count_words(text: str) -> int:
    return len(text.split())


def article(article_id: str, article_type: str = "article", text: str = "one two three") -> Article:
    return Article(
        id=article_id,
        title=article_id,
        url=f"https://example.com/{article_id}",
        article_type=article_type,
        contents=text,
        html="<h2>Steps</h2><p>...</p><h3>Notes</h3>",
    )


def question(question_id: str, subset: str, gold: list[str], text: str) -> Question:
    return Question(
        id=question_id, subset=subset, question=text, answer="Do this.", gold_ids=tuple(gold)
    )


def sample_report() -> str:
    articles = [article("k1"), article("k2", "feature_request"), article("k3", text="word " * 50)]
    questions = {
        "expertwritten": [
            question("ew-000", "expertwritten", ["k1"], "How do I add a page?"),
            question("ew-001", "expertwritten", ["k1", "k2"], "Can I restore deleted products?"),
        ],
        "simulated": [question("sim-000", "simulated", ["gone"], "How do I add a page?")],
        "synthetic": [
            question("syn-0000", "synthetic", ["k1"], "What does k1 cover?"),
            question("syn-0001", "synthetic", ["k2"], "What does k2 cover?"),
        ],
    }
    return build_report(articles, questions, count_tokens=count_words)


def test_report_has_every_section():
    report = sample_report()
    for heading in ("## Knowledge base", "## Questions", "## Gold articles", "## Data checks"):
        assert heading in report


def test_report_counts_feature_request_gold_articles():
    assert "at least one feature request among their gold articles: 1 of 3" in sample_report()


def test_data_checks_catch_problems():
    report = sample_report()
    assert "| simulated: gold IDs missing from the KB | 1 |" in report
    assert "| Questions in both expertwritten and simulated | 1 |" in report
    assert "| KB articles with no synthetic question (expected: few) | 1 |" in report
