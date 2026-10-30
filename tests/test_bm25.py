from support_rag.bm25 import BM25Retriever, tokenize
from support_rag.records import Article


def _article(article_id: str, contents: str) -> Article:
    return Article(article_id, contents, "", "article", contents, "")


def _simple(text: str) -> list[str]:
    return text.lower().split()


# tokenize()


def test_tokenize_lowercases():
    assert tokenize("Domain", stem=False) == ["domain"]


def test_tokenize_splits_glued_sentences():
    assert tokenize("switch to ADI.Choose a template", stem=False) == [
        "switch",
        "adi",
        "choose",
        "template",
    ]


def test_tokenize_keeps_numbers():
    assert tokenize("error 404", stem=False) == ["error", "404"]


def test_tokenize_treats_non_breaking_space_as_a_separator():
    assert tokenize("Wix\u00a0Editor", stem=False) == ["wix", "editor"]


def test_tokenize_drops_stopwords():
    assert tokenize("the domain is not connected", stem=False) == ["domain", "connected"]


def test_tokenize_stems_by_default():
    assert tokenize("connecting domains") == ["connect", "domain"]


def test_tokenize_without_stemming_keeps_word_forms():
    assert tokenize("connecting domains", stem=False) == ["connecting", "domains"]


# BM25Retriever: these use a plain split tokenizer, so they don't depend on tokenize().


def test_search_ranks_the_best_match_first():
    articles = [
        _article("buy", "buy a domain from wix"),
        _article("connect", "connect a domain you bought elsewhere"),
        _article("font", "change the font on your site"),
    ]
    hits = BM25Retriever(articles, tokenizer=_simple).search("connect domain", k=3)
    assert [hit.article_id for hit in hits] == ["connect", "buy"]


def test_search_returns_no_hits_when_no_word_matches():
    articles = [_article("a", "connect a domain"), _article("b", "change the font")]
    assert BM25Retriever(articles, tokenizer=_simple).search("payout") == []


def test_search_caps_k_at_the_corpus_size():
    articles = [_article("a", "connect a domain"), _article("b", "domain settings")]
    hits = BM25Retriever(articles, tokenizer=_simple).search("domain", k=10)
    assert len(hits) == 2


def test_articles_and_queries_share_the_stemming_tokenizer():
    articles = [_article("a", "Connecting a domain"), _article("b", "Changing your font")]
    hits = BM25Retriever(articles).search("connect domains")
    assert hits[0].article_id == "a"
