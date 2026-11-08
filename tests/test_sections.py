from pathlib import Path

import pytest

from support_rag.records import Article
from support_rag.sections import (
    UNITS,
    cut_articles,
    heading_pieces,
    heading_starts,
    window_pieces,
    with_title,
)

ALL = {"h2", "h3", "h4"}


def _article(contents, html="", title="Title"):
    return Article("a", title, "", "article", contents, html)


# As in the real corpus: `contents` is the title, then the HTML's text with its spacing changed.
HTML = (
    "<div>Intro. Learn how to:</div><ul><li>Add a strip</li></ul>"
    '<div><h2 id="x" level="3">Add a strip</h2></div><div>Drag it in.</div>'
    "<h4>Why?</h4><div>Because.</div>"
    "<h2>Remove a strip</h2><div>Delete it.</div>"
)
CONTENTS = (
    "Title\nIntro. Learn how to:\nAdd a stripAdd a stripDrag it in.\n"
    "Why?Because.Remove a stripDelete it."
)


def test_heading_pieces_rejoin_to_the_exact_contents():
    article = _article(CONTENTS, HTML)
    assert "".join(heading_pieces(article, {"h2"})) == CONTENTS
    assert "".join(heading_pieces(article, ALL)) == CONTENTS


def test_a_heading_repeated_in_the_intro_list_cuts_at_the_real_heading():
    article = _article(CONTENTS, HTML)
    first_mention = CONTENTS.find("Add a strip")
    assert heading_starts(article, {"h2"})[0] == CONTENTS.find("Add a strip", first_mention + 1)


def test_the_unit_decides_which_headings_cut():
    article = _article(CONTENTS, HTML)
    assert heading_pieces(article, {"h2"}) == [
        "Title\nIntro. Learn how to:\nAdd a strip",
        "Add a stripDrag it in.\nWhy?Because.",
        "Remove a stripDelete it.",
    ]
    assert heading_pieces(article, ALL) == [
        "Title\nIntro. Learn how to:\nAdd a strip",
        "Add a stripDrag it in.\n",
        "Why?Because.",
        "Remove a stripDelete it.",
    ]


def test_an_article_with_no_headings_stays_in_one_piece():
    article = _article("Title\nJust text.", "<div>Just text.</div>")
    assert heading_pieces(article, ALL) == ["Title\nJust text."]


def test_an_empty_heading_marks_nothing():
    article = _article("Title\nOne.Two.", "<div>One.</div><h2> </h2><div>Two.</div>")
    assert heading_pieces(article, ALL) == ["Title\nOne.Two."]


def test_script_text_outside_contents_does_not_shift_the_cuts():
    html = "<script>var hidden = 1;</script><div>One.</div><h2>Two</h2><div>Three.</div>"
    assert heading_pieces(_article("Title\nOne.TwoThree.", html), ALL) == [
        "Title\nOne.",
        "TwoThree.",
    ]


def test_html_entities_count_as_the_characters_they_stand_for():
    html = "<div>Q&amp;A</div><h2>Next</h2>"
    assert heading_pieces(_article("Title\nQ&ANext", html), ALL) == ["Title\nQ&A", "Next"]


def test_html_that_disagrees_with_contents_is_refused():
    with pytest.raises(ValueError, match="isn't where it should be"):
        heading_pieces(_article("Title\nSomething else entirely.", HTML), ALL)


def test_windows_overlap_and_cover_every_word():
    text = " ".join(f"w{i}" for i in range(10))
    pieces = window_pieces(_article(text), size=4, overlap=1)
    assert pieces == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_a_window_stops_once_the_last_word_is_in():
    text = " ".join(f"w{i}" for i in range(7))
    assert window_pieces(_article(text), size=4, overlap=1) == ["w0 w1 w2 w3", "w3 w4 w5 w6"]


def test_an_article_shorter_than_a_window_is_one_piece_without_its_edge_spaces():
    assert window_pieces(_article(" one  two "), size=4, overlap=1) == ["one  two"]


def test_a_title_goes_in_front_of_a_piece_once():
    assert with_title("Title", "Title\nIntro.") == "Title\nIntro."
    assert with_title("Title", "To cancel:") == "Title\nTo cancel:"


def test_cut_articles_keeps_article_order_and_says_whose_piece_it_is():
    one = Article("a", "A", "", "article", "A\nOne.TwoThree.", "<div>One.</div><h2>Two</h2>")
    two = Article("b", "B", "", "article", "B\nFour.", "<div>Four.</div>")
    sections = cut_articles([one, two], "h2")
    assert [(s.article_id, s.text) for s in sections] == [
        ("a", "A\nOne."),
        ("a", "A\nTwoThree."),
        ("b", "B\nFour."),
    ]


def test_without_titles_only_the_first_piece_has_one():
    one = Article("a", "A", "", "article", "A\nOne.TwoThree.", "<div>One.</div><h2>Two</h2>")
    assert [s.text for s in cut_articles([one], "h2", title=False)] == ["A\nOne.", "TwoThree."]


def test_a_one_piece_article_keeps_the_text_it_had_as_a_whole_article():
    article = _article("Title\nJust text.", "<div>Just text.</div>")
    assert [s.text for s in cut_articles([article], "headings")] == [article.contents]


def test_the_cli_offers_every_unit():
    from support_rag import cli

    source = Path(cli.__file__).read_text(encoding="utf-8")
    for name in UNITS:
        assert f'"{name}"' in source


@pytest.mark.parametrize(
    "flags",
    [
        ["--retriever", "bm25", "--unit", "h2"],
        ["--retriever", "hybrid", "--unit", "h2"],
        ["--retriever", "rerank", "--unit", "h2", "--candidates", "both"],
        ["--retriever", "dense", "--no-title"],
        ["--retriever", "rerank", "--unit", "h2", "--no-title"],
        ["--retriever", "bm25", "--store", "postgres"],
        ["--retriever", "dense", "--unit", "h2", "--store", "postgres"],
        ["--retriever", "dense", "--index", "hnsw"],
    ],
)
def test_the_cli_refuses_section_options_that_would_be_ignored(flags):
    from support_rag.cli import main

    with pytest.raises(SystemExit, match="error"):
        main(["eval-retrieval", *flags])
