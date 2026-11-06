"""Sections: cut an article into pieces that each get their own vector (D-039).

One vector has to stand for everything in an article, while a question is usually about
one part of it. A section's vector stands for less, so it can match more sharply. An
article then scores as its best section, and the metrics still count articles.

Heading pieces rejoin to exactly the article's `contents`: only the unit changes, so a
difference from whole articles can't come from cleaning the text (D-012).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Sequence
from functools import partial
from html.parser import HTMLParser
from itertools import pairwise

from support_rag.records import Article, Section

WINDOW_WORDS = 200
WINDOW_OVERLAP = 40


class _HeadingFinder(HTMLParser):
    """Counts the non-space characters of the page's text as it goes, and notes the count
    at the start of each heading."""

    def __init__(self, levels: Collection[str]) -> None:
        super().__init__(convert_charrefs=True)
        self._levels = levels
        self._hidden = 0  # inside <script> or <style>, whose text isn't in `contents`
        self._open: str | None = None
        self._start = 0
        self._text: list[str] = []
        self.seen = 0
        self.headings: list[tuple[int, str]] = []  # (non-space characters before it, its text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._hidden += 1
        if tag in self._levels:
            self._open, self._start, self._text = tag, self.seen, []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._hidden -= 1
        if tag == self._open:
            if self._text:  # one heading in the corpus is empty, which marks nothing
                self.headings.append((self._start, "".join(self._text)))
            self._open = None

    def handle_data(self, data: str) -> None:
        if self._hidden:
            return
        compact = "".join(data.split())
        self.seen += len(compact)
        if self._open and compact:
            self._text.append(compact)


def heading_starts(article: Article, levels: Collection[str]) -> list[int]:
    """Where each heading starts in `contents`.

    Searching `contents` for a heading's text isn't safe: "In this article" lists repeat
    the headings, so a search can land in the list and cut there. The HTML's text is
    `contents` minus the title, apart from whitespace, so counting non-space characters
    gives the true position.

    Raises ValueError if a heading isn't where the count puts it: a knowledge base whose
    HTML and text disagree should stop here, not get cut in the wrong places.
    """
    finder = _HeadingFinder(levels)
    finder.feed(article.html)
    finder.close()
    # The nth non-space character of `contents` is at offsets[n].
    offsets = [i for i, char in enumerate(article.contents) if not char.isspace()]
    compact = "".join(article.contents.split())
    title = len("".join(article.title.split()))
    starts = []
    for seen, heading in finder.headings:
        if not compact.startswith(heading, title + seen):
            raise ValueError(f"article {article.id}: heading {heading!r} isn't where it should be")
        starts.append(offsets[title + seen])
    return starts


def heading_pieces(article: Article, levels: Collection[str]) -> list[str]:
    """`contents`, cut just before each heading. The pieces rejoin to exactly `contents`,
    and an article with no such headings stays in one piece."""
    cuts = [0, *heading_starts(article, levels), len(article.contents)]
    return [article.contents[start:end] for start, end in pairwise(cuts) if start < end]


def window_pieces(
    article: Article, *, size: int = WINDOW_WORDS, overlap: int = WINDOW_OVERLAP
) -> list[str]:
    """Runs of `size` words, each sharing `overlap` words with the one before, ignoring
    headings. Cut from `contents`, so the text inside a window is unchanged."""
    words = [match.span() for match in re.finditer(r"\S+", article.contents)]
    pieces = []
    for first in range(0, len(words), size - overlap):
        last = min(first + size, len(words)) - 1
        pieces.append(article.contents[words[first][0] : words[last][1]])
        if last == len(words) - 1:
            break
    return pieces


UNITS: dict[str, Callable[[Article], list[str]]] = {
    "h2": partial(heading_pieces, levels={"h2"}),
    "headings": partial(heading_pieces, levels={"h2", "h3", "h4"}),
    "window": window_pieces,
}


def with_title(title: str, piece: str) -> str:
    """The article's title, then the piece. A piece like "To cancel your site plan:" doesn't
    say which product it's about. The first piece already starts with the title."""
    return piece if piece.startswith(title) else f"{title}\n{piece}"


def cut_articles(articles: Sequence[Article], unit: str, *, title: bool = True) -> list[Section]:
    """Every article's sections, in article order."""
    cut = UNITS[unit]
    return [
        Section(article.id, with_title(article.title, piece) if title else piece)
        for article in articles
        for piece in cut(article)
    ]
