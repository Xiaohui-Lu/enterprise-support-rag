import random
from dataclasses import replace

from support_rag.answer_eval import answer_question, summarize_answers
from support_rag.context_rot import (
    RANDOM_POOL,
    gold_depths,
    hard_pool,
    question_rng,
    random_pool,
    rot_context,
)
from support_rag.llm import LLMResult
from support_rag.records import Article, Hit, Question


def _article(article_id, words=9):
    return Article(article_id, article_id, "", "article", " ".join(["w"] * words), "")


def count_words(text):
    """A stand-in for the token estimate: "[1] w w w" is 4 tokens."""
    return len(text.split())


GOLD = _article("G")  # 10 tokens with its number
POOL = [_article(f"D{i}") for i in range(1, 11)]  # 10 tokens each


def _ids(context):
    return [a.id for a in context]


def test_distractors_fill_the_prompt_up_to_the_size():
    # 20 used + 10 of gold leaves room for 4 distractors of 10 under a size of 70.
    context = rot_context(
        [GOLD], [0.0], POOL, size=70, count=count_words, used=20, rng=random.Random(0)
    )
    assert len(context) == 5
    assert set(_ids(context)) == {"G", "D1", "D2", "D3", "D4"}  # from the start of the pool


def test_a_gold_article_has_its_depth_share_of_distractors_in_front():
    def gold_at(depth):
        context = rot_context(
            [GOLD], [depth], POOL, size=70, count=count_words, used=20, rng=random.Random(0)
        )
        return _ids(context).index("G")

    assert gold_at(0.0) == 0
    assert gold_at(0.5) == 2  # 2 of the 4 distractors in front
    assert gold_at(0.99) == 4  # last


def test_two_gold_articles_each_keep_their_own_depth():
    second = _article("H")
    context = rot_context(
        [GOLD, second], [0.75, 0.0], POOL, size=80, count=count_words, used=20,
        rng=random.Random(0),
    )  # fmt: skip
    ids = _ids(context)
    assert len(ids) == 6  # 4 distractors
    assert ids.index("H") == 0
    assert ids.index("G") == 4  # 3 distractors and H in front


def test_a_larger_size_keeps_the_smaller_ones_distractors():
    def distractors(size):
        context = rot_context(
            [GOLD], [0.5], POOL, size=size, count=count_words, used=20, rng=random.Random(1)
        )
        return set(_ids(context)) - {"G"}

    assert distractors(50) < distractors(90)


def test_the_shuffle_depends_only_on_the_seed():
    def order(seed):
        return _ids(
            rot_context(
                [GOLD], [0.5], POOL, size=130, count=count_words, used=20, rng=random.Random(seed)
            )
        )

    assert order(3) == order(3)
    assert order(3) != order(4)


def test_gold_only_when_no_distractor_fits():
    context = rot_context(
        [GOLD], [0.5], POOL, size=35, count=count_words, used=20, rng=random.Random(0)
    )
    assert _ids(context) == ["G"]


def test_a_questions_draws_do_not_depend_on_the_rest_of_the_run():
    question = Question("ew-001", "expertwritten", "q", "", ("G", "H"))
    assert gold_depths(question) == gold_depths(question)
    assert question_rng("ew-001", "depths").random() == gold_depths(question)[0]
    assert question_rng("ew-001", "random").random() != question_rng("ew-002", "random").random()


def test_the_random_pool_never_holds_a_gold_article():
    articles = {f"A{i}": _article(f"A{i}") for i in range(RANDOM_POOL + 5)}
    question = Question("ew-001", "expertwritten", "q", "", ("A1", "A2"))
    pool = random_pool(question, articles)
    assert len(pool) == RANDOM_POOL
    assert not {"A1", "A2"} & set(_ids(pool))
    assert _ids(pool) == _ids(random_pool(question, articles))


def test_the_hard_pool_is_the_ranking_without_the_gold():
    class Ranked:
        def search(self, query, k=10):
            return [Hit(f"A{i}", 1 / i) for i in range(1, k + 1)]

    articles = {f"A{i}": _article(f"A{i}") for i in range(1, 300)}
    question = Question("ew-001", "expertwritten", "q", "", ("A2",))
    pool = hard_pool(question, Ranked(), articles)
    assert _ids(pool)[:3] == ["A1", "A3", "A4"]
    assert "A2" not in _ids(pool)


def test_a_summary_counts_answers_that_cite_a_distractor():
    class FakeLLM:
        def __init__(self, text):
            self.text = text

        def complete(self, request):
            usage = {"input_tokens": 40_000, "output_tokens": 500}
            return LLMResult(self.text, "end_turn", usage, 0.0, True, request["model"], batch=True)

    question = Question("ew-001", "expertwritten", "q", "", ("G",))
    rot = {
        "distractors": "hard",
        "size": 40_000,
        "n_distractors": 1,
        "gold_depths": [0.5],
        "gold_positions": [2],
        "estimated_prompt_tokens": 39_000,
    }
    results = [
        answer_question(FakeLLM(text), question, [POOL[0], GOLD], effort="medium")
        for text in ("See [1].", "See [2].")
    ]
    summary = summarize_answers([replace(r, rot=rot) for r in results])["rot"]
    assert summary["cited_a_distractor"] == 1
    assert summary["over_price_line"] == 0
    assert summary["estimate_error"]["max"] > 0  # the estimate was 1,000 low
