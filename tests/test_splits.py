import pytest

from support_rag.records import Question
from support_rag.splits import make_splits, write_splits


def make_questions(prefix: str, subset: str, n: int, n_multi: int) -> list[Question]:
    return [
        Question(
            id=f"{prefix}-{i:04d}",
            subset=subset,
            question=f"question {i}",
            answer="answer",
            gold_ids=("k1", "k2") if i < n_multi else ("k1",),
        )
        for i in range(n)
    ]


def benchmark_shaped_inputs() -> list[list[Question]]:
    """Same sizes and multi-article counts as WixQA's real subsets."""
    return [
        make_questions("ew", "expertwritten", 200, 52),
        make_questions("sim", "simulated", 200, 27),
        make_questions("syn", "synthetic", 1000, 0),
    ]


@pytest.fixture
def splits() -> dict:
    return make_splits(*benchmark_shaped_inputs())


def test_split_sizes(splits):
    counts = splits["counts"]
    assert counts["dev_real"] == 50
    assert counts["test"] == 350
    assert counts["dev_synthetic"] == 500
    assert counts["synthetic_reference"] == 500


def test_no_question_is_in_two_splits_or_lost(splits):
    dev, test = set(splits["dev_real"]), set(splits["test"])
    assert not dev & test
    assert len(dev | test) == 400
    assert not set(splits["dev_synthetic"]) & set(splits["synthetic_reference"])


def test_dev_keeps_the_multi_article_share(splits):
    # expertwritten: 25 * 52/200 = 6.5, rounds to 6 (Python rounds halves to even)
    # simulated:     25 * 27/200 = 3.4, rounds to 3
    assert splits["counts"]["dev_real_multi_article"] == 9
    assert splits["counts"]["test_multi_article"] == 79 - 9


def test_same_seed_gives_the_same_split():
    inputs = benchmark_shaped_inputs()
    assert make_splits(*inputs, seed=1) == make_splits(*inputs, seed=1)
    assert make_splits(*inputs, seed=1)["test"] != make_splits(*inputs, seed=2)["test"]


def test_existing_split_is_never_overwritten(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="locked"):
        write_splits(path=path)
