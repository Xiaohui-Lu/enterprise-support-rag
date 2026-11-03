import pytest

from support_rag.labels import draw_sample, label_session, load_labels, write_sample
from support_rag.records import Article, Question
from support_rag.runs import append_result, start_run

QUESTION_IDS = [f"q{n}" for n in range(10)]
ARTICLES = {a: Article(a, f"Title {a}", "", "article", f"Text {a}", "") for a in ("A", "B")}
QUESTIONS = {
    q: Question(q, "expertwritten", f"Question {q}?", f"Reference {q}.", ("A",))
    for q in QUESTION_IDS
}


def _answer(question_id, *, declined=False, cited=("B",)):
    return {
        "question_id": question_id,
        "text": "NOT_FOUND" if declined else f"Answer to {question_id} [2].",
        "context_ids": ["A", "B"],
        "cited_ids": [] if declined else list(cited),
        "declined": declined,
    }


def _answers():
    return {
        "s0": [_answer(q, cited=()) for q in QUESTION_IDS],
        "s1": [_answer(q, declined=q == "q3") for q in QUESTION_IDS],
        "s_star": [_answer(q) for q in QUESTION_IDS],
    }


def _sample(**kwargs):
    return draw_sample(_answers(), **{"target": 12, "n_grounded": 4, "exclude": (), **kwargs})


class Replies:
    """Stands in for input(): returns the scripted replies, then quits."""

    def __init__(self, *replies):
        self.replies = list(replies)

    def __call__(self, prompt):
        return self.replies.pop(0) if self.replies else "q"


def _label(sample, path, *replies):
    shown = []
    labeled = label_session(
        sample, QUESTIONS, ARTICLES, labels_path=path, ask=Replies(*replies), show=shown.append
    )
    return labeled, "\n".join(shown)


# draw_sample()


def test_sample_takes_whole_questions_and_the_reserve_shares_none():
    sample = _sample()
    sampled = {item["question_id"] for item in sample["items"]}
    assert sampled == set(sample["labeled_questions"])
    assert not sampled & set(sample["reserve_questions"])
    assert len(sample["items"]) >= 12


def test_declines_are_never_sampled():
    sample = _sample(target=100)
    assert all(item["text"] != "NOT_FOUND" for item in sample["items"])
    assert len(sample["items"]) == 29  # 30 answers, one declined


def test_an_excluded_question_is_neither_labeled_nor_in_reserve():
    sample = _sample(exclude=("q5",))
    assert "q5" not in sample["labeled_questions"] + sample["reserve_questions"]


def test_the_same_seed_draws_the_same_sample():
    assert _sample() == _sample()
    assert _sample()["items"] != _sample(seed=1)["items"]


def test_groundedness_is_checked_only_on_answers_with_citations():
    checked = [item for item in _sample()["items"] if item["check_grounded"]]
    assert len(checked) == 4
    assert all(item["cited_ids"] for item in checked)


def test_runs_over_different_questions_are_rejected():
    answers = _answers()
    answers["s0"] = answers["s0"][:5]
    with pytest.raises(ValueError):
        draw_sample(answers)


def test_an_existing_sample_is_never_overwritten(tmp_path):
    runs = []
    for strategy, answers in _answers().items():
        config = {"kind": "answers", "strategy": strategy, "split": "dev_real", "model": "m"}
        run_dir = start_run(
            strategy, {**config, "effort": "medium", "prompt_version": "v"}, tmp_path
        )
        for answer in answers:
            append_result(run_dir, answer)
        runs.append(run_dir)
    path = tmp_path / "sample.json"
    write_sample(runs, path)
    with pytest.raises(FileExistsError):
        write_sample(runs, path)


# label_session()


def test_labels_are_saved_as_given_and_a_session_resumes(tmp_path):
    sample, path = _sample(n_grounded=0), tmp_path / "labels.jsonl"
    assert _label(sample, path, "c", "P skips a step")[0] == 2
    first, second = sample["items"][:2]
    labels = load_labels(path)
    assert labels[first["item"]]["correctness"] == "correct"
    assert labels[second["item"]]["correctness"] == "partial"
    assert labels[second["item"]]["note"] == "skips a step"

    _label(sample, path, "i")  # resumes at the third item
    assert load_labels(path)[sample["items"][2]["item"]]["correctness"] == "incorrect"


def test_undo_relabels_the_previous_item(tmp_path):
    sample, path = _sample(n_grounded=0), tmp_path / "labels.jsonl"
    _label(sample, path, "c", "u", "i")
    assert load_labels(path)[sample["items"][0]["item"]]["correctness"] == "incorrect"


def test_a_reply_without_a_valid_grade_is_asked_again(tmp_path):
    sample, path = _sample(n_grounded=0), tmp_path / "labels.jsonl"
    _label(sample, path, "x", "correct-ish", "", "p")
    assert load_labels(path)[sample["items"][0]["item"]]["correctness"] == "partial"


def test_a_grounded_check_shows_the_cited_articles_by_their_number(tmp_path):
    citing = {"s_star": _answers()["s_star"]}  # every answer cites [2]
    sample = draw_sample(citing, target=3, n_grounded=3, exclude=())
    path = tmp_path / "labels.jsonl"
    _, shown = _label(sample, path, "c", "n")
    first = sample["items"][0]
    assert first["check_grounded"]
    assert "[2] Title B\nText B" in shown
    assert load_labels(path)[first["item"]]["grounded"] == "no"


def test_the_labeler_never_sees_the_strategy(tmp_path):
    sample, path = _sample(n_grounded=4), tmp_path / "labels.jsonl"
    _, shown = _label(sample, path, *["c", "y"] * 20)
    for hint in ("s0", "s1", "s_star", "strategy", "closed book"):
        assert hint not in shown.lower()
