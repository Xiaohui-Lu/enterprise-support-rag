from support_rag.judge_eval import judge_request
from support_rag.panel import PANEL, format_agreement, panel_agreement, panel_requests
from support_rag.records import Article, Question
from support_rag.runs import append_result, start_run

ARTICLES = {a: Article(a, f"Title {a}", "", "article", f"Text {a}", "") for a in ("A", "B")}
QUESTIONS = {"q1": Question("q1", "expertwritten", "Question?", "Reference.", ("B",))}
ITEMS = [
    {
        "item": 1,
        "question_id": "q1",
        "strategy": "s1",
        "text": "Answer [2].",
        "context_ids": ["A", "B"],
        "cited_ids": ["B"],
        "check_grounded": False,
    }
]


def test_the_panel_sees_exactly_what_the_judge_sees():
    judge = judge_request({**ITEMS[0], "declined": False}, QUESTIONS["q1"], ARTICLES, effort="x")
    for request, (model, effort) in zip(
        panel_requests(ITEMS, QUESTIONS, ARTICLES), PANEL, strict=True
    ):
        assert (request["model"], request["output_config"]["effort"]) == (model, effort)
        same = {**request, "model": judge["model"]}
        same["output_config"] = {**request["output_config"], "effort": "x"}
        assert same == judge


def _member_runs(tmp_path, votes_by_item, failed=()):
    """One run per panel member; votes_by_item maps item -> four correctness grades."""
    run_dirs = []
    for n, (model, effort) in enumerate(PANEL):
        run_dir = start_run(f"m{n}", {"model": model, "effort": effort}, tmp_path)
        for item, votes in votes_by_item.items():
            grade = None if (item, n) in failed else votes[n]
            usage = {"output_tokens": 100}
            record = {"item": item, "correctness": grade, "grounded": "yes", "usage": usage}
            append_result(run_dir, record | {"cost_usd": 0.01})
        run_dirs.append(run_dir)
    return run_dirs


SAMPLE = {"items": [{"item": i, "strategy": "s0" if i < 3 else "s1"} for i in range(1, 6)]}
VOTES = {
    1: ["correct"] * 4,
    2: ["correct", "correct", "correct", "partial"],
    3: ["partial", "partial", "incorrect", "incorrect"],
    4: ["incorrect"] * 4,
    5: ["correct", "partial", "correct", "incorrect"],
}


def test_agreement_counts_unanimous_and_split_items(tmp_path):
    agreement = panel_agreement(_member_runs(tmp_path, VOTES), SAMPLE)
    c = agreement["correctness"]
    assert (c["unanimous"], c["split"]) == (2, 3)
    assert c["split_shapes"] == {"3-1": 1, "2-2": 1, "2-1-1": 1}
    assert c["odd_one_out_in_3_1"] == {"fable-5-1-max": 1}
    assert c["split_by_strategy"] == {"s0": "1 of 2", "s1": "2 of 3"}
    # Opus high and Opus max differ only on item 5.
    assert c["pairwise_agreement_pct"]["opus-5-5-high / opus-5-5-max"] == 80.0


def test_an_item_a_member_failed_is_left_out_of_agreement(tmp_path):
    agreement = panel_agreement(_member_runs(tmp_path, VOTES, failed={(1, 2)}), SAMPLE)
    assert agreement["judge_failed"]["fable-5-1-high"] == 1
    assert agreement["correctness"]["graded_by_all"] == 4


def test_the_printed_summary_never_shows_a_grade(tmp_path):
    text = format_agreement(panel_agreement(_member_runs(tmp_path, VOTES), SAMPLE))
    for grade in ("correct", "partial", "incorrect"):
        assert f"'{grade}'" not in text and f" {grade}" not in text.replace("Correctness", "")
