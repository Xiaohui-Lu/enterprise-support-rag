from support_rag.charts import answers_chart, rot_chart


def _strategy(short_name, cost, pct, latency_ms):
    return {"short_name": short_name, "cost_per_answer_usd": cost, "pct_correct": pct,
            "ci": (pct - 10, pct + 10), "latency_p50_ms": latency_ms}  # fmt: skip


ANSWERS = {
    "strategies": {
        "s0": _strategy("S0", 0.0004, 30.0, 4000.0),
        "s1-dense": _strategy("S1 dense", 0.0012, 80.0, 3500.0),
        "s3-k100": _strategy("S3 K=100", 0.008, 81.0, 5000.0),
        "s_star": _strategy("S*", 0.0004, 75.0, 2500.0),
    },
    "frontier": {"cost": {"keys": ["s0", "s1-dense"]}, "latency": {"keys": ["s1-dense"]}},
}


def _rot_row(arm, tokens, pct):
    return {"arm": arm, "mean_prompt_tokens": tokens, "pct": pct, "ci": (pct - 8, pct + 8)}


ROT = {
    "rows": [
        _rot_row(("gold only", 0), 1500, 75.0),
        _rot_row(("hard", 10_000), 8500, 72.0),
        _rot_row(("random", 10_000), 9200, 74.0),
        _rot_row(("hard", 90_000), 89_000, 64.0),
        _rot_row(("random", 90_000), 88_600, 70.0),
    ]
}


def test_a_rerun_draws_the_same_answers_chart(tmp_path):
    first = answers_chart(ANSWERS, tmp_path / "a.svg").read_bytes()
    second = answers_chart(ANSWERS, tmp_path / "b.svg").read_bytes()
    assert first == second
    assert b"S1 dense" in first  # names stay text


def test_a_rerun_draws_the_same_rot_chart(tmp_path):
    first = rot_chart(ROT, tmp_path / "a.svg").read_bytes()
    assert first == rot_chart(ROT, tmp_path / "b.svg").read_bytes()
    assert b"Gold only" in first
